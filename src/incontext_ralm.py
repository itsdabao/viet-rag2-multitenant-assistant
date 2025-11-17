import json
import os
import time
from typing import List, Dict, Tuple

import numpy as np
from llama_index.core import Settings, VectorStoreIndex
from src.config import (
    USE_BM25,
    BM25_TOP_K,
    HYBRID_ALPHA,
    BM25_MAX_CHARS,
    DEBUG_VERBOSE,
    DEBUG_TOPN_PRINT,
    DEBUG_SHOW_PROMPT,
    MAX_PROMPT_CHARS,
    PER_CHUNK_PROMPT_MAX_CHARS,
    PROMPT_TOP_CONTEXTS,
    HISTORY_ENABLED,
    HISTORY_MAX_TURNS,
    HISTORY_MSG_MAX_CHARS,
    RERANK_USE_COSINE,
    RERANK_TOP_M,
    RERANK_WEIGHT,
    LLM_MAX_RETRIES,
    LLM_RETRY_INITIAL_DELAY,
    LLM_RETRY_BACKOFF,
    SYSTEM_PROMPT_PATH,
)
from src.lexical_bm25 import bm25_retrieve, bm25_retrieve_debug


def _embed(text: str) -> np.ndarray:
    model = Settings.embed_model
    if model is None:
        raise RuntimeError("Embed model is not initialized. Call setup_embedding() first.")
    # HuggingFaceEmbedding exposes get_text_embedding
    if hasattr(model, "get_text_embedding"):
        vec = model.get_text_embedding(text)
    elif hasattr(model, "embed"):
        vec = model.embed(text)
    else:
        raise RuntimeError("Embed model does not support text embedding.")
    return np.array(vec, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _dbg_print(header: str):
    if DEBUG_VERBOSE:
        print(f"[DEBUG] {header}")


def _dbg_block(lines: List[str]):
    if DEBUG_VERBOSE:
        for ln in lines:
            print(f"[DEBUG] {ln}")


def load_fewshot_examples(path: str) -> List[Dict[str, str]]:
    """Load few-shot Q/A examples from JSON file. Schema: [{"question": str, "answer": str}, ...]"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out: List[Dict[str, str]] = []
        for item in data if isinstance(data, list) else []:
            q = item.get("question")
            a = item.get("answer")
            if isinstance(q, str) and isinstance(a, str):
                out.append({"question": q, "answer": a})
        return out
    except Exception:
        return []


def rank_examples_by_similarity(query: str, examples: List[Dict[str, str]], top_k: int) -> List[Dict[str, str]]:
    if not examples or top_k <= 0:
        return []
    q_vec = _embed(query)
    scored: List[Tuple[float, Dict[str, str]]] = []
    for ex in examples:
        try:
            s = _cosine(q_vec, _embed(ex["question"]))
            scored.append((s, ex))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:top_k]]


def _load_system_prompt(path: str) -> str:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    return txt
    except Exception:
        pass
    return (
        "Bạn là trợ lý RAG nói tiếng Việt. Trả lời ngắn gọn, rõ ràng, "
        "dựa trên ngữ cảnh được cung cấp; nếu không đủ thông tin thì nói không đủ dữ liệu; "
        "không bịa; luôn trích nguồn cuối câu trả lời."
    )


def _render_history(history: List[Dict[str, str]] | None) -> List[str]:
    if not HISTORY_ENABLED or not history:
        return []
    # Lấy N message gần nhất
    recent = history[-HISTORY_MAX_TURNS:]
    lines: List[str] = ["Conversation History:"]
    for i, m in enumerate(recent, 1):
        role = m.get("role", "user").lower()
        content = (m.get("content", "") or "")[:HISTORY_MSG_MAX_CHARS]
        tag = "User" if role == "user" else "Assistant"
        lines.append(f"{tag}: {content}")
    return lines


def build_prompt(query: str, examples: List[Dict[str, str]], retrieved_texts: List[str], history: List[Dict[str, str]] | None = None) -> str:
    lines: List[str] = []
    # System instruction
    lines.append(_load_system_prompt(SYSTEM_PROMPT_PATH))
    # History (nếu có)
    hist_lines = _render_history(history)
    if hist_lines:
        lines.append("")
        lines.extend(hist_lines)
    if examples:
        lines.append("\nExamples:")
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i} - Q: {ex['question']}")
            lines.append(f"Example {i} - A: {ex['answer']}")
    if retrieved_texts:
        lines.append("\nRetrieved Knowledge:")
        for i, t in enumerate(retrieved_texts, 1):
            lines.append(f"[Doc {i}] {t}")
    lines.append("\nUser Question:")
    lines.append(query)
    lines.append("\nAnswer clearly and concisely. If unsure, say you are unsure.")
    return "\n".join(lines)


def _extract_meta(n) -> Dict[str, str]:
    meta = {}
    try:
        meta = n.metadata
    except Exception:
        try:
            meta = n.node.metadata
        except Exception:
            meta = {}
    return meta if isinstance(meta, dict) else {}


def _vector_retrieve(index: VectorStoreIndex, user_query: str, top_k: int, tenant_id: str | None = None) -> List[Dict[str, object]]:
    # Try apply metadata filter if available; otherwise post-filter
    try:
        retriever = index.as_retriever(similarity_top_k=top_k)
        # Some versions of LlamaIndex support setting filters on retriever
        if tenant_id and hasattr(retriever, "metadata_filters"):
            # best-effort; API differences across versions
            try:
                from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

                retriever.metadata_filters = MetadataFilters(filters=[ExactMatchFilter(key="tenant_id", value=tenant_id)])
            except Exception:
                pass
    except Exception:
        retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(user_query)
    results: List[Dict[str, object]] = []
    for n in nodes:
        try:
            txt = n.get_text()
        except Exception:
            txt = getattr(n, "text", "")
        score = 0.0
        try:
            score = float(getattr(n, "score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        results.append({"text": (txt or "")[:1200], "score": score, "meta": _extract_meta(n)})
    # Post-filter by tenant if needed
    if tenant_id:
        filtered = []
        for r in results:
            m = r.get("meta", {}) or {}
            if m.get("tenant_id") == tenant_id:
                filtered.append(r)
        return filtered
    return results


def _hybrid_fuse(vec_res: List[Dict[str, object]], bm25_res: List[Dict[str, object]], *, final_top_k: int) -> List[Dict[str, object]]:
    def _normalize(res: List[Dict[str, object]]):
        max_s = max((float(r.get("score", 0.0)) for r in res), default=0.0)
        for r in res:
            s = float(r.get("score", 0.0))
            r["_norm"] = (s / max_s) if max_s > 0 else 0.0

    _normalize(vec_res)
    _normalize(bm25_res)

    fused: Dict[str, Dict[str, object]] = {}
    for r in vec_res:
        key = r["text"]
        fused[key] = {"text": r["text"], "meta": r.get("meta", {}), "score": HYBRID_ALPHA * r["_norm"]}
    for r in bm25_res:
        key = r["text"]
        if key in fused:
            fused[key]["score"] += (1.0 - HYBRID_ALPHA) * r["_norm"]
        else:
            fused[key] = {"text": r["text"], "meta": r.get("meta", {}), "score": (1.0 - HYBRID_ALPHA) * r["_norm"]}

    fused_list = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return fused_list[:final_top_k]


def query_with_incontext_ralm(
    user_query: str,
    index: VectorStoreIndex,
    *,
    fewshot_path: str,
    top_k_ctx: int = 5,
    top_k_examples: int = 3,
    tenant_id: str | None = None,
    history: List[Dict[str, str]] | None = None,
):
    # Retrieve knowledge: vector + optional BM25, then fuse
    _dbg_print("Starting retrieval pipeline")
    _dbg_block([
        f"Query: {user_query}",
        f"Vector top_k={top_k_ctx}, BM25 enabled={USE_BM25}, bm25_top_k={BM25_TOP_K}",
    ])
    vec_res = _vector_retrieve(index, user_query, top_k_ctx, tenant_id=tenant_id)
    bm25_res: List[Dict[str, object]] = []
    if USE_BM25:
        bm25_dbg = bm25_retrieve_debug(user_query, top_k=BM25_TOP_K, max_chars=BM25_MAX_CHARS, tenant_id=tenant_id)
        bm25_tokens = bm25_dbg.get("q_tokens", [])
        bm25_res = bm25_dbg.get("results", [])
        _dbg_block(["BM25 tokens: " + str(bm25_tokens), "BM25 top results (score, src):"])
        for r in bm25_res[:DEBUG_TOPN_PRINT]:
            m = r.get("meta", {}) or {}
            src = m.get("file_name") or m.get("file_path") or "unknown"
            _dbg_block([f"  {r.get('score', 0.0):.4f} | {src}"])
    _dbg_block(["Vector top results (score, src):"])
    for r in vec_res[:DEBUG_TOPN_PRINT]:
        m = r.get("meta", {}) or {}
        src = m.get("file_name") or m.get("file_path") or "unknown"
        _dbg_block([f"  {r.get('score', 0.0):.4f} | {src}"])
    fused = _hybrid_fuse(vec_res, bm25_res, final_top_k=top_k_ctx)
    _dbg_block(["Fused results (norm-weighted score, src):"])
    for r in fused[:DEBUG_TOPN_PRINT]:
        m = r.get("meta", {}) or {}
        src = m.get("file_name") or m.get("file_path") or "unknown"
        _dbg_block([f"  {r.get('score', 0.0):.4f} | {src}"])

    # Optional rerank by cosine to query
    if RERANK_USE_COSINE and fused:
        try:
            q_vec = _embed(user_query)
            # take top M to re-score
            pool = fused[: max(RERANK_TOP_M, len(fused))]
            # normalize fused score
            max_f = max((float(it.get("score", 0.0)) for it in pool), default=0.0)
            rescored = []
            cos_scores = []
            for it in pool:
                cos = _cosine(q_vec, _embed(it["text"]))
                cos_scores.append(cos)
            # min-max normalize cos
            if cos_scores:
                mn, mx = min(cos_scores), max(cos_scores)
                rng = (mx - mn) if mx > mn else 1.0
                cos_norm = [ (c - mn) / rng for c in cos_scores ]
            else:
                cos_norm = []
            for i, it in enumerate(pool):
                f_norm = (float(it.get("score", 0.0)) / max_f) if max_f > 0 else 0.0
                combined = (1.0 - RERANK_WEIGHT) * f_norm + RERANK_WEIGHT * (cos_norm[i] if i < len(cos_norm) else 0.0)
                rescored.append({**it, "score": combined})
            rescored.sort(key=lambda x: x["score"], reverse=True)
            fused = rescored[: top_k_ctx]
            _dbg_block(["After rerank (combined score, src):"])
            for r in fused[:DEBUG_TOPN_PRINT]:
                m = r.get("meta", {}) or {}
                src = m.get("file_name") or m.get("file_path") or "unknown"
                _dbg_block([f"  {r.get('score', 0.0):.4f} | {src}"])
        except Exception as e:
            _dbg_block([f"Rerank failed: {e}"])

    # Clamp exact top-N contexts for the prompt (post-rerank)
    selected = fused[: max(0, min(PROMPT_TOP_CONTEXTS, len(fused)))]

    # Apply prompt budget on selected contexts
    retrieved_texts: List[str] = []
    budget = MAX_PROMPT_CHARS
    overhead = 1200  # rough headroom for instructions + examples
    budget = max(1000, budget - overhead)
    used = 0
    for it in selected:
        t = it.get("text", "")
        if not t:
            continue
        t = t[:PER_CHUNK_PROMPT_MAX_CHARS]
        if used + len(t) > budget:
            break
        retrieved_texts.append(t)
        used += len(t)

    # Select few-shot examples
    examples_all = load_fewshot_examples(fewshot_path)
    examples = rank_examples_by_similarity(user_query, examples_all, top_k_examples)
    if examples:
        _dbg_block(["Few-shot selected (Q -> A):"])
        for i, ex in enumerate(examples[:DEBUG_TOPN_PRINT], 1):
            _dbg_block([f"  {i}. Q: {ex['question']}", f"     A: {ex['answer']}"])

    # Build prompt and call LLM
    # Guardrail: if no context and no examples, avoid calling LLM
    if not retrieved_texts and not examples:
        return {"answer": "Sorry, I don't have enough context to answer.", "sources": []}

    prompt = build_prompt(user_query, examples, retrieved_texts, history=history)
    if DEBUG_SHOW_PROMPT:
        head = prompt[:600].replace("\n", " ")
        _dbg_block([f"Prompt head(600): {head} ...", f"Prompt length: {len(prompt)} chars"])
    llm = Settings.llm
    if llm is None:
        raise RuntimeError("LLM is not initialized. Call init_llm_from_env() first.")
    # LLM call with retry/backoff
    delay = LLM_RETRY_INITIAL_DELAY
    last_err = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = llm.complete(prompt)
            break
        except Exception as e:
            last_err = e
            _dbg_block([f"LLM call failed (attempt {attempt+1}/{LLM_MAX_RETRIES}): {e}"])
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= LLM_RETRY_BACKOFF
            else:
                raise
    answer_text = getattr(resp, "text", str(resp))

    # Collect sources from fused items
    sources: List[str] = []
    for r in fused:
        m = r.get("meta", {}) or {}
        src = m.get("file_name") or m.get("file_path") or m.get("source") or "unknown"
        sources.append(str(src))

    return {"answer": answer_text, "sources": sources}
