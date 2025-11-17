import math
import re
import os
import json
from typing import List, Dict, Tuple, Optional

from llama_index.core.node_parser import SentenceSplitter
from src.data_loader import load_documents
from src.config import (
    DATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BM25_SOURCE,
    NODES_CACHE_PATH,
    BM25_K1,
    BM25_B,
)


def _tokenize(text: str) -> List[str]:
    # Simple unicode-aware word tokenizer
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def _split_into_chunks(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    if not text:
        return []
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = 0
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.tf: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}

    def build(self, texts: List[str]):
        self.N = len(texts)
        self.doc_len = []
        self.tf = []
        self.df = {}
        for t in texts:
            tokens = _tokenize(t)
            self.doc_len.append(len(tokens))
            tf_doc: Dict[str, int] = {}
            for tok in tokens:
                tf_doc[tok] = tf_doc.get(tok, 0) + 1
            self.tf.append(tf_doc)
            for tok in tf_doc.keys():
                self.df[tok] = self.df.get(tok, 0) + 1
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) if self.N else 0.0

    def score(self, q_tokens: List[str], idx: int) -> float:
        dl = self.doc_len[idx] or 1
        tf_doc = self.tf[idx]
        score = 0.0
        for term in q_tokens:
            f = tf_doc.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * (f * (self.k1 + 1)) / (denom or 1)
        return score

    def query(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.N == 0:
            return []
        q_tokens = _tokenize(query)
        scored: List[Tuple[int, float]] = []
        for i in range(self.N):
            s = self.score(q_tokens, i)
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# Cache theo tenant_id (None -> default)
_BM25_CACHE: Dict[Optional[str], Dict[str, object]] = {}


def _load_from_nodes_cache(path: str = NODES_CACHE_PATH) -> Optional[Tuple[list, list]]:
    if not os.path.exists(path):
        return None
    texts: list = []
    metas: list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get('text') or ''
            meta = obj.get('metadata') or {}
            texts.append(text)
            metas.append(meta if isinstance(meta, dict) else {})
    return texts, metas


def get_bm25_state(data_path: str = DATA_PATH, max_chars: int = 800, *, tenant_id: Optional[str] = None):
    global _BM25_CACHE
    state = _BM25_CACHE.get(tenant_id)
    if state and state.get("index") is not None:
        return state

    texts: List[str] = []
    metas: List[Dict[str, str]] = []

    # Preferred source: persisted nodes file
    if BM25_SOURCE == "nodes_file":
        # chọn path theo tenant nếu có
        cache_path = NODES_CACHE_PATH
        if tenant_id:
            base = os.path.dirname(NODES_CACHE_PATH)
            cache_path = os.path.join(base, tenant_id, "nodes.jsonl")
        loaded = _load_from_nodes_cache(cache_path)
        if loaded is not None:
            texts, metas = loaded
        else:
            # Fallback: rebuild from files using the same SentenceSplitter
            docs = load_documents(data_path)
            splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            nodes = splitter.get_nodes_from_documents(docs)
            for n in nodes:
                try:
                    t = n.get_text()
                except Exception:
                    t = getattr(n, 'text', '')
                md = {}
                try:
                    md = n.metadata
                    if not isinstance(md, dict):
                        md = {}
                except Exception:
                    md = {}
                texts.append(t)
                metas.append(md)
    else:
        # legacy files mode: split by characters if SentenceSplitter is not desired
        docs = load_documents(data_path)
        for d in docs:
            try:
                txt = getattr(d, "text", "") or d.get("text", "")
            except Exception:
                txt = ""
            chunks = _split_into_chunks(txt, max_chars=max_chars)
            for ch in chunks:
                texts.append(ch)
                meta = {}
                try:
                    md = getattr(d, "metadata", {})
                    if not isinstance(md, dict):
                        md = {}
                except Exception:
                    md = {}
                src = md.get("file_name") or md.get("file_path") or md.get("source") or "unknown"
                metas.append({"file_name": src})

    bm25 = BM25Index(k1=BM25_K1, b=BM25_B)
    bm25.build(texts)
    state = {"index": bm25, "texts": texts, "metas": metas}
    _BM25_CACHE[tenant_id] = state
    return state


def bm25_retrieve(query: str, top_k: int = 5, *, max_chars: int = 800, tenant_id: Optional[str] = None):
    state = get_bm25_state(max_chars=max_chars, tenant_id=tenant_id)
    pairs = state["index"].query(query, top_k=top_k)
    results: List[Dict[str, object]] = []
    for idx, score in pairs:
        results.append({
            "text": state["texts"][idx],
            "score": float(score),
            "meta": state["metas"][idx],
        })
    return results


def bm25_retrieve_debug(query: str, top_k: int = 5, *, max_chars: int = 800, tenant_id: Optional[str] = None):
    state = get_bm25_state(max_chars=max_chars, tenant_id=tenant_id)
    pairs = state["index"].query(query, top_k=top_k)
    q_tokens = _tokenize(query)
    results: List[Dict[str, object]] = []
    for idx, score in pairs:
        results.append({
            "text": state["texts"][idx],
            "score": float(score),
            "meta": state["metas"][idx],
            "_idx": idx,
        })
    return {
        "q_tokens": q_tokens,
        "results": results,
        "pairs": [{"idx": i, "score": float(s)} for i, s in pairs],
    }
