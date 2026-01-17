import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_agent_env() -> None:
    """
    Re-exec in conda env `agent` to ensure deps exist (llama-index, sqlalchemy, psycopg2).
    Mirrors the pattern used in scripts/eval_discount_tool.py.
    """
    if os.getenv("EVAL8_NO_REEXEC"):
        return
    try:
        import llama_index  # type: ignore  # noqa: F401
        import sqlalchemy  # type: ignore  # noqa: F401
        import psycopg2  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    cmd = ["conda", "run", "-n", "agent", "python", "-X", "utf8", str(Path(__file__).resolve()), *sys.argv[1:]]
    env = dict(os.environ)
    env["EVAL8_NO_REEXEC"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        r = subprocess.run(cmd, env=env)
        raise SystemExit(r.returncode)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Cannot find `conda` to re-exec into env `agent`. "
            "Run inside the correct env:\n"
            "  conda run -n agent python scripts/eval_day8.py"
        ) from e


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _fmt_vnd(v: int) -> str:
    return f"{int(v):,}".replace(",", ".") + " VND"


def _extract_money_like(text: str) -> List[int]:
    """
    Best-effort parse of money mentions like 9.270.000 or 10tr from an answer string.
    """
    vals: List[int] = []
    t = (text or "").lower()
    for m in re.finditer(r"(\d[\d\.\, ]{4,})\s*(?:vnd|dong)?", t):
        num = re.sub(r"[^\d]", "", m.group(1))
        if len(num) >= 5:
            try:
                vals.append(int(num))
            except Exception:
                pass
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*tr\b", t):
        try:
            vals.append(int(round(float(m.group(1).replace(",", ".")) * 1_000_000)))
        except Exception:
            pass
    return sorted(set([v for v in vals if v > 0]))


def _contains_all(text: str, needles: List[str]) -> bool:
    t = (text or "").lower()
    return all((n or "").lower() in t for n in needles if isinstance(n, str) and n.strip())


def _ensure_session_reset(session_id: str, tenant_id: str) -> None:
    from app.services.memory.store import get_or_create_session, save_session

    st = get_or_create_session(session_id=session_id, tenant_id=tenant_id)
    st.rolling_summary = ""
    st.recent_messages_buffer = []
    st.entity_memory = {}
    save_session(state=st)


def _leakage_check_contexts(contexts: List[Dict[str, Any]], tenant_id: str) -> Tuple[bool, str]:
    """
    Fail if any context meta indicates a different tenant_id.
    """
    for c in contexts or []:
        if not isinstance(c, dict):
            continue
        meta = c.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        t = meta.get("tenant_id")
        if t is None:
            continue
        if str(t) != str(tenant_id):
            return False, f"tenant_leakage meta.tenant_id={t} expected={tenant_id}"
    return True, "ok"


def _run_one_case(
    case: Dict[str, Any],
    *,
    with_index: bool,
    allow_llm: bool,
) -> Dict[str, Any]:
    # Pre-route to decide whether this case requires LLM.
    from app.services.agentic.preprocess import preprocess_query
    from app.services.agentic.router import route_query

    tenant_id = str(case.get("tenant_id") or "").strip()
    session_id = str(case.get("session_id") or "").strip()
    if not tenant_id:
        raise ValueError("case.tenant_id is required")
    if not session_id:
        session_id = f"{tenant_id}:eval_{case.get('id') or 'case'}"

    question = str(case.get("question") or "").strip()
    if not question:
        raise ValueError("case.question is required")

    decision = route_query(preprocess_query(question))
    llm_required = decision.route in ("course_search",)
    if llm_required and not allow_llm:
        return {
            "id": case.get("id"),
            "tenant_id": tenant_id,
            "session_id": session_id,
            "question": question,
            "skipped": True,
            "skip_reason": f"llm_required_route={decision.route}",
            "passed": False,
            "ts": datetime.datetime.now().isoformat(),
        }

    from app.core.bootstrap import bootstrap_runtime
    import app.core.config as cfg
    from app.services.rag_service import build_index, rag_query
    from app.services.rag.incontext_ralm import retrieve_hybrid_contexts

    # Keep eval output stable and reduce noisy Unicode logs (conda run can be fragile on Windows).
    # Some modules read debug flags at import time; also patch the module-level globals.
    cfg.DEBUG_VERBOSE = False
    cfg.DEBUG_SHOW_PROMPT = False
    try:
        import app.services.rag.incontext_ralm as ralm

        ralm.DEBUG_VERBOSE = False
        ralm.DEBUG_SHOW_PROMPT = False
    except Exception:
        pass

    bootstrap_runtime()
    index = build_index() if with_index else None

    _ensure_session_reset(session_id, tenant_id)

    # Optional setup turns to warm memory.
    setup_turns = case.get("setup_turns") if isinstance(case.get("setup_turns"), list) else []
    if setup_turns:
        if allow_llm:
            for t in setup_turns:
                if not isinstance(t, dict):
                    continue
                if str(t.get("role") or "").lower() != "user":
                    continue
                q = str(t.get("content") or "").strip()
                if not q:
                    continue
                _ = rag_query(q, tenant_id=tenant_id, session_id=session_id, channel="cli", history=[])
        else:
            # Inject memory directly without calling LLM, so we can still test memory-related behaviors.
            from app.services.memory.store import get_or_create_session, save_session, append_messages

            st = get_or_create_session(session_id=session_id, tenant_id=tenant_id)
            st = append_messages(state=st, messages=setup_turns, max_messages=50)
            save_session(state=st)

    # Main call
    res = rag_query(question, tenant_id=tenant_id, session_id=session_id, channel="cli", history=[])
    answer = str(res.get("answer") or "")
    route = str(res.get("route") or "")
    sources = res.get("sources") or []
    tool_md = res.get("tool_metadata") if isinstance(res.get("tool_metadata"), dict) else {}

    # --- Cheap checks ---
    checks: List[Dict[str, Any]] = []

    # 1) Must include keywords
    must_include = case.get("must_include") if isinstance(case.get("must_include"), list) else []
    checks.append({"name": "must_include", "pass": _contains_all(answer, must_include), "detail": must_include})

    # 2) Expected sources (by filename)
    expected_sources = case.get("expected_sources") if isinstance(case.get("expected_sources"), list) else []
    if expected_sources:
        src_join = " | ".join([str(s) for s in sources])
        checks.append({"name": "expected_sources", "pass": _contains_all(src_join, expected_sources), "detail": expected_sources})
    else:
        checks.append({"name": "expected_sources", "pass": True, "detail": "skipped"})

    # 3) Route expectation (if provided)
    expect_route = case.get("expect_route_one_of") if isinstance(case.get("expect_route_one_of"), list) else []
    if expect_route:
        checks.append({"name": "route_one_of", "pass": route in set(map(str, expect_route)), "detail": {"route": route, "expect": expect_route}})
    else:
        checks.append({"name": "route_one_of", "pass": True, "detail": "skipped"})

    # 4) Tenant leakage check (use retrieval contexts, not just sources)
    contexts: List[Dict[str, Any]] = []
    if index is not None:
        try:
            retrieved = retrieve_hybrid_contexts(question, index, tenant_id=tenant_id, branch_id=None, top_k_ctx=5)
            contexts = retrieved.get("contexts") or []
        except Exception:
            contexts = []
    ok, detail = _leakage_check_contexts(contexts, tenant_id)
    checks.append({"name": "tenant_leakage", "pass": ok, "detail": detail})

    # 5) Business consistency: if tool computed_final_vnd exists, answer should contain it.
    computed_final = tool_md.get("computed_final_vnd")
    if isinstance(computed_final, (int, float)):
        money_mentions = _extract_money_like(answer)
        checks.append(
            {
                "name": "calculator_consistency",
                "pass": int(computed_final) in set(money_mentions),
                "detail": {"computed_final_vnd": int(computed_final), "answer_money_mentions": money_mentions},
            }
        )
    else:
        checks.append({"name": "calculator_consistency", "pass": True, "detail": "skipped"})

    passed = all(bool(c.get("pass")) for c in checks)
    return {
        "id": case.get("id"),
        "tenant_id": tenant_id,
        "session_id": session_id,
        "question": question,
        "route": route,
        "answer": answer,
        "sources": sources,
        "tool_metadata": tool_md,
        "memory": res.get("memory"),
        "eval_contexts": [
            {
                "text": str(c.get("text") or "")[:1200],
                "meta": c.get("meta") if isinstance(c.get("meta"), dict) else {},
                "score": c.get("score"),
            }
            for c in (contexts or [])
            if isinstance(c, dict)
        ],
        "checks": checks,
        "passed": passed,
        "ts": datetime.datetime.now().isoformat(),
    }


def main() -> None:
    _ensure_agent_env()
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Day 8 evaluation runner (cheap checks + optional RAGAS hook)")
    parser.add_argument("--tenant", default="elitespeak", help="Tenant id (default: elitespeak)")
    parser.add_argument("--golden", default=None, help="Path to golden JSONL (default: data/eval/golden/<tenant>.jsonl)")
    parser.add_argument("--with-index", action="store_true", help="Use Qdrant index during eval (recommended)")
    parser.add_argument("--allow-llm", action="store_true", help="Allow LLM calls (needed for course_search + summary roll-up)")
    parser.add_argument("--with-ragas", action="store_true", help="Compute RAGAS scores when available (requires deps + LLM)")
    parser.add_argument("--out", default=None, help="Output report path (default: data/eval/runs/<date>_<tenant>.jsonl)")
    args = parser.parse_args()

    # If LLM is disabled for this run, also disable memory roll-up summary to avoid accidental LLM calls.
    if not args.allow_llm:
        os.environ.setdefault("MEMORY_SUMMARY_ENABLED", "0")

    tenant = str(args.tenant).strip()
    golden_path = Path(args.golden) if args.golden else Path("data/eval/golden") / f"{tenant}.jsonl"
    cases = _read_jsonl(golden_path)
    if not cases:
        raise SystemExit(f"No cases found in {golden_path}")

    out_path = Path(args.out) if args.out else Path("data/eval/runs") / f"{datetime.date.today().isoformat()}_{tenant}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    n_pass = 0
    n_run = 0
    n_skip = 0
    for c in cases:
        try:
            r = _run_one_case(c, with_index=bool(args.with_index), allow_llm=bool(args.allow_llm))
        except Exception as e:
            r = {
                "id": c.get("id"),
                "tenant_id": c.get("tenant_id"),
                "session_id": c.get("session_id"),
                "question": c.get("question"),
                "passed": False,
                "error": str(e),
                "ts": datetime.datetime.now().isoformat(),
            }
        results.append(r)
        if r.get("skipped"):
            n_skip += 1
            continue
        n_run += 1
        if r.get("passed"):
            n_pass += 1

    # Optional: RAGAS hook (best-effort). We keep it separate so cheap-checks can run offline.
    if args.with_ragas:
        try:
            from ragas import evaluate  # type: ignore
            from ragas.metrics import faithfulness, answer_relevance  # type: ignore
            import pandas as pd  # type: ignore

            rows = []
            for r in results:
                # RAGAS needs question, answer, contexts.
                ctx = r.get("eval_contexts") if isinstance(r.get("eval_contexts"), list) else []
                ctx_texts = [str(x.get("text") or "") for x in ctx if isinstance(x, dict) and x.get("text")]
                rows.append({"question": r.get("question"), "answer": r.get("answer"), "contexts": ctx_texts})
            ds = pd.DataFrame(rows)
            scores = evaluate(ds, metrics=[faithfulness, answer_relevance])
            # Attach aggregate scores only (per-item scores vary by ragas versions).
            agg = {}
            try:
                agg = {
                    "faithfulness": float(getattr(scores, "get", lambda k, d=None: d)("faithfulness", None)),
                    "answer_relevance": float(getattr(scores, "get", lambda k, d=None: d)("answer_relevance", None)),
                }
            except Exception:
                agg = {}
            for r in results:
                r.setdefault("ragas", {})
                r["ragas"]["aggregate"] = agg
        except Exception as e:
            for r in results:
                r.setdefault("ragas", {})
                r["ragas"]["error"] = str(e)

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"OK wrote {out_path} (pass={n_pass}/{max(1, n_run)} run={n_run} skipped={n_skip} total={len(results)})")


if __name__ == "__main__":
    main()
