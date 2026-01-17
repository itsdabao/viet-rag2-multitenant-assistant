import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# In this repo's sandboxed environments, network egress may be blocked via a dummy proxy.
# Set offline mode early to avoid long HuggingFace download retry loops during index init.
_proxy_vals = {
    (os.getenv("HTTP_PROXY") or "").strip().lower(),
    (os.getenv("HTTPS_PROXY") or "").strip().lower(),
    (os.getenv("ALL_PROXY") or "").strip().lower(),
}
_blocked_proxy = any(p in ("http://127.0.0.1:9", "https://127.0.0.1:9") for p in _proxy_vals)
if _blocked_proxy and not os.getenv("ALLOW_HF_DOWNLOAD"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def process_and_log_query(query: str):
    # Backward-compatible wrapper: return (answer, trace)
    ans, trace, _log_file = process_and_log_query_v2(query)
    return ans, trace

def _ensure_agent_env() -> None:
    """
    Re-exec in conda env `agent` to ensure deps exist (llama-index, etc.).
    """
    if os.getenv("EVAL_NO_REEXEC"):
        return
    try:
        import llama_index  # type: ignore  # noqa: F401

        return
    except Exception:
        pass

    # Preserve CLI args when re-executing.
    cmd = ["conda", "run", "-n", "agent", "python", "-X", "utf8", str(Path(__file__).resolve()), *sys.argv[1:]]
    env = dict(os.environ)
    env["EVAL_NO_REEXEC"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        msg = ("Re-running with: " + " ".join(cmd)).encode("ascii", "backslashreplace").decode("ascii")
    except Exception:
        msg = "Re-running with: (failed to render command)"
    # Keep stdout clean for tool-only mode; only print re-exec hint when requested.
    if os.getenv("EVAL_VERBOSE"):
        print(msg, flush=True, file=sys.stderr)
    try:
        r = subprocess.run(cmd, env=env)
        raise SystemExit(r.returncode)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Cannot find `conda` to re-exec into env `agent`. "
            "Run inside the correct env:\n"
            "  conda run -n agent python scripts/eval_discount_tool.py"
        ) from e


CASES_PATH = Path("data/.cache/eval_discount_cases.jsonl")
REPORT_PATH = Path("data/.cache/eval_discount_report.json")
LOG_DIR = Path("data/logs/traces")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_trace(trace: Dict[str, Any]) -> Path:
    log_file = LOG_DIR / f"trace_{datetime.date.today().isoformat()}.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return log_file


def _norm_ascii(s: str) -> str:
    try:
        import unicodedata
        import re

        t = unicodedata.normalize("NFD", s or "")
        t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
        t = t.replace("đ", "d").replace("Đ", "D").lower()
        t = re.sub(r"\s+", " ", t).strip()
        return t
    except Exception:
        return (s or "").lower()


def process_and_log_query_v2(query: str) -> tuple[str, Dict[str, Any], Path]:
    """
    Execute preprocess -> route -> tool (if applicable), and append a JSONL trace.
    Returns (final_answer, trace, log_file_path).
    """
    from app.services.agentic.preprocess import preprocess_query
    from app.services.agentic.router import route_query
    from app.services.agentic.tools import tuition_calculator_tool

    trace: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input": query,
        "steps": [],
        "status": "INIT",
    }

    final_ans = ""
    try:
        t_pre0 = time.perf_counter()
        p_res = preprocess_query(query)
        trace["steps"].append(
            {
                "step": "preprocessing",
                "ms": round((time.perf_counter() - t_pre0) * 1000.0, 1),
                "output": {"lang": p_res.language, "toxic": p_res.toxic, "phone": p_res.phone},
            }
        )

        if p_res.toxic:
            trace["status"] = "REJECTED_TOXIC"
            final_ans = "Câu hỏi vi phạm chính sách."
            trace["final_output"] = final_ans
            log_file = _append_trace(trace)
            return final_ans, trace, log_file

        t_r0 = time.perf_counter()
        decision = route_query(p_res)
        trace["steps"].append(
            {
                "step": "routing",
                "ms": round((time.perf_counter() - t_r0) * 1000.0, 1),
                "decision": decision.route,
                "reason": decision.reason,
            }
        )

        if decision.route == "tuition_calculator":
            t_tool0 = time.perf_counter()
            tool_res = tuition_calculator_tool(query, index=None)
            trace["steps"].append(
                {
                    "step": "tool_execution",
                    "ms": round((time.perf_counter() - t_tool0) * 1000.0, 1),
                    "tool": "tuition_calculator",
                    "metadata": tool_res.metadata,
                    "raw_answer": tool_res.answer,
                }
            )
            final_ans = tool_res.answer
        else:
            final_ans = "Chưa hỗ trợ route này."

        trace["status"] = "SUCCESS"
        trace["final_output"] = final_ans
    except Exception as e:
        trace["status"] = "ERROR"
        trace["error_detail"] = str(e)
        final_ans = "Đã có lỗi xảy ra."
        trace["final_output"] = final_ans

    log_file = _append_trace(trace)
    return final_ans, trace, log_file


DEFAULT_CASES: List[Dict[str, Any]] = [
    {"id": "pct_10tr_10", "query": "Học phí 10tr giảm 10% còn bao nhiêu?", "expected": {"final_vnd": 9000000}},
    {
        "id": "pct_10tr_fee300k_10_tuition",
        "query": "Học phí 10tr, phí giáo trình 300k, giảm 10% gói học.",
        "expected": {"final_vnd": 9300000},
    },
    {"id": "pct_9500k_10", "query": "9.500.000đ giảm 10% còn bao nhiêu?", "expected": {"final_vnd": 8550000}},
    {"id": "amt_10tr_500k", "query": "Học phí 10tr giảm 500k còn bao nhiêu?", "expected": {"final_vnd": 9500000}},
    {"id": "amt_9500k_1tr", "query": "9.500.000 VND giảm 1tr còn bao nhiêu?", "expected": {"final_vnd": 8500000}},
    {"id": "invalid_pct", "query": "10tr giảm 120% còn bao nhiêu?", "expected": {"error": True}},
    {"id": "invalid_amt", "query": "10tr giảm 20tr còn bao nhiêu?", "expected": {"error": True}},
    {"id": "missing_discount", "query": "10tr sau giảm còn bao nhiêu?", "expected": {"needs_more_info": True}},
]


def _load_cases() -> List[Dict[str, Any]]:
    existing: List[Dict[str, Any]] = []
    if CASES_PATH.exists():
        for ln in CASES_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("query"):
                existing.append(obj)

    # Merge defaults so new built-in cases are picked up automatically.
    by_id: Dict[str, Dict[str, Any]] = {}
    for c in existing:
        cid = str(c.get("id") or "").strip()
        if cid:
            by_id[cid] = c
    added = False
    for c in DEFAULT_CASES:
        cid = str(c.get("id") or "").strip()
        if cid and cid not in by_id:
            by_id[cid] = c
            added = True

    merged = list(by_id.values()) if by_id else list(DEFAULT_CASES)
    if (not CASES_PATH.exists()) or added:
        CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
        CASES_PATH.write_text("\n".join([json.dumps(x, ensure_ascii=False) for x in merged]) + "\n", encoding="utf-8")
    return merged


def _run_eval_suite() -> None:
    cases = _load_cases()
    t0 = time.perf_counter()

    results: List[Dict[str, Any]] = []
    n_pass = 0
    n_fail = 0

    for c in cases:
        case_id = str(c.get("id") or "")
        q = str(c.get("query") or "").strip()
        exp = c.get("expected") if isinstance(c.get("expected"), dict) else {}
        if not q:
            continue

        ans, trace, log_file = process_and_log_query_v2(q)
        route = None
        route_reason = None
        tool_md: Dict[str, Any] = {}
        computed = None
        for step in trace.get("steps", []) if isinstance(trace.get("steps"), list) else []:
            if not isinstance(step, dict):
                continue
            if step.get("step") == "routing":
                route = step.get("decision")
                route_reason = step.get("reason")
            if step.get("step") == "tool_execution":
                tool_md = step.get("metadata") or {}
                computed = tool_md.get("computed_final_vnd")

        ok_route = route == "tuition_calculator"
        ok = True
        reason = []

        if exp.get("final_vnd") is not None:
            ok = (computed == int(exp["final_vnd"])) and ok_route
            if not ok_route:
                reason.append(f"route={route}")
            if computed != int(exp["final_vnd"]):
                reason.append(f"computed={computed} expected={int(exp['final_vnd'])}")
        elif exp.get("error"):
            ok = ("khong hop le" in _norm_ascii(ans)) and ok_route
            if not ok:
                reason.append("expected_error")
        elif exp.get("needs_more_info"):
            ok = ("cho em xin" in _norm_ascii(ans)) and ok_route
            if not ok:
                reason.append("expected_needs_more_info")

        results.append(
            {
                "id": case_id,
                "query": q,
                "route": route,
                "route_reason": route_reason,
                "final_answer": ans,
                "tool_metadata": tool_md,
                "trace_status": trace.get("status"),
                "trace_log_file": str(log_file),
                "pass": bool(ok),
                "fail_reason": "; ".join(reason),
            }
        )
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    report = {
        "cases_total": len(results),
        "pass": n_pass,
        "fail": n_fail,
        "pass_rate": (n_pass / max(1, len(results))),
        "elapsed_ms": round(elapsed_ms, 1),
        "cases_path": str(CASES_PATH),
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote report to {REPORT_PATH} (pass={n_pass} fail={n_fail})")
    if n_fail:
        print("Hint: open the report and inspect fail_reason/final_answer/tool_metadata.")


def _run_tool_only(query: str) -> str:
    """
    Tool-only runner:
    - If router matches a supported tool, execute it and print its answer.
    - Otherwise print 'none'.

    Note: This runner intentionally does NOT call LLM-backed RAG.
    """
    raise RuntimeError("Use _run_tool_once() (new API).")


def _try_build_index() -> Tuple[object | None, str | None]:
    """
    Best-effort index loader; returns (index, error_message).
    """
    try:
        from app.services.rag_service import build_index

        return build_index(), None
    except Exception as e:
        return None, str(e)


def _run_tool_once(
    query: str,
    *,
    with_index: bool,
    allow_llm: bool,
    tenant_id: str | None = None,
    branch_id: str | None = None,
) -> str:
    """
    Tool-only runner:
    - If router matches a supported tool, execute it and print its answer.
    - Otherwise print 'none'.
    """
    from app.services.agentic.preprocess import preprocess_query
    from app.services.agentic.router import route_query, out_of_domain_answer
    from app.services.agentic.tools import comparison_tool, course_search_tool, create_ticket_tool, tuition_calculator_tool
    from app.core.config import FEWSHOT_PATH

    t0 = time.perf_counter()
    p = preprocess_query(query)
    trace: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input": query,
        "steps": [{"step": "preprocessing", "output": {"lang": p.language, "toxic": p.toxic, "phone": p.phone}}],
        "status": "INIT",
    }

    if p.toxic:
        trace["status"] = "REJECTED_TOXIC"
        trace["final_output"] = None
        _append_trace(trace)
        return "none"

    decision = route_query(p)
    trace["steps"].append({"step": "routing", "decision": decision.route, "reason": decision.reason})

    # Build index only if needed.
    index = None
    index_err = None
    if with_index and decision.route in ("comparison", "course_search", "tuition_calculator"):
        index, index_err = _try_build_index()
        trace["steps"].append({"step": "index_load", "ok": bool(index is not None), "error": index_err})

    try:
        if decision.route == "out_of_domain":
            trace["status"] = "NO_TOOL_MATCH"
            trace["final_output"] = None
            _append_trace(trace)
            return "none"

        if decision.route == "smalltalk":
            trace["status"] = "NO_TOOL_MATCH"
            trace["final_output"] = None
            _append_trace(trace)
            return "none"

        if decision.route == "create_ticket":
            tool = create_ticket_tool(query, tenant_id=tenant_id, branch_id=branch_id, user_id=None)
            trace["steps"].append({"step": "tool_execution", "tool": "create_ticket", "metadata": tool.metadata, "raw_answer": tool.answer})
            trace["status"] = "SUCCESS"
            trace["final_output"] = tool.answer
            trace["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            _append_trace(trace)
            return tool.answer

        if decision.route == "tuition_calculator":
            tool = tuition_calculator_tool(
                query,
                index=index if isinstance(index, object) else None,
                tenant_id=tenant_id,
                branch_id=branch_id,
                allow_llm=bool(allow_llm),
            )
            trace["steps"].append({"step": "tool_execution", "tool": "tuition_calculator", "metadata": tool.metadata, "raw_answer": tool.answer})
            trace["status"] = "SUCCESS"
            trace["final_output"] = tool.answer
            trace["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            _append_trace(trace)
            return tool.answer

        if decision.route == "comparison":
            tool = comparison_tool(query, index=index, tenant_id=tenant_id, branch_id=branch_id)
            trace["steps"].append({"step": "tool_execution", "tool": "comparison", "metadata": tool.metadata, "raw_answer": tool.answer})
            trace["status"] = "SUCCESS"
            trace["final_output"] = tool.answer
            trace["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            _append_trace(trace)
            return tool.answer

        if decision.route == "course_search":
            # This tool is LLM-backed. Only run when explicitly allowed.
            if not allow_llm or index is None:
                trace["status"] = "NO_TOOL_MATCH"
                trace["final_output"] = None
                _append_trace(trace)
                return "none"
            tool = course_search_tool(
                query,
                index=index,
                fewshot_path=FEWSHOT_PATH,
                tenant_id=tenant_id,
                branch_id=branch_id,
                history=[],
            )
            trace["steps"].append({"step": "tool_execution", "tool": "course_search", "metadata": tool.metadata, "raw_answer": tool.answer})
            trace["status"] = "SUCCESS"
            trace["final_output"] = tool.answer
            trace["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            _append_trace(trace)
            return tool.answer

        trace["status"] = "NO_TOOL_MATCH"
        trace["final_output"] = None
        _append_trace(trace)
        return "none"
    except Exception as e:
        trace["status"] = "ERROR"
        trace["error_detail"] = str(e)
        trace["final_output"] = None
        trace["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        _append_trace(trace)
        return "none"


def _safe_ascii(s: str) -> str:
    """
    Windows terminals can display mojibake depending on codepage.
    Print an ASCII-safe representation when requested.
    """
    if not os.getenv("EVAL_ASCII_ONLY"):
        return str(s or "")
    try:
        return (s or "").encode("ascii", "backslashreplace").decode("ascii")
    except Exception:
        return str(s or "")


def _interactive_loop() -> None:
    with_index = False
    allow_llm = False
    tenant_id: str | None = None
    branch_id: str | None = None

    def show_help() -> None:
        print(
            _safe_ascii(
                "Commands:\n"
                "  /help                 Show help\n"
                "  /exit                 Exit\n"
                "  /index on|off          Enable/disable index usage (Qdrant) for comparison/course_search\n"
                "  /llm on|off            Allow/deny LLM-backed tool (course_search)\n"
                "  /tenant <id>           Set tenant_id\n"
                "  /branch <id>           Set branch_id\n"
                "  /state                Show current state\n"
                "\n"
                "Notes:\n"
                "- This runner prints 'none' when no supported tool is matched.\n"
                "- All traces are logged to data/logs/traces/trace_YYYY-MM-DD.jsonl\n"
            )
        )

    def show_state() -> None:
        print(
            _safe_ascii(
                f"state: index={'on' if with_index else 'off'} llm={'on' if allow_llm else 'off'} "
                f"tenant={tenant_id or '-'} branch={branch_id or '-'}"
            )
        )

    show_help()
    show_state()

    while True:
        try:
            line = input("\nQuery> ").strip()
        except EOFError:
            break
        if not line:
            continue
        low = line.lower()
        if low in ("exit", "quit", "/exit", "/quit"):
            break
        if low in ("/help", "help", "?"):
            show_help()
            continue
        if low == "/state":
            show_state()
            continue
        if low.startswith("/index"):
            parts = line.split(maxsplit=1)
            arg = (parts[1].strip().lower() if len(parts) > 1 else "")
            if arg in ("on", "true", "1"):
                with_index = True
            elif arg in ("off", "false", "0"):
                with_index = False
            show_state()
            continue
        if low.startswith("/llm"):
            parts = line.split(maxsplit=1)
            arg = (parts[1].strip().lower() if len(parts) > 1 else "")
            if arg in ("on", "true", "1"):
                allow_llm = True
            elif arg in ("off", "false", "0"):
                allow_llm = False
            show_state()
            continue
        if low.startswith("/tenant"):
            parts = line.split(maxsplit=1)
            tenant_id = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
            show_state()
            continue
        if low.startswith("/branch"):
            parts = line.split(maxsplit=1)
            branch_id = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
            show_state()
            continue

        ans = _run_tool_once(
            line,
            with_index=with_index,
            allow_llm=allow_llm,
            tenant_id=tenant_id,
            branch_id=branch_id,
        )
        print(_safe_ascii(ans))


def _run_batch_args(argv: List[str]) -> None:
    """
    Allow passing the same interactive slash-commands as CLI args, e.g.:
      python scripts/eval_discount_tool.py /index on /llm off /tenant elitespeak /state
    If a non-slash token is encountered, the remainder is treated as a query and executed once.
    """
    with_index = False
    allow_llm = False
    tenant_id: str | None = None
    branch_id: str | None = None

    def show_help() -> None:
        print(
            _safe_ascii(
                "Commands:\n"
                "  /help                 Show help\n"
                "  /exit                 Exit\n"
                "  /index on|off          Enable/disable index usage (Qdrant) for comparison/course_search\n"
                "  /llm on|off            Allow/deny LLM-backed tool (course_search)\n"
                "  /tenant <id>           Set tenant_id\n"
                "  /branch <id>           Set branch_id\n"
                "  /state                Show current state\n"
            )
        )

    def show_state() -> None:
        print(
            _safe_ascii(
                f"state: index={'on' if with_index else 'off'} llm={'on' if allow_llm else 'off'} "
                f"tenant={tenant_id or '-'} branch={branch_id or '-'}"
            )
        )

    i = 0
    printed_any = False
    while i < len(argv):
        tok = argv[i].strip()
        if not tok:
            i += 1
            continue

        if not tok.startswith("/"):
            # Treat the remainder as a query and run once.
            q = " ".join([t for t in argv[i:] if t is not None]).strip()
            if q:
                ans = _run_tool_once(
                    q,
                    with_index=with_index,
                    allow_llm=allow_llm,
                    tenant_id=tenant_id,
                    branch_id=branch_id,
                )
                print(_safe_ascii(ans))
                printed_any = True
            return

        cmd = tok.lower()
        if cmd in ("/exit", "/quit"):
            return
        if cmd in ("/help", "/?"):
            show_help()
            printed_any = True
            i += 1
            continue
        if cmd == "/state":
            show_state()
            printed_any = True
            i += 1
            continue
        if cmd == "/index":
            arg = (argv[i + 1].strip().lower() if i + 1 < len(argv) else "")
            if arg in ("on", "true", "1"):
                with_index = True
            elif arg in ("off", "false", "0"):
                with_index = False
            i += 2 if i + 1 < len(argv) else 1
            continue
        if cmd == "/llm":
            arg = (argv[i + 1].strip().lower() if i + 1 < len(argv) else "")
            if arg in ("on", "true", "1"):
                allow_llm = True
            elif arg in ("off", "false", "0"):
                allow_llm = False
            i += 2 if i + 1 < len(argv) else 1
            continue
        if cmd == "/tenant":
            tenant_id = (argv[i + 1].strip() if i + 1 < len(argv) and argv[i + 1].strip() else None)
            i += 2 if i + 1 < len(argv) else 1
            continue
        if cmd == "/branch":
            branch_id = (argv[i + 1].strip() if i + 1 < len(argv) and argv[i + 1].strip() else None)
            i += 2 if i + 1 < len(argv) else 1
            continue

        # Unknown slash command; ignore it.
        i += 1

    # If only state-change commands were passed, print state so it's not silent.
    if not printed_any:
        show_state()


def main() -> None:
    _ensure_agent_env()
    try:
        if os.name == "nt":
            try:
                import ctypes

                ctypes.windll.kernel32.SetConsoleOutputCP(65001)
                ctypes.windll.kernel32.SetConsoleCP(65001)
            except Exception:
                pass
            os.environ.setdefault("PYTHONIOENCODING", "utf-8")

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # Default: interactive mode (no flags needed).
    if len(sys.argv) == 1:
        _interactive_loop()
        return
    # If user passed interactive commands as argv (e.g. /index on), run in batch mode.
    if any(a.startswith("/") for a in sys.argv[1:]):
        _run_batch_args(sys.argv[1:])
        return

    parser = argparse.ArgumentParser(description="Discount tool eval + interactive tool runner")
    parser.add_argument("--eval", action="store_true", help="Run the built-in eval suite and write report JSON")
    parser.add_argument("--query", default=None, help="Single query to run tool-only (prints answer or 'none')")
    parser.add_argument("--interactive", action="store_true", help="Interactive tool-only mode")
    parser.add_argument("--with-index", action="store_true", help="Try load Qdrant index to enable comparison/course_search")
    parser.add_argument("--allow-llm", action="store_true", help="Allow LLM-backed tool execution (course_search)")
    parser.add_argument("--tenant", default=None, help="Tenant ID (optional)")
    parser.add_argument("--branch", default=None, help="Branch ID (optional)")
    args = parser.parse_args()

    if args.eval:
        _run_eval_suite()
        return

    if args.query is not None:
        print(
            _run_tool_once(
                args.query,
                with_index=bool(args.with_index),
                allow_llm=bool(args.allow_llm),
                tenant_id=args.tenant,
                branch_id=args.branch,
            )
        )
        return

    if args.interactive:
        _interactive_loop()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
