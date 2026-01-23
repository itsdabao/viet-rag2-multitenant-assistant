from __future__ import annotations

import logging
from typing import Dict, List, Optional

from llama_index.core import VectorStoreIndex

from app.core.config import ENABLE_SEMANTIC_ROUTER, FEWSHOT_PATH, TOXIC_MESSAGE

from .preprocess import preprocess_query
from .router import out_of_domain_answer, route_query
from .arguments import extract_comparison_args, extract_ticket_args, extract_tuition_calculator_args
from .tools import (
    ToolResult,
    comparison_tool,
    course_search_tool,
    create_ticket_tool,
    tuition_calculator_tool,
)


logger = logging.getLogger(__name__)


def agentic_query(
    question: str,
    *,
    index: VectorStoreIndex,
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[str] = None,
) -> Dict[str, object]:
    """
    Day 4 agentic entrypoint: preprocessing -> semantic router -> tool execution.
    Returns a dict compatible with existing API: {answer, sources, ...}.
    """
    if not ENABLE_SEMANTIC_ROUTER:
        tr = course_search_tool(
            question,
            index=index,
            fewshot_path=FEWSHOT_PATH,
            tenant_id=tenant_id,
            branch_id=branch_id,
            history=history or [],
        )
        return {"answer": tr.answer, "sources": tr.sources, "route": "course_search", "contexts": tr.context_texts or []}

    p = preprocess_query(question)
    if p.toxic:
        return {"answer": TOXIC_MESSAGE, "sources": [], "route": "toxic"}

    decision = route_query(p)
    logger.info(
        "router route=%s conf=%.2f reason=%s tenant=%s branch=%s",
        decision.route,
        float(decision.confidence),
        decision.reason,
        tenant_id or "-",
        branch_id or "-",
    )

    tr: ToolResult
    if decision.route == "smalltalk":
        return {"answer": decision.smalltalk_answer or "", "sources": [], "route": "smalltalk"}
    if decision.route == "out_of_domain":
        # Check specifically for language mismatch reason from router
        if "language_mismatch" in decision.reason:
            return {
                "answer": "Sorry I don't support this language. This chatbot is only supported in Vietnamese only.",
                "sources": [],
                "route": "language_guard"
            }
        return {"answer": out_of_domain_answer(), "sources": [], "route": "out_of_domain", "contexts": []}
    if decision.route == "create_ticket":
        tr = create_ticket_tool(question, tenant_id=tenant_id, branch_id=branch_id, user_id=user_id)
        out: Dict[str, object] = {"answer": tr.answer, "sources": tr.sources, "route": "create_ticket", "contexts": tr.context_texts or []}
        if tr.metadata:
            out["tool_metadata"] = tr.metadata
        try:
            out["tool_metadata"] = dict(out.get("tool_metadata") or {})
            out["tool_metadata"]["extracted_args"] = extract_ticket_args(question)
        except Exception:
            pass
        return out
    if decision.route == "comparison":
        tr = comparison_tool(question, index=index, tenant_id=tenant_id, branch_id=branch_id)
        out = {"answer": tr.answer, "sources": tr.sources, "route": "comparison", "contexts": tr.context_texts or []}
        if tr.metadata:
            out["tool_metadata"] = tr.metadata
        try:
            out["tool_metadata"] = dict(out.get("tool_metadata") or {})
            out["tool_metadata"]["extracted_args"] = extract_comparison_args(question)
        except Exception:
            pass
        return out
    if decision.route == "tuition_calculator":
        tr = tuition_calculator_tool(question, index=index, tenant_id=tenant_id, branch_id=branch_id, allow_llm=True)
        out = {"answer": tr.answer, "sources": tr.sources, "route": "tuition_calculator", "contexts": tr.context_texts or []}
        if tr.metadata:
            out["tool_metadata"] = tr.metadata
        try:
            out["tool_metadata"] = dict(out.get("tool_metadata") or {})
            out["tool_metadata"]["extracted_args"] = extract_tuition_calculator_args(question)
        except Exception:
            pass
        return out

    # Default: full RAG answer
    tr = course_search_tool(
        question,
        index=index,
        fewshot_path=FEWSHOT_PATH,
        tenant_id=tenant_id,
        branch_id=branch_id,
        history=history or [],
    )
    out = {"answer": tr.answer, "sources": tr.sources, "route": "course_search", "contexts": tr.context_texts or []}
    if tr.metadata:
        out["tool_metadata"] = tr.metadata
    return out


def semantic_router_response(
    question: str,
    *,
    index: Optional[VectorStoreIndex] = None,
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[str] = None,
) -> object:
    """
    Semantic-router output mode:
    - If a tool is needed: return a JSON-serializable dict {tool_name, arguments, thought}
    - Otherwise: return a plain text answer (string)
    """
    p = preprocess_query(question)
    if p.toxic:
        return TOXIC_MESSAGE

    decision = route_query(p) if ENABLE_SEMANTIC_ROUTER else None
    route = decision.route if decision is not None else "course_search"

    if route == "smalltalk":
        return (decision.smalltalk_answer or "") if decision is not None else ""

    if route == "out_of_domain":
        if decision is not None and "language_mismatch" in decision.reason:
            return "Sorry I don't support this language. This chatbot is only supported in Vietnamese only."
        return out_of_domain_answer()

    if route == "create_ticket":
        args = extract_ticket_args(question)
        if not args.get("phone"):
            return "Dạ anh/chị cho em xin **SĐT** và **khung giờ thuận tiện** để tư vấn viên liên hệ hỗ trợ chi tiết nhé."
        if tenant_id:
            args["tenant_id"] = tenant_id
        if branch_id:
            args["branch_id"] = branch_id
        if user_id:
            args["user_id"] = user_id
        return {
            "tool_name": "create_ticket_tool",
            "arguments": args,
            "thought": "Người dùng muốn tư vấn/chuyển tư vấn viên, đã có SĐT nên tạo ticket để CSKH liên hệ.",
        }

    if route == "comparison":
        args = extract_comparison_args(question)
        if tenant_id:
            args["tenant_id"] = tenant_id
        if branch_id:
            args["branch_id"] = branch_id
        return {
            "tool_name": "comparison_tool",
            "arguments": args,
            "thought": "Câu hỏi yêu cầu so sánh giữa các khóa học, cần trích xuất tên khóa và tiêu chí.",
        }

    if route == "tuition_calculator":
        args = extract_tuition_calculator_args(question)
        if tenant_id:
            args["tenant_id"] = tenant_id
        if branch_id:
            args["branch_id"] = branch_id
        return {
            "tool_name": "tuition_calculator_tool",
            "arguments": args,
            "thought": "Câu hỏi về học phí/giảm giá/phụ phí, cần trích xuất biến tài chính để tính toán.",
        }

    if index is None:
        raise RuntimeError("semantic_router_response requires `index` for course_search answers.")

    tr = course_search_tool(
        question,
        index=index,
        fewshot_path=FEWSHOT_PATH,
        tenant_id=tenant_id,
        branch_id=branch_id,
        history=history or [],
    )
    return tr.answer
