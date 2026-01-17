from __future__ import annotations

import logging
from typing import Dict, List, Optional

from llama_index.core import VectorStoreIndex

from app.core.config import ENABLE_SEMANTIC_ROUTER, FEWSHOT_PATH, TOXIC_MESSAGE

from .preprocess import preprocess_query
from .router import out_of_domain_answer, route_query
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
        return {"answer": tr.answer, "sources": tr.sources, "route": "course_search"}

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
        return {"answer": out_of_domain_answer(), "sources": [], "route": "out_of_domain"}
    if decision.route == "create_ticket":
        tr = create_ticket_tool(question, tenant_id=tenant_id, branch_id=branch_id, user_id=user_id)
        out: Dict[str, object] = {"answer": tr.answer, "sources": tr.sources, "route": "create_ticket"}
        if tr.metadata:
            out["tool_metadata"] = tr.metadata
        return out
    if decision.route == "comparison":
        tr = comparison_tool(question, index=index, tenant_id=tenant_id, branch_id=branch_id)
        out = {"answer": tr.answer, "sources": tr.sources, "route": "comparison"}
        if tr.metadata:
            out["tool_metadata"] = tr.metadata
        return out
    if decision.route == "tuition_calculator":
        tr = tuition_calculator_tool(question, index=index, tenant_id=tenant_id, branch_id=branch_id, allow_llm=True)
        out = {"answer": tr.answer, "sources": tr.sources, "route": "tuition_calculator"}
        if tr.metadata:
            out["tool_metadata"] = tr.metadata
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
    out = {"answer": tr.answer, "sources": tr.sources, "route": "course_search"}
    if tr.metadata:
        out["tool_metadata"] = tr.metadata
    return out
