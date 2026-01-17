from __future__ import annotations

from typing import Any, Dict, Optional

from llama_index.core import VectorStoreIndex

from app.core.config import DATABASE_URL, MEMORY_ENABLED
from app.services.agentic.service import agentic_query

from .manager import build_history_from_session, maybe_rollup_summary, update_session_after_turn
from .store import get_or_create_session


def build_session_id(*, tenant_id: str, channel: str, user_id: str) -> str:
    """
    SaaS-safe session key: {tenant}:{channel}:{user_id}
    """
    return f"{tenant_id}:{channel}:{user_id}"


def memory_rag_query(
    question: str,
    *,
    index: VectorStoreIndex,
    tenant_id: str,
    branch_id: Optional[str] = None,
    channel: str = "cli",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, object]:
    """
    Day 6-7 wrapper: persistent memory (Postgres) + optional roll-up summary (LLM).
    Returns the same dict shape as `agentic_query`.
    """
    if not MEMORY_ENABLED or not DATABASE_URL:
        return agentic_query(question, index=index, tenant_id=tenant_id, branch_id=branch_id, history=[], user_id=user_id)

    sid = (session_id or "").strip()
    if not sid:
        if not user_id:
            raise ValueError("memory_rag_query requires either session_id or user_id.")
        sid = build_session_id(tenant_id=tenant_id, channel=channel, user_id=str(user_id))

    # Ensure CLI convention tenant:session_id
    if ":" not in sid and tenant_id:
        sid = f"{tenant_id}:{sid}"
    if not sid.startswith(f"{tenant_id}:"):
        # Fail-closed to avoid accidental cross-tenant session load.
        sid = f"{tenant_id}:{sid}"

    state = get_or_create_session(session_id=sid, tenant_id=tenant_id)
    mem_ctx = build_history_from_session(state)

    result = agentic_query(
        question,
        index=index,
        tenant_id=tenant_id,
        branch_id=branch_id,
        history=mem_ctx.history,
        user_id=user_id,
    )

    answer = str(result.get("answer", "") or "")
    tool_md = result.get("tool_metadata") if isinstance(result.get("tool_metadata"), dict) else None
    state = update_session_after_turn(state=state, user_text=question, assistant_text=answer, tool_metadata=tool_md)

    # Roll up if needed
    _state2, roll_metrics = maybe_rollup_summary(state=state)
    result["memory"] = {
        "session_id": sid,
        "token_estimate": mem_ctx.token_estimate,
        "rolled_up": bool(roll_metrics.get("rolled_up")),
        "rollup_metrics": roll_metrics,
    }
    return result

