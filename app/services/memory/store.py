from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column, Index, String, Text, TIMESTAMP, create_engine, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, declarative_base

from app.core.config import CHAT_SESSIONS_TABLE, DATABASE_URL


Base = declarative_base()


class ChatSession(Base):
    __tablename__ = CHAT_SESSIONS_TABLE

    id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), nullable=False)
    entity_memory = Column(JSONB().with_variant(JSON(), "sqlite"), nullable=False, default=dict)
    rolling_summary = Column(Text, nullable=True)
    recent_messages_buffer = Column(JSONB().with_variant(JSON(), "sqlite"), nullable=False, default=list)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("idx_chat_sessions_tenant", "tenant_id"),)


_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set. Please export DATABASE_URL or put it in .env.")
    _ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
    return _ENGINE


def ensure_tables_exist() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine, checkfirst=True)


@dataclass
class SessionState:
    id: str
    tenant_id: str
    entity_memory: Dict[str, Any]
    rolling_summary: str
    recent_messages_buffer: List[Dict[str, Any]]
    updated_at: Optional[float] = None


def _now_ts() -> int:
    return int(time.time())


def get_or_create_session(*, session_id: str, tenant_id: str) -> SessionState:
    ensure_tables_exist()
    engine = get_engine()
    with Session(engine) as db:
        row = db.get(ChatSession, session_id)
        if row is None:
            row = ChatSession(
                id=session_id,
                tenant_id=tenant_id,
                entity_memory={},
                rolling_summary="",
                recent_messages_buffer=[],
            )
            db.add(row)
            db.commit()
            db.refresh(row)
        # Fail-closed: if tenant_id mismatches, do not leak other tenant's session.
        if str(row.tenant_id) != str(tenant_id):
            raise RuntimeError("chat_sessions tenant_id mismatch for session_id (refuse to load).")
        return SessionState(
            id=str(row.id),
            tenant_id=str(row.tenant_id),
            entity_memory=(row.entity_memory or {}) if isinstance(row.entity_memory, dict) else {},
            rolling_summary=str(row.rolling_summary or ""),
            recent_messages_buffer=list(row.recent_messages_buffer or []) if isinstance(row.recent_messages_buffer, list) else [],
            updated_at=None,
        )


def save_session(*, state: SessionState) -> None:
    ensure_tables_exist()
    engine = get_engine()
    with Session(engine) as db:
        row = db.get(ChatSession, state.id)
        if row is None:
            row = ChatSession(
                id=state.id,
                tenant_id=state.tenant_id,
                entity_memory=state.entity_memory or {},
                rolling_summary=state.rolling_summary or "",
                recent_messages_buffer=state.recent_messages_buffer or [],
            )
            db.add(row)
        else:
            if str(row.tenant_id) != str(state.tenant_id):
                raise RuntimeError("chat_sessions tenant_id mismatch for session_id (refuse to update).")
            row.entity_memory = state.entity_memory or {}
            row.rolling_summary = state.rolling_summary or ""
            row.recent_messages_buffer = state.recent_messages_buffer or []
        db.commit()


def append_messages(
    *,
    state: SessionState,
    messages: List[Dict[str, Any]],
    max_messages: int,
) -> SessionState:
    buf = list(state.recent_messages_buffer or [])
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = str(m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        buf.append({"role": role, "content": content, "ts": int(m.get("ts") or _now_ts())})
    if max_messages > 0 and len(buf) > max_messages:
        buf = buf[-max_messages:]
    state.recent_messages_buffer = buf
    return state


def merge_entity_memory(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_entity_memory(out.get(k) or {}, v)
        else:
            out[k] = v
    return out
