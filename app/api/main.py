import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.bootstrap import bootstrap_runtime
from app.core.config import PROJECT_ROOT
from app.services.rag_service import build_index, rag_query
from app.services.analytics.store import (
    insert_feedback,
    insert_trace,
    list_handoffs,
    list_tenants,
    list_traces,
    metrics,
    new_trace_id,
)


logger = logging.getLogger(__name__)

app = FastAPI(title="Viet RAG2 Multitenant Assistant")

# CORS (phục vụ phát triển frontend; có thể siết lại origins khi deploy)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    tenant_id: Optional[str] = None
    branch_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    trace_id: Optional[str] = None
    time_ms: Optional[float] = None
    route: Optional[str] = None


class FeedbackRequest(BaseModel):
    trace_id: str
    tenant_id: Optional[str] = None
    rating: int  # 1=up, -1=down
    comment: Optional[str] = None


def _parse_date_like(s: Optional[str]) -> Optional[int]:
    """
    Parse YYYY-MM-DD or epoch seconds to int epoch seconds.
    """
    if not s:
        return None
    raw = str(s).strip()
    if not raw:
        return None
    if raw.isdigit():
        try:
            return int(raw)
        except Exception:
            return None
    try:
        # YYYY-MM-DD
        parts = raw.split("T", 1)[0].split("-")
        if len(parts) == 3:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            import datetime as _dt

            return int(_dt.datetime(y, m, d).timestamp())
    except Exception:
        return None
    return None


@app.on_event("startup")
def startup_event() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logger.info("Starting RAG backend...")
    bootstrap_runtime()
    build_index()
    logger.info("RAG backend is ready.")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


WEB_DIR = PROJECT_ROOT / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/admin", include_in_schema=False)
def admin_ui() -> FileResponse:
    return FileResponse(str(WEB_DIR / "admin.html"))


@app.get("/admin/api/tenants")
def admin_tenants() -> Dict[str, object]:
    return {"tenants": list_tenants()}


@app.get("/admin/api/metrics")
def admin_metrics(tenant_id: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None) -> Dict[str, object]:
    return metrics(tenant_id=tenant_id, since_ts=_parse_date_like(since), until_ts=_parse_date_like(until))


@app.get("/admin/api/logs")
def admin_logs(
    tenant_id: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    route: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, object]:
    rows = list_traces(
        tenant_id=tenant_id,
        since_ts=_parse_date_like(since),
        until_ts=_parse_date_like(until),
        route=route,
        status=status,
        q=q,
        limit=limit,
        offset=offset,
    )
    return {"rows": rows}


@app.get("/admin/api/handoffs")
def admin_handoffs(
    tenant_id: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, object]:
    rows = list_handoffs(
        tenant_id=tenant_id,
        since_ts=_parse_date_like(since),
        until_ts=_parse_date_like(until),
        status=status,
        limit=limit,
        offset=offset,
    )
    return {"rows": rows}


@app.post("/admin/api/feedback")
def admin_feedback(payload: FeedbackRequest) -> Dict[str, object]:
    rating = 1 if int(payload.rating) > 0 else -1
    fb_id = insert_feedback(trace_id=payload.trace_id, tenant_id=payload.tenant_id, rating=rating, comment=payload.comment)
    return {"ok": True, "id": fb_id}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest) -> QueryResponse:
    trace_id = new_trace_id()
    start = time.perf_counter()
    result = None
    err = None
    try:
        result = rag_query(
            question=payload.question,
            tenant_id=payload.tenant_id,
            branch_id=payload.branch_id,
            history=payload.history or [],
            channel="web",
            session_id=payload.session_id,
            user_id=payload.user_id,
        )
    except Exception as e:
        err = str(e)
        result = {"answer": "", "sources": [], "route": "error"}
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    sources = (result.get("sources", []) or []) if isinstance(result, dict) else []
    route = (result.get("route") if isinstance(result, dict) else None) or None
    tool_md = (result.get("tool_metadata") if isinstance(result, dict) else None) or {}
    status = "ERROR" if err else "SUCCESS"
    logger.info(
        "query tenant=%s branch=%s len_q=%d time_ms=%.1f sources=%d",
        payload.tenant_id or "-",
        payload.branch_id or "-",
        len(payload.question or ""),
        elapsed_ms,
        len(sources),
    )
    try:
        insert_trace(
            trace_id=trace_id,
            tenant_id=payload.tenant_id,
            branch_id=payload.branch_id,
            channel="web",
            session_id=payload.session_id,
            user_id=payload.user_id,
            question=payload.question or "",
            answer=str(result.get("answer", "") if isinstance(result, dict) else ""),
            sources=[str(s) for s in sources],
            route=str(route) if route else None,
            status=status,
            latency_ms=float(elapsed_ms),
            tool_metadata=tool_md if isinstance(tool_md, dict) else {},
            error=err,
        )
    except Exception as e:
        logger.warning("Failed to insert trace: %s", e)
    return QueryResponse(
        answer=str(result.get("answer", "") if isinstance(result, dict) else ""),
        sources=[str(s) for s in sources],
        trace_id=trace_id,
        time_ms=round(elapsed_ms, 1),
        route=str(route) if route else None,
    )


@app.websocket("/ws/query")
async def websocket_query(ws: WebSocket) -> None:
    """
    WebSocket endpoint để stream câu trả lời dần dần xuống frontend.

    Giao thức đơn giản:
    - Client gửi JSON: {"question": str, "tenant_id": str | null, "history": [...]}
    - Server trả lần lượt:
        {"type": "meta", "sources": [...], "time_ms": float}
        {"type": "chunk", "text": "..."}  (lặp nhiều lần)
        {"type": "end"}
      Hoặc nếu lỗi:
        {"type": "error", "message": "..."}
    """
    await ws.accept()
    logger.info("WebSocket connected")
    try:
        while True:
            data = await ws.receive_json()
            question = (data.get("question") or "").strip()
            tenant_id = data.get("tenant_id")
            branch_id = data.get("branch_id")
            session_id = data.get("session_id")
            user_id = data.get("user_id")
            history = data.get("history") or []
            if not question:
                await ws.send_json({"type": "error", "message": "Câu hỏi trống."})
                continue

            start = time.perf_counter()
            loop = asyncio.get_running_loop()
            trace_id = new_trace_id()
            err = None
            result = await loop.run_in_executor(
                None,
                lambda: rag_query(
                    question=question,
                    tenant_id=tenant_id,
                    branch_id=branch_id,
                    history=history,
                    channel="web",
                    session_id=session_id,
                    user_id=user_id,
                ),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            answer = str(result.get("answer", "") or "")
            sources = result.get("sources", []) or []
            route = (result.get("route") if isinstance(result, dict) else None) or None
            tool_md = (result.get("tool_metadata") if isinstance(result, dict) else None) or {}

            logger.info(
                "ws_query tenant=%s branch=%s len_q=%d time_ms=%.1f sources=%d",
                tenant_id or "-",
                branch_id or "-",
                len(question),
                elapsed_ms,
                len(sources),
            )

            try:
                insert_trace(
                    trace_id=trace_id,
                    tenant_id=tenant_id,
                    branch_id=branch_id,
                    channel="web_ws",
                    session_id=session_id,
                    user_id=user_id,
                    question=question,
                    answer=answer,
                    sources=[str(s) for s in sources],
                    route=str(route) if route else None,
                    status="SUCCESS",
                    latency_ms=float(elapsed_ms),
                    tool_metadata=tool_md if isinstance(tool_md, dict) else {},
                )
            except Exception as e:
                logger.warning("Failed to insert ws trace: %s", e)

            await ws.send_json(
                {
                    "type": "meta",
                    "time_ms": round(elapsed_ms, 1),
                    "sources": [str(s) for s in sources],
                    "trace_id": trace_id,
                    "route": str(route) if route else None,
                }
            )

            chunk_size = 40
            if not answer:
                await ws.send_json({"type": "chunk", "text": "(Không có câu trả lời) "})
            else:
                for i in range(0, len(answer), chunk_size):
                    chunk = answer[i : i + chunk_size]
                    await ws.send_json({"type": "chunk", "text": chunk})
                    await asyncio.sleep(0.03)

            await ws.send_json({"type": "end"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
