"""
Compatibility entrypoint for the FastAPI app.

Prefer running: `uvicorn app.api.main:app --reload --port 8000`
"""

from app.api.main import app  # noqa: F401


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
    history: Optional[List[Dict[str, str]]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


@app.on_event("startup")
def startup_event() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logger.info("Starting RAG backend...")
    init_llm_from_env()
    build_index()
    logger.info("RAG backend is ready.")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    result = rag_query(
        question=payload.question,
        tenant_id=payload.tenant_id,
        history=payload.history or [],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    sources = result.get("sources", []) or []
    logger.info(
        "query tenant=%s len_q=%d time_ms=%.1f sources=%d",
        payload.tenant_id or "-",
        len(payload.question or ""),
        elapsed_ms,
        len(sources),
    )
    return QueryResponse(
        answer=str(result.get("answer", "")),
        sources=[str(s) for s in sources],
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
            history = data.get("history") or []
            if not question:
                await ws.send_json({"type": "error", "message": "Câu hỏi trống."})
                continue

            start = time.perf_counter()
            # Gọi RAG trong thread riêng để tránh lỗi asyncio.run() bên trong LLM
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: rag_query(
                    question=question,
                    tenant_id=tenant_id,
                    history=history,
                ),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            answer = str(result.get("answer", "") or "")
            sources = result.get("sources", []) or []

            logger.info(
                "ws_query tenant=%s len_q=%d time_ms=%.1f sources=%d",
                tenant_id or "-",
                len(question),
                elapsed_ms,
                len(sources),
            )

            # Gửi meta trước
            await ws.send_json(
                {
                    "type": "meta",
                    "time_ms": round(elapsed_ms, 1),
                    "sources": [str(s) for s in sources],
                }
            )

            # Stream answer theo từng đoạn nhỏ để tạo cảm giác "gõ dần"
            chunk_size = 40
            if not answer:
                await ws.send_json({"type": "chunk", "text": "(Không có câu trả lời) "})
            else:
                for i in range(0, len(answer), chunk_size):
                    chunk = answer[i : i + chunk_size]
                    await ws.send_json({"type": "chunk", "text": chunk})
                    # Delay nhỏ cho UX, có thể giảm/tăng hoặc bỏ đi
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
