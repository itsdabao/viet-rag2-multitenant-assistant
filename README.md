# Viet RAG2 Multitenant Assistant

Trợ lý hỏi‑đáp tiếng Việt dựa trên RAG 2.0, dùng LlamaIndex + Qdrant + BM25 + Gemini. Hệ thống hỗ trợ multi‑tenant (nhiều khách hàng), hybrid search, rerank bằng cosine và có cả CLI lẫn backend FastAPI.

## Tính năng chính

- Ingest tài liệu (PDF, DOCX, TXT, MD, RTF) vào Qdrant bằng LlamaIndex.
- Hybrid retrieval: kết hợp vector search (Qdrant) và BM25 trên corpus đã chunk.
- Rerank nhẹ bằng cosine giữa truy vấn và context (có thể bật/tắt qua config).
- Smalltalk + out-of-domain guard: trả lời hội thoại đơn giản và chặn câu hỏi ngoài phạm vi để giảm số lần gọi LLM (cấu hình trong `app/core/config.py`, dữ liệu tại `app/resources/smalltalk_vi.json` và `app/resources/domain_anchors_vi.json`).
- Hỗ trợ multi‑tenant thông qua metadata `tenant_id` khi ingest và query.
- Giao diện CLI hỏi‑đáp và backend FastAPI với endpoint `/query`.

## 1. Chuẩn bị môi trường

Yêu cầu:
- Python 3.10–3.11
- Docker (để chạy Qdrant)

Khuyến nghị tạo môi trường ảo riêng (ví dụ conda):

Tạo file `.env` ở root:

```bash
# (Optional) ép provider để tránh tự fallback sang Gemini/OpenAI
LLM_PROVIDER=groq

# Primary (recommended): Groq (OpenAI-compatible)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_BASE_URL=https://api.groq.com/openai/v1
# Groq được gọi qua OpenAI-compatible endpoint `GROQ_BASE_URL` (cách 2).

# Optional: LlamaParse (Modern Ingestion cho PDF phức tạp)
# LLAMA_CLOUD_API_KEY=your_llama_cloud_key_here

# Optional fallback: Gemini
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
```

## 2. Chạy Qdrant

Chạy Qdrant bằng Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Mặc định code trỏ tới `localhost:6333` (có thể chỉnh trong `app/core/config.py`).

## 3. Ingest dữ liệu

Ingest toàn bộ file trong `data/knowledge_base`:

```bash
python scripts/ingest.py --auto-from-filenames
```

- Tên file dạng `tenant_*.pdf` sẽ được suy ra `tenant_id`.
- Hoặc ingest thủ công:

```bash
python scripts/ingest.py --tenant brightpathacademy --file data/knowledge_base/tenant_brightpathacademy.pdf
```

Nếu PDF có bảng biểu/phức tạp và bạn có `LLAMA_CLOUD_API_KEY`, có thể bật LlamaParse:

```bash
python scripts/ingest.py --pdf-engine llamaparse --tenant brightpathacademy --file data/knowledge_base/tenant_brightpathacademy.pdf
```

## 4. Chạy CLI hỏi‑đáp

Sau khi ingest và Qdrant đã chạy:

```bash
python scripts/query.py --mode hybrid_rerank --tenant brightpathacademy
```

- Gõ câu hỏi, `exit` để thoát, `/reset` để xoá lịch sử hội thoại.
- Các mode:
  - `vector` – chỉ dùng vector search.
  - `hybrid` – vector + BM25, không rerank cosine.
  - `hybrid_rerank` – hybrid + rerank cosine (đề xuất).

## 5. Chạy backend FastAPI

Khởi động backend:

```bash
uvicorn app.api.main:app --reload --port 8000
```

Truy cập:
- Docs tự động: `http://localhost:8000/docs`
- Test nhanh endpoint `/query` với body mẫu:

```json
{
  "question": "Trung tâm BrightPath có những chương trình nào?",
  "tenant_id": "brightpathacademy",
  "branch_id": null,
  "history": []
}
```

Backend sẽ trả:

```json
{
  "answer": "...",
  "sources": ["tenant_brightpathacademy.pdf", "..."]
}
```

## 6. Cấu trúc thư mục

- `src/config.py` – cấu hình chung (Qdrant, embedding, BM25, RAG).
- `src/embedding_model.py` – khởi tạo model embedding `BAAI/bge-m3`.
- `src/vector_store.py` – kết nối và tạo collection Qdrant.
- `src/ingest_pipeline.py` – pipeline ingest + chunk + persist nodes.
- `src/lexical_bm25.py` – triển khai BM25 và hybrid retrieval.
- `src/incontext_ralm.py` – logic RAG 2.0 + few-shot + rerank cosine.
- `src/rag_engine.py` – “trái tim” RAG dùng chung cho CLI và backend.
- `scripts/ingest.py` – CLI để ingest dữ liệu.
- `scripts/query.py` – CLI chat với RAG.
- `app/api/main.py` – FastAPI backend (healthcheck + `/query`).
- `web/` – frontend demo (HTML/CSS/JS).

## 7. Ghi chú phát triển

Các ý tưởng, roadmap và log phát triển chi tiết nằm trong:
- `data/giai đoạn phát triển.docx`
- `data/Log_phat_trien.docx`

Đây là nơi mô tả các giai đoạn RAG 1.0 → 2.0, hybrid, multi‑tenant và kế hoạch đánh giá. 
