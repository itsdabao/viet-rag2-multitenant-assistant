# Backend (Owner Console MVP – Local-first)

File này là **phần Backend theo hướng “Owner Console MVP – local-first”** để đưa vào tài liệu/đồ án, **khác mục đích** với:
- `docs/backend.md`: tổng hợp backend “as-built” theo các thay đổi Day 4 → Day 9.
- `docs/backend_owner_mvp.md`: spec ngắn gọn về Owner Console MVP.

Nội dung dưới đây **tổng hợp và chuẩn hoá** theo 2 file trên, đồng thời **gắn với cấu trúc code hiện có** để dễ triển khai/mở rộng.

---

## 1) Mục tiêu & phạm vi

### 1.1 Mục tiêu hệ thống
Backend phục vụ hệ thống trợ lý RAG tiếng Việt đa tenant (SaaS-ready) cho trung tâm Anh ngữ/doanh nghiệp nội bộ:
- Trả lời hỏi-đáp dựa trên tri thức nội bộ đã ingest.
- Cô lập dữ liệu theo tenant (và tuỳ chọn theo chi nhánh/branch).
- Agentic workflow: tiền xử lý → định tuyến (router) → gọi tool hoặc RAG.
- Lưu memory bền vững theo session (phục vụ hội thoại dài).
- Ghi trace/feedback/handoff để có dashboard quan sát và debug.

### 1.2 Phạm vi MVP Owner Console (local-first)
MVP tập trung vào **Owner (quản trị viên hệ thống/chủ platform)**:
- Quan sát tổng quan theo tenant/thời gian: tổng request, lỗi, p95 latency, satisfaction, handoff rate.
- Xem danh sách logs/traces, lọc theo tenant/route/status/từ khoá.
- Xem danh sách handoff tickets và trạng thái `new|contacted|closed`.
- Frontend Owner Console là **web tĩnh** (HTML/CSS/JS) được FastAPI serve; logic “call model” vẫn nằm ở backend.

---

## 2) Kiến trúc tổng quan

### 2.1 Các thành phần chính
- **FastAPI API server**: cung cấp HTTP + WebSocket, serve UI web tĩnh.
- **RAG Core**: pipeline retrieval (Qdrant + BM25) + rerank + prompt + LLM.
- **Agentic Router + Tools**: quyết định route và chạy các tool nghiệp vụ (ví dụ tính học phí, so sánh, tạo ticket/handoff).
- **Qdrant (Vector DB)**: lưu embeddings + payload metadata, bắt buộc filter theo tenant để tránh leakage.
- **Postgres (hoặc DB qua SQLAlchemy)**:
  - Memory theo session (`chat_sessions`).
  - Analytics (`request_traces`, `user_feedback`, `handoff_tickets`).
- **Owner Console (Static Web)**: dashboard/filters, gọi các endpoint “owner/admin API”.

Gợi ý mapping vào code hiện có:
- API: `app/api/main.py`
- RAG entrypoint: `app/services/rag_service.py`
- Config runtime: `app/core/config.py`
- Analytics store: `app/services/analytics/store.py`
- Memory store: `app/services/memory/store.py`

### 2.2 Dataflow chat (HTTP/WS)
1) Client gửi request:
   - HTTP: `POST /query` (demo/public) hoặc `POST /chat` (protected)
   - WS: `WS /ws/query` (streaming)
2) Backend gọi `rag_query()`:
   - Nếu có `tenant_id` và có `session_id`/`user_id` → dùng nhánh **memory_rag_query** (memory bền vững).
   - Ngược lại → dùng **agentic_query** (router → tool hoặc RAG).
3) Nếu route = RAG (course_search):
   - Retrieve hybrid: Qdrant vector + BM25 lexical
   - (Optional) cosine rerank
   - Build prompt (system + history + contexts + few-shot) → gọi LLM provider.
4) Backend ghi trace (`request_traces`) + (optional) feedback/handoff.
5) Trả về `answer`, `sources[]`, `route`, `trace_id`, `time_ms`.

### 2.3 Dataflow Owner Console (dashboard/logs/handoffs)
1) Owner mở trang dashboard (web tĩnh) được serve bởi backend.
2) UI gọi API để lấy metrics/logs/handoffs (có filter).
3) Backend query dữ liệu analytics đã ghi trong DB và trả JSON để UI render.

---

## 3) Thiết kế lưu trữ dữ liệu

### 3.1 Qdrant (Vector DB)
Vai trò: lưu chunks đã embedding để retrieval theo ngữ nghĩa.

Các điểm bắt buộc để “multi-tenant isolation”:
- Payload metadata phải có `tenant_id` và được filter khi retrieve.
- Khi bật multi-branch: thêm `branch_id` vào metadata và filter kèm.

Mapping config hiện có:
- Collection: `COLLECTION_NAME` (mặc định `RAG_docs`) trong `app/core/config.py`
- Keys filter:
  - `TENANT_FIELD = "metadata.tenant_id"`
  - `BRANCH_FIELD = "metadata.branch_id"`
- Quy tắc an toàn: nếu không áp được filter ở retrieval layer → **fail-closed** (trả no results).

### 3.2 Postgres (Memory + Analytics)

#### A) Memory theo session: `chat_sessions`
Mục tiêu: hỗ trợ hội thoại dài với chi phí prompt hợp lý:
- `entity_memory` (JSONB): các thuộc tính/biến nghiệp vụ quan trọng (ví dụ tổng thanh toán, scope giảm…).
- `rolling_summary` (TEXT): tóm tắt diễn biến hội thoại.
- `recent_messages_buffer` (JSONB): giữ last N turns.

Quy tắc an toàn:
- Khi load/update session phải check `tenant_id` khớp `session_id` (fail-closed nếu mismatch).

Mapping hiện có: `app/services/memory/store.py`.

#### B) Analytics: `request_traces`, `user_feedback`, `handoff_tickets`
Mục tiêu: dashboard và debug theo tenant/route/thời gian.

1) `request_traces` (1 dòng / request)
- Fields tối thiểu nên có: `trace_id`, `ts`, `tenant_id`, `route`, `status`, `latency_ms`, `sources`, `tool_metadata`, `error`, `question`, `answer`.

2) `user_feedback`
- Fields: `id`, `ts`, `trace_id`, `tenant_id`, `rating` (1 hoặc -1), `comment`.

3) `handoff_tickets`
- Fields: `id`, `ts`, `tenant_id`, `branch_id`, `user_id`, `phone`, `message`, `status`, `meta`.

Mapping hiện có: `app/services/analytics/store.py`.

Chỉ mục khuyến nghị (MVP):
- `request_traces(ts)`, `request_traces(tenant_id)`, `request_traces(route)`
- `handoff_tickets(ts)`, `handoff_tickets(tenant_id)`
- `user_feedback(trace_id)`

---

## 4) Thiết kế API (hướng Owner Console MVP)

### 4.1 Nhóm chat (public demo)
Phục vụ demo/web streaming; có thể mở (không auth) ở môi trường dev:
- `POST /query`
  - Input: `question, tenant_id?, branch_id?, history?, session_id?, user_id?`
  - Output: `answer, sources[], trace_id, time_ms, route`
- `WS /ws/query`
  - Streaming: `meta (sources/time_ms/trace_id/route)` + `chunk` + `end`.

Mapping hiện có: `app/api/main.py`.

### 4.2 Nhóm chat protected (tenant)
Phục vụ “tenant chat” khi có auth:
- `POST /chat`
  - Header: `Authorization: Bearer <firebase_id_token>`
  - tenant_id lấy từ token (custom claim hoặc fallback uid) và enforce tenant isolation.

Mapping hiện có: `app/api/main.py`, `app/api/deps.py`.

### 4.3 Nhóm Owner Console (spec local-first)
Trong MVP Owner Console “chuẩn spec”:
- `POST /owner/auth/login` → set cookie httpOnly `owner_token=<jwt>`
- `POST /owner/auth/logout` → clear cookie
- Tất cả `/owner/api/*` đi qua `require_owner()`
- `GET /owner` → serve UI Owner Console
- `GET /owner/api/metrics?tenant_id=&since=&until=`
- `GET /owner/api/logs?tenant_id=&since=&until=&route=&status=&q=&limit=&offset=`
- `GET /owner/api/logs/{trace_id}`
- `GET /owner/api/handoffs?tenant_id=&since=&until=&status=&limit=&offset=`

Lưu ý triển khai:
- Code hiện tại đã có nhóm endpoint tương đương dưới `/admin` (dashboard/logs/handoffs/feedback). Owner Console MVP có thể:
  1) Giữ `/admin` cho prototype nội bộ, hoặc
  2) Tách hẳn thành `/owner` + auth (JWT cookie) đúng spec trong `docs/backend_owner_mvp.md`.

---

## 5) Observability & Trace

### 5.1 Trace “đủ để debug”
Mỗi request chat nên ghi:
- `trace_id`, `ts`
- `tenant_id`, `branch_id` (optional), `channel`
- `session_id`, `user_id`
- `route`, `status`, `latency_ms`
- `sources[]`, `tool_metadata{}`, `error` (nếu có)

### 5.2 Token usage / cost (phase sau)
Nếu provider trả usage thì lưu `input_tokens/output_tokens/total_tokens`; nếu không có thì ước tính `len(chars)/4`.

---

## 6) Bảo mật & chống leak dữ liệu

Checklist bắt buộc:
1) Retrieval từ Qdrant phải filter theo `metadata.tenant_id` (và `metadata.branch_id` nếu bật branch).
2) Memory session phải fail-closed khi `tenant_id` mismatch.
3) Khi không áp được filters → trả no results (không “đoán bừa” dựa trên dữ liệu khác tenant).
4) PII: số điện thoại nên mask khi hiển thị trên UI logs, nhưng có thể lưu raw trong `handoff_tickets.phone`.

---

## 7) Runtime config quan trọng (gợi ý)

- Postgres/DB: `DATABASE_URL`
- Qdrant: `QDRANT_HOST`, `QDRANT_PORT`, `COLLECTION_NAME`
- Memory: `MEMORY_ENABLED`, `MEMORY_LAST_TURNS`, `MEMORY_BUDGET_TOKENS`, `MEMORY_SUMMARY_ENABLED`
- Owner auth (local-first spec): `OWNER_USERNAME`, `OWNER_PASSWORD`, `JWT_SECRET`, `JWT_EXPIRE_MIN`
- Tenant protected chat: `FIREBASE_SERVICE_ACCOUNT_PATH`
- LLM provider: `LLM_PROVIDER` + key/model tương ứng (Groq/Gemini/OpenAI-compatible…)

