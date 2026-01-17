# Backend Spec (Owner Console MVP – Local-first)

Ref: `backend.docx` (Chương 6)  
Scope của file này: chốt **giai đoạn Owner Console (Quản trị viên hệ thống)** chạy **local** trước; phần Tenant Admin (khách hàng) để Phase sau.

---

## 0) Mục tiêu

Backend phục vụ hệ thống RAG đa tenant (SaaS-ready) cho trung tâm Anh ngữ/doanh nghiệp nội bộ.

Trong MVP này, tập trung vào **Owner (quản trị viên hệ thống/chủ platform)**:
- Có dashboard theo tenant / theo thời gian.
- Xem logs/traces theo tenant/route/status.
- Xem handoff tickets (new/contacted/closed).
- UI là **web tĩnh** (HTML/CSS/JS) nhưng vẫn **call LLM** thông qua API backend.

---

## 1) Khái niệm & vai trò

### 1.1 Owner (Quản trị viên hệ thống)
Người tạo ra platform, nắm toàn bộ cấu hình và dữ liệu vận hành:
- Quan sát toàn bộ tenants (hoặc giai đoạn MVP có thể chỉ 1 tenant).
- Xem lưu lượng, latency, lỗi, tỷ lệ hài lòng, tỷ lệ handoff.
- Xem chi tiết traces để debug: route nào chạy, nguồn nào được trích, tool nào được dùng.

### 1.2 Tenant Admin (Khách hàng)
Phase sau. Trong docx có, nhưng MVP Owner chưa cần đầy đủ.

---

## 2) “Web tĩnh nhưng vẫn call model” nghĩa là gì?

- **Web tĩnh**: frontend chỉ là file `*.html/*.css/*.js` được FastAPI serve qua `/static/*` hoặc `/owner/*`.
- **Call model**: xảy ra ở backend khi endpoint chat (`/query` hoặc `/ws/query` hoặc `/chat`) gọi `rag_query()` → gọi LLM provider (Groq/Gemini/OpenAI-compatible).

=> Không cần Next.js vẫn có thể gọi LLM/RAG bình thường, vì việc chạy model nằm ở backend.

---

## 3) Tech Stack (local-first)

### 3.1 Backend
- **FastAPI**: HTTP API + WebSocket streaming.
- **Postgres**: lưu memory + analytics + handoff.
- **Qdrant**: lưu vector embeddings (chunks).
- **LlamaIndex**: orchestration RAG (retrieve + prompt + LLM).

### 3.2 Frontend (Owner Console)
- **Static web** (HTML/CSS/JS) để phản hồi nhanh, setup đơn giản.
- Owner Console gọi các endpoint `/owner/api/*`.

---

## 4) Database (MVP Owner)

### 4.1 Analytics tables (đã/đang dùng)
- `request_traces`
  - 1 dòng / request.
  - Fields tối thiểu: `trace_id`, `ts`, `tenant_id`, `route`, `status`, `latency_ms`, `sources`, `tool_metadata`, `error`, `question`, `answer`.
- `user_feedback`
  - Fields: `id`, `ts`, `trace_id`, `tenant_id`, `rating` (1 hoặc -1), `comment`.
- `handoff_tickets`
  - Fields: `id`, `ts`, `tenant_id`, `branch_id`, `user_id`, `phone`, `message`, `status`, `meta`.

### 4.2 Memory table (đã dùng cho chat)
- `chat_sessions`
  - `entity_memory` (JSONB), `rolling_summary`, `recent_messages_buffer`, `tenant_id`, timestamps.

### 4.3 Index (khuyến nghị)
- `request_traces(ts)`, `request_traces(tenant_id)`, `request_traces(route)`.
- `handoff_tickets(ts)`, `handoff_tickets(tenant_id)`.
- `user_feedback(trace_id)`.

---

## 5) Auth (Owner hardcode – local)

### 5.1 Env vars
- `OWNER_USERNAME`
- `OWNER_PASSWORD`
- `JWT_SECRET`
- `JWT_EXPIRE_MIN` (ví dụ 1440)

### 5.2 Endpoints (Owner Auth)
- `POST /owner/auth/login`
  - Input: `{ "username": "...", "password": "..." }`
  - Output: set cookie httpOnly (vd: `owner_token=<jwt>`), `{ "ok": true }`
- `POST /owner/auth/logout`
  - Output: clear cookie `owner_token`, `{ "ok": true }`

### 5.3 Rule
- Tất cả `/owner/api/*` phải đi qua dependency `require_owner()`.
- `require_owner()` lấy token từ cookie hoặc `Authorization: Bearer`.
- Nếu thiếu/invalid → `401 Unauthorized`.

---

## 6) API Surface (Owner Console MVP)

### 6.1 Serve UI
- `GET /owner` → trả trang Owner Console (HTML)
- `GET /static/*` → serve assets CSS/JS

### 6.2 Metrics (Dashboard)
- `GET /owner/api/metrics?tenant_id=&since=&until=`
  - `tenant_id` optional: nếu empty → metrics toàn hệ thống
  - `since/until` dạng `YYYY-MM-DD` hoặc epoch seconds
  - Response (gợi ý):
    - `total_requests`, `error_requests`
    - `avg_time_ms`, `p50_ms`, `p95_ms`
    - `satisfaction_rate` (up/(up+down)), `feedback_total`
    - `handoff_count`, `handoff_rate`

### 6.3 Logs / Traces list
- `GET /owner/api/logs?tenant_id=&since=&until=&route=&status=&q=&limit=&offset=`
  - Filter theo tenant/time/route/status/text search.
  - Response list:
    - `trace_id`, `ts`, `tenant_id`, `route`, `status`, `latency_ms`, `sources_count`, `question_preview`

### 6.4 Trace detail
- `GET /owner/api/logs/{trace_id}`
  - Response:
    - `trace_id`, `tenant_id`, `question`, `answer`, `sources[]`, `tool_metadata{}`, `latency_ms`, `error`

### 6.5 Handoffs
- `GET /owner/api/handoffs?tenant_id=&since=&until=&status=&limit=&offset=`
  - status: `new|contacted|closed` (optional)
  - Response list:
    - `id`, `ts`, `tenant_id`, `phone`, `message`, `status`

---

## 7) Observability: Trace “đủ để debug”

### 7.1 Trace fields tối thiểu
Mỗi request chat nên ghi:
- `trace_id`
- `tenant_id`, `branch_id` (optional)
- `channel` (cli/web/web_ws/tenant_chat…)
- `route` (course_search/tuition_calculator/comparison/create_ticket/out_of_domain…)
- `status` (SUCCESS/ERROR)
- `latency_ms`
- `sources[]` (file_name/doc_id)
- `tool_metadata` (JSON): computed numbers, fee extraction flags, retrieval metrics…
- `error` (nếu có)

### 7.2 Token usage (nếu muốn đưa vào dashboard)
MVP có thể:
- **Ưu tiên**: lấy `usage` từ response LLM (nếu provider hỗ trợ).
- **Fallback**: ước tính tokens = `len(chars)/4`.

Field gợi ý:
- `input_tokens`, `output_tokens`, `total_tokens`, `cost_estimate` (optional)
- `llm_provider`, `llm_model`

---

## 8) Luồng nghiệp vụ (MVP Owner)

### 8.1 Chat request → trace
1) User hỏi (web demo/cli/channel).
2) Backend: preprocess → router → tool hoặc course_search.
3) Nếu course_search: retrieve Qdrant (filter tenant) + BM25 + rerank + LLM.
4) Ghi `request_traces`.
5) Trả `trace_id` về frontend để copy/debug.

### 8.2 Handoff
1) Nếu router/tool quyết định handoff → tạo ticket.
2) Ghi `handoff_tickets` (để dashboard tính handoff rate).

### 8.3 Feedback
1) Web demo gửi 👍/👎 kèm `trace_id`.
2) Ghi `user_feedback`.
3) Dashboard tính satisfaction rate.

---

## 9) Quy tắc bảo mật (bắt buộc)

### 9.1 Tenant isolation
- Retrieval vector (Qdrant) luôn filter theo `tenant_id` (và `branch_id` nếu bật).
- Memory load/update phải check `tenant_id` khớp `session_id`.
- Owner console được xem mọi tenant (vì là chủ platform).

### 9.2 Fail-closed
- Nếu không áp được metadata filters khi retrieve → trả no result (tránh leak).

### 9.3 PII
- SĐT có thể mask trong logs UI (vd hiển thị `09****123`) nhưng vẫn lưu raw trong field `phone` của ticket.

---

## 10) Phase sau (không làm trong Owner MVP)

### 10.1 Tenant Admin Console
Gồm:
- Ingestion UI (upload/url/import) + job monitor
- Usage/token theo tenant
- Chat demo theo tenant (protected)

### 10.2 URL Import / ScrapeGraphAI (giải thích)
- “Import URL” = cào nội dung từ website (scrape) → chuyển thành markdown → chunk/embedding → index vào vector DB.
- ScrapeGraphAI chỉ là một cách triển khai (optional). MVP Owner không cần.
