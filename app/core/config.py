from __future__ import annotations

import os
from pathlib import Path

from qdrant_client.http.models import Distance


PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    # Load repo-root `.env` so env-driven settings work even when `cwd` is different.
    load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"))
except Exception:
    pass

# Prompt templates
SYSTEM_PROMPT_PATH = str(PROJECT_ROOT / "app" / "resources" / "prompts" / "system_vi.md")

# Smalltalk / cheap routing (avoid LLM calls)
SMALLTALK_PATH = str(PROJECT_ROOT / "app" / "resources" / "smalltalk_vi.json")
ENABLE_SMALLTALK = True
SMALLTALK_COSINE_THRESHOLD = 0.78

# Agentic router (Day 4)
ENABLE_SEMANTIC_ROUTER = True
TOXIC_MESSAGE = "Dạ em xin phép không hỗ trợ nội dung này. Anh/chị cần tư vấn khóa học/học phí/lịch học nào của trung tâm ạ?"

# In-domain anchors (cheap pre-check before retrieval)
DOMAIN_ANCHORS_PATH = str(PROJECT_ROOT / "app" / "resources" / "domain_anchors_vi.json")
DOMAIN_ANCHOR_COSINE_THRESHOLD = 0.25
DOMAIN_KEYWORDS = [
    # Vietnamese (with/without accents are handled by code)
    "trung tâm",
    "trung tam",
    "ở đâu",
    "o dau",
    "địa điểm",
    "dia diem",
    "học phí",
    "hoc phi",
    "lịch học",
    "lich hoc",
    "khai giảng",
    "khai giang",
    "ưu đãi",
    "uu dai",
    "địa chỉ",
    "dia chi",
    "cơ sở",
    "co so",
    "chi nhánh",
    "chi nhanh",
    "giao tiếp",
    "giao tiep",
    "ielts",
    "toeic",
    "thiếu nhi",
    "thieu nhi",
    "đầu vào",
    "dau vao",
    "xếp lớp",
    "xep lop",
    "lộ trình",
    "lo trinh",
    "giáo viên",
    "giao vien",
]

# Qdrant connection and collection
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "RAG_docs"

# Multi-tenant / multi-branch isolation
# NOTE:
# - Node metadata keys are `tenant_id` / `branch_id` (see `app/services/ingestion.py`).
# - Qdrant stores node metadata under the payload key `metadata` (so the Qdrant *payload path* is `metadata.tenant_id`).
# LlamaIndex will translate metadata filters to the underlying vector-store filter format, so keep these as metadata keys.
TENANT_FIELD = "tenant_id"
BRANCH_FIELD = "branch_id"
ENABLE_BRANCH_FILTER = False
# If True: refuse to retrieve without tenant_id (recommended for production SaaS)
REQUIRE_TENANT_ID = False
# If True: if metadata filters cannot be applied at retrieval layer, return no results (avoid leakage)
ENFORCE_METADATA_FILTERS = True

# Vector settings (must match embedding model dimension)
VECTOR_SIZE = 1024
VECTOR_DISTANCE = Distance.COSINE

# Embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Data locations
DATA_PATH = str(PROJECT_ROOT / "data" / "knowledge_base")
NODES_CACHE_PATH = str(PROJECT_ROOT / "data" / ".cache" / "nodes.jsonl")

# Chunking (shared for ingest and BM25 fallback)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# In-Context RALM
FEWSHOT_PATH = str(PROJECT_ROOT / "app" / "resources" / "eval" / "fewshot_examples.json")
RETRIEVAL_TOP_K = 5 
EXAMPLES_TOP_K = 3

# BM25 / Hybrid retrieval
USE_BM25 = True # bật tắt hybrid/lexcial search 
BM25_SOURCE = "nodes_file"  # options: "nodes_file", "files"
BM25_TOP_K = 5
BM25_MAX_CHARS = 800  # used only in legacy files mode
BM25_K1 = 1.5
BM25_B = 0.75
HYBRID_ALPHA = 0.5  # 1.0 = vector-only, 0.0 = BM25-only

# Debug/trace controls (default OFF to avoid leaking prompts/PII into logs)
DEBUG_VERBOSE = (os.getenv("DEBUG_VERBOSE") or "0").strip().lower() in ("1", "true", "yes", "on")
DEBUG_TOPN_PRINT = int(os.getenv("DEBUG_TOPN_PRINT") or "3")
DEBUG_SHOW_PROMPT = (os.getenv("DEBUG_SHOW_PROMPT") or "0").strip().lower() in ("1", "true", "yes", "on")

# Postgres (Day 6-7 persistent memory)
# Prefer configuring via env/.env; keep code default empty to avoid hardcoding credentials.
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
CHAT_SESSIONS_TABLE = (os.getenv("CHAT_SESSIONS_TABLE") or "chat_sessions").strip()

# Memory controls (Day 6-7)
MEMORY_ENABLED = (os.getenv("MEMORY_ENABLED") or "1").strip().lower() in ("1", "true", "yes", "on")
MEMORY_LAST_TURNS = int(os.getenv("MEMORY_LAST_TURNS") or "6")  # 6 turns = 12 messages (user+assistant)
MEMORY_BUDGET_TOKENS = int(os.getenv("MEMORY_BUDGET_TOKENS") or "1000")  # apply to (summary + last N turns)
MEMORY_SUMMARY_ENABLED = (os.getenv("MEMORY_SUMMARY_ENABLED") or "1").strip().lower() in ("1", "true", "yes", "on")
MEMORY_SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("MEMORY_SUMMARY_MAX_OUTPUT_TOKENS") or "350")

# Prompt budget controls
MAX_PROMPT_CHARS = 9000
PER_CHUNK_PROMPT_MAX_CHARS = 1000
PROMPT_TOP_CONTEXTS = 3  # clamp exact top-N contexts sent to LLM after rerank

# Conversation memory (stateful)
HISTORY_ENABLED = True
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS") or "12")  # default ~6 turns (user+assistant)
HISTORY_MSG_MAX_CHARS = 300      # cắt mỗi message để giữ prompt gọn

# Reranking controls
RERANK_USE_COSINE = True
RERANK_TOP_M = 20
RERANK_WEIGHT = 0.5  # combine fused score and cosine (0..1)

# In-domain / out-of-domain guard (avoid LLM calls on low-confidence retrieval)
ENABLE_DOMAIN_GUARD = True
# cosine(query, best_chunk_text) below this → treat as out-of-domain
DOMAIN_COSINE_THRESHOLD = 0.33
OUT_OF_DOMAIN_MESSAGE = (
    "Dạ em chỉ hỗ trợ tư vấn các thông tin liên quan đến trung tâm (khóa học, học phí, lịch học, ưu đãi...). "
    "Nếu anh/chị cho em xin **SĐT** và **nhu cầu học**, em sẽ chuyển tư vấn viên liên hệ hỗ trợ chi tiết hơn ạ."
)

# In-domain but not enough evidence in the tenant knowledge base.
NO_MATCH_MESSAGE = (
    "Dạ hiện tại em **chưa tìm thấy thông tin phù hợp trong tài liệu của trung tâm** để trả lời câu này. "
    "Anh/chị có thể hỏi cụ thể hơn (tên khóa/IELTS/TOEIC/lịch học/học phí/ưu đãi...) giúp em được không ạ? "
    "Nếu anh/chị cho em xin **SĐT** và **nhu cầu học**, em sẽ chuyển tư vấn viên hỗ trợ chi tiết hơn ạ."
)

# LLM reliability
LLM_MAX_RETRIES = 3
LLM_RETRY_INITIAL_DELAY = 1.0
LLM_RETRY_BACKOFF = 2.0
LLM_429_SLEEP_SECS = 20
LLM_429_JITTER_SECS = 10

# LLM fallback behavior (when quota/network/provider errors happen)
LLM_FALLBACK_TO_CONTEXT_ON_ERROR = True
LLM_FALLBACK_CONTEXT_SNIPPET_CHARS = 1200
