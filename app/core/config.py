from __future__ import annotations

from pathlib import Path

from qdrant_client.http.models import Distance


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Prompt templates
SYSTEM_PROMPT_PATH = str(PROJECT_ROOT / "app" / "resources" / "prompts" / "system_vi.md")

# Smalltalk / cheap routing (avoid LLM calls)
SMALLTALK_PATH = str(PROJECT_ROOT / "app" / "resources" / "smalltalk_vi.json")
ENABLE_SMALLTALK = True
SMALLTALK_COSINE_THRESHOLD = 0.78

# In-domain anchors (cheap pre-check before retrieval)
DOMAIN_ANCHORS_PATH = str(PROJECT_ROOT / "app" / "resources" / "domain_anchors_vi.json")
DOMAIN_ANCHOR_COSINE_THRESHOLD = 0.25
DOMAIN_KEYWORDS = [
    # Vietnamese (with/without accents are handled by code)
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
# NOTE: LlamaIndex Qdrant integration stores node metadata under the payload key `metadata` by default.
# Using dot-path keys ensures filters match the stored payload structure and prevents cross-tenant leakage.
TENANT_FIELD = "metadata.tenant_id"
BRANCH_FIELD = "metadata.branch_id"
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
HYBRID_ALPHA = 0.4  # 1.0 = vector-only, 0.0 = BM25-only

# Debug/trace controls
DEBUG_VERBOSE = True
DEBUG_TOPN_PRINT = 3
DEBUG_SHOW_PROMPT = True

# Prompt budget controls
MAX_PROMPT_CHARS = 9000
PER_CHUNK_PROMPT_MAX_CHARS = 1000
PROMPT_TOP_CONTEXTS = 3  # clamp exact top-N contexts sent to LLM after rerank

# Conversation memory (stateful)
HISTORY_ENABLED = True
HISTORY_MAX_TURNS = 4            # số message gần nhất (user/assistant) đưa vào prompt
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

# LLM reliability
LLM_MAX_RETRIES = 3
LLM_RETRY_INITIAL_DELAY = 1.0
LLM_RETRY_BACKOFF = 2.0
LLM_429_SLEEP_SECS = 20
LLM_429_JITTER_SECS = 10

# LLM fallback behavior (when quota/network/provider errors happen)
LLM_FALLBACK_TO_CONTEXT_ON_ERROR = True
LLM_FALLBACK_CONTEXT_SNIPPET_CHARS = 1200
