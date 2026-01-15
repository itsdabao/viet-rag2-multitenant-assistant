from pathlib import Path
from qdrant_client.http.models import Distance

# Prompt templates
SYSTEM_PROMPT_PATH = "src/prompts/system_vi.md"

# Qdrant connection and collection
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "RAG_docs"

# Vector settings (must match embedding model dimension)
VECTOR_SIZE = 1024
VECTOR_DISTANCE = Distance.COSINE

# Embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Data locations
DATA_PATH = str(Path("data") / "knowledge_base")
NODES_CACHE_PATH = str(Path("data") / ".cache" / "nodes.jsonl")

# Chunking (shared for ingest and BM25 fallback)
# Options: "fixed_size", "document_based" (structure-based với ##/###), "semantic"
CHUNKING_STRATEGY = "document_based"  # "document_based" = Structure-based với Markdown ##/###
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Document-based (Structure-based) chunking settings
# Chia theo cấu trúc Markdown: ## (Section) và ### (Sub-section)
# Context Injection: Sub-section ### sẽ tự động có tiêu đề cha ## ở đầu
#
# === CHUNK SIZE RECOMMENDATIONS ===
# MIN_CHUNK_SIZE: Khuyến nghị >= 350 chars vì:
#   1. Chunks quá nhỏ (<200) không đủ context cho LLM hiểu
#   2. Embedding quality giảm với text quá ngắn (thiếu semantic information)
#   3. Tăng số lượng chunks làm chậm retrieval và tăng noise
#   4. Mỗi chunk cần đủ thông tin để "độc lập" trả lời câu hỏi
#
# MAX_CHUNK_SIZE: Khuyến nghị 1000-2000 chars vì:
#   1. Quá dài sẽ dilute relevant information
#   2. Embedding models có giới hạn tokens hiệu quả
#   3. Context window của LLM có limite
#
DOC_BASED_MIN_CHUNK_SIZE = 350  # Tăng từ 200 để đảm bảo đủ context
DOC_BASED_MAX_CHUNK_SIZE = 1500  # Kích thước tối đa của mỗi chunk

# In-Context RALM
FEWSHOT_PATH = str(Path("src") / "eval" / "test_case.json")
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

# LLM reliability
LLM_MAX_RETRIES = 3
LLM_RETRY_INITIAL_DELAY = 1.0
LLM_RETRY_BACKOFF = 2.0
LLM_429_SLEEP_SECS = 20
LLM_429_JITTER_SECS = 10