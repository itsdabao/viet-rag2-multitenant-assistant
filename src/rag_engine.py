import logging
import os
from typing import List, Dict, Optional

import qdrant_client
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    FEWSHOT_PATH,
    RETRIEVAL_TOP_K,
    EXAMPLES_TOP_K,
)
from src.embedding_model import setup_embedding
from src.incontext_ralm import query_with_incontext_ralm


logger = logging.getLogger(__name__)

_INDEX: Optional[VectorStoreIndex] = None


def init_llm_from_env() -> None:
    """Khởi tạo LLM Google Gemini từ biến môi trường / file .env."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Không tìm thấy GOOGLE_API_KEY. Vui lòng thêm vào .env hoặc biến môi trường."
        )
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    Settings.llm = GoogleGenAI(model_name=model, api_key=api_key)
    logger.info("Đã khởi tạo LLM GoogleGenAI với model: %s", model)


def build_index() -> VectorStoreIndex:
    """
    Khởi tạo embedding, kết nối Qdrant và load VectorStoreIndex dùng chung.
    Được cache vào biến toàn cục để tái sử dụng giữa các request.
    """
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    setup_embedding()

    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info("Đã kết nối Qdrant tại %s:%s", QDRANT_HOST, QDRANT_PORT)

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    _INDEX = VectorStoreIndex.from_vector_store(vector_store)
    logger.info("Đã tải index từ collection: %s", COLLECTION_NAME)
    return _INDEX


def rag_query(
    question: str,
    *,
    tenant_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, object]:
    """
    Hàm tiện ích RAG dùng chung cho CLI và backend.
    """
    index = build_index()
    result = query_with_incontext_ralm(
        user_query=question,
        index=index,
        fewshot_path=FEWSHOT_PATH,
        top_k_ctx=RETRIEVAL_TOP_K,
        top_k_examples=EXAMPLES_TOP_K,
        tenant_id=tenant_id,
        history=history,
    )
    return result

