import logging
from typing import Dict, List, Optional

import qdrant_client
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.core.config import (
    COLLECTION_NAME,
    EXAMPLES_TOP_K,
    FEWSHOT_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    RETRIEVAL_TOP_K,
)
from app.core.llama import init_llm_from_env, setup_embedding
from app.services.retrieval.vector_store import init_qdrant_collection
from app.services.rag.incontext_ralm import query_with_incontext_ralm


logger = logging.getLogger(__name__)

_INDEX: Optional[VectorStoreIndex] = None


def build_index() -> VectorStoreIndex:
    """
    Khởi tạo embedding, kết nối Qdrant và load VectorStoreIndex dùng chung.
    Được cache vào biến toàn cục để tái sử dụng giữa các request.
    """
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    if Settings.embed_model is None:
        setup_embedding()

    # Ensure collection + payload indexes exist before loading the vector store
    client = init_qdrant_collection()
    logger.info("Qdrant ready at %s:%s", QDRANT_HOST, QDRANT_PORT)

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    _INDEX = VectorStoreIndex.from_vector_store(vector_store)
    logger.info("Đã tải index từ collection: %s", COLLECTION_NAME)
    return _INDEX


def rag_query(
    question: str,
    *,
    tenant_id: Optional[str] = None,
    branch_id: Optional[str] = None,
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
        branch_id=branch_id,
        history=history,
    )
    return result
