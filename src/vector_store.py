import qdrant_client
from qdrant_client.http.models import VectorParams
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from src.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, VECTOR_SIZE, VECTOR_DISTANCE


def init_qdrant_collection():
    """
    Kết nối tới Qdrant và đảm bảo collection tồn tại.
    Nếu collection chưa tồn tại -> tạo mới.
    Nếu đã tồn tại -> giữ nguyên dữ liệu cũ.
    """
    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("Đã kết nối tới Qdrant thành công.")

    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=VECTOR_DISTANCE,
            ),
        )
        print(f"Collection '{COLLECTION_NAME}' chưa tồn tại -> đã tạo mới thành công.")
    else:
        print(f"Collection '{COLLECTION_NAME}' đã tồn tại -> đang lưu dữ liệu cũ.")

    # Create payload index for tenant_id to enable fast filter per tenant
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="tenant_id",
            field_schema="keyword",
        )
        print("Đã tạo payload index cho 'tenant_id'.")
    except Exception:
        # ignore if exists or server doesn't support
        pass
    return client


def get_storage_context(client):
    """Tạo StorageContext từ Qdrant client để dùng trong ingest hoặc query."""
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("Storage context đã khởi tạo thành công.")
    return storage_context
