import qdrant_client
from qdrant_client.http.models import VectorParams
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
import types
from app.core.config import (
    BRANCH_FIELD,
    COLLECTION_NAME,
    ENABLE_BRANCH_FILTER,
    QDRANT_HOST,
    QDRANT_PORT,
    TENANT_FIELD,
    VECTOR_DISTANCE,
    VECTOR_SIZE,
)

def _ensure_qdrant_client_compat(client) -> None:
    """
    Ensure qdrant-client API compatibility for dependencies that expect `client.search(...)`.

    Some qdrant-client versions expose `search_points(...)` but not `search(...)`.
    LlamaIndex's Qdrant integration may call `client.search(...)`.
    """
    if hasattr(client, "search"):
        return

    def _search(self, collection_name, query_vector=None, query_filter=None, **kwargs):
        # Map common parameter names across versions.
        if "query_vector" in kwargs and query_vector is None:
            query_vector = kwargs.pop("query_vector")
        if "query_filter" in kwargs and query_filter is None:
            query_filter = kwargs.pop("query_filter")
        if "search_params" in kwargs and "params" not in kwargs:
            kwargs["params"] = kwargs.pop("search_params")

        # qdrant-client newer API uses `vector` + `filter`
        if "vector" not in kwargs:
            kwargs["vector"] = query_vector
        if "filter" not in kwargs and query_filter is not None:
            kwargs["filter"] = query_filter

        if hasattr(self, "search_points"):
            return self.search_points(collection_name=collection_name, **kwargs)

        # Fallback to the generated HTTP client if present
        http = getattr(self, "http", None)
        points_api = getattr(http, "points_api", None) if http is not None else None
        if points_api is not None and hasattr(points_api, "search_points"):
            try:
                from qdrant_client.http.models import SearchRequest

                req = SearchRequest(
                    vector=kwargs.get("vector"),
                    filter=kwargs.get("filter"),
                    limit=kwargs.get("limit"),
                    with_payload=kwargs.get("with_payload"),
                    with_vectors=kwargs.get("with_vectors"),
                    score_threshold=kwargs.get("score_threshold"),
                    params=kwargs.get("params"),
                )
                return points_api.search_points(collection_name=collection_name, search_request=req)
            except Exception:
                pass

        raise AttributeError("Qdrant client has neither `search` nor compatible `search_points` methods.")

    # Some qdrant-client builds may restrict setting new instance attributes; patch class if needed.
    try:
        client.search = types.MethodType(_search, client)
    except Exception:
        try:
            setattr(client.__class__, "search", _search)
        except Exception:
            pass


def init_qdrant_collection():
    """
    Kết nối tới Qdrant và đảm bảo collection tồn tại.
    Nếu collection chưa tồn tại -> tạo mới.
    Nếu đã tồn tại -> giữ nguyên dữ liệu cũ.
    """
    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    _ensure_qdrant_client_compat(client)
    # Eager connectivity check so we fail fast with a clear message.
    try:
        _ = client.get_collections()
    except Exception as e:
        raise RuntimeError(
            f"Không kết nối được Qdrant tại http://{QDRANT_HOST}:{QDRANT_PORT}.\n"
            "Hãy chắc chắn Qdrant đang chạy (ví dụ Docker):\n"
            "  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant\n"
            "Hoặc chỉnh host/port trong `app/core/config.py` nếu bạn chạy Qdrant ở nơi khác.\n"
            f"Lỗi gốc: {e}"
        ) from e
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

    # Create payload index for tenant_id / branch_id to enable fast filter per tenant
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=TENANT_FIELD,
            field_schema="keyword",
        )
        print(f"Đã tạo payload index cho '{TENANT_FIELD}'.")
    except Exception:
        # ignore if exists or server doesn't support
        pass

    if ENABLE_BRANCH_FILTER:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=BRANCH_FIELD,
                field_schema="keyword",
            )
            print(f"Đã tạo payload index cho '{BRANCH_FIELD}'.")
        except Exception:
            pass
    return client


def get_storage_context(client):
    """Tạo StorageContext từ Qdrant client để dùng trong ingest hoặc query."""
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("Storage context đã khởi tạo thành công.")
    return storage_context
