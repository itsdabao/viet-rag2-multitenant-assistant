import os
import sys
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

import qdrant_client
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.google_genai import GoogleGenAI

# Đảm bảo import được gói src/* khi chạy trực tiếp file này
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
import src.config as cfg


def init_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )


def init_llm_from_env():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Không tìm thấy GOOGLE_API_KEY. Vui lòng thêm vào .env hoặc biến môi trường."
        )
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    Settings.llm = GoogleGenAI(model_name=model, api_key=api_key)
    logging.info(f"Đã khởi tạo LLM GoogleGenAI với model: {model}")


def build_index():
    # Khởi tạo embedding (dùng chung)
    setup_embedding()

    # Kết nối Qdrant và tải index
    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logging.info(f"Đã kết nối Qdrant tại {QDRANT_HOST}:{QDRANT_PORT}")

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store)
    logging.info(f"Đã tải index từ collection: {COLLECTION_NAME}")

    return index


def main():
    init_logging()
    parser = argparse.ArgumentParser(description="Hybrid RAG (In-Context RALM)")
    parser.add_argument("--debug", action="store_true", help="Bật debug chi tiết")
    parser.add_argument(
        "--mode",
        choices=["vector", "hybrid", "hybrid_rerank"],
        default=None,
        help="Chế độ: vector | hybrid | hybrid_rerank",
    )
    parser.add_argument("--tenant", default=None, help="Tenant ID để lọc dữ liệu truy hồi")
    args = parser.parse_args()

    # Ghi đè cấu hình theo cờ tối giản
    if args.debug:
        cfg.DEBUG_VERBOSE = True
        cfg.DEBUG_SHOW_PROMPT = True
    if args.mode:
        if args.mode == "vector":
            cfg.USE_BM25 = False
            cfg.RERANK_USE_COSINE = False
        elif args.mode == "hybrid":
            cfg.USE_BM25 = True
            cfg.RERANK_USE_COSINE = False
        else:  # hybrid_rerank
            cfg.USE_BM25 = True
            cfg.RERANK_USE_COSINE = True

    print("Đang khởi tạo. Vui lòng chờ...")

    try:
        init_llm_from_env()
        index = build_index()
        print('--- Sẵn sàng. Nhập câu hỏi (gõ "exit" để thoát, gõ "/reset" để xoá lịch sử) ---')

        if args.tenant:
            print(f"Tenant: {args.tenant}")
        history: list[dict] = []
        while True:
            user_query = input("\nBạn hỏi: ")
            if user_query.strip().lower() == "exit":
                print("Tạm biệt!")
                break
            if user_query.strip().lower() == "/reset":
                history.clear()
                print("Đã xoá lịch sử hội thoại.")
                continue

            result = query_with_incontext_ralm(
                user_query,
                index,
                fewshot_path=FEWSHOT_PATH,
                top_k_ctx=RETRIEVAL_TOP_K,
                top_k_examples=EXAMPLES_TOP_K,
                tenant_id=args.tenant,
                history=history,
            )

            print("\nAnswer:")
            print(result.get("answer", ""))
            sources = result.get("sources", [])
            if sources:
                print("\nSources:")
                for s in sources:
                    print(f" - {s}")

            # Lưu lịch sử hội thoại (stateful theo session CLI)
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": str(result.get("answer", ""))})
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")


if __name__ == "__main__":
    main()
