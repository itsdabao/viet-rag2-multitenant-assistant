import sys
import logging
import argparse
from pathlib import Path

# Đảm bảo import được gói src/* khi chạy trực tiếp file này
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.config as cfg
from src.rag_engine import init_llm_from_env, build_index, rag_query


def init_logging() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )


def main() -> None:
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
        build_index()
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

            result = rag_query(
                user_query,
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

