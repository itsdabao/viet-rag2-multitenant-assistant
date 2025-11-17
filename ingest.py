import argparse
import logging
from pathlib import Path
import re
from src.ingest_pipeline import run_ingestion
from src.config import DATA_PATH

logging.basicConfig(
    filename="logs/ingest.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def _derive_tenant_from_filename(path: Path) -> str:
    name = path.stem
    # Nếu tên bắt đầu bằng tenant_, lấy phần sau; ngược lại dùng nguyên tên
    if name.lower().startswith("tenant_"):
        name = name[len("tenant_") :]
    # Chuẩn hoá: chữ thường, chỉ giữ a-z0-9 và _
    safe = re.sub(r"[^a-z0-9_]+", "", name.lower())
    return safe or name.lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest pipeline")
    parser.add_argument("--tenant", default=None, help="Tenant ID (gắn vào metadata)")
    parser.add_argument("--file", default=None, help="Đường dẫn file đơn lẻ để ingest")
    parser.add_argument(
        "--auto-from-filenames",
        action="store_true",
        help="Quét DATA_PATH và ingest từng file, suy ra tenant từ tên file (tenant_*.pdf)",
    )
    args = parser.parse_args()

    try:
        if args.auto_from_filenames:
            base = Path(DATA_PATH)
            if not base.exists():
                raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {base}")
            # Chỉ ingest các định dạng phổ biến
            exts = {".pdf", ".md", ".txt", ".docx", ".rtf"}
            files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts]
            if not files:
                print("Không tìm thấy file nào để ingest.")
            else:
                print(f"Tìm thấy {len(files)} file để ingest trong {base.resolve()}.")
            for fp in files:
                tenant = _derive_tenant_from_filename(fp)
                print(f"\n[Auto] Ingest tenant={tenant} file={fp}")
                run_ingestion(tenant_id=tenant, input_files=[str(fp)])
            print("\nIngest hàng loạt hoàn tất.")
        else:
            if args.file:
                run_ingestion(tenant_id=args.tenant, input_files=[args.file])
            else:
                run_ingestion(tenant_id=args.tenant)
            print("Ingest hoàn tất.")
    except Exception as e:
        logging.error(f"Lỗi khi ingest: {e}")
        print(f"Lỗi: {e}")
