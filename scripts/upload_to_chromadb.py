import argparse
import json
import os
from pathlib import Path

from google.cloud import firestore_admin_v1


def _load_project_id(service_account_path: str) -> str:
    p = Path(service_account_path)
    if not p.exists():
        raise FileNotFoundError(f"Service account file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    pid = str(data.get("project_id") or "").strip()
    if not pid:
        raise ValueError("Missing `project_id` in service account JSON.")
    return pid


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan Firestore databases using a service account JSON.")
    parser.add_argument(
        "--service-account",
        default=(os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH") or "").strip(),
        help="Path to service account JSON (or set GOOGLE_APPLICATION_CREDENTIALS / FIREBASE_SERVICE_ACCOUNT_PATH).",
    )
    args = parser.parse_args()

    if not args.service_account:
        raise SystemExit(
            "Missing service account path. Provide --service-account or set GOOGLE_APPLICATION_CREDENTIALS / FIREBASE_SERVICE_ACCOUNT_PATH."
        )

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.service_account
    project_id = _load_project_id(args.service_account)

    print(f"--- Đang quét toàn bộ database trong Project: {project_id} ---")
    client = firestore_admin_v1.FirestoreAdminClient()

    parent = f"projects/{project_id}"
    databases = client.list_databases(parent=parent)

    found = False
    for db in databases:
        found = True
        print("Tìm thấy Database:")
        print(f" - ID: {db.name.split('/')[-1]}")
        print(f" - Location: {db.location_id}")
        print(f" - Type: {db.type_}")
        print(f" - State: {db.state}")

    if not found:
        print("❌ KHÔNG tìm thấy bất kỳ database nào. Hãy kiểm tra lại project_id trong file JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
