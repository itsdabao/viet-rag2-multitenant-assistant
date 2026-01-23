import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from app.services.ingestion import run_ingestion
from app.core.config import DATA_PATH

def ingest_global():
    print("Ingesting Global Knowledge...")
    file_path = os.path.join(DATA_PATH, "general_concepts.md")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Ingest with special tenant_id
    run_ingestion(
        tenant_id="global_public",
        input_files=[file_path]
    )
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_global()
