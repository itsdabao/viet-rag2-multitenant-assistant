from pathlib import Path
from typing import List, Optional
from llama_index.core import VectorStoreIndex
from src.data_loader import load_documents
from src.vector_store import init_qdrant_collection, get_storage_context
from src.embedding_model import setup_embedding
from src.config import DATA_PATH
 
def run_ingestion(tenant_id: Optional[str] = None, input_files: Optional[List[str]] = None):
    # New: chunk with SentenceSplitter, index nodes, and dump nodes cache
    from llama_index.core.node_parser import SentenceSplitter
    import os, json
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP, NODES_CACHE_PATH

    print("Starting ingestion pipeline ...")

    setup_embedding()
    client = init_qdrant_collection()
    storage_context = get_storage_context(client)

    documents = load_documents(DATA_PATH, input_files=input_files)
    node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = node_parser.get_nodes_from_documents(documents)

    # Attach tenant_id to node metadata if provided
    if tenant_id:
        for n in nodes:
            try:
                md = n.metadata
                if not isinstance(md, dict):
                    md = {}
            except Exception:
                md = {}
            md["tenant_id"] = tenant_id
            try:
                n.metadata = md
            except Exception:
                pass

    # Index from nodes to align with BM25
    # For current LlamaIndex version, use from_documents() which accepts nodes as input
    _ = VectorStoreIndex.from_documents(nodes, storage_context=storage_context)

    # Persist nodes to JSONL for BM25 corpus
    cache_path = NODES_CACHE_PATH
    if tenant_id:
        base = Path(NODES_CACHE_PATH).parent
        cache_path = str(base / tenant_id / "nodes.jsonl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        for n in nodes:
            try:
                text = n.get_text()
            except Exception:
                text = getattr(n, 'text', '')
            try:
                md = n.metadata
                if not isinstance(md, dict):
                    md = {}
            except Exception:
                md = {}
            if tenant_id:
                md["tenant_id"] = tenant_id
            json.dump({'text': text, 'metadata': md}, f, ensure_ascii=False)
            f.write('\n')

    print("Done. Data has been written to Qdrant and nodes cached.")


def _legacy_run_ingestion():
    print("🚀 Bắt đầu quá trình ingest dữ liệu...")

    setup_embedding()
    client = init_qdrant_collection()
    storage_context = get_storage_context(client)

    documents = load_documents(DATA_PATH)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    print("✅ HOÀN TẤT INGEST! Dữ liệu đã được ghi vào Qdrant.")
