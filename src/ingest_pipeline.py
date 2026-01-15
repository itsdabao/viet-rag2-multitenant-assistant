from pathlib import Path
from typing import List, Optional
from llama_index.core import VectorStoreIndex
from src.data_loader import load_documents
from src.vector_store import init_qdrant_collection, get_storage_context
from src.embedding_model import setup_embedding
from src.config import DATA_PATH
 
def run_ingestion(tenant_id: Optional[str] = None, input_files: Optional[List[str]] = None):
    """
    Ingest documents into vector store with configurable chunking strategy.
    
    Args:
        tenant_id: Optional tenant identifier for multi-tenancy
        input_files: Optional list of specific files to ingest
    """
    import os, json
    from src.config import CHUNKING_STRATEGY, NODES_CACHE_PATH
    from src.chunking_strategies import get_node_parser

    print("=" * 60)
    print("Starting ingestion pipeline...")
    print(f"📋 Chunking Strategy: {CHUNKING_STRATEGY}")
    print("=" * 60)

    setup_embedding()
    client = init_qdrant_collection()
    storage_context = get_storage_context(client)

    documents = load_documents(DATA_PATH, input_files=input_files)
    print(f"📄 Loaded {len(documents)} document(s) from PDF pages")
    
    # Merge all pages/documents into a single document for better chunking
    # PDF pages are loaded as separate documents, but we want to process as one
    if len(documents) > 1:
        # Get metadata from first document
        first_meta = documents[0].metadata.copy() if hasattr(documents[0], 'metadata') else {}
        
        # Merge all text
        all_texts = [doc.get_content() for doc in documents]
        merged_text = "\n\n".join(all_texts)
        
        # Create single merged document
        from llama_index.core import Document as LIDocument
        merged_doc = LIDocument(text=merged_text, metadata=first_meta)
        documents = [merged_doc]
        print(f"📎 Merged into 1 document ({len(merged_text):,} chars)")
    
    # Sử dụng node parser theo strategy được config
    node_parser = get_node_parser(strategy=CHUNKING_STRATEGY)
    print(f"🔧 Using parser: {type(node_parser).__name__}")
    
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"✂️  Created {len(nodes)} chunk(s)")

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
