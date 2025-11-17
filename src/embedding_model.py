from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from src.config import EMBEDDING_MODEL_NAME


def setup_embedding():
    print(f"Khởi tạo mô hình embedding: {EMBEDDING_MODEL_NAME}")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model đã sẵn sàng.")

