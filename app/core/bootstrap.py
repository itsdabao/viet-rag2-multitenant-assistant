from __future__ import annotations

from llama_index.core import Settings

from app.core.llama import init_llm_from_env, setup_embedding

_BOOTSTRAPPED_EMBEDDINGS = False
_BOOTSTRAPPED_RUNTIME = False


def bootstrap_embeddings_only() -> None:
    global _BOOTSTRAPPED_EMBEDDINGS
    if _BOOTSTRAPPED_EMBEDDINGS:
        return

    # Always override to the project embedding to avoid accidental defaults (e.g., OpenAIEmbedding).
    setup_embedding()
    _BOOTSTRAPPED_EMBEDDINGS = True



def bootstrap_runtime() -> None:
    """
    Initialize LlamaIndex `Settings` for runtime Q&A:
    - embeddings
    - LLM (Gemini by default)
    Safe to call multiple times.
    """
    global _BOOTSTRAPPED_RUNTIME
    if _BOOTSTRAPPED_RUNTIME:
        return
    bootstrap_embeddings_only()
    # Always override to the configured provider to avoid accidental defaults (e.g., OpenAI LLM).
    init_llm_from_env()
    _BOOTSTRAPPED_RUNTIME = True
