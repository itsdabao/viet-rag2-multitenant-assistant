from __future__ import annotations

import os

# In some sandboxed/dev environments, network egress is blocked via a dummy proxy.
# Set offline mode early (before importing HuggingFace/transformers) to avoid long retry loops.
_proxy_vals = {
    (os.getenv("HTTP_PROXY") or "").strip().lower(),
    (os.getenv("HTTPS_PROXY") or "").strip().lower(),
    (os.getenv("ALL_PROXY") or "").strip().lower(),
}
_blocked_proxy = any(p in ("http://127.0.0.1:9", "https://127.0.0.1:9") for p in _proxy_vals)
if _blocked_proxy and not os.getenv("ALLOW_HF_DOWNLOAD"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from llama_index.core import Settings

from app.core.llama import init_llm_from_env, setup_embedding

_BOOTSTRAPPED_EMBEDDINGS = False
_BOOTSTRAPPED_RUNTIME = False


def bootstrap_embeddings_only() -> None:
    global _BOOTSTRAPPED_EMBEDDINGS
    if _BOOTSTRAPPED_EMBEDDINGS:
        return

    # Always override to the project embedding to avoid accidental defaults (e.g., OpenAIEmbedding).
    try:
        setup_embedding()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize embeddings. If you're offline/sandboxed, pre-download the HuggingFace model "
            f"or set EMBEDDING_MODEL_NAME to a local path. Original error: {e}"
        ) from e
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
    # [LOCAL OVERRIDE] Force Qwen 2.5 local model for fee_extractor / pipeline
    # User requested: "D:\AI_Agent\models\qwen2.5-3b-instruct-q4_k_m.gguf"
    os.environ["LLM_PROVIDER"] = "llama_cpp"
    os.environ["LLAMA_CPP_MODEL_PATH"] = r"D:\AI_Agent\models\qwen2.5-3b-instruct-q4_k_m.gguf"
    os.environ["LLAMA_CPP_N_CTX"] = "4096"
    os.environ["LLAMA_CPP_N_GPU_LAYERS"] = "-1"
    
    init_llm_from_env()
    _BOOTSTRAPPED_RUNTIME = True
