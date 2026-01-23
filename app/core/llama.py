from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from app.core.config import EMBEDDING_MODEL_NAME


def setup_embedding():
    print(f"Khởi tạo mô hình embedding: {EMBEDDING_MODEL_NAME}")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model đã sẵn sàng.")


def init_llm_from_env() -> None:
    """
    Khởi tạo LLM từ biến môi trường / file .env.

    Ưu tiên:
    1) Groq (OpenAI-compatible) nếu có `GROQ_API_KEY` (mặc định model Llama 4 Scout)
    2) Google Gemini nếu có `GOOGLE_API_KEY`
    3) OpenAI nếu có `OPENAI_API_KEY`

    Có thể ép provider bằng `LLM_PROVIDER`:
    - `groq` (OpenAI-compatible via `GROQ_BASE_URL`)
    - `openai_compat` (local/remote OpenAI-compatible via `OPENAI_COMPAT_BASE_URL`)
    - `gemini`
    - `openai`
    - `none`
    """
    import logging
    import os

    from dotenv import load_dotenv

    logger = logging.getLogger(__name__)

    load_dotenv()

    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    logger.info(
        "LLM provider=%s (auto=%s) env_keys: GROQ=%s COMPAT=%s LLAMA_CPP=%s GOOGLE=%s OPENAI=%s",
        provider or "auto",
        "yes" if not provider else "no",
        "yes" if bool(os.getenv("GROQ_API_KEY")) else "no",
        "yes" if bool(os.getenv("OPENAI_COMPAT_BASE_URL")) else "no",
        "yes" if bool(os.getenv("LLAMA_CPP_MODEL_PATH")) else "no",
        "yes" if bool(os.getenv("GOOGLE_API_KEY")) else "no",
        "yes" if bool(os.getenv("OPENAI_API_KEY")) else "no",
    )

    if provider == "none":
        Settings.llm = None
        logger.info("LLM is disabled by LLM_PROVIDER=none")
        return

    # 1) llama-cpp-python (in-process GGUF)
    model_path = (os.getenv("LLAMA_CPP_MODEL_PATH") or "").strip()
    if provider in ("llama_cpp", "llama-cpp") or (not provider and model_path):
        if not model_path:
            raise ValueError("LLM_PROVIDER=llama_cpp nhưng thiếu LLAMA_CPP_MODEL_PATH trong .env / biến môi trường.")
        from app.core.llama_cpp_llm import LlamaCppConfig, LlamaCppLLM

        n_ctx = int(os.getenv("LLAMA_CPP_N_CTX") or "2048")
        n_gpu_layers = int(os.getenv("LLAMA_CPP_N_GPU_LAYERS") or "0")
        n_threads = int(os.getenv("LLAMA_CPP_N_THREADS") or "0")
        chat_format = (os.getenv("LLAMA_CPP_CHAT_FORMAT") or "chatml").strip()
        verbose = (os.getenv("LLAMA_CPP_VERBOSE") or "0").strip().lower() in ("1", "true", "yes", "on")

        Settings.llm = LlamaCppLLM(
            LlamaCppConfig(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                verbose=verbose,
                temperature=float(os.getenv("LLAMA_CPP_TEMPERATURE") or "0.2"),
                max_tokens=int(os.getenv("LLAMA_CPP_MAX_TOKENS") or "1024"),
            )
        )
        logger.info("Đã khởi tạo LLM llama.cpp (llama-cpp-python) với model_path=%s", model_path)
        return

    # 2) Generic OpenAI-compatible (local/remote)
    compat_base = (os.getenv("OPENAI_COMPAT_BASE_URL") or "").strip()
    if provider in ("openai_compat", "local_openai_compat") or (not provider and compat_base):
        if not compat_base:
            raise ValueError("LLM_PROVIDER=openai_compat nhưng thiếu OPENAI_COMPAT_BASE_URL trong .env / biến môi trường.")

        compat_key = (os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("OPENAI_API_KEY") or "local").strip()
        compat_model = (os.getenv("OPENAI_COMPAT_MODEL") or os.getenv("OPENAI_MODEL") or "local-model").strip()

        from app.core.openai_compat_llm import OpenAICompatConfig, OpenAICompatLLM

        Settings.llm = OpenAICompatLLM(OpenAICompatConfig(api_key=compat_key, base_url=compat_base, model=compat_model))
        logger.info("Đã khởi tạo LLM OpenAI-compatible với model: %s (base_url=%s)", compat_model, compat_base)
        return

    # 3) Groq (OpenAI-compatible)
    groq_key = os.getenv("GROQ_API_KEY")
    if provider in ("groq", "groq_openai_compat") or (not provider and groq_key):
        if not groq_key:
            raise ValueError("LLM_PROVIDER=groq nhưng thiếu GROQ_API_KEY trong .env / biến môi trường.")

        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

        # Cách 2 (OpenAI-compatible endpoint) nhưng dùng custom LlamaIndex LLM để:
        # - pass `Settings.llm` type check (LLM instance)
        # - không validate model name theo catalog của OpenAI
        from app.core.openai_compat_llm import OpenAICompatConfig, OpenAICompatLLM

        Settings.llm = OpenAICompatLLM(OpenAICompatConfig(api_key=groq_key, base_url=base_url, model=model))
        logger.info("Đã khởi tạo LLM Groq(OpenAI-compatible) với model: %s (base_url=%s)", model, base_url)
        return

    # 4) Google Gemini
    google_key = os.getenv("GOOGLE_API_KEY")
    if provider == "gemini" or (not provider and google_key):
        from llama_index.llms.google_genai import GoogleGenAI
        import inspect

        if provider == "gemini" and not google_key:
            raise ValueError("LLM_PROVIDER=gemini nhưng thiếu GOOGLE_API_KEY trong .env / biến môi trường.")

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

        # Different llama-index versions may use different constructor arg names.
        # We detect the supported one to avoid silent fallback to default models.
        kwargs = {"api_key": google_key}
        try:
            sig = inspect.signature(GoogleGenAI.__init__)
            if "model_name" in sig.parameters:
                kwargs["model_name"] = model
            elif "model" in sig.parameters:
                kwargs["model"] = model
            elif "model_id" in sig.parameters:
                kwargs["model_id"] = model
            else:
                # Best-effort fallback
                kwargs["model_name"] = model
        except Exception:
            kwargs["model_name"] = model

        Settings.llm = GoogleGenAI(**kwargs)
        # Log what we requested; the underlying SDK should honor this and not fallback.
        logger.info("Đã khởi tạo LLM GoogleGenAI với model: %s", model)
        return

    # 5) OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if provider == "openai" or (not provider and openai_key):
        from llama_index.llms.openai import OpenAI

        if provider == "openai" and not openai_key:
            raise ValueError("LLM_PROVIDER=openai nhưng thiếu OPENAI_API_KEY trong .env / biến môi trường.")

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        Settings.llm = OpenAI(model=model, api_key=openai_key)
        logger.info("Đã khởi tạo LLM OpenAI với model: %s", model)
        return

    raise ValueError(
        "Không tìm thấy API key cho LLM. Vui lòng set GROQ_API_KEY (Groq), "
        "GOOGLE_API_KEY (Gemini) hoặc OPENAI_API_KEY (OpenAI) trong .env / biến môi trường."
    )
