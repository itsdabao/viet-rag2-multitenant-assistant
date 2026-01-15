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
        "LLM provider=%s (auto=%s) env_keys: GROQ=%s GOOGLE=%s OPENAI=%s",
        provider or "auto",
        "yes" if not provider else "no",
        "yes" if bool(os.getenv("GROQ_API_KEY")) else "no",
        "yes" if bool(os.getenv("GOOGLE_API_KEY")) else "no",
        "yes" if bool(os.getenv("OPENAI_API_KEY")) else "no",
    )

    if provider == "none":
        Settings.llm = None
        logger.info("LLM is disabled by LLM_PROVIDER=none")
        return

    groq_key = os.getenv("GROQ_API_KEY")
    if provider in ("groq", "groq_openai_compat", "openai_compat") or (not provider and groq_key):
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
