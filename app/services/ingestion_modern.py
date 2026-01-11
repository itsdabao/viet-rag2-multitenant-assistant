from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class IngestionOptions:
    """
    Day-2 ingestion options:
    - Prefer LlamaParse for complex PDFs (tables/forms), output markdown.
    - Parse markdown into elements (heading/table/list) before chunking.
    """

    pdf_engine: str = "auto"  # auto | llamaparse | simple
    use_markdown_element_parser: bool = True
    llamaparse_result_type: str = "markdown"
    language: str = "vi"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def _split_files(data_path: str, input_files: Optional[List[str]]) -> List[Path]:
    if input_files:
        return [Path(p) for p in input_files]
    base = Path(data_path)
    if not base.exists():
        raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {base}")
    exts = {".pdf", ".md", ".txt", ".docx", ".rtf"}
    return [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def _simple_reader_load(files: List[Path]):
    from llama_index.core import SimpleDirectoryReader

    reader = SimpleDirectoryReader(input_files=[str(p) for p in files])
    return reader.load_data()


def _llamaparse_load_pdf(path: Path, *, result_type: str, language: str):
    """
    Parse a single PDF via LlamaParse -> list[Document].
    Requires LLAMA_CLOUD_API_KEY (or LLAMAPARSE_API_KEY).
    """
    import os

    _load_env()
    api_key = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu LLAMA_CLOUD_API_KEY (hoặc LLAMAPARSE_API_KEY) để dùng LlamaParse.")

    # Prefer llama_parse SDK; fall back to llama-index reader if needed.
    try:
        from llama_parse import LlamaParse  # type: ignore

        parser = LlamaParse(api_key=api_key, result_type=result_type, language=language)
        docs = parser.load_data(str(path))
        return docs
    except Exception:
        from llama_index.readers.llama_parse import LlamaParseReader  # type: ignore

        reader = LlamaParseReader(api_key=api_key, result_type=result_type, language=language)
        return reader.load_data(str(path))


def load_documents_for_ingestion(
    data_path: str,
    *,
    input_files: Optional[List[str]] = None,
    opts: IngestionOptions,
):
    """
    Load Documents for ingestion with an optional LlamaParse path for PDFs.

    Strategy:
    - Non-PDF always uses SimpleDirectoryReader.
    - PDF uses:
      - `llamaparse` if forced, or if `auto` and key exists.
      - otherwise SimpleDirectoryReader.
    """
    import os

    files = _split_files(data_path, input_files)
    if not files:
        return []

    pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
    others = [p for p in files if p.suffix.lower() != ".pdf"]

    # Always load non-PDF via simple reader
    documents = []
    if others:
        documents.extend(_simple_reader_load(others))

    engine = (opts.pdf_engine or "auto").strip().lower()
    _load_env()
    has_key = bool(os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY"))
    use_llamaparse = engine == "llamaparse" or (engine == "auto" and has_key)

    if pdfs and use_llamaparse:
        for pdf in pdfs:
            parsed = _llamaparse_load_pdf(pdf, result_type=opts.llamaparse_result_type, language=opts.language)
            # Ensure source metadata exists for traceability
            for d in parsed:
                try:
                    md = d.metadata or {}
                except Exception:
                    md = {}
                md.setdefault("file_name", pdf.name)
                md.setdefault("source", pdf.name)
                try:
                    d.metadata = md
                except Exception:
                    pass
            documents.extend(parsed)
    elif pdfs:
        documents.extend(_simple_reader_load(pdfs))

    return documents


def build_nodes_for_ingestion(documents, *, chunk_size: int, chunk_overlap: int, use_markdown_elements: bool):
    """
    Build nodes with structure preservation when possible.

    - If markdown element parser is available and enabled, parse into elements first.
    - Then chunk large nodes with SentenceSplitter to keep chunk size stable.
    """
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    nodes = None
    if use_markdown_elements:
        try:
            from llama_index.core.node_parser import MarkdownElementNodeParser  # type: ignore

            md_parser = MarkdownElementNodeParser()
            nodes = md_parser.get_nodes_from_documents(documents)
        except Exception:
            nodes = None

    if nodes is None:
        return splitter.get_nodes_from_documents(documents)

    # Chunk large element-nodes while keeping metadata (element_type, heading, page, ...)
    try:
        from llama_index.core import Document
    except Exception:
        Document = None  # type: ignore

    out = []
    for n in nodes:
        try:
            text = n.get_text()
        except Exception:
            text = getattr(n, "text", "") or ""
        if len(text) <= chunk_size * 2 or Document is None:
            out.append(n)
            continue
        try:
            md = getattr(n, "metadata", {}) or {}
            d = Document(text=text, metadata=dict(md))
            out.extend(splitter.get_nodes_from_documents([d]))
        except Exception:
            out.append(n)
    return out

