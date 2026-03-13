##############################################################################
# HELPER FUNCTIONS (PDF-PARSING + SEMANTIC RETRIEVAL)
##############################################################################
from __future__ import annotations

import logging
import os
from pathlib import Path

import pymupdf

from opal.config import SemanticRetrievalConfig
from opal.embedding import SemanticRetriever

logger = logging.getLogger(__name__)

##############################################################################
# SHARED CONSTANTS
##############################################################################
PATH_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.getcwd()))
PATH_FINANCE_BENCH = PATH_ROOT + "/finance_bench"
PATH_PDFS = PATH_FINANCE_BENCH + "/pdfs/"


def get_pdf_text(doc: str) -> str:
    """Extract all text from a PDF file.

    Args:
        doc: Document name (without .pdf extension).

    Returns:
        Full text content of the PDF.
    """
    path_doc = f"{PATH_PDFS}/{doc}.pdf"
    pdf = pymupdf.open(path_doc)
    pages = [page.get_text() for page in pdf]
    pdf.close()
    return "\n".join(pages)


def chunk_text(
    text: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 30,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def build_retriever(
    docs: list[str] | str,
    retrieval_config: SemanticRetrievalConfig | None = None,
) -> SemanticRetriever:
    """Build a SemanticRetriever indexed on the given PDF documents.

    Args:
        docs: A single document name, a list of document names,
              or ``"all"`` to index every PDF in the pdfs folder.
        retrieval_config: Semantic retrieval configuration. Uses defaults
            when ``None``.

    Returns:
        A ``SemanticRetriever`` ready for search queries.
    """
    if retrieval_config is None:
        retrieval_config = SemanticRetrievalConfig()

    if docs == "all":
        pdf_dir = Path(PATH_PDFS)
        docs = [p.stem for p in sorted(pdf_dir.glob("*.pdf"))]
    elif isinstance(docs, str):
        docs = [docs]

    all_chunks: list[str] = []
    for doc in docs:
        logger.info("Extracting text from %s", doc)
        text = get_pdf_text(doc)
        chunks = chunk_text(
            text,
            chunk_size=retrieval_config.chunk_size,
            chunk_overlap=retrieval_config.chunk_overlap,
        )
        logger.info("  -> %d chunks", len(chunks))
        all_chunks.extend(chunks)

    logger.info(
        "Indexing %d total chunks from %d documents", len(all_chunks), len(docs)
    )
    retriever = SemanticRetriever(
        model_name=retrieval_config.model_name,
        reranker_model_name=retrieval_config.reranker_model_name,
    )
    retriever.index(all_chunks, num_docs=len(docs))
    return retriever
