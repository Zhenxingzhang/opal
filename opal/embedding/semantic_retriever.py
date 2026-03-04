"""Semantic retriever: index documents and retrieve the most relevant ones for a query."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opal.embedding.embedding_model_cache import get_sentence_transformer

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get("PROJECT_ROOT", os.path.abspath(os.getcwd())),
    ".retriever_cache",
)


def _compute_cache_key(documents: list[str], model_name: str) -> str:
    """Compute a deterministic cache key from documents and model name."""
    hasher = hashlib.sha256()
    hasher.update(model_name.encode())
    for doc in documents:
        hasher.update(doc.encode())
    return hasher.hexdigest()


@dataclass
class SearchResult:
    """A single search result with its document text, index, and similarity score."""

    text: str
    index: int
    score: float


class SemanticRetriever:
    """Index a collection of text chunks and retrieve the most similar ones for a query.

    Uses a SentenceTransformer model (via ``EmbeddingModelCache``) to encode
    chunks into dense vectors and performs cosine-similarity search at query
    time.

    The retriever distinguishes between *documents* (source files / logical
    units) and *chunks* (the text pieces actually embedded and searched).
    When indexing pre-chunked text, pass ``num_docs`` to record how many
    source documents the chunks originated from.

    Supports an on-disk cache so that repeated calls to :meth:`index` with the
    same chunks skip re-encoding.  Pass ``cache_dir`` to control where the
    cache is stored (defaults to ``$PROJECT_ROOT/.retriever_cache``), or set it
    to ``None`` to disable caching entirely.

    Example::

        retriever = SemanticRetriever(cache_dir=".retriever_cache")
        retriever.index(chunks, num_docs=3)
        results = retriever.search("What pets are friendly?", top_k=1)

        # Second call loads from cache – no model invocation:
        retriever2 = SemanticRetriever(cache_dir=".retriever_cache")
        retriever2.index(chunks, num_docs=3)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str | Path | None = _DEFAULT_CACHE_DIR,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._chunks: list[str] = []
        self._num_docs: int = 0
        self._embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(
        self,
        chunks: list[str],
        *,
        num_docs: int | None = None,
        force: bool = False,
    ) -> None:
        """Replace the current index with *chunks*.

        When a ``cache_dir`` is configured the embeddings are persisted to
        disk.  Subsequent calls with the same chunks (and model) will
        load from cache instead of re-encoding.

        Args:
            chunks: List of text chunks to index.
            num_docs: Number of source documents the chunks originated from.
                      Defaults to ``len(chunks)`` (i.e. one chunk per doc).
            force: If ``True``, re-encode even when a cached index exists.
        """
        if not chunks:
            self._chunks = []
            self._num_docs = 0
            self._embeddings = None
            return

        self._num_docs = num_docs if num_docs is not None else len(chunks)

        # Try loading from cache
        if not force and self.cache_dir is not None:
            cached = self._load_cache(chunks)
            if cached is not None:
                self._chunks, self._embeddings = cached
                logger.info(
                    "Loaded %d chunks from %d docs from cache (dim=%d)",
                    len(self._chunks),
                    self._num_docs,
                    self._embeddings.shape[1],
                )
                return

        # Encode from scratch
        model = get_sentence_transformer(self.model_name)
        self._chunks = list(chunks)
        self._embeddings = model.encode(chunks, convert_to_numpy=True)
        logger.info(
            "Indexed %d chunks from %d docs (dim=%d)",
            len(chunks),
            self._num_docs,
            self._embeddings.shape[1],
        )

        # Persist to cache
        if self.cache_dir is not None:
            self._save_cache()

    def add(self, chunks: list[str], *, num_docs: int | None = None) -> None:
        """Append *chunks* to the existing index.

        Note: ``add`` does **not** update the on-disk cache automatically.
        Call :meth:`save_cache` explicitly after adding if persistence is
        desired.

        Args:
            chunks: Additional text chunks to add.
            num_docs: Number of new source documents these chunks came from.
                      Defaults to ``len(chunks)``.
        """
        if not chunks:
            return

        self._num_docs += num_docs if num_docs is not None else len(chunks)

        model = get_sentence_transformer(self.model_name)
        new_embeddings: np.ndarray = model.encode(chunks, convert_to_numpy=True)

        self._chunks.extend(chunks)
        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        logger.info(
            "Added %d chunks (total: %d chunks, %d docs)",
            len(chunks),
            len(self._chunks),
            self._num_docs,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the *top_k* most similar documents to *query*.

        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            List of :class:`SearchResult` ordered by descending similarity.

        Raises:
            ValueError: If no documents have been indexed yet.
        """
        if self._embeddings is None or len(self._chunks) == 0:
            raise ValueError("No documents indexed. Call index() or add() first.")

        model = get_sentence_transformer(self.model_name)
        query_embedding: np.ndarray = model.encode(query, convert_to_numpy=True)

        scores = self._cosine_similarity(query_embedding, self._embeddings)
        top_k = min(top_k, len(self._chunks))
        top_indices = np.argpartition(-scores, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [
            SearchResult(text=self._chunks[i], index=int(i), score=float(scores[i]))
            for i in top_indices
        ]

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    def save_cache(self) -> None:
        """Manually persist the current index to the cache directory."""
        if self.cache_dir is None:
            logger.warning("No cache_dir configured; nothing to save.")
            return
        if self._embeddings is None or not self._chunks:
            logger.warning("No index to save.")
            return
        self._save_cache()

    def _cache_path(self, chunks: list[str] | None = None) -> Path:
        """Return the cache directory for the given (or current) chunks."""
        docs = chunks if chunks is not None else self._chunks
        key = _compute_cache_key(docs, self.model_name)
        assert self.cache_dir is not None
        return self.cache_dir / key

    def _save_cache(self) -> None:
        assert self.cache_dir is not None and self._embeddings is not None
        cache_path = self._cache_path()
        cache_path.mkdir(parents=True, exist_ok=True)

        np.save(cache_path / "embeddings.npy", self._embeddings)
        with open(cache_path / "documents.json", "w") as f:
            json.dump(self._chunks, f)

        logger.debug("Saved index cache to %s", cache_path)

    def _load_cache(self, chunks: list[str]) -> tuple[list[str], np.ndarray] | None:
        """Try to load a cached index for *chunks*. Returns None on miss."""
        assert self.cache_dir is not None
        cache_path = self._cache_path(chunks)
        embeddings_path = cache_path / "embeddings.npy"
        documents_path = cache_path / "documents.json"

        if not embeddings_path.exists() or not documents_path.exists():
            return None

        try:
            embeddings = np.load(embeddings_path)
            with open(documents_path) as f:
                cached_docs: list[str] = json.load(f)

            if len(cached_docs) != embeddings.shape[0]:
                logger.warning("Cache inconsistency, ignoring cache at %s", cache_path)
                return None

            return cached_docs, embeddings
        except Exception:
            logger.warning("Failed to load cache from %s, re-indexing", cache_path, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def doc_count(self) -> int:
        """Number of source documents indexed."""
        return self._num_docs

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently indexed."""
        return len(self._chunks)

    @property
    def document_count(self) -> int:
        """Number of source documents indexed.

        .. deprecated:: Use :attr:`doc_count` or :attr:`chunk_count` instead.
        """
        return self._num_docs

    def summary(self) -> str:
        """Return a summary string with key retriever information."""
        dim = self._embeddings.shape[1] if self._embeddings is not None else 0
        return (
            f"SemanticRetriever(model={self.model_name!r}, "
            f"docs={self._num_docs}, chunks={len(self._chunks)}, dim={dim})"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity between vector *a* and matrix *b*."""
        return (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))
