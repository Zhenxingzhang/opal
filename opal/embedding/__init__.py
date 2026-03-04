"""Embedding module exports."""

from opal.embedding.embedding_model_cache import (
    EmbeddingModelCache as EmbeddingModelCache,
    get_cross_encoder as get_cross_encoder,
    get_sentence_transformer as get_sentence_transformer,
)
from opal.embedding.semantic_retriever import (
    SearchResult as SearchResult,
    SemanticRetriever as SemanticRetriever,
)
