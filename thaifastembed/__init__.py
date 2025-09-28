"""
Thai FastEmbed - BM25 Sparse Embeddings for Thai Language

A specialized BM25 implementation for Thai language that generates sparse embeddings
compatible with Qdrant's idf modifier for hybrid search applications.
"""

from .thaifastembed_rust import SparseEmbedding, ThaiBm25, Tokenizer, TextProcessor, StopwordsFilter

__version__ = "0.1.3"
__all__ = [
    "ThaiBm25",
    "SparseEmbedding",
    "Tokenizer",
    "TextProcessor",
    "StopwordsFilter",
]
