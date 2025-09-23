"""
Thai FastEmbed - BM25 Sparse Embeddings for Thai Language

A specialized BM25 implementation for Thai language that generates sparse embeddings
compatible with Qdrant's idf modifier for hybrid search applications.
"""

from .sparse_embedding import SparseEmbedding
from .bm25 import ThaiBm25
from .types import NumpyArray, IntArray, Tokenizer
from .text_processor import TextProcessor, StopwordsFilter, Stemmer, PyThaiNLPTokenizer
from .utils import load_custom_dict

__version__ = "0.1.0"
__all__ = [
    "ThaiBm25",
    "SparseEmbedding",
    "NumpyArray",
    "IntArray",
    "Tokenizer",
    "TextProcessor",
    "StopwordsFilter",
    "Stemmer",
    "PyThaiNLPTokenizer",
    "load_custom_dict",
]
