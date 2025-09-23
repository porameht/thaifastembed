"""
Thai BM25 implementation for sparse embeddings compatible with Qdrant.

This implementation provides BM25 sparse embeddings specifically designed for Thai language
with support for custom tokenizers and stopwords through dependency injection.

Thank you for the original implementation: https://github.com/fastembed
"""

from collections import defaultdict
from typing import List, Optional, Iterable, Union
import mmh3
import numpy as np
from .sparse_embedding import SparseEmbedding
from .text_processor import TextProcessor


class ThaiBm25:
    """Implements traditional Thai BM25 in a form of sparse embeddings.
    Uses a count of tokens in the document to evaluate the importance of the token.

    WARNING: This model is expected to be used with `modifier="idf"` in the sparse vector index of Qdrant.

    BM25 formula:

    score(q, d) = SUM[ IDF(q_i) * (f(q_i, d) * (k + 1)) / (f(q_i, d) + k * (1 - b + b * (|d| / avg_len))) ],

    where IDF is the inverse document frequency, computed on Qdrant's side
    f(q_i, d) is the term frequency of the token q_i in the document d
    k, b, avg_len are hyperparameters, described below.

    Args:
        text_processor: TextProcessor instance for handling text processing.
                       If None, creates a default TextProcessor with PyThaiNLP tokenizer.
        k: BM25 k parameter (term frequency saturation). Defaults to 1.2.
        b: BM25 b parameter (document length normalization). Defaults to 0.75.
        avg_len: Average document length for normalization. Defaults to 256.0.
    """

    def __init__(
        self,
        text_processor: Optional[TextProcessor] = None,
        k: float = 1.2,
        b: float = 0.75,
        avg_len: float = 256.0,
    ):
        # Text processing - use default if none provided
        self.text_processor = text_processor if text_processor is not None else TextProcessor()

        # BM25 parameters
        self.k = k
        self.b = b
        self.avg_len = avg_len

    @classmethod
    def compute_token_id(cls, token: str) -> int:
        return mmh3.hash(token, signed=False)

    def _clean_and_tokenize(self, text: str) -> List[str]:
        return self.text_processor.process_text(text)

    def _term_frequency(self, tokens: list[str]) -> dict[int, float]:
        """Calculate the term frequency part of the BM25 formula.

        (
            f(q_i, d) * (k + 1)
        ) / (
            f(q_i, d) + k * (1 - b + b * (|d| / avg_len))
        )

        Args:
            tokens (list[str]): The list of tokens in the document.

        Returns:
            dict[int, float]: The token_id to term frequency mapping.
        """
        tf_map: dict[int, float] = {}
        counter: defaultdict[str, int] = defaultdict(int)
        for stemmed_token in tokens:
            counter[stemmed_token] += 1

        doc_len = len(tokens)
        for stemmed_token in counter:
            token_id = self.compute_token_id(stemmed_token)
            num_occurrences = counter[stemmed_token]
            tf_map[token_id] = num_occurrences * (self.k + 1)
            tf_map[token_id] /= num_occurrences + self.k * (
                1 - self.b + self.b * doc_len / self.avg_len
            )
        return tf_map

    def embed_tokens(self, tokenized_documents: List[List[str]]) -> List[SparseEmbedding]:
        """
        Embed each tokens in the documents into sparse BM25 vectors.
        """

        embeddings: List[SparseEmbedding] = []
        for tokens in tokenized_documents:
            token_id2value = self._term_frequency(tokens)
            embeddings.append(SparseEmbedding.from_dict(token_id2value))
        return embeddings

    def embed(self, documents: List[str], tokenize: bool = True) -> List[SparseEmbedding]:
        """
        Embed documents into sparse BM25 vectors.

        Args:
            documents: List of documents to embed

        Returns:
            List of SparseEmbedding objects
        """
        embeddings: List[SparseEmbedding] = []
        for document in documents:
            tokens = self._clean_and_tokenize(document)
            token_id2value = self._term_frequency(tokens)
            embeddings.append(SparseEmbedding.from_dict(token_id2value))

        return embeddings

    def query_embed(self, query: Union[str, Iterable[str]]) -> Iterable[SparseEmbedding]:
        """
        Embed query into sparse BM25 vectors.

        Args:
            query: Query to embed

        Returns:
            List of SparseEmbedding objects
        """
        if isinstance(query, str):
            query = [query]

        for text in query:
            tokens = self._clean_and_tokenize(text)
            token_ids = np.array(
                list(set(self.compute_token_id(token) for token in tokens)),
                dtype=np.uint32,
            )
            values = np.ones_like(token_ids)
            yield SparseEmbedding(indices=token_ids, values=values)
