"""
Tests for Thai BM25 implementation.
"""

import pytest
import numpy as np
from typing import List

from thaifastembed import ThaiBm25, SparseEmbedding
from thaifastembed.types import Tokenizer
from thaifastembed.text_processor import TextProcessor, StopwordsFilter


class MockTokenizer(Tokenizer):
    """Simple tokenizer for testing."""

    def tokenize(self, text: str) -> List[str]:
        return text.split()


class TestSparseEmbedding:
    """Test SparseEmbedding class."""

    def test_initialization_and_conversion(self):
        """Test SparseEmbedding initialization and conversion methods."""
        # Test from arrays
        indices = np.array([1, 5, 10])
        values = np.array([0.5, 1.2, 0.8])
        embedding = SparseEmbedding(values=values, indices=indices)

        assert len(embedding.indices) == 3
        assert len(embedding.values) == 3
        assert np.array_equal(embedding.indices, indices)
        assert np.array_equal(embedding.values, values)

        # Test from dict
        token_dict = {100: 0.5, 200: 1.2, 50: 0.8}
        embedding_from_dict = SparseEmbedding.from_dict(token_dict)

        # Test conversions
        result_dict = embedding_from_dict.as_dict()
        for key in token_dict:
            assert key in result_dict
            assert abs(result_dict[key] - token_dict[key]) < 1e-6

        obj = embedding_from_dict.as_object()
        assert "indices" in obj
        assert "values" in obj
        assert isinstance(obj["indices"], np.ndarray)
        assert isinstance(obj["values"], np.ndarray)

    def test_empty_embedding(self):
        """Test empty embedding."""
        embedding = SparseEmbedding.from_dict({})
        assert len(embedding.indices) == 0
        assert len(embedding.values) == 0


class TestThaiBm25:
    """Test ThaiBm25 class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tokenizer = MockTokenizer()
        self.stopwords = {"และ", "ของ", "ที่"}
        self.stopwords_filter = StopwordsFilter(self.stopwords)
        self.text_processor = TextProcessor(
            tokenizer=self.tokenizer, stopwords_filter=self.stopwords_filter
        )

    def test_initialization(self):
        """Test ThaiBm25 initialization with default and custom parameters."""
        # Default parameters
        bm25 = ThaiBm25(text_processor=self.text_processor)
        assert bm25.k == 1.2
        assert bm25.b == 0.75
        assert bm25.avg_len == 256.0

        # Custom parameters
        bm25_custom = ThaiBm25(text_processor=self.text_processor, k=1.5, b=0.8, avg_len=512.0)
        assert bm25_custom.k == 1.5
        assert bm25_custom.b == 0.8
        assert bm25_custom.avg_len == 512.0

        # Test with default text processor when none provided
        bm25_default = ThaiBm25()  # Should work with default TextProcessor
        assert bm25_default.text_processor is not None

    def test_token_processing(self):
        """Test token ID computation and text cleaning."""
        bm25 = ThaiBm25(text_processor=self.text_processor)

        # Token ID computation should be consistent
        token = "ภาษาไทย"
        id1 = ThaiBm25.compute_token_id(token)
        id2 = ThaiBm25.compute_token_id(token)
        assert isinstance(id1, int) and id1 >= 0
        assert id1 == id2  # Same token, same ID

        # Text cleaning and tokenization
        text = "นี่ คือ เอกสาร และ ของ โลก Hello WORLD"
        tokens = bm25._clean_and_tokenize(text)

        # Should filter out stopwords and convert to lowercase
        assert "และ" not in tokens
        assert "ของ" not in tokens
        assert "นี่" in tokens
        assert "hello" in tokens
        assert "world" in tokens
        assert "Hello" not in tokens  # Should be lowercase

    def test_document_embedding(self):
        """Test document embedding generation."""
        bm25 = ThaiBm25(text_processor=self.text_processor)

        documents = ["นี่ คือ เอกสาร แรก", "this is the second document"]

        embeddings = bm25.embed(documents)

        assert len(embeddings) == 2
        for embedding in embeddings:
            assert isinstance(embedding, SparseEmbedding)
            assert len(embedding.indices) == len(embedding.values)

    def test_query_embedding(self):
        """Test query embedding generation."""
        bm25 = ThaiBm25(text_processor=self.text_processor)

        # Single query
        query = "ค้นหา เอกสาร"
        embeddings = list(bm25.query_embed(query))
        assert len(embeddings) == 1
        embedding = embeddings[0]
        assert isinstance(embedding, SparseEmbedding)
        # Query embeddings should have binary values (1.0)
        if len(embedding.values) > 0:
            assert all(v == 1.0 for v in embedding.values)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        bm25 = ThaiBm25(text_processor=self.text_processor)

        # Empty input
        empty_doc = bm25.embed([""])
        assert len(empty_doc[0].indices) == 0

        empty_query = list(bm25.query_embed([""]))
        assert len(empty_query[0].indices) == 0

        # All stopwords
        all_stopwords = "และ ของ ที่"
        tokens = bm25._clean_and_tokenize(all_stopwords)
        assert len(tokens) == 0

        embeddings = bm25.embed([all_stopwords])
        assert len(embeddings[0].indices) == 0

    def test_embed_tokens(self):
        """Test embed_tokens method."""
        bm25 = ThaiBm25(text_processor=self.text_processor)
        tokenized_documents = [
            ["นี่", "คือ", "เอกสาร", "แรก"],
            ["this", "is", "the", "second", "document"],
        ]
        embeddings = bm25.embed_tokens(tokenized_documents)
        assert len(embeddings) == 2
        for embedding in embeddings:
            assert isinstance(embedding, SparseEmbedding)
            assert len(embedding.indices) == len(embedding.values)
        assert len(embeddings[0].indices) == 4
        assert len(embeddings[1].indices) == 5


def test_integration():
    """Integration test for complete workflow."""
    tokenizer = MockTokenizer()
    stopwords = {"เป็น", "ของ"}
    stopwords_filter = StopwordsFilter(stopwords)
    text_processor = TextProcessor(tokenizer=tokenizer, stopwords_filter=stopwords_filter)
    bm25 = ThaiBm25(text_processor=text_processor)

    # Test data
    documents = [
        "ภาษาไทย เป็น ภาษาราชการ ของ ประเทศไทย",
        "การ ประมวลผล ภาษาธรรมชาติ เป็น สาขา ของ ปัญญาประดิษฐ์",
    ]
    query = "ภาษาไทย"

    # Generate embeddings
    doc_embeddings = bm25.embed(documents)
    query_embeddings = list(bm25.query_embed([query]))

    # Validate results
    assert len(doc_embeddings) == 2
    assert len(query_embeddings) == 1

    # Test data structure compatibility
    for embedding in doc_embeddings:
        result_dict = embedding.as_dict()
        assert isinstance(result_dict, dict)

        obj = embedding.as_object()
        assert "indices" in obj and "values" in obj
        assert isinstance(obj["indices"], np.ndarray)
        assert isinstance(obj["values"], np.ndarray)

    # Query embedding should be binary
    query_embedding = query_embeddings[0]
    if len(query_embedding.values) > 0:
        assert all(v == 1.0 for v in query_embedding.values)
