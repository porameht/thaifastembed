"""
Tests for Thai BM25 implementation.
"""

import pytest
import numpy as np
from typing import List

from thaifastembed import ThaiBm25, SparseEmbedding, Tokenizer, TextProcessor, StopwordsFilter


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
        assert np.allclose(embedding.values, values.astype(np.float32))

        # Test from dict
        token_dict = {100: 0.5, 200: 1.2, 50: 0.8}
        embedding_from_dict = SparseEmbedding.from_dict(token_dict)

        # Test conversions
        result_dict = embedding_from_dict.as_dict()
        for key in token_dict:
            assert key in result_dict
            assert abs(result_dict[key] - token_dict[key]) < 1e-6

        # Test basic properties
        assert len(embedding_from_dict.indices) > 0
        assert len(embedding_from_dict.values) > 0
        assert isinstance(embedding_from_dict.indices, np.ndarray)
        assert isinstance(embedding_from_dict.values, np.ndarray)

    def test_empty_embedding(self):
        """Test empty embedding."""
        embedding = SparseEmbedding.from_dict({})
        assert len(embedding.indices) == 0
        assert len(embedding.values) == 0


class TestThaiBm25:
    """Test ThaiBm25 class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tokenizer = Tokenizer()
        self.stopwords_filter = StopwordsFilter()
        self.text_processor = TextProcessor(
            self.tokenizer, lowercase=True, stopwords_filter=self.stopwords_filter, min_token_len=1
        )

    def test_initialization(self):
        """Test ThaiBm25 initialization with default and custom parameters."""
        # Default parameters
        bm25 = ThaiBm25(text_processor=self.text_processor)
        assert isinstance(bm25, ThaiBm25)

        # Custom parameters
        bm25_custom = ThaiBm25(text_processor=self.text_processor, k=1.5, b=0.8, avg_len=512.0)
        assert isinstance(bm25_custom, ThaiBm25)

        # Test with default text processor when none provided
        bm25_default = ThaiBm25()  # Should work with default TextProcessor
        assert isinstance(bm25_default, ThaiBm25)

    def test_token_processing(self):
        """Test token ID computation and text cleaning."""
        bm25 = ThaiBm25(text_processor=self.text_processor)

        # Token ID computation should be consistent
        token = "ภาษาไทย"
        id1 = ThaiBm25.compute_token_id(token)
        id2 = ThaiBm25.compute_token_id(token)
        assert isinstance(id1, int) and id1 >= 0
        assert id1 == id2  # Same token, same ID

        # Test basic functionality
        text = "ประเทศไทยมีวัฒนธรรมที่หลากหลาย"
        tokens = self.text_processor.process_text(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

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
        query = "ค้นหาเอกสาร"
        embedding = bm25.query_embed(query)
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

        empty_query = bm25.query_embed("")
        assert len(empty_query.indices) == 0

    def test_embed_tokens(self):
        """Test basic embedding functionality."""
        bm25 = ThaiBm25(text_processor=self.text_processor)
        documents = [
            "ประเทศไทยมีวัฒนธรรมที่หลากหลาย",
            "อาหารไทยมีรสชาติเผ็ดหวานเปรียวเค็ม",
        ]
        embeddings = bm25.embed(documents)
        assert len(embeddings) == 2
        for embedding in embeddings:
            assert isinstance(embedding, SparseEmbedding)
            assert len(embedding.indices) == len(embedding.values)


def test_integration():
    """Integration test for complete workflow."""
    tokenizer = Tokenizer()
    stopwords_filter = StopwordsFilter()
    text_processor = TextProcessor(tokenizer, lowercase=True, stopwords_filter=stopwords_filter, min_token_len=1)
    bm25 = ThaiBm25(text_processor=text_processor)

    # Test data
    documents = [
        "ภาษาไทยเป็นภาษาราชการของประเทศไทย",
        "การประมวลผลภาษาธรรมชาติเป็นสาขาของปัญญาประดิษฐ์",
    ]
    query = "ภาษาไทย"

    # Generate embeddings
    doc_embeddings = bm25.embed(documents)
    query_embedding = bm25.query_embed(query)

    # Validate results
    assert len(doc_embeddings) == 2
    assert isinstance(query_embedding, SparseEmbedding)

    # Test data structure compatibility
    for embedding in doc_embeddings:
        assert isinstance(embedding, SparseEmbedding)
        assert len(embedding.indices) == len(embedding.values)

    # Query embedding should be binary
    if len(query_embedding.values) > 0:
        assert all(v == 1.0 for v in query_embedding.values)
