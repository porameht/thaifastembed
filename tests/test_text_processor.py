"""
Tests for TextProcessor and its components.
"""

from typing import List

from thaifastembed import TextProcessor, Tokenizer, StopwordsFilter


class TestStopwordsFilter:
    """Test StopwordsFilter class."""
    
    def test_default_stopwords(self):
        """Test default Thai stopwords."""
        filter = StopwordsFilter()
        
        # Thai stopwords
        assert filter.is_stopword("และ")
        assert filter.is_stopword("ของ")
        
        # Non-stopwords
        assert not filter.is_stopword("python")
        assert not filter.is_stopword("the")  # English stopwords not included
    
    def test_stopwords_length(self):
        """Test stopwords filter length method."""
        filter = StopwordsFilter()
        assert filter.len() > 0


class TestTokenizer:
    """Test Tokenizer class."""
    
    def test_thai_tokenization(self):
        """Test Thai text tokenization."""
        tokenizer = Tokenizer()
        
        text = "ภาษาไทย เป็น ภาษาราชการ"
        tokens = tokenizer.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert any("ภาษา" in token for token in tokens)
    
    def test_mixed_text_tokenization(self):
        """Test mixed Thai-English tokenization."""
        tokenizer = Tokenizer()
        
        text = "ภาษาไทย Hello World"
        tokens = tokenizer.tokenize(text)
        
        assert len(tokens) > 0
        has_content = any("ภาษา" in token or "Hello" in token for token in tokens)
        assert has_content
    
    def test_empty_text(self):
        """Test empty text tokenization."""
        tokenizer = Tokenizer()
        assert tokenizer.tokenize("") == []


class TestTextProcessor:
    """Test TextProcessor class."""
    
    def test_initialization(self):
        """Test TextProcessor initialization."""
        tokenizer = Tokenizer()
        processor = TextProcessor(tokenizer)
        assert processor is not None
    
    def test_process_token(self):
        """Test token processing."""
        tokenizer = Tokenizer()
        processor = TextProcessor(tokenizer)
        
        # Normal token - should be processed
        result = processor.process_token("Hello")
        assert result is not None
        assert isinstance(result, str)
        
        # Empty token
        result = processor.process_token("")
        assert result is None
    
    def test_process_text(self):
        """Test text processing."""
        tokenizer = Tokenizer()
        processor = TextProcessor(tokenizer)
        
        text = "Hello world test good"
        result = processor.process_text(text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Empty/invalid inputs
        assert processor.process_text("") == []
    
    def test_thai_text_processing(self):
        """Test Thai text processing."""
        tokenizer = Tokenizer()
        processor = TextProcessor(tokenizer)
        
        # Thai text
        text = "ภาษาไทย เป็น ภาษา ของ ประเทศไทย"
        result = processor.process_text(text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # English text
        text = "The quick brown fox"
        result = processor.process_text(text)
        
        assert isinstance(result, list)
        assert len(result) > 0


class TestEdgeCases:
    """Test edge cases."""
    
    def test_special_cases(self):
        """Test special characters and edge cases."""
        tokenizer = Tokenizer()
        processor = TextProcessor(tokenizer)
        
        # Unicode/emoji
        result = processor.process_text("สวัสดี 🙏")
        assert isinstance(result, list)
        
        # Special characters
        result = processor.process_text("Hello! @#$%")
        assert isinstance(result, list)
        
        # Numbers and mixed
        result = processor.process_text("ปี 2024 Good")
        assert isinstance(result, list)
        assert len(result) > 0