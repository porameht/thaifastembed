"""
Tests for TextProcessor and its components.
"""

from typing import List

from thaifastembed.text_processor import (
    TextProcessor, 
    PyThaiNLPTokenizer, 
    Stemmer, 
    StopwordsFilter
)
from thaifastembed.types import Tokenizer


class MockTokenizer(Tokenizer):
    """Mock tokenizer for testing."""
    
    def tokenize(self, text: str) -> List[str]:
        return text.split()


class TestStemmer:
    """Test Stemmer class."""
    
    def test_basic_stemming(self):
        """Test basic English word stemming."""
        stemmer = Stemmer("english")
        
        assert stemmer.stem("running") == "run"
        assert stemmer.stem("dogs") == "dog"
        assert stemmer.stem("") == ""
        assert stemmer.stem("123") == "123"


class TestStopwordsFilter:
    """Test StopwordsFilter class."""
    
    def test_default_stopwords(self):
        """Test default Thai and English stopwords."""
        filter = StopwordsFilter()
        
        # Thai stopwords
        assert filter.is_stopword("และ")
        assert filter.is_stopword("ของ")
        
        # English stopwords
        assert filter.is_stopword("the")
        assert filter.is_stopword("and")
        
        # Non-stopwords
        assert not filter.is_stopword("python")
    
    def test_custom_stopwords(self):
        """Test custom stopwords set."""
        custom_stopwords = {"test", "custom"}
        filter = StopwordsFilter(custom_stopwords)
        
        assert filter.is_stopword("test")
        assert not filter.is_stopword("the")  # Default not included


class TestPyThaiNLPTokenizer:
    """Test PyThaiNLP tokenizer."""
    
    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = PyThaiNLPTokenizer()
        assert tokenizer.engine == "newmm"

        tokenizer_no_dict = PyThaiNLPTokenizer(use_custom_dict=False)
        assert tokenizer_no_dict.custom_dict is None
    
    def test_thai_tokenization(self):
        """Test Thai text tokenization."""
        tokenizer = PyThaiNLPTokenizer()
        
        text = "ภาษาไทย เป็น ภาษาราชการ"
        tokens = tokenizer.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert any("ภาษา" in token for token in tokens)
    
    def test_mixed_text_tokenization(self):
        """Test mixed Thai-English tokenization."""
        tokenizer = PyThaiNLPTokenizer()
        
        text = "ภาษาไทย Hello World"
        tokens = tokenizer.tokenize(text)
        
        assert len(tokens) > 0
        has_content = any("ภาษา" in token or "Hello" in token for token in tokens)
        assert has_content


class TestTextProcessor:
    """Test TextProcessor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_tokenizer = MockTokenizer()
        self.custom_stopwords = {"test", "stop"}
        self.stopwords_filter = StopwordsFilter(self.custom_stopwords)
    
    def test_initialization(self):
        """Test TextProcessor initialization."""
        processor = TextProcessor()
        
        assert isinstance(processor.tokenizer, PyThaiNLPTokenizer)
        assert isinstance(processor.stemmer, Stemmer)  # enabled by default
        assert processor.lowercase is True
    
    def test_process_token(self):
        """Test token processing."""
        processor = TextProcessor(
            tokenizer=self.mock_tokenizer,
            stopwords_filter=self.stopwords_filter
        )
        
        # Normal token - lowercase
        assert processor.process_token("Hello") == "hello"
        
        # Stopword - filtered
        assert processor.process_token("test") is None
        
        # Empty - filtered
        assert processor.process_token("") is None
    
    def test_token_length_filtering(self):
        """Test token length filtering."""
        processor = TextProcessor(
            tokenizer=self.mock_tokenizer,
            min_token_len=3,
            max_token_len=8
        )
        
        assert processor.process_token("ab") is None  # Too short
        assert processor.process_token("verylongtoken") is None  # Too long
        assert processor.process_token("good") == "good"  # Just right
    
    def test_process_text(self):
        """Test text processing."""
        processor = TextProcessor(
            tokenizer=self.mock_tokenizer,
            stopwords_filter=self.stopwords_filter
        )
        
        text = "Hello world test good"
        result = processor.process_text(text)
        
        # "test" filtered as stopword
        assert "hello" in result
        assert "world" in result
        assert "test" not in result
        
        # Empty/invalid inputs
        assert processor.process_text("") == []
        assert processor.process_text(None) == []
    
    def test_thai_english_mixed(self):
        """Test Thai-English mixed text processing."""
        processor = TextProcessor()
        
        # Thai text with stopwords
        text = "ภาษาไทย เป็น ภาษา ของ ประเทศไทย"
        result = processor.process_text(text)
        
        # Thai stopwords filtered
        assert "เป็น" not in result
        assert "ของ" not in result
        assert len(result) > 0
        
        # English text with stopwords
        text = "The quick brown fox"
        result = processor.process_text(text)
        
        assert "the" not in result
        assert "quick" in result


class TestEdgeCases:
    """Test edge cases."""
    
    def test_special_cases(self):
        """Test special characters and edge cases."""
        processor = TextProcessor()
        
        # Unicode/emoji
        assert isinstance(processor.process_text("สวัสดี 🙏"), list)
        
        # Special characters
        assert isinstance(processor.process_text("Hello! @#$%"), list)
        
        # Numbers and mixed
        result = processor.process_text("ปี 2024 Good")
        assert len(result) > 0