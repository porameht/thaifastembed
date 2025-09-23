"""
Text processing utilities for Thai language with support for tokenization, stemming, and stopwords.
"""

from typing import Optional, Set, List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from py_rust_stemmers import SnowballStemmer
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus.common import thai_words
from pythainlp.util import dict_trie
from .types import Tokenizer
from .utils import load_custom_dict



class Stemmer:
    """Stemmer class for stemming words."""

    def __init__(self, language: str = "english"):
        """Initialize the Stemmer class."""
        self.stemmer = SnowballStemmer(language)

    def stem(self, word: str) -> str:
        """Stem a word to its root form."""
        return self.stemmer.stem_word(word)


class StopwordsFilter:
    """Filter for handling stopwords."""

    def __init__(self, stopwords: Optional[Set[str]] = None):
        if stopwords is not None:
            self.stopwords = stopwords
        else:
            self.stopwords = set(thai_stopwords()).union(ENGLISH_STOP_WORDS)

    def is_stopword(self, word: str) -> bool:
        """Check if a word is a stopword."""
        return word in self.stopwords


class PyThaiNLPTokenizer(Tokenizer):
    """
    Thai tokenizer with enhanced custom dictionary support.

    - Optional custom dictionary from words.txt combined with PyThaiNLP's Thai words
    - Configurable tokenization engine
    """

    def __init__(self, engine: str = "newmm", use_custom_dict: bool = True):
        """
        Initialize the PyThaiNLP tokenizer.

        Args:
            engine: Tokenization engine ("newmm", "longest", "attacut", etc.)
            use_custom_dict: Whether to use enhanced custom dictionary (combines words.txt + Thai words)
        """
        self.engine = engine
        self.use_custom_dict = use_custom_dict
        self.custom_dict = self._create_dictionary()

    def _create_dictionary(self):
        """Create the appropriate dictionary based on configuration."""
        if not self.use_custom_dict:
            return None
        
        # Load custom words from words.txt
        custom_trie = load_custom_dict()
        custom_words = list(custom_trie) if custom_trie else []
        
        if not custom_words:
            return None
        
        # combine custom words with PyThaiNLP's Thai words
        thai_nlp_words = list(thai_words())
        all_words = list(set(custom_words + thai_nlp_words))
        
        # Create dictionary with combined words
        return dict_trie(dict_source=all_words)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using PyThaiNLP with optional enhanced dictionary.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokenized words
        """
        if not text or not isinstance(text, str):
            return []
        
        return word_tokenize(
            text, 
            custom_dict=self.custom_dict, 
            engine=self.engine, 
            keep_whitespace=False
        )


class TextProcessor:
    """
    Processes tokens for indexing. Applies all configured options to the token.

    Similar to the Rust implementation, this processes tokens with:
    - Lowercase conversion
    - Stopword filtering
    - Stemming (optional)
    - Token length validation

    Args:
        tokenizer: Optional tokenizer instance for breaking text into tokens.
                  If None, uses PyThaiNLP's enhanced tokenizer with custom dictionary.
        lowercase: Whether to convert tokens to lowercase
        stopwords_filter: Optional StopwordsFilter instance for filtering stopwords.
                         If None, uses PyThaiNLP's default Thai stopwords.
        stemmer: Optional stemmer for word stemming
        min_token_len: Minimum token length (in characters)
        max_token_len: Optional maximum token length (in characters)
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        lowercase: Optional[bool] = True,
        stopwords_filter: Optional[StopwordsFilter] = None,
        disable_stemmer: Optional[bool] = False,
        min_token_len: Optional[int] = None,
        max_token_len: Optional[int] = None,
    ):
        # Use enhanced PyThaiNLP tokenizer as default if none provided
        self.tokenizer = tokenizer if tokenizer is not None else PyThaiNLPTokenizer()

        # Use default Thai stopwords if no filter provided
        self.stopwords_filter = (
            stopwords_filter if stopwords_filter is not None else StopwordsFilter()
        )

        # Use default stemmer in english if not provided
        self.stemmer = None if disable_stemmer else Stemmer()

        self.lowercase = lowercase
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len

    def process_token(self, token: str, check_max_len: bool = True) -> Optional[str]:
        """
        Process a single token for indexing.

        Returns None if:
        - The token is empty
        - The token is a stopword
        - The token's character length is outside the configured range

        Args:
            token: The token to process
            check_max_len: Whether to check maximum token length

        Returns:
            Processed token or None if it should be filtered out
        """
        if not token:
            return None

        # Handle lowercase
        processed_token = token.lower() if self.lowercase else token

        # Handle stopwords
        if self.stopwords_filter.is_stopword(processed_token):
            return None

        # Handle stemming
        if self.stemmer:
            processed_token = self.stemmer.stem(processed_token)

        # Handle token length
        token_len = len(processed_token)

        if self.min_token_len is not None and token_len < self.min_token_len:
            return None

        if check_max_len and self.max_token_len is not None and token_len > self.max_token_len:
            return None

        return processed_token

    def process_text(self, text: str, check_max_len: bool = True) -> list[str]:
        """
        Process a complete text string into filtered tokens.

        Args:
            text: The text to process
            check_max_len: Whether to check maximum token length

        Returns:
            List of processed tokens
        """
        if not text or not isinstance(text, str):
            return []

        tokens = self.tokenizer.tokenize(text)
        processed_tokens = []

        for token in tokens:
            processed_token = self.process_token(token, check_max_len)
            if processed_token is not None:
                processed_tokens.append(processed_token)

        return processed_tokens
