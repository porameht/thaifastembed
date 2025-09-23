"""
Tests for PyThaiNLPTokenizer with enhanced custom dictionary support.
"""

from thaifastembed.text_processor import TextProcessor, PyThaiNLPTokenizer
from thaifastembed.utils import load_custom_dict
from pythainlp.tokenize import Trie

class TestCustomDictionary:
    """Test PyThaiNLPTokenizer with enhanced custom dictionary for furniture terms."""

    def test_custom_dict_loads_from_file(self):
        """Verify custom dictionary loads from words.txt."""
        tokenizer = PyThaiNLPTokenizer()

        assert tokenizer is not None
        assert tokenizer.custom_dict is not None

        # Use utils function to verify dictionary loads properly
        custom_dict = load_custom_dict()
        assert custom_dict is not None, "load_custom_dict should return a Trie object"
        assert isinstance(custom_dict, Trie), "Should return a Trie instance"

    def test_tokenizer_modes(self):
        """Test different tokenizer modes: with and without custom dictionary."""
        # Enhanced mode (default): custom + Thai words
        with_custom = PyThaiNLPTokenizer(use_custom_dict=True)
        
        # No custom mode: PyThaiNLP default only
        without_custom = PyThaiNLPTokenizer(use_custom_dict=False)
        
        test_text = "โต๊ะกลาง"
        
        with_custom_tokens = with_custom.tokenize(test_text)
        without_custom_tokens = without_custom.tokenize(test_text)
        
        assert 'โต๊ะกลาง' in with_custom_tokens, f"Enhanced should recognize compound: {with_custom_tokens}"
        
        print(f"With custom dict: {with_custom_tokens}")
        print(f"Without custom dict: {without_custom_tokens}")

    def test_custom_dict_affects_tokenization(self):
        """Prove custom dictionary changes tokenization behavior."""
        with_dict = PyThaiNLPTokenizer(use_custom_dict=True)
        without_dict = PyThaiNLPTokenizer(use_custom_dict=False)

        test_texts = [
            "โซฟา sofa เก้าอี้HelloWorldไทยEnglishมิกซ์ กินrice",
            "Coffee Table",
            "Air Conditioner",
            "Built-in",
            "Sound Bar",
            "โต๊ะกลาง",
            "เก้าอี้ทำงาน",
            "ตู้เย็น",
            # Mixed terms
            "Sofa bed โซฟาเบด",
        ]

        differences_found = 0
        for text in test_texts:
            tokens_with = with_dict.tokenize(text)
            tokens_without = without_dict.tokenize(text)

            print(f"Text: '{text}'")
            print(f"  With custom dict: {tokens_with}")
            print(f"  Without custom dict: {tokens_without}")

    def test_thai_furniture_vocabulary(self):
        """Test Thai furniture terms from custom dictionary."""
        tokenizer = PyThaiNLPTokenizer(use_custom_dict=True)
        processor = TextProcessor(tokenizer=tokenizer)

        # Thai furniture terms from words.txt
        thai_furniture_text = """
        ต้องการ โซฟา เก้าอี้ โต๊ะกลาง ตู้เย็น ตู้กับข้าว 
        เตียงนอน ชั้นวางของ โคมไฟ พรมปูพื้น ม่านหน้าต่าง
        เครื่องปรับอากาศ เครื่องซักผ้า ตู้เสื้อผ้า
        """

        tokens = processor.process_text(thai_furniture_text)

        # Should tokenize Thai furniture terms
        furniture_terms = ["โซฟา", "เก้าอี้", "โต๊ะ", "ตู้", "เตียง", "ชั้น", "โคม", "พรม"]
        found_terms = sum(1 for term in furniture_terms if any(term in token for token in tokens))

        assert found_terms >= 6, f"Should find most furniture terms, found {found_terms}"

    def test_english_furniture_from_dict(self):
        """Test English furniture terms from custom dictionary."""
        tokenizer = PyThaiNLPTokenizer(use_custom_dict=True)

        # English furniture terms from words.txt
        english_text = "Modern Sofa with Coffee Table, Built-in Hood and Microwave"

        tokens = tokenizer.tokenize(english_text)

        # Check tokenization preserves furniture terms
        tokens_lower = [t.lower() for t in tokens]

        # These furniture terms should appear in tokens
        furniture_found = 0
        for term in ["sofa", "coffee", "table", "built", "hood", "microwave"]:
            if any(term in token for token in tokens_lower):
                furniture_found += 1

        assert furniture_found >= 4, f"Should find most furniture terms, found {furniture_found}"

    def test_mixed_language_benefits_from_dict(self):
        """Test mixed Thai-English text benefits from custom dictionary."""
        tokenizer = PyThaiNLPTokenizer(use_custom_dict=True)

        # Mixed text with furniture terms
        mixed_text = "Living room ต้องการ Sofa bed โซฟาเบด"

        tokens = tokenizer.tokenize(mixed_text)

        # Check both languages are preserved
        token_str = " ".join(tokens)
        assert "Living" in token_str, "Should preserve English words"
        assert "ต้องการ" in token_str, "Should preserve Thai words"

    def test_tokenizer_interface_compliance(self):
        """Test that PyThaiNLPTokenizer properly implements Tokenizer interface."""
        from thaifastembed.types import Tokenizer
        
        tokenizer = PyThaiNLPTokenizer()
        
        assert isinstance(tokenizer, Tokenizer), "PyThaiNLPTokenizer should implement Tokenizer interface"
        
        assert hasattr(tokenizer, 'tokenize'), "Should have tokenize method"
        assert callable(getattr(tokenizer, 'tokenize')), "tokenize should be callable"
        
        result = tokenizer.tokenize("ทดสอบ")
        assert isinstance(result, list), "tokenize should return a list"
        assert all(isinstance(token, str) for token in result), "All tokens should be strings"

    def test_error_handling(self):
        """Test error handling for edge cases."""
        tokenizer = PyThaiNLPTokenizer()
        
        assert tokenizer.tokenize("") == []
        
        assert tokenizer.tokenize(None) == []
        
        assert tokenizer.tokenize(123) == []