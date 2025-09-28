"""
Tests for Tokenizer functionality with various text inputs.
"""

from thaifastembed import TextProcessor, Tokenizer

class TestTokenizer:
    """Test Tokenizer with various text inputs."""

    @property
    def test_furniture_words(self):
        """Test furniture vocabulary for testing purposes."""
        return [
            # Thai furniture terms
            "โซฟา", "เก้าอี้", "โต๊ะกลาง", "ตู้เย็น", "ตู้กับข้าว",
            "เตียงนอน", "ชั้นวางของ", "โคมไฟ", "พรมปูพื้น", "ม่านหน้าต่าง",
            "เครื่องปรับอากาศ", "เครื่องซักผ้า", "ตู้เสื้อผ้า", "เก้าอี้ทำงาน", "โซฟาเบด",
            # English furniture terms
            "Sofa", "Coffee Table", "Built-in", "Hood", "Microwave",
            "Air Conditioner", "Sound Bar", "Living Room", "Bedroom",
            "Kitchen", "Dining Room", "Office Chair", "Wardrobe",
            "Refrigerator", "Washing Machine"
        ]

    def test_tokenizer_initialization(self):
        """Verify tokenizer initialization."""
        tokenizer = Tokenizer()
        assert tokenizer is not None

    def test_thai_furniture_vocabulary(self):
        """Test Thai furniture terms tokenization."""
        tokenizer = Tokenizer()
        processor = TextProcessor(tokenizer)

        # Thai furniture terms
        thai_furniture_text = """
        ต้องการ โซฟา เก้าอี้ โต๊ะกลาง ตู้เย็น ตู้กับข้าว 
        เตียงนอน ชั้นวางของ โคมไฟ พรมปูพื้น ม่านหน้าต่าง
        เครื่องปรับอากาศ เครื่องซักผ้า ตู้เสื้อผ้า
        """

        tokens = tokenizer.tokenize(thai_furniture_text)
        processed_tokens = processor.process_text(thai_furniture_text)

        # Should tokenize Thai furniture terms
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert isinstance(processed_tokens, list)
        assert len(processed_tokens) > 0

    def test_english_furniture_tokenization(self):
        """Test English furniture terms tokenization."""
        tokenizer = Tokenizer()

        # English furniture terms
        english_text = "Modern Sofa with Coffee Table, Built-in Hood and Microwave"

        tokens = tokenizer.tokenize(english_text)

        # Check tokenization preserves furniture terms
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        tokens_lower = [t.lower() for t in tokens]
        
        # These furniture terms should appear in tokens
        furniture_found = 0
        for term in ["sofa", "coffee", "table", "built", "hood", "microwave"]:
            if any(term in token for token in tokens_lower):
                furniture_found += 1

        assert furniture_found >= 3, f"Should find most furniture terms, found {furniture_found}"

    def test_mixed_language_tokenization(self):
        """Test mixed Thai-English text tokenization."""
        tokenizer = Tokenizer()

        # Mixed text with furniture terms
        mixed_text = "Living room ต้องการ Sofa bed โซฟาเบด"

        tokens = tokenizer.tokenize(mixed_text)

        # Check both languages are preserved
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        token_str = " ".join(tokens)
        assert len(token_str) > 0

    def test_tokenizer_interface_compliance(self):
        """Test that Tokenizer properly implements expected interface."""
        tokenizer = Tokenizer()
        
        assert hasattr(tokenizer, 'tokenize'), "Should have tokenize method"
        assert callable(getattr(tokenizer, 'tokenize')), "tokenize should be callable"
        
        result = tokenizer.tokenize("ทดสอบ")
        assert isinstance(result, list), "tokenize should return a list"
        assert all(isinstance(token, str) for token in result), "All tokens should be strings"

    def test_error_handling(self):
        """Test error handling for edge cases."""
        tokenizer = Tokenizer()
        
        # Empty string
        assert tokenizer.tokenize("") == []
        
        # Test various inputs
        assert isinstance(tokenizer.tokenize("test"), list)

    def test_text_processor_with_tokenizer(self):
        """Test TextProcessor with various inputs."""
        tokenizer = Tokenizer()
        processor = TextProcessor(tokenizer)
        
        # Thai text
        thai_result = processor.process_text("ภาษาไทยสวยงาม")
        assert isinstance(thai_result, list)
        
        # English text  
        english_result = processor.process_text("Beautiful language")
        assert isinstance(english_result, list)
        
        # Mixed text
        mixed_result = processor.process_text("ภาษาไทย Beautiful language")
        assert isinstance(mixed_result, list)