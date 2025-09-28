use hashbrown::HashSet;
use nlpo3::tokenizer::newmm::NewmmTokenizer;
use nlpo3::tokenizer::tokenizer_trait::Tokenizer as TokenizerTrait;
use pyo3::prelude::*;
use unicode_segmentation::UnicodeSegmentation;

/// Thai tokenizer using nlpo3 NewMM algorithm with default Thai words
#[pyclass]
pub struct Tokenizer {
    tokenizer: NewmmTokenizer,
}

impl Tokenizer {
    /// Load default Thai words from words_th.txt file
    fn load_default_thai_words() -> Vec<String> {
        let words_content = include_str!("data/words_th.txt");
        words_content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .map(|word| word.to_string())
            .collect()
    }
}

impl Clone for Tokenizer {
    fn clone(&self) -> Self {
        // Create a new tokenizer with default Thai words
        let tokenizer = NewmmTokenizer::from_word_list(Self::load_default_thai_words());
        Tokenizer { tokenizer }
    }
}

#[pymethods]
impl Tokenizer {
    #[new]
    #[pyo3(signature = (custom_words=None))]
    pub fn new(custom_words: Option<Vec<String>>) -> PyResult<Self> {
        let tokenizer = if let Some(words) = custom_words {
            NewmmTokenizer::from_word_list(words)
        } else {
            // Use Thai words from words_th.txt file
            NewmmTokenizer::from_word_list(Self::load_default_thai_words())
        };

        Ok(Tokenizer { tokenizer })
    }

    /// Tokenize Thai text using NewMM algorithm
    pub fn tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        match TokenizerTrait::segment(&self.tokenizer, text, true, false) {
            Ok(tokens) => {
                let filtered_tokens: Vec<String> = tokens
                    .into_iter()
                    .filter(|token| !token.trim().is_empty())
                    .collect();
                Ok(filtered_tokens)
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Tokenization error: {}",
                e
            ))),
        }
    }
}

/// Thai stopwords filter using comprehensive stopwords_th.txt
#[pyclass]
#[derive(Clone)]
pub struct StopwordsFilter {
    stopwords: HashSet<String>,
}

impl Default for StopwordsFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl StopwordsFilter {
    /// Load Thai stopwords from stopwords_th.txt file
    fn load_thai_stopwords() -> HashSet<String> {
        let stopwords_content = include_str!("data/stopwords_th.txt");
        stopwords_content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !line.starts_with('\u{feff}')) // Filter BOM
            .map(|word| word.to_string())
            .collect()
    }
}

#[pymethods]
impl StopwordsFilter {
    #[new]
    pub fn new() -> Self {
        StopwordsFilter {
            stopwords: Self::load_thai_stopwords(),
        }
    }

    pub fn is_stopword(&self, word: &str) -> bool {
        self.stopwords.contains(word)
    }

    /// Get number of loaded stopwords
    pub fn len(&self) -> usize {
        self.stopwords.len()
    }

    /// Check if stopwords collection is empty
    pub fn is_empty(&self) -> bool {
        self.stopwords.is_empty()
    }
}

/// Text processor for filtering and normalizing tokens
#[pyclass]
#[derive(Clone)]
pub struct TextProcessor {
    tokenizer: Tokenizer,
    lowercase: bool,
    stopwords_filter: Option<StopwordsFilter>,
    min_token_len: Option<usize>,
}

#[pymethods]
impl TextProcessor {
    #[new]
    #[pyo3(signature = (tokenizer, lowercase=true, stopwords_filter=None, min_token_len=None))]
    pub fn new(
        tokenizer: Tokenizer,
        lowercase: bool,
        stopwords_filter: Option<StopwordsFilter>,
        min_token_len: Option<usize>,
    ) -> Self {
        TextProcessor {
            tokenizer,
            lowercase,
            stopwords_filter,
            min_token_len,
        }
    }

    /// Process text through the complete pipeline
    pub fn process_text(&self, text: &str) -> PyResult<Vec<String>> {
        // Tokenize first
        let tokens = self.tokenizer.tokenize(text)?;

        // Then filter and process
        let processed_tokens: Vec<String> = tokens
            .into_iter()
            .filter_map(|token| self.process_token(&token))
            .collect();

        Ok(processed_tokens)
    }

    /// Process a single token
    fn process_token(&self, token: &str) -> Option<String> {
        if token.is_empty() {
            return None;
        }

        let processed = if self.lowercase {
            token.to_lowercase()
        } else {
            token.to_string()
        };

        // Check stopwords
        if let Some(ref filter) = self.stopwords_filter {
            if filter.is_stopword(&processed) {
                return None;
            }
        }

        // Check minimum token length
        let token_len = processed.graphemes(true).count();
        if let Some(min_len) = self.min_token_len {
            if token_len < min_len {
                return None;
            }
        }

        Some(processed)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    // Mock structures for testing tokenizer logic without PyO3
    struct MockTokenizer;

    impl MockTokenizer {
        fn new(_custom_words: Option<Vec<String>>) -> Self {
            Self
        }

        fn tokenize(&self, text: &str) -> Vec<String> {
            if text.is_empty() {
                return vec![];
            }

            // Simple tokenization: split by whitespace and punctuation
            text.split_whitespace()
                .flat_map(|word| word.split(',').flat_map(|w| w.split('.')))
                .filter(|token| !token.trim().is_empty())
                .map(|token| token.to_string())
                .collect()
        }
    }

    struct MockStopwordsFilter {
        stopwords: HashSet<String>,
    }

    impl MockStopwordsFilter {
        fn new() -> Self {
            let mut stopwords = HashSet::new();
            stopwords.insert("และ".to_string());
            stopwords.insert("ของ".to_string());
            stopwords.insert("ที่".to_string());
            stopwords.insert("เป็น".to_string());
            Self { stopwords }
        }

        fn is_stopword(&self, word: &str) -> bool {
            self.stopwords.contains(word)
        }

        fn len(&self) -> usize {
            self.stopwords.len()
        }
    }

    struct MockTextProcessor {
        tokenizer: MockTokenizer,
        lowercase: bool,
        stopwords_filter: Option<MockStopwordsFilter>,
        min_token_len: usize,
    }

    impl MockTextProcessor {
        fn new(
            tokenizer: MockTokenizer,
            lowercase: bool,
            stopwords_filter: Option<MockStopwordsFilter>,
            min_token_len: usize,
        ) -> Self {
            Self {
                tokenizer,
                lowercase,
                stopwords_filter,
                min_token_len,
            }
        }

        fn process_token(&self, token: &str) -> Option<String> {
            if token.trim().is_empty() {
                return None;
            }

            let processed = if self.lowercase {
                token.to_lowercase()
            } else {
                token.to_string()
            };

            if processed.len() < self.min_token_len {
                return None;
            }

            if let Some(ref filter) = self.stopwords_filter {
                if filter.is_stopword(&processed) {
                    return None;
                }
            }

            Some(processed)
        }

        fn process_text(&self, text: &str) -> Vec<String> {
            if text.is_empty() {
                return vec![];
            }

            self.tokenizer
                .tokenize(text)
                .into_iter()
                .filter_map(|token| self.process_token(&token))
                .collect()
        }
    }

    #[test]
    fn test_tokenizer_empty_input() {
        let tokenizer = MockTokenizer::new(None);
        let result = tokenizer.tokenize("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_tokenizer_basic_functionality() {
        let tokenizer = MockTokenizer::new(None);
        let result = tokenizer.tokenize("hello world test");
        assert_eq!(result.len(), 3);
        assert_eq!(result, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_tokenizer_punctuation_handling() {
        let tokenizer = MockTokenizer::new(None);
        let result = tokenizer.tokenize("hello, world. test");
        assert!(result.len() >= 3);
        assert!(result.contains(&"hello".to_string()));
        assert!(result.contains(&"world".to_string()));
        assert!(result.contains(&"test".to_string()));
    }

    #[test]
    fn test_stopwords_filter() {
        let filter = MockStopwordsFilter::new();

        // Thai stopwords
        assert!(filter.is_stopword("และ"));
        assert!(filter.is_stopword("ของ"));

        // Non-stopwords
        assert!(!filter.is_stopword("python"));
        assert!(!filter.is_stopword("the"));

        assert!(filter.len() > 0);
    }

    #[test]
    fn test_text_processor_basic() {
        let tokenizer = MockTokenizer::new(None);
        let processor = MockTextProcessor::new(tokenizer, true, None, 1);

        let result = processor.process_text("Hello World Test");
        assert!(!result.is_empty());
        assert!(result.contains(&"hello".to_string()));
        assert!(result.contains(&"world".to_string()));
    }

    #[test]
    fn test_text_processor_with_stopwords() {
        let tokenizer = MockTokenizer::new(None);
        let filter = MockStopwordsFilter::new();
        let processor = MockTextProcessor::new(tokenizer, true, Some(filter), 1);

        let result = processor.process_text("hello และ world");
        assert!(result.contains(&"hello".to_string()));
        assert!(result.contains(&"world".to_string()));
        assert!(!result.contains(&"และ".to_string()));
    }

    #[test]
    fn test_text_processor_min_length() {
        let tokenizer = MockTokenizer::new(None);
        let processor = MockTextProcessor::new(tokenizer, false, None, 3);

        let result = processor.process_text("a bb hello world");
        assert!(!result.contains(&"a".to_string()));
        assert!(!result.contains(&"bb".to_string()));
        assert!(result.contains(&"hello".to_string()));
        assert!(result.contains(&"world".to_string()));
    }

    #[test]
    fn test_text_processor_lowercase() {
        let tokenizer = MockTokenizer::new(None);
        let processor = MockTextProcessor::new(tokenizer, true, None, 1);

        let result = processor.process_text("Hello WORLD");
        assert!(result.contains(&"hello".to_string()));
        assert!(result.contains(&"world".to_string()));
        assert!(!result.contains(&"Hello".to_string()));
        assert!(!result.contains(&"WORLD".to_string()));
    }

    #[test]
    fn test_process_token_empty() {
        let tokenizer = MockTokenizer::new(None);
        let processor = MockTextProcessor::new(tokenizer, true, None, 1);

        let result = processor.process_token("");
        assert!(result.is_none());

        let result = processor.process_token("   ");
        assert!(result.is_none());
    }

    #[test]
    fn test_thai_text_handling() {
        let tokenizer = MockTokenizer::new(None);
        let processor = MockTextProcessor::new(tokenizer, true, None, 1);

        // Basic Thai text test (simplified tokenization)
        let result = processor.process_text("ภาษาไทย สวยงาม");
        assert!(!result.is_empty());
    }
}
