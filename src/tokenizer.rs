use pyo3::prelude::*;
use nlpo3::tokenizer::newmm::NewmmTokenizer;
use nlpo3::tokenizer::tokenizer_trait::Tokenizer as TokenizerTrait;
use hashbrown::HashSet;
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
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Tokenization error: {}", e)
            ))
        }
    }
}

/// Thai stopwords filter using comprehensive stopwords_th.txt
#[pyclass]
#[derive(Clone)]
pub struct StopwordsFilter {
    stopwords: HashSet<String>,
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