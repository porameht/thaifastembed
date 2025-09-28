use hashbrown::HashSet;
use murmur3::murmur3_32;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::io;

use crate::sparse_embedding::SparseEmbedding;
use crate::tokenizer::TextProcessor;

/// Thai BM25 implementation with sparse embeddings
#[pyclass]
pub struct ThaiBm25 {
    text_processor: Option<TextProcessor>,
    k: f32,
    b: f32,
    avg_len: f32,
}

#[pymethods]
impl ThaiBm25 {
    #[new]
    #[pyo3(signature = (text_processor=None, k=1.2, b=0.75, avg_len=256.0))]
    pub fn new(
        text_processor: Option<TextProcessor>,
        k: f32,
        b: f32,
        avg_len: f32,
    ) -> PyResult<Self> {
        Ok(ThaiBm25 {
            text_processor,
            k,
            b,
            avg_len,
        })
    }

    #[staticmethod]
    pub fn compute_token_id(token: &str) -> u32 {
        // Use murmur3_32 to match Python mmh3.hash(token, signed=False)
        murmur3_32(&mut io::Cursor::new(token.as_bytes()), 0).unwrap()
    }

    fn _clean_and_tokenize(&self, text: &str) -> PyResult<Vec<String>> {
        if let Some(processor) = &self.text_processor {
            processor.process_text(text)
        } else {
            // Default: simple whitespace split and lowercase
            Ok(text
                .to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect())
        }
    }

    fn _term_frequency(&self, py: Python, tokens: Vec<String>) -> PyResult<Py<PyDict>> {
        let mut token_counts: HashMap<String, u32> = HashMap::new();

        // Count token occurrences
        for token in &tokens {
            *token_counts.entry(token.clone()).or_insert(0) += 1;
        }

        let doc_len = tokens.len() as f32;
        let tf_dict = PyDict::new_bound(py);

        for (token, count) in token_counts {
            let token_id = Self::compute_token_id(&token);
            let num_occurrences = count as f32;

            // BM25 term frequency calculation
            let tf = num_occurrences * (self.k + 1.0);
            let denominator =
                num_occurrences + self.k * (1.0 - self.b + self.b * doc_len / self.avg_len);

            tf_dict.set_item(token_id, tf / denominator)?;
        }

        Ok(tf_dict.unbind())
    }

    /// Embed documents into sparse BM25 vectors
    pub fn embed(&self, py: Python, documents: Vec<String>) -> PyResult<Vec<SparseEmbedding>> {
        let mut embeddings = Vec::new();

        for document in documents {
            let tokens = self._clean_and_tokenize(&document)?;
            let token_id2value = self._term_frequency(py, tokens)?;
            let token_dict = token_id2value.bind(py);
            let embedding = SparseEmbedding::from_dict(
                &py.get_type_bound::<SparseEmbedding>(),
                py,
                token_dict,
            )?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Embed query into sparse BM25 vector
    pub fn query_embed(&self, py: Python, query: &str) -> PyResult<SparseEmbedding> {
        let tokens = self._clean_and_tokenize(query)?;

        // For queries, use unique token IDs with value 1.0
        let mut unique_tokens: HashSet<String> = HashSet::new();
        for token in tokens {
            unique_tokens.insert(token);
        }

        let mut token_ids: Vec<u32> = unique_tokens
            .iter()
            .map(|token| Self::compute_token_id(token))
            .collect();
        token_ids.sort();

        let values = vec![1.0; token_ids.len()];

        SparseEmbedding::new(py, token_ids, values)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    // Mock structures for testing BM25 logic without PyO3
    #[derive(Debug, Clone)]
    struct MockSparseEmbedding {
        indices: Vec<u32>,
        values: Vec<f32>,
    }

    impl MockSparseEmbedding {
        fn new(values: Vec<f32>, indices: Vec<u32>) -> Self {
            Self { indices, values }
        }

        fn from_dict(dict: HashMap<u32, f32>) -> Self {
            if dict.is_empty() {
                return Self {
                    indices: vec![],
                    values: vec![],
                };
            }

            let mut pairs: Vec<(u32, f32)> = dict.into_iter().collect();
            pairs.sort_by_key(|&(idx, _)| idx);

            let (indices, values): (Vec<u32>, Vec<f32>) = pairs.into_iter().unzip();
            Self { indices, values }
        }
    }

    struct MockTextProcessor;

    impl MockTextProcessor {
        fn new() -> Self {
            Self
        }

        fn process_text(&self, text: &str) -> Vec<String> {
            if text.is_empty() {
                return vec![];
            }
            text.split_whitespace().map(|s| s.to_lowercase()).collect()
        }
    }

    struct MockThaiBm25 {
        text_processor: Option<MockTextProcessor>,
        k: f32,
        b: f32,
        avg_len: f32,
    }

    impl MockThaiBm25 {
        fn new(text_processor: Option<MockTextProcessor>, k: f32, b: f32, avg_len: f32) -> Self {
            Self {
                text_processor,
                k,
                b,
                avg_len,
            }
        }

        fn default() -> Self {
            Self::new(Some(MockTextProcessor::new()), 1.2, 0.75, 256.0)
        }

        fn compute_token_id(token: &str) -> u32 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            token.hash(&mut hasher);
            hasher.finish() as u32
        }

        fn clean_and_tokenize(&self, text: &str) -> Vec<String> {
            if let Some(ref processor) = self.text_processor {
                processor.process_text(text)
            } else {
                text.to_lowercase()
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect()
            }
        }

        fn term_frequency(&self, tokens: Vec<String>) -> HashMap<u32, f32> {
            let mut token_counts: HashMap<String, u32> = HashMap::new();

            for token in &tokens {
                *token_counts.entry(token.clone()).or_insert(0) += 1;
            }

            let doc_len = tokens.len() as f32;
            let mut tf_dict = HashMap::new();

            for (token, count) in token_counts {
                let token_id = Self::compute_token_id(&token);
                let num_occurrences = count as f32;

                let tf = num_occurrences * (self.k + 1.0);
                let denominator =
                    num_occurrences + self.k * (1.0 - self.b + self.b * doc_len / self.avg_len);

                tf_dict.insert(token_id, tf / denominator);
            }

            tf_dict
        }

        fn embed(&self, documents: &[&str]) -> Vec<MockSparseEmbedding> {
            documents
                .iter()
                .map(|document| {
                    let tokens = self.clean_and_tokenize(document);
                    let token_dict = self.term_frequency(tokens);
                    MockSparseEmbedding::from_dict(token_dict)
                })
                .collect()
        }

        fn query_embed(&self, query: &str) -> MockSparseEmbedding {
            let tokens = self.clean_and_tokenize(query);

            let mut unique_tokens: HashSet<String> = HashSet::new();
            for token in tokens {
                unique_tokens.insert(token);
            }

            let mut token_ids: Vec<u32> = unique_tokens
                .iter()
                .map(|token| Self::compute_token_id(token))
                .collect();
            token_ids.sort();

            let values = vec![1.0; token_ids.len()];
            MockSparseEmbedding::new(values, token_ids)
        }
    }

    #[test]
    fn test_bm25_initialization() {
        let bm25 = MockThaiBm25::default();
        assert_eq!(bm25.k, 1.2);
        assert_eq!(bm25.b, 0.75);
        assert_eq!(bm25.avg_len, 256.0);

        let bm25_custom = MockThaiBm25::new(Some(MockTextProcessor::new()), 1.5, 0.8, 512.0);
        assert_eq!(bm25_custom.k, 1.5);
        assert_eq!(bm25_custom.b, 0.8);
        assert_eq!(bm25_custom.avg_len, 512.0);
    }

    #[test]
    fn test_token_id_computation() {
        let token = "ภาษาไทย";
        let id1 = MockThaiBm25::compute_token_id(token);
        let id2 = MockThaiBm25::compute_token_id(token);
        assert_eq!(id1, id2);

        let token2 = "ภาษาอังกฤษ";
        let id3 = MockThaiBm25::compute_token_id(token2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_text_processing() {
        let bm25 = MockThaiBm25::default();

        let text = "ประเทศไทยมีวัฒนธรรมที่หลากหลาย";
        let tokens = bm25.clean_and_tokenize(text);
        assert!(!tokens.is_empty());

        let empty_tokens = bm25.clean_and_tokenize("");
        assert!(empty_tokens.is_empty());
    }

    #[test]
    fn test_document_embedding() {
        let bm25 = MockThaiBm25::default();

        let documents = vec!["นี่ คือ เอกสาร แรก", "this is the second document"];
        let embeddings = bm25.embed(&documents);

        assert_eq!(embeddings.len(), 2);
        for embedding in &embeddings {
            assert_eq!(embedding.indices.len(), embedding.values.len());
        }
    }

    #[test]
    fn test_query_embedding() {
        let bm25 = MockThaiBm25::default();

        let query = "ค้นหาเอกสาร";
        let embedding = bm25.query_embed(query);

        if !embedding.values.is_empty() {
            for &value in &embedding.values {
                assert_eq!(value, 1.0);
            }
        }
    }

    #[test]
    fn test_empty_inputs() {
        let bm25 = MockThaiBm25::default();

        let empty_doc = bm25.embed(&[""]);
        assert_eq!(empty_doc[0].indices.len(), 0);

        let empty_query = bm25.query_embed("");
        assert_eq!(empty_query.indices.len(), 0);
    }

    #[test]
    fn test_term_frequency_calculation() {
        let bm25 = MockThaiBm25::default();

        let tokens = vec!["test".to_string(), "test".to_string(), "word".to_string()];
        let tf_dict = bm25.term_frequency(tokens);

        assert!(!tf_dict.is_empty());

        // Check that repeated tokens have different scores than single tokens
        let single_tokens = vec!["test".to_string(), "word".to_string()];
        let single_tf_dict = bm25.term_frequency(single_tokens);

        // The dictionaries should have different values for "test"
        let test_id = MockThaiBm25::compute_token_id("test");
        if tf_dict.contains_key(&test_id) && single_tf_dict.contains_key(&test_id) {
            assert_ne!(tf_dict[&test_id], single_tf_dict[&test_id]);
        }
    }

    #[test]
    fn test_bm25_parameters_effect() {
        let bm25_low_k = MockThaiBm25::new(Some(MockTextProcessor::new()), 0.5, 0.75, 256.0);
        let bm25_high_k = MockThaiBm25::new(Some(MockTextProcessor::new()), 2.0, 0.75, 256.0);

        let tokens = vec!["test".to_string(), "test".to_string()];
        let tf_low_k = bm25_low_k.term_frequency(tokens.clone());
        let tf_high_k = bm25_high_k.term_frequency(tokens);

        // Different k values should produce different scores
        let test_id = MockThaiBm25::compute_token_id("test");
        if tf_low_k.contains_key(&test_id) && tf_high_k.contains_key(&test_id) {
            assert_ne!(tf_low_k[&test_id], tf_high_k[&test_id]);
        }
    }

    #[test]
    fn test_integration_workflow() {
        let bm25 = MockThaiBm25::default();

        let documents = vec![
            "ภาษาไทยเป็นภาษาราชการของประเทศไทย",
            "การประมวลผลภาษาธรรมชาติเป็นสาขาของปัญญาประดิษฐ์",
        ];
        let query = "ภาษาไทย";

        let doc_embeddings = bm25.embed(&documents);
        let query_embedding = bm25.query_embed(query);

        assert_eq!(doc_embeddings.len(), 2);

        for embedding in &doc_embeddings {
            assert_eq!(embedding.indices.len(), embedding.values.len());
        }

        if !query_embedding.values.is_empty() {
            for &value in &query_embedding.values {
                assert_eq!(value, 1.0);
            }
        }
    }
}
