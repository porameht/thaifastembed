use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use murmur3::murmur3_32;
use std::io;
use hashbrown::HashSet;

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
            let denominator = num_occurrences + self.k * (1.0 - self.b + self.b * doc_len / self.avg_len);
            
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
            let embedding = SparseEmbedding::from_dict(&py.get_type_bound::<SparseEmbedding>(), py, token_dict)?;
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
        
        let mut token_ids: Vec<u32> = unique_tokens.iter()
            .map(|token| Self::compute_token_id(token))
            .collect();
        token_ids.sort();
        
        let values = vec![1.0; token_ids.len()];
        
        SparseEmbedding::new(py, token_ids, values)
    }
}