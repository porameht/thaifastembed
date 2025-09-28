#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

// Module declarations
mod bm25;
mod sparse_embedding;
mod tokenizer;

// Re-export public types - simple and clean
pub use bm25::ThaiBm25;
pub use sparse_embedding::SparseEmbedding;
pub use tokenizer::{StopwordsFilter, TextProcessor, Tokenizer};

/// A Python module implemented in Rust
#[pymodule]
fn thaifastembed_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SparseEmbedding>()?;
    m.add_class::<Tokenizer>()?;
    m.add_class::<TextProcessor>()?;
    m.add_class::<StopwordsFilter>()?;
    m.add_class::<ThaiBm25>()?;
    Ok(())
}
