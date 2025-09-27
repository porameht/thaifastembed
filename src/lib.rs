use pyo3::prelude::*;

// Module declarations
mod sparse_embedding;
mod tokenizer;
mod bm25;

// Re-export public types - simple and clean
pub use sparse_embedding::SparseEmbedding;
pub use tokenizer::{Tokenizer, TextProcessor, StopwordsFilter};
pub use bm25::ThaiBm25;

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