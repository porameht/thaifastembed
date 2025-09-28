use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Sparse embedding representation compatible with Python interface
#[pyclass]
pub struct SparseEmbedding {
    #[pyo3(get)]
    pub indices: Py<PyArray1<u32>>,
    #[pyo3(get)]
    pub values: Py<PyArray1<f32>>,
}

#[pymethods]
impl SparseEmbedding {
    #[new]
    pub fn new(py: Python, indices: Vec<u32>, values: Vec<f32>) -> PyResult<Self> {
        let indices_array = indices.to_pyarray_bound(py).unbind();
        let values_array = values.to_pyarray_bound(py).unbind();

        Ok(SparseEmbedding {
            indices: indices_array,
            values: values_array,
        })
    }

    #[classmethod]
    pub fn from_dict(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python,
        token_dict: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        if token_dict.is_empty() {
            return Ok(SparseEmbedding {
                indices: [].to_pyarray_bound(py).unbind(),
                values: [].to_pyarray_bound(py).unbind(),
            });
        }

        let mut pairs: Vec<(u32, f32)> = Vec::new();
        for (key, value) in token_dict.iter() {
            let idx: u32 = key.extract()?;
            let val: f32 = value.extract()?;
            pairs.push((idx, val));
        }
        pairs.sort_by_key(|&(idx, _)| idx);

        let (indices, values): (Vec<u32>, Vec<f32>) = pairs.into_iter().unzip();

        Ok(SparseEmbedding {
            indices: indices.to_pyarray_bound(py).unbind(),
            values: values.to_pyarray_bound(py).unbind(),
        })
    }

    pub fn as_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let indices = self.indices.bind(py).try_readonly()?;
        let values = self.values.bind(py).try_readonly()?;

        let indices_slice = indices.as_slice()?;
        let values_slice = values.as_slice()?;

        let dict = PyDict::new_bound(py);
        for (&idx, &val) in indices_slice.iter().zip(values_slice.iter()) {
            dict.set_item(idx, val)?;
        }

        Ok(dict.unbind())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    // Test helper to simulate SparseEmbedding logic without PyO3
    #[derive(Debug, Clone)]
    struct TestSparseEmbedding {
        indices: Vec<u32>,
        values: Vec<f32>,
    }

    impl TestSparseEmbedding {
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

        fn as_dict(&self) -> HashMap<u32, f32> {
            self.indices
                .iter()
                .zip(self.values.iter())
                .map(|(&idx, &val)| (idx, val))
                .collect()
        }
    }

    #[test]
    fn test_sparse_embedding_ordering() {
        let mut token_dict = HashMap::new();
        token_dict.insert(300, 1.0);
        token_dict.insert(100, 0.5);
        token_dict.insert(200, 1.2);

        let embedding = TestSparseEmbedding::from_dict(token_dict);

        // Indices should be sorted
        for i in 1..embedding.indices.len() {
            assert!(
                embedding.indices[i] > embedding.indices[i - 1],
                "Indices should be sorted"
            );
        }
    }

    #[test]
    fn test_empty_sparse_embedding() {
        let empty_dict = HashMap::new();
        let embedding = TestSparseEmbedding::from_dict(empty_dict);
        assert_eq!(embedding.indices.len(), 0);
        assert_eq!(embedding.values.len(), 0);
    }

    #[test]
    fn test_round_trip_conversion() {
        let mut original_dict = HashMap::new();
        original_dict.insert(42, 3.14);
        original_dict.insert(17, 2.71);
        original_dict.insert(99, 1.41);

        let embedding = TestSparseEmbedding::from_dict(original_dict.clone());
        let converted_dict = embedding.as_dict();

        assert_eq!(original_dict.len(), converted_dict.len());
        for (key, value) in original_dict {
            assert!((converted_dict[&key] - value).abs() < 1e-6);
        }
    }

    #[test]
    fn test_large_indices() {
        let mut token_dict = HashMap::new();
        token_dict.insert(1_000_000, 1.0);
        token_dict.insert(2_000_000, 2.0);

        let embedding = TestSparseEmbedding::from_dict(token_dict);
        assert_eq!(embedding.indices.len(), 2);
        assert_eq!(embedding.values.len(), 2);
        assert_eq!(embedding.indices.len(), embedding.values.len());
    }
}
