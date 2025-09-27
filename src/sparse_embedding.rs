use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, ToPyArray, PyArrayMethods};

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
    pub fn from_dict(_cls: &Bound<'_, pyo3::types::PyType>, py: Python, token_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        if token_dict.is_empty() {
            return Ok(SparseEmbedding {
                indices: vec![].to_pyarray_bound(py).unbind(),
                values: vec![].to_pyarray_bound(py).unbind(),
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