//! JAMMA native extensions for high-performance linear algebra.
//!
//! This module provides Rust implementations of computationally intensive
//! operations, starting with eigendecomposition of kinship matrices.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Eigendecompose a symmetric kinship matrix.
///
/// Returns (eigenvalues, eigenvectors) tuple where eigenvalues are sorted
/// ascending and eigenvectors are column vectors. Eigenvalues below the
/// threshold are zeroed (GEMMA compatibility).
///
/// # Arguments
/// * `k` - Symmetric kinship matrix (n x n)
/// * `threshold` - Eigenvalues with absolute value below this are zeroed (default: 1e-10)
///
/// # Returns
/// * Tuple of (eigenvalues, eigenvectors) as NumPy arrays
///
/// # Errors
/// * `ValueError` if matrix is not square or is empty
#[pyfunction]
#[pyo3(signature = (k, threshold = 1e-10))]
#[allow(clippy::type_complexity)]
fn eigendecompose_kinship<'py>(
    py: Python<'py>,
    k: PyReadonlyArray2<'py, f64>,
    threshold: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let k_array = k.as_array();
    let (nrows, ncols) = k_array.dim();

    // Validate input
    if nrows != ncols {
        return Err(PyValueError::new_err(format!(
            "Kinship matrix must be square, got {}x{}",
            nrows, ncols
        )));
    }
    if nrows == 0 {
        return Err(PyValueError::new_err("Kinship matrix cannot be empty"));
    }

    // Copy to contiguous vec (required for py.allow_threads)
    // NumPy is row-major, we'll handle conversion in compute_eigen_internal
    let k_vec: Vec<f64> = k_array.iter().copied().collect();

    // Release GIL during heavy computation
    let (eigenvalues, eigenvectors_flat) =
        py.allow_threads(|| compute_eigen_internal(&k_vec, nrows, threshold));

    // Convert results to NumPy arrays
    let eigenvalues_arr = eigenvalues.into_pyarray(py);
    let eigenvectors_arr = Array2::from_shape_vec((nrows, ncols), eigenvectors_flat)
        .expect("eigenvector reshape failed")
        .into_pyarray(py);

    Ok((eigenvalues_arr, eigenvectors_arr))
}

/// Internal computation (runs without GIL).
fn compute_eigen_internal(k_flat: &[f64], n: usize, threshold: f64) -> (Vec<f64>, Vec<f64>) {
    use faer::Mat;

    // Convert row-major input to faer column-major Mat
    // k_flat[i * n + j] is element (i, j) in row-major
    let k_mat: Mat<f64> = Mat::from_fn(n, n, |i, j| k_flat[i * n + j]);

    // Compute self-adjoint eigendecomposition
    // faer returns eigenvalues sorted ascending (nondecreasing order)
    let eigen = k_mat
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("eigendecomposition failed");
    let s = eigen.S().column_vector(); // eigenvalues as column vector
    let u = eigen.U(); // eigenvectors as MatRef (columns)

    // Extract eigenvalues, zeroing values below threshold (GEMMA compatibility)
    let eigenvalues: Vec<f64> = (0..n)
        .map(|i| if s[i].abs() < threshold { 0.0 } else { s[i] })
        .collect();

    // Convert eigenvectors to row-major for NumPy
    let eigenvectors: Vec<f64> = (0..n)
        .flat_map(|i| (0..n).map(move |j| u[(i, j)]))
        .collect();

    (eigenvalues, eigenvectors)
}

/// JAMMA native extensions module.
#[pymodule]
fn jamma_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eigendecompose_kinship, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_eigendecomp() {
        let n = 3;
        let k_flat: Vec<f64> = (0..n * n)
            .map(|idx| if idx / n == idx % n { 1.0 } else { 0.0 })
            .collect();

        let (eigenvalues, eigenvectors) = compute_eigen_internal(&k_flat, n, 1e-10);

        // All eigenvalues should be 1.0
        for &v in &eigenvalues {
            assert!((v - 1.0).abs() < 1e-10);
        }

        // Eigenvectors should reconstruct identity
        assert_eq!(eigenvectors.len(), n * n);
    }

    #[test]
    fn test_threshold_zeroing() {
        // Matrix with very small eigenvalue
        let n = 2;
        // [[1, 0], [0, 1e-11]] has eigenvalues 1 and 1e-11
        let k_flat = vec![1.0, 0.0, 0.0, 1e-11];

        let (eigenvalues, _) = compute_eigen_internal(&k_flat, n, 1e-10);

        // Smaller eigenvalue should be zeroed
        assert!(eigenvalues.iter().any(|&v| v == 0.0));
        assert!(eigenvalues.iter().any(|&v| (v - 1.0).abs() < 1e-10));
    }
}
