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
/// * `k` - Symmetric kinship matrix (n x n). Only the lower triangle is read;
///   the matrix is assumed symmetric. For best performance, pass a symmetric
///   matrix; asymmetric input will use lower triangle values.
/// * `threshold` - Eigenvalues with absolute value below this are zeroed (default: 1e-10).
///   Must be non-negative and finite.
///
/// # Returns
/// * Tuple of (eigenvalues, eigenvectors) as NumPy arrays
///
/// # Errors
/// * `ValueError` if matrix is not square, is empty, contains NaN/Inf,
///   or threshold is invalid
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

    // Validate input dimensions
    if nrows != ncols {
        return Err(PyValueError::new_err(format!(
            "Kinship matrix must be square, got {}x{}",
            nrows, ncols
        )));
    }
    if nrows == 0 {
        return Err(PyValueError::new_err("Kinship matrix cannot be empty"));
    }

    // Validate threshold
    if threshold < 0.0 {
        return Err(PyValueError::new_err(format!(
            "threshold must be non-negative, got {}",
            threshold
        )));
    }
    if !threshold.is_finite() {
        return Err(PyValueError::new_err(format!(
            "threshold must be finite, got {}",
            threshold
        )));
    }

    // Copy to contiguous vec and check for NaN/Inf
    // NumPy is row-major, we'll handle conversion in compute_eigen_internal
    let mut k_vec: Vec<f64> = Vec::with_capacity(nrows * ncols);
    for &val in k_array.iter() {
        if !val.is_finite() {
            return Err(PyValueError::new_err(
                "Kinship matrix contains NaN or Inf values",
            ));
        }
        k_vec.push(val);
    }

    // Release GIL during heavy computation
    let result = py.detach(|| compute_eigen_internal(&k_vec, nrows, threshold));

    // Handle computation errors
    let (eigenvalues, eigenvectors_flat) = result.map_err(PyValueError::new_err)?;

    // Convert results to NumPy arrays
    let eigenvalues_arr = eigenvalues.into_pyarray(py);
    let eigenvectors_arr = Array2::from_shape_vec((nrows, ncols), eigenvectors_flat)
        .map_err(|e| PyValueError::new_err(format!("eigenvector reshape failed: {}", e)))?
        .into_pyarray(py);

    Ok((eigenvalues_arr, eigenvectors_arr))
}

/// Internal computation using faer (runs without GIL).
///
/// Returns `Ok((eigenvalues, eigenvectors))` on success, or `Err(message)` on failure.
fn compute_eigen_faer(
    k_flat: &[f64],
    n: usize,
    threshold: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    use faer::Mat;

    // Convert row-major input to faer column-major Mat
    // k_flat[i * n + j] is element (i, j) in row-major
    let k_mat: Mat<f64> = Mat::from_fn(n, n, |i, j| k_flat[i * n + j]);

    // Compute self-adjoint eigendecomposition
    // faer returns eigenvalues sorted ascending (nondecreasing order)
    // Note: faer uses only the lower triangle (Side::Lower), assuming symmetry
    let eigen = k_mat
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| format!("eigendecomposition failed: {:?}", e))?;
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

    Ok((eigenvalues, eigenvectors))
}

/// Internal computation using scirs2-linalg/OxiBLAS (runs without GIL).
///
/// Returns `Ok((eigenvalues, eigenvectors))` on success, or `Err(message)` on failure.
fn compute_eigen_scirs2(
    k_flat: &[f64],
    n: usize,
    threshold: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    use ndarray::Array2;
    use scirs2_linalg::eigh;

    // Convert row-major flat array to ndarray Array2
    let k_array: Array2<f64> = Array2::from_shape_vec((n, n), k_flat.to_vec())
        .map_err(|e| format!("array conversion failed: {}", e))?;

    // Compute symmetric eigendecomposition
    let (eigenvalues_arr, eigenvectors_arr): (ndarray::Array1<f64>, Array2<f64>) =
        eigh(&k_array.view(), None).map_err(|e| format!("eigendecomposition failed: {:?}", e))?;

    // Apply threshold zeroing (GEMMA compatibility)
    let eigenvalues: Vec<f64> = eigenvalues_arr
        .iter()
        .map(|&v: &f64| if v.abs() < threshold { 0.0 } else { v })
        .collect();

    // Convert eigenvectors to row-major flat vec for NumPy
    let eigenvectors: Vec<f64> = eigenvectors_arr.iter().cloned().collect();

    Ok((eigenvalues, eigenvectors))
}

/// Wrapper that calls the default backend (faer).
fn compute_eigen_internal(
    k_flat: &[f64],
    n: usize,
    threshold: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    compute_eigen_faer(k_flat, n, threshold)
}

/// Eigendecompose using scirs2-linalg/OxiBLAS backend.
///
/// Same interface as eigendecompose_kinship but uses OxiBLAS instead of faer.
#[pyfunction]
#[pyo3(signature = (k, threshold = 1e-10))]
#[allow(clippy::type_complexity)]
fn eigendecompose_kinship_scirs2<'py>(
    py: Python<'py>,
    k: PyReadonlyArray2<'py, f64>,
    threshold: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let k_array = k.as_array();
    let (nrows, ncols) = k_array.dim();

    // Validate input dimensions
    if nrows != ncols {
        return Err(PyValueError::new_err(format!(
            "Kinship matrix must be square, got {}x{}",
            nrows, ncols
        )));
    }
    if nrows == 0 {
        return Err(PyValueError::new_err("Kinship matrix cannot be empty"));
    }

    // Validate threshold
    if threshold < 0.0 {
        return Err(PyValueError::new_err(format!(
            "threshold must be non-negative, got {}",
            threshold
        )));
    }
    if !threshold.is_finite() {
        return Err(PyValueError::new_err(format!(
            "threshold must be finite, got {}",
            threshold
        )));
    }

    // Copy to contiguous vec and check for NaN/Inf
    let mut k_vec: Vec<f64> = Vec::with_capacity(nrows * ncols);
    for &val in k_array.iter() {
        if !val.is_finite() {
            return Err(PyValueError::new_err(
                "Kinship matrix contains NaN or Inf values",
            ));
        }
        k_vec.push(val);
    }

    // Release GIL during heavy computation - use scirs2 backend
    let result = py.detach(|| compute_eigen_scirs2(&k_vec, nrows, threshold));

    // Handle computation errors
    let (eigenvalues, eigenvectors_flat) = result.map_err(PyValueError::new_err)?;

    // Convert results to NumPy arrays
    let eigenvalues_arr = eigenvalues.into_pyarray(py);
    let eigenvectors_arr = Array2::from_shape_vec((nrows, ncols), eigenvectors_flat)
        .map_err(|e| PyValueError::new_err(format!("eigenvector reshape failed: {}", e)))?
        .into_pyarray(py);

    Ok((eigenvalues_arr, eigenvectors_arr))
}

/// JAMMA native extensions module.
#[pymodule]
fn jamma_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eigendecompose_kinship, m)?)?;
    m.add_function(wrap_pyfunction!(eigendecompose_kinship_scirs2, m)?)?;
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

        let (eigenvalues, eigenvectors) =
            compute_eigen_internal(&k_flat, n, 1e-10).expect("should succeed");

        // All eigenvalues should be 1.0
        for &v in &eigenvalues {
            assert!((v - 1.0).abs() < 1e-10);
        }

        // Eigenvectors should reconstruct identity
        assert_eq!(eigenvectors.len(), n * n);

        // Verify reconstruction: K = U @ diag(S) @ U.T
        for i in 0..n {
            for j in 0..n {
                let mut reconstructed = 0.0;
                for k in 0..n {
                    // eigenvectors[i * n + k] is U[i, k]
                    // eigenvectors[j * n + k] is U[j, k]
                    reconstructed +=
                        eigenvectors[i * n + k] * eigenvalues[k] * eigenvectors[j * n + k];
                }
                let original = k_flat[i * n + j];
                assert!(
                    (reconstructed - original).abs() < 1e-10,
                    "reconstruction mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    reconstructed,
                    original
                );
            }
        }
    }

    #[test]
    fn test_threshold_zeroing() {
        // Matrix with very small eigenvalue
        let n = 2;
        // [[1, 0], [0, 1e-11]] has eigenvalues 1 and 1e-11
        let k_flat = vec![1.0, 0.0, 0.0, 1e-11];

        let (eigenvalues, _) = compute_eigen_internal(&k_flat, n, 1e-10).expect("should succeed");

        // Smaller eigenvalue should be zeroed
        assert!(eigenvalues.iter().any(|&v| v == 0.0));
        assert!(eigenvalues.iter().any(|&v| (v - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_symmetric_reconstruction() {
        // Random symmetric 4x4 matrix
        let n = 4;
        // Symmetric matrix:
        // [[ 2.0,  0.5, -0.3,  0.1],
        //  [ 0.5,  3.0,  0.2, -0.4],
        //  [-0.3,  0.2,  1.5,  0.6],
        //  [ 0.1, -0.4,  0.6,  2.5]]
        #[rustfmt::skip]
        let k_flat = vec![
            2.0,  0.5, -0.3,  0.1,
            0.5,  3.0,  0.2, -0.4,
           -0.3,  0.2,  1.5,  0.6,
            0.1, -0.4,  0.6,  2.5,
        ];

        let (eigenvalues, eigenvectors) =
            compute_eigen_internal(&k_flat, n, 0.0).expect("should succeed");

        // Verify reconstruction: K = U @ diag(S) @ U.T
        for i in 0..n {
            for j in 0..n {
                let mut reconstructed = 0.0;
                for k in 0..n {
                    reconstructed +=
                        eigenvectors[i * n + k] * eigenvalues[k] * eigenvectors[j * n + k];
                }
                let original = k_flat[i * n + j];
                assert!(
                    (reconstructed - original).abs() < 1e-10,
                    "reconstruction mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    reconstructed,
                    original
                );
            }
        }
    }
}
