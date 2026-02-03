# Rust Backend for JAMMA

JAMMA includes an optional Rust-based eigendecomposition backend that provides
CPU-optimized performance without requiring MKL or external BLAS libraries.

## When to Use

The Rust backend is automatically selected when:
- No GPU is available (JAX cannot detect CUDA/ROCm devices)
- You explicitly set `JAMMA_BACKEND=rust`

The JAX backend is preferred when a GPU is available, as GPU-accelerated
eigendecomposition is faster for large matrices.

## Building the Rust Extension

The Rust extension is built separately from the main JAMMA package using maturin.

### Prerequisites

- Rust toolchain (install via https://rustup.rs/)
- maturin (`pip install maturin`)

### Build Steps

```bash
# Navigate to the Rust crate directory
cd rust/jamma-core

# Build and install in development mode
maturin develop --release

# Or build a wheel for distribution
maturin build --release
```

### Verifying Installation

```python
from jamma.core import get_backend_info
print(get_backend_info())
# Should show: {'selected': 'rust', 'rust_available': True, ...}
```

Or via CLI:

```bash
jamma --version
# Shows: Backend: rust, Rust available: True
```

## Backend Selection

JAMMA automatically selects the best backend based on available hardware:

| Condition | Backend Selected |
|-----------|-----------------|
| GPU available | JAX |
| No GPU, Rust available | Rust |
| No GPU, Rust unavailable | JAX (scipy fallback) |

Override with environment variable:
```bash
export JAMMA_BACKEND=rust  # Force Rust backend
export JAMMA_BACKEND=jax   # Force JAX backend
```

## Performance Characteristics

- **Rust/faer**: Uses faer library with rayon parallelism. No external BLAS required.
  Good CPU performance without MKL configuration complexity.
- **JAX/scipy**: Uses system BLAS (OpenBLAS or MKL). Can be faster with MKL but
  requires proper configuration.

For most CPU-only workloads, the Rust backend provides comparable performance
with simpler setup.

## Numerical Parity

The Rust backend is validated to produce numerically identical results to scipy:
- Eigenvalues match within 1e-10 relative tolerance
- Full LMM workflow produces identical p-values
- GEMMA reference validation passes with both backends

Parity tests are in `tests/test_rust_parity.py`.

## Implementation Details

The Rust backend uses:
- **faer**: Pure Rust linear algebra library for eigendecomposition
- **rayon**: Parallelism framework for multi-threaded computation
- **PyO3**: Python bindings with GIL release during computation
- **ndarray**: NumPy-compatible array interface

Key implementation points:
- GIL is released during eigendecomposition (`py.allow_threads()`)
- Row-major (NumPy) to column-major (faer) conversion handled internally
- Eigenvalue threshold zeroing matches scipy path (GEMMA compatibility)

## Troubleshooting

### Rust backend not detected

```python
from jamma.core import is_rust_available
print(is_rust_available())  # Should be True
```

If False, rebuild the extension:
```bash
cd rust/jamma-core
maturin develop --release
```

### Performance issues

The Rust backend should be within 2-5x of scipy/OpenBLAS performance. If slower:
1. Ensure you built with `--release` flag
2. Check rayon thread count: `RAYON_NUM_THREADS=N` to limit parallelism
3. For very large matrices (>50k samples), scipy with MKL may be faster

### Backend override not working

The backend is cached at first use. Clear the cache:
```python
from jamma.core.backend import get_compute_backend
get_compute_backend.cache_clear()
```

Or set the environment variable before importing jamma.
