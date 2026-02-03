# JAMMA Compute Backends

JAMMA supports multiple compute backends for different hardware configurations and scale requirements.

## Backend Overview

| Backend | Eigendecomp | LMM Runtime | Status | Best For |
|---------|-------------|-------------|--------|----------|
| `jax.scipy` | scipy/LAPACK | JAX | Stable | GPU systems, small-medium scale |
| `jax.rust` | faer/Rust | JAX | Stable | CPU-only, 100k+ samples |
| `rust` | faer/Rust | Pure Rust | Planned | Future: memory-constrained systems |

## Backend Details

### jax.scipy (JAX pipeline + scipy eigendecomp)

The fallback backend when `jamma_core` is not installed. Uses:

- **Eigendecomposition**: scipy.linalg.eigh (LAPACK)
- **LMM computation**: JAX with XLA compilation

**Best for:**

- Systems with GPU acceleration
- Small to medium scale (< 50k samples)
- When scipy/OpenBLAS/MKL is well-configured

**Potential issues:**

- OpenBLAS can crash at 100k+ samples due to threading bugs
- Requires thread limiting workarounds for large matrices

### jax.rust (JAX pipeline + Rust eigendecomp)

The **recommended backend** for production use. Uses:

- **Eigendecomposition**: faer (pure Rust via jamma_core)
- **LMM computation**: JAX with XLA compilation

**Best for:**

- CPU-only workloads
- Large scale (100k+ samples)
- Avoiding OpenBLAS threading issues

**Requirements:**

- `jamma_core` extension (built separately with maturin)

**Why jax.rust over jax.scipy?**

OpenBLAS has known threading bugs that cause SIGSEGV crashes during eigendecomposition at 100k+ samples. Previous workarounds included:

- Installing MKL (complex, licensing concerns)
- Thread limiting (works but slower)

The `jax.rust` backend eliminates these issues entirely with faer's pure Rust implementation.

### rust (Pure Rust) - Not Yet Implemented

A future backend that will implement the entire LMM pipeline in Rust:

- **Eigendecomposition**: faer
- **LMM computation**: Pure Rust (no JAX dependency)

**This will enable:**

- Minimal dependencies
- Embedded/constrained environments
- Potentially better memory efficiency

**Status**: Stub only. Selecting this backend will error with instructions to use `jax.scipy` or `jax.rust`.

## Backend Selection

### Automatic Selection

By default, JAMMA selects the best available backend:

1. If `jamma_core` is available: `jax.rust`
2. Otherwise: `jax.scipy` (with warning about missing jamma_core)

**Note**: GPU presence does not affect automatic selection. The `jax.rust` backend is preferred for its stability at scale.

### Manual Selection

#### CLI Flag (-be)

```bash
# Use scipy eigendecomp
jamma lmm -bfile data -k kinship.txt -be jax.scipy

# Use Rust eigendecomp (requires jamma_core)
jamma lmm -bfile data -k kinship.txt -be jax.rust

# Auto-select (default)
jamma lmm -bfile data -k kinship.txt -be auto
```

#### Environment Variable

```bash
# Set for all commands
export JAMMA_BACKEND=jax.rust

# Or per-command
JAMMA_BACKEND=jax.scipy jamma lmm -bfile data -k kinship.txt
```

**Precedence**: CLI `-be` flag overrides `JAMMA_BACKEND` environment variable.

### Checking Current Backend

Via CLI:

```bash
jamma --version
# Output:
# JAMMA version X.Y.Z
# Backend: jax.rust
#   (JAX pipeline + faer/Rust eigendecomp)
# jax.rust available: True
# GPU available: False
```

Via Python:

```python
from jamma.core import get_backend_info

print(get_backend_info())
# {'selected': 'jax.rust', 'rust_available': True, 'jax_rust_available': True, 'gpu_available': False, 'override': None}
```

## Building jamma_core

The `jax.rust` backend requires the `jamma_core` Rust extension.

### Prerequisites

- Rust toolchain: https://rustup.rs/
- maturin: `pip install maturin`

### Build Steps

```bash
# Navigate to the Rust crate directory
cd rust/jamma-core

# Build and install in development mode
maturin develop --release

# Or build a wheel for distribution
maturin build --release
```

### Verification

```python
from jamma.core import is_rust_available

print(is_rust_available())  # Should be True
```

Or via CLI:

```bash
jamma --version
# Shows: jax.rust available: True
```

## Numerical Parity

Both `jax.scipy` and `jax.rust` backends produce identical numerical results:

- Eigenvalues match within 1e-10 relative tolerance
- Full LMM workflow produces identical p-values
- GEMMA reference validation passes with either backend

Parity is verified in `tests/test_rust_parity.py`.

## Implementation Details

The Rust backend (`jamma_core`) uses:

- **faer**: Pure Rust linear algebra library for eigendecomposition
- **rayon**: Parallelism framework for multi-threaded computation
- **PyO3**: Python bindings with GIL release during computation
- **ndarray**: NumPy-compatible array interface

Key implementation points:

- GIL is released during eigendecomposition (`py.allow_threads()`)
- Row-major (NumPy) to column-major (faer) conversion handled internally
- Eigenvalue threshold zeroing matches scipy path (GEMMA compatibility)

## Troubleshooting

### "Backend 'jax.rust' requires jamma_core"

The Rust extension is not installed. Build it:

```bash
cd rust/jamma-core && maturin develop --release
```

### "Backend 'rust' not implemented"

You requested pure Rust LMM, which is not yet available. Use:

- `jax.scipy` for scipy eigendecomp with JAX LMM
- `jax.rust` for Rust eigendecomp with JAX LMM

### Deprecation Warnings

Old backend names are deprecated:

- `JAMMA_BACKEND=jax` is deprecated. Use `jax.scipy` instead.

### Performance Issues

**For `jax.rust`:**

1. Ensure built with `--release` flag
2. Check rayon thread count: `RAYON_NUM_THREADS=N` to limit parallelism
3. For very large matrices (>50k samples), scipy with MKL may be faster

**For `jax.scipy`:**

1. Install MKL for best performance: `pip install mkl`
2. For 100k+ samples, consider `jax.rust` to avoid crashes

### Backend Override Not Working

The backend is cached at first use. Clear the cache:

```python
from jamma.core.backend import get_compute_backend

get_compute_backend.cache_clear()
```

Or set the environment variable before importing jamma.

## Backend Comparison

| Aspect | jax.scipy | jax.rust |
|--------|-----------|----------|
| Dependencies | scipy, BLAS | jamma_core (Rust) |
| 100k+ samples | May crash (OpenBLAS bugs) | Stable |
| GPU acceleration | Full support | Full support (JAX LMM) |
| Setup complexity | Needs BLAS config | Build once with maturin |
| Performance | Faster with MKL | Competitive, more stable |
