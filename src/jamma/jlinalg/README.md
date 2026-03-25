# jlinalg

jlinalg is JAMMA's controlled C compute layer. It dispatches to vendor
ILP64 BLAS/LAPACK (MKL, Accelerate, OpenBLAS) for the operations JAMMA
needs -- dgemm, dsyrk, eigh, qr, svd -- falling back to NumPy when no
ILP64 vendor is available. This avoids numpy BLAS pitfalls: LP64 integer
overflow at >46k samples, scipy ILP64 incompatibility, and inconsistent
BLAS backends across platforms.

Level 1/2 BLAS primitives (ddot, dnrm2, daxpy, dscal, dgemv) and dsyr2k
are pure NumPy implementations -- they were removed from the C extension
in ABI version 12.

## Public API

All functions accept and return `numpy.ndarray` (float64, C-contiguous).

### Level 1/2 BLAS (NumPy-only)

| Function | Signature | Description |
|----------|-----------|-------------|
| `ddot` | `(x, y) -> float` | Inner product |
| `dnrm2` | `(x) -> float` | Euclidean norm |
| `daxpy` | `(alpha, x, y) -> None` | y += alpha * x (in-place) |
| `dscal` | `(alpha, x) -> None` | x *= alpha (in-place) |
| `dgemv` | `(A, x) -> ndarray` | Matrix-vector product A @ x |
| `dsyr2k` | `(C, A, B) -> ndarray` | Symmetric rank-2k update C - A @ B.T - B @ A.T |

### Level 3 BLAS (vendor dispatch)

| Function | Signature | Description |
|----------|-----------|-------------|
| `dgemm` | `(A, B, transa='N', transb='N') -> ndarray` | Matrix multiply op(A) @ op(B) |
| `dsyrk` | `(X) -> ndarray` | Symmetric rank-k update X @ X.T |

### LAPACK (vendor dispatch)

| Function | Signature | Description |
|----------|-----------|-------------|
| `eigh` | `(K, inplace=False) -> tuple[ndarray, ndarray]` | Eigenvalues and eigenvectors of symmetric K |
| `qr` | `(A) -> tuple[ndarray, ndarray]` | Reduced QR factorization (Q, R) |
| `svd` | `(A, compute_uv=True) -> tuple \| ndarray` | SVD of tall-skinny matrix (m >= n required) |

### Introspection

| Name | Type | Description |
|------|------|-------------|
| `jlinalg_isa` | `str` | Active ISA: "AVX2", "NEON", "generic", or "numpy-fallback" |
| `blas_backend` | `str` | Active BLAS backend: "MKL-ILP64", "Accelerate-ILP64", "numpy-fallback", etc. |
| `blas_is_ilp64` | `int` | 1 if active BLAS uses 64-bit integers |
| `blas_has_dsyevd` | `int` | 1 if vendor DSYEVD available |
| `blas_has_dsyevr` | `int` | 1 if vendor DSYEVR available |
| `blas_has_dsyrk` | `int` | 1 if vendor DSYRK available |
| `blas_has_dgeqrf` | `int` | 1 if vendor QR (DGEQRF + DORGQR) available |
| `blas_has_dgesvd` | `int` | 1 if vendor SVD (DGESVD) available |
| `HAS_C_EXTENSION` | `bool` | True if the compiled C extension is loaded |
| `HAS_OPENMP` | `bool` | True if compiled with OpenMP |
| `ABI_VERSION` | `int` | C extension ABI version (0 = numpy fallback) |
| `get_n_threads()` | `-> int` | Current thread count |
| `set_n_threads(n)` | `-> int` | Set thread count, returns previous |

## Platform Support

| Platform | ISA | Vendor Dispatch |
|----------|-----|-----------------|
| Linux x86_64 | AVX2 | MKL-ILP64, OpenBLAS-ILP64 |
| macOS ARM (M1+) | NEON | Accelerate-ILP64 |
| macOS x86_64 | AVX2 | Accelerate |
| Other / fallback | generic | NumPy fallback |

Vendor dispatch is ILP64-only. LP64 backends are detected but not wired
into the dispatch table -- NumPy fallback is used instead for numerical
consistency with GEMMA validation tolerances (different FP accumulation
order in LP64 backends causes subtle result differences).

## Build

**Dev-mode compilation** (after `uv sync`):

```bash
uv run python -c "from jamma.jlinalg._compile_jlinalg import compile_extension; compile_extension()"
```

**Wheel builds:** `hatch_build.py` handles compilation automatically.

Both `_compile_jlinalg.py` and `hatch_build.py` must list every C source file.
LAPACK sources use strict IEEE 754 flags (`-O2 -fno-fast-math`).

## Further Reading

- [Architecture and Contributing Guide](../../docs/JLINALG_ARCHITECTURE.md) -- layer diagrams, file structure, benchmarking guide
- [Algorithm Notes](../../docs/JLINALG_ALGORITHMS.md) -- vendor DSYEVD algorithm notes, Golub-Kahan SVD
