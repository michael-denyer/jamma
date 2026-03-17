# jlinalg

jlinalg is JAMMA's controlled C compute layer. It provides the specific BLAS
and LAPACK operations JAMMA needs -- dgemm, dsyrk, dsyr2k, eigh, qr, svd --
with vendor dispatch when ILP64 system BLAS is available, falling back to its
own C implementations with ISA-specific microkernels (AVX2/NEON). This avoids
numpy BLAS pitfalls: LP64 integer overflow at >46k samples, scipy ILP64
incompatibility, and inconsistent BLAS backends across platforms.

## Public API

All functions accept and return `numpy.ndarray` (float64, C-contiguous).

### Level 1 BLAS

| Function | Signature | Description |
|----------|-----------|-------------|
| `ddot` | `(x, y) -> float` | Inner product |
| `dnrm2` | `(x) -> float` | Euclidean norm |
| `daxpy` | `(alpha, x, y) -> None` | y += alpha * x (in-place) |
| `dscal` | `(alpha, x) -> None` | x *= alpha (in-place) |

### Level 2 BLAS

| Function | Signature | Description |
|----------|-----------|-------------|
| `dgemv` | `(A, x) -> ndarray` | Matrix-vector product A @ x |

### Level 3 BLAS

| Function | Signature | Description |
|----------|-----------|-------------|
| `dgemm` | `(A, B, transa='N', transb='N') -> ndarray` | Matrix multiply op(A) @ op(B) |
| `dsyrk` | `(X) -> ndarray` | Symmetric rank-k update X @ X.T |
| `dsyr2k` | `(C, A, B) -> ndarray` | Symmetric rank-2k update C - A @ B.T - B @ A.T |

### LAPACK

| Function | Signature | Description |
|----------|-----------|-------------|
| `eigh` | `(K) -> tuple[ndarray, ndarray]` | Eigenvalues and eigenvectors of symmetric K (K overwritten) |
| `qr` | `(A) -> tuple[ndarray, ndarray]` | Reduced QR factorization (Q, R) |
| `svd` | `(A, compute_uv=True) -> tuple \| ndarray` | SVD of tall-skinny matrix (m >= n required) |

### Introspection

| Name | Type | Description |
|------|------|-------------|
| `jlinalg_isa` | `str` | Active ISA: "AVX2", "NEON", "generic", or "numpy-fallback" |
| `blas_backend` | `str` | Active BLAS backend: "MKL-ILP64", "Accelerate-ILP64", "jlinalg-own", etc. |
| `blas_is_ilp64` | `int` | 1 if active BLAS uses 64-bit integers |
| `blas_has_dsyevd` | `int` | 1 if vendor DSYEVD available |
| `blas_has_dsyevr` | `int` | 1 if vendor DSYEVR available |
| `blas_has_dsyrk` | `int` | 1 if vendor DSYRK available |
| `blas_has_dgeqrf` | `int` | 1 if vendor QR (DGEQRF + DORGQR) available |
| `blas_has_dgesvd` | `int` | 1 if vendor SVD (DGESVD) available |
| `HAS_C_EXTENSION` | `bool` | True if the compiled C extension is loaded |
| `HAS_OPENMP` | `bool` | True if compiled with OpenMP |
| `ABI_VERSION` | `int` | C extension ABI version (0 = numpy fallback) |
| `JLINALG_MR` | `int` | Microkernel row tile size |
| `JLINALG_NR` | `int` | Microkernel column tile size |
| `JLINALG_KC` | `int` | Blocking depth (L1 target) |
| `JLINALG_MC` | `int` | Row panel size (L2 target) |
| `JLINALG_NC` | `int` | Column panel size (L3 target) |
| `get_n_threads()` | `-> int` | Current thread count |
| `set_n_threads(n)` | `-> int` | Set thread count, returns previous |

## Platform Support

| Platform | ISA | Vendor Dispatch | Own Kernels |
|----------|-----|-----------------|-------------|
| Linux x86_64 | AVX2 | MKL-ILP64, OpenBLAS-ILP64 | 6x8 FMA microkernel |
| macOS ARM (M1+) | NEON | Accelerate-ILP64 | 8x4 microkernel |
| macOS x86_64 | AVX2 | Accelerate | 6x8 FMA microkernel |
| Other / fallback | generic | numpy fallback | scalar C fallback |

Vendor dispatch is ILP64-only. LP64 backends are detected but not wired into
the dispatch table -- jlinalg's own dgemm is preferred for numerical
consistency with GEMMA validation tolerances (different FP accumulation order
in LP64 backends causes subtle result differences).

## Build

**Dev-mode compilation** (after `uv sync`):

```bash
uv run python -c "from jamma.jlinalg._compile_jlinalg import compile_extension; compile_extension()"
```

**Wheel builds:** `hatch_build.py` handles compilation automatically.

Both `_compile_jlinalg.py` and `hatch_build.py` must list every C source file.
LAPACK sources use strict IEEE 754 flags (`-O2 -fno-fast-math`).

## Further Reading

- [Architecture and Contributing Guide](../../docs/JLINALG_ARCHITECTURE.md) -- layer diagrams, file structure, microkernel tutorial, benchmarking guide
- [Algorithm Notes](../../docs/JLINALG_ALGORITHMS.md) -- cache blocking theory, D&C eigendecomposition, Golub-Kahan SVD
