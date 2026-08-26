# jlinalg

jlinalg is JAMMA's controlled C compute layer. It dispatches to vendor
ILP64 BLAS/LAPACK (MKL, Accelerate, OpenBLAS) for the operations JAMMA
needs -- dgemm, dsyrk, eigh -- falling back to NumPy when no
ILP64 vendor is available. This avoids numpy BLAS pitfalls: LP64 integer
overflow at >46k samples, scipy ILP64 incompatibility, and inconsistent
BLAS backends across platforms.

## Public API

All functions accept and return `numpy.ndarray` (float64, C-contiguous).

### Level 3 BLAS (vendor dispatch)

| Function | Signature | Description |
|----------|-----------|-------------|
| `dgemm` | `(A, B, transa='N', transb='N') -> ndarray` | Matrix multiply op(A) @ op(B) |
| `dsyrk` | `(X, *, out=None, beta=0.0) -> ndarray` | Symmetric rank-k update X @ X.T + beta*out |
| `dsyrk_scratch_bytes` | `(n) -> int` | Upper bound on what one `dsyrk` call holds beyond its n-by-n output. Zero on the native backend; the NumPy fallback needs a block, which a memory pre-flight has to budget for |

### Kernels

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_snp_stats_chunk` | `(data, means, miss_counts, variances, n_aa=None, n_ab=None, n_bb=None) -> None` | Single-pass per-SNP mean, population variance, and missing count into pre-allocated outputs. The three optional arrays collect genotype counts for HWE testing |

### LAPACK (vendor dispatch)

| Function | Signature | Description |
|----------|-----------|-------------|
| `eigh` | `(K, inplace=False) -> tuple[ndarray, ndarray]` | Eigenvalues and eigenvectors of symmetric K |

### Introspection

| Name | Type | Description |
|------|------|-------------|
| `jlinalg_isa` | `str` | Active ISA: "AVX2", "NEON", "generic", or "numpy-fallback" |
| `blas_backend` | `str` | Active BLAS backend: "MKL-ILP64", "Accelerate-ILP64", "numpy-fallback", etc. |
| `blas_is_ilp64` | `int` | 1 if active BLAS uses 64-bit integers |
| `blas_has_dgemm` | `int` | 1 if vendor DGEMM is wired; when 0, `dgemm` is the NumPy implementation |
| `blas_has_dsyevd` | `int` | 1 if vendor DSYEVD available |
| `blas_has_lapacke_dsyevd` | `int` | 1 if the vendor exposes DSYEVD through the LAPACKE row-major interface |
| `blas_has_dsyevr` | `int` | 1 if vendor DSYEVR available |
| `blas_has_dsyrk` | `int` | 1 if vendor DSYRK available |
| `HAS_C_EXTENSION` | `bool` | True if the compiled C extension is loaded |
| `HAS_OPENMP` | `bool` | True if compiled with OpenMP |
| `ABI_VERSION` | `int` | C extension ABI version (0 = numpy fallback) |
| `get_n_threads()` | `-> int` | Current thread count |
| `set_n_threads(n)` | `-> int` | Set thread count, returns previous |

## Platform Support

| Platform | ISA | Vendor Dispatch |
|----------|-----|-----------------|
| Linux x86_64 | AVX2 | MKL-ILP64, OpenBLAS-ILP64 |
| macOS ARM (M1+) | NEON | Accelerate-ILP64 (macOS 13.3+) |
| macOS x86_64 | AVX2 | NumPy fallback (LP64 Accelerate not wired) |
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

C sources, compile flags, and link flags are owned by
`src/jamma/_build_support/compile_and_link.py` (`BASELINE_SOURCES`,
`LAPACK_SOURCES`, `BASE_CFLAGS`, `LAPACK_CFLAGS`). Three compile entry
points (`hatch_build.py`, `_compile_jlinalg.py`,
`src/jamma/lmm/_compile_accel.py`) all import from `_build_support` so
they stay in sync. Bare flag literals outside `_build_support/` fail
the `check_compile_flag_literals.py` pre-commit hook. LAPACK sources
(`eigh.c`) use strict IEEE 754 flags (`-O2 -fno-fast-math`); other
sources use `BASE_CFLAGS` (`-O3 -ftree-vectorize ...`).

## Debugging

Set `JAMMA_FORCE_NUMPY_FALLBACK=1` to force the entire jlinalg layer
onto the NumPy fallback even when vendor BLAS is loaded -- useful for
isolating numerical differences between vendor LAPACK and NumPy, and
required by the weekly sanitizer workflow. The narrower
`JLINALG_NO_VENDOR_LAPACK` only affects eigendecomposition (in
`lmm/eigen.py`), not the BLAS primitives. `JLINALG_NO_VENDOR_DGEMM=1`
is narrower still: dispatch leaves vendor dgemm unwired, so
`blas_has_dgemm` reports 0 with the extension loaded -- the state an
LP64-only host is permanently in.

## Further Reading

- [Architecture and Contributing Guide](../../../docs/JLINALG_ARCHITECTURE.md) -- layer diagrams, file structure, benchmarking guide
- [Algorithm Notes](../../../docs/JLINALG_ALGORITHMS.md) -- vendor DSYEVD algorithm notes
