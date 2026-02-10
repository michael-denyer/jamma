# Performance Summary

## v1.4 — Memory Optimization and Scale Validation

v1.4 targeted memory optimization and correctness at production scale (85k+ real samples). The primary achievement is **validated GEMMA equivalence at 85,000 samples on 91,613 real SNPs** with 100% agreement on significance calls, effect directions, and SNP rankings.

### Changes Applied

| Change | Impact |
|--------|--------|
| Phase-specific LMM memory estimates | Fixed false MemoryError at 100k samples (was demanding 320GB pipeline peak when only 96GB needed) |
| JAX async dispatch: `block_until_ready()` | Progress bars and timing now reflect actual compute, not async dispatch time |
| Progress bar lifecycle fix | Bars complete cleanly (no hanging on final iteration) |
| Vectorized per-SNP imputation | Streaming runner imputation ~2x faster |
| Top-level `gwas()` API | Single-call Python entry point for full GWAS pipeline |
| GEMMA comparison notebook | Compare-only mode with OOM-safe kinship comparison at 85k scale |

### 90k Synthetic Baseline (Phase 19, Databricks)

Measured on 32-core Databricks VM with MKL ILP64, 90k synthetic samples x 90k SNPs:

| Phase | Time | % of Total |
|-------|------|-----------|
| Kinship | 1,440s (24 min) | 25% |
| Eigendecomp | 3,114s (52 min) | 54% |
| LMM Association | 1,211s (20 min) | 21% |
| **Total** | **5,764s (96 min)** | **100%** |

### 85k Real Data Validation (v1.4.3, Databricks)

JAMMA vs GEMMA on 85,000 real samples, 91,613 SNPs:

| Metric | Result |
|--------|--------|
| **Kinship Spearman rho** | 1.00000000 |
| Kinship max abs diff | 1.09e-05 |
| Kinship mean abs diff | 1.17e-07 |
| Kinship Frobenius relative | 1.52e-05 |
| **Association Spearman rho (-log10 p)** | 1.000000 |
| Significance agree (p < 0.05) | 91,613/91,613 (100%) |
| Significance agree (p < 5e-8) | 91,613/91,613 (100%) |
| Effect direction agreement | 100.0% |
| Max relative p-value diff | 2.10e-03 |

### Bottleneck Breakdown

All three pipeline phases are dominated by BLAS/LAPACK calls. No Python-level optimization can improve these:

| Phase | Bottleneck | Notes |
|-------|-----------|-------|
| Eigendecomp (54%) | LAPACK dsyevd — O(n³) | Single call, irreducible. 90k at 32 cores ≈ 3,100s, matching theoretical floor (~9.7e14 FLOPs / ~310 GFLOPS effective) |
| Kinship (25%) | JAX-batched dgemm | Already JIT-compiled matrix multiply |
| LMM Association (21%) | JAX JIT + golden section per SNP | Rotation is a single dgemm per chunk |

### What v1.4 Did Not Change

- **Wall-clock time**: Eigendecomp, kinship, and LMM times are unchanged from Phase 19 baseline. JAMMA was already operating at the hardware-limited floor for CPU eigendecomposition.
- **Thread configuration**: MKL was already running at 32 threads on Databricks. The thread-pinning code (`_pin_blas_threads`) was a no-op because MKL loads during `import jax` before pinning runs. v1.4 formalized this into `blas_threads()` context managers but the runtime behavior is identical.

### Configuration Guide

| Scale | Samples | RAM Required | MKL Build |
|-------|---------|-------------|-----------|
| Small | ≤10k | 8 GB | Any |
| Medium | 10–50k | 64 GB | LP64 or ILP64 |
| Large | 50–100k | 256 GB | ILP64 required |
| XLarge | 100–200k | 512 GB+ | ILP64 required |

See [USER_GUIDE.md](USER_GUIDE.md) for installation instructions and [GEMMA_DIVERGENCES.md](GEMMA_DIVERGENCES.md) for documented divergences from GEMMA.

### Test Suite

448 tests passing (1 skipped for missing optional fixture). Zero tolerance changes from v1.3. Tolerance constants in `src/jamma/validation/tolerances.py` unchanged.

---
*Created: 2026-02-10*
