# Performance Summary

## v4.2.0 — 125k Scale (Latest)

v4.2.0 at 125,632 samples on 91,586 real SNPs. **~10x faster than GEMMA** (2h 29m vs ~27h). 19% faster than v2.10.1 thanks to jlinalg eigendecomp and C extension LMM improvements. Eigendecomp used DSYEVR (memory-constrained fallback from DSYEVD).

**Note**: GEMMA was compiled with default OpenBLAS, not MKL. Building GEMMA against ILP64 MKL is non-trivial (requires Makefile patches and ILP64 linking for matrices >46k). The comparison reflects typical deployment: GEMMA as-distributed vs JAMMA with ILP64 numpy-mkl.

### 125k Real Data Benchmark (v4.2.0, Databricks)

Hardware: Azure E96ds_v6 (Intel Xeon Platinum 8573C, 48 physical / 96 logical cores, 672 GB RAM). numpy 2.4.2 with MKL ILP64, Python 3.12, Databricks Runtime 16.4 LTS.

| Phase | Time | % of Total |
|-------|------|-----------|
| Kinship compute | 1,591s (27 min) | 18% |
| Eigendecomp (DSYEVR) | 6,427s (1h 47m) | 72% |
| LMM (C ext) | 887s (15 min) | 10% |
| **Total** | **~8,942s (2h 29m)** | **100%** |

Throughput: 12.5 SNPs/sec (eigen+LMM), 10.2 SNPs/sec end-to-end. Peak RSS: 380.6 GB (after eigendecomp), 320.6 GB (LMM phase).

### 125k Validation: JAMMA vs GEMMA (v4.2.0)

| Metric | Result |
|--------|--------|
| **Kinship Spearman rho** | 1.00000000 |
| Kinship max abs diff | 5.00e-11 |
| Kinship mean abs diff | 1.24e-12 |
| Kinship max relative diff | 3.50e-06 |
| Kinship Frobenius relative | 1.45e-10 |
| **Association Spearman rho (-log10 p)** | 1.000000 |
| Significance agree (p < 0.05) | 91,586/91,586 (100%) |
| Significance agree (p < 5e-8) | 91,586/91,586 (100%) |
| Effect direction agreement | 100.0% |
| Max relative p-value diff | 9.14e-04 |

### Progression: 125k benchmarks across versions

All runs on the same hardware (E96ds_v6) and dataset (125,632 × 91,586).

| Phase | v4.2.0 (latest) | v2.10.1 | v2.5.6 (1 dev) | v4.2 vs v2.10 |
|-------|----------------|---------|----------------|---------------|
| Kinship compute | 1,591s | 2,047s | 2,068s | **-22%** |
| Eigendecomp | 6,427s | — | — | — |
| LMM (C ext) | 887s | — | — | — |
| Eigen+LMM | 7,314s | 8,437s | 9,365s | **-13%** |
| **Pipeline total** | **8,942s** | **11,040s** | **12,008s** | **-19%** |

v4.2.0 is 2,098s (35 min) faster than v2.10.1. Kinship is 22% faster (jlinalg DGEMM improvements). Eigen+LMM is 13% faster despite using DSYEVR (memory-constrained fallback) instead of DSYEVD. LMM alone dropped from ~2,000s (JAX) to 887s (C ext with 48 OpenMP threads).

### Scaling from 90k to 125k

| Phase | v2.3 (90k, 32 cores) | v4.2.0 (125k, 48 cores) |
|-------|------|-----------|
| Kinship compute | 1,440s (24 min) | 1,591s (27 min) |
| Eigendecomp | 3,114s (52 min) | 6,427s (1h 47m) |
| LMM | 1,211s (20 min) | 887s (15 min) |
| **Total** | **5,764s (96 min)** | **8,942s (2h 29m)** |

Eigendecomp dominates the increase: O(n³) scaling from 90k→125k is ~2.1× (DSYEVR). LMM actually got faster at 125k than 90k was at v2.3 thanks to the C extension replacing JAX. The 126 GB eigenvector matrices exceed L3 cache, making eigendecomp memory-bandwidth bound.

### Full Pipeline Scaling (v4.6.1, 95k SNPs, 48 cores)

| Phase | 5k×95k | 20k×95k | 50k×95k | 75k×95k | 125k×92k (real) |
|-------|--------|---------|---------|---------|-----------------|
| Kinship compute | 10s | 67s (1 min) | 284s (5 min) | 500s (8 min) | 1,591s (27 min)† |
| Eigendecomp | 1s | 44s | 516s (9 min) | 1,478s (25 min) | 6,427s (1h 47m)†‡ |
| LMM (C ext) | 8s | 42s | 182s (3 min) | 362s (6 min) | 887s (15 min)† |
| **Total (C ext)** | **19s** | **155s (3 min)** | **988s (16 min)** | **2,353s (39 min)** | **8,942s (2h 29m)** |

†125k numbers from v4.2.0 (same hardware, not re-benchmarked). ‡125k used DSYEVR (memory-constrained fallback); all others used DSYEVD. Eigendecomp scales O(n³): 516s at 50k → 1,478s at 75k (2.9× for 1.5× samples). LMM scales roughly O(n²) due to rotation dominance. v4.6.1 LMM is 14–16% faster than v4.2.0 at 50k–75k thanks to centralized jlinalg thread control and pthreads-based SNP stats.

---

## C Extension LMM Acceleration (NumPy Backend)

The NumPy backend includes an optional C extension (`_lmm_accel.c`) with OpenMP parallelism
that replaces the Python loop over SNPs for Wald test computation. The extension uses a
workspace API (pre-allocated per-thread buffers). The primary path (fused kernel) takes
utg_t in (n_snps, n_samples) layout directly from DGEMM TRANSA, computing wx/xx/xy
on-the-fly without a separate SoA Uab buffer. The SoA Uab layout with invariant precompute
is retained as a fallback when the fused C extension is unavailable. Mean imputation of
missing genotypes is done in-place on the chunk buffer (no copy), so the per-chunk memory
footprint equals the rotation output buffer only.

### C Extension vs JAX Backend (E96ds_v6, 48 cores, synthetic data)

**5k SNPs (LMM only, pre-computed eigen):**

| Scale    | C ext LMM | JAX LMM | C ext Speedup |
|----------|-----------|---------|---------------|
| 5k×5k    | 1.1s      | 2.6s    | **2.3x**      |
| 20k×5k   | 4.1s      | 7.5s    | **1.8x**      |
| 50k×5k   | 14.6s     | 20.9s   | **1.4x**      |
| 75k×5k   | 27.1s     | 40.6s   | **1.5x**      |

**95k SNPs (LMM phase only):**

| Scale    | C ext LMM | JAX LMM | C ext Speedup |
|----------|-----------|---------|---------------|
| 5k×95k   | 11.5s     | 31.8s   | **2.8x**      |
| 20k×95k  | 56.6s     | 145.0s  | **2.6x**      |
| 50k×95k  | 216.8s    | 502.0s  | **2.3x**      |
| 75k×95k  | 418.3s    | 693.0s  | **1.7x**      |

The C extension wins at all scales. v4.2.0 widened the gap significantly vs prior versions
(2.3–2.8× at small/medium scales vs ~1.5× previously) thanks to jlinalg and C extension
improvements. The gap narrows at 75k where UT@G rotation (identical DGEMM in both backends)
becomes a larger fraction of total time.

### C Extension Scaling (LMM timing breakdown, 95k SNPs)

| Scale    | UT@G Rotation | Compute  | LMM Total | RSS      |
|----------|---------------|----------|-----------|----------|
| 5k×95k   | 2.8s          | 3.6s     | 11.5s     | 6.6 GB   |
| 20k×95k  | 22.5s         | 15.2s    | 56.6s     | 27.0 GB  |
| 50k×95k  | 125.8s        | 44.8s    | 216.8s    | 60.3 GB  |
| 75k×95k  | 285.1s        | 61.3s    | 418.3s    | 95.1 GB  |
| 125k×92k | 652.0s        | 93.1s    | 882.1s    | 320.6 GB |

### C Extension Scaling (LMM timing breakdown, 5k SNPs)

| Scale    | UT@G Rotation | Compute  | LMM Total | RSS      |
|----------|---------------|----------|-----------|----------|
| 5k×5k    | 0.18s         | 0.51s    | 1.1s      | 1.0 GB   |
| 20k×5k   | 1.17s         | 1.71s    | 4.0s      | 5.7 GB   |
| 50k×5k   | 6.23s         | 5.23s    | 14.4s     | 24.6 GB  |
| 75k×5k   | 13.59s        | 8.81s    | 26.9s     | 51.4 GB  |

Compute scales O(n_samples). Rotation scales O(n² × n_snps) and dominates at 20k+.
Both use MKL DGEMM with 48 threads.

---

## v2.0 — Production GWAS Features

v2.0 added LOCO kinship, eigendecomposition reuse, SNP filtering, HWE QC, and phenotype selection. No performance regressions from v1.4.

### New Features Performance Characteristics

| Feature | Scaling | Notes |
|---------|---------|-------|
| LOCO kinship | O(n_chr × n²) | Linear in chromosomes; one eigendecomp per chromosome |
| Eigendecomp reuse (`-d`/`-u`) | Eliminates O(n³) | Skips eigendecomp entirely on subsequent phenotypes |
| SNP filtering (`-snps`/`-ksnps`) | O(log n) per chunk | Searchsorted-based chunk filtering |
| HWE filtering (`-hwe`) | O(1) per SNP | Piggybacks on pass-1 streaming (no extra disk pass) |

### LOCO Scaling

LOCO analysis processes each chromosome independently: compute K_loco via streaming subtraction, eigendecompose, run LMM. Total time scales linearly with the number of chromosomes (typically 22 for human data). Each per-chromosome eigendecomp is the same O(n³) as a standard run, so total LOCO wall time is approximately `n_chr × single_eigendecomp_time + LMM_time`.

### Test Suite

Default test suite passing (`uv run pytest tests/ -x`), which excludes `slow` and `tier2` markers per `pyproject.toml` defaults. Run `uv run pytest tests/ -x -m ""` for the full suite including slow/tier2 tests. Tolerance constants in `src/jamma/validation/tolerances.py` unchanged from v1.3 (kinship tolerance aligned to 1e-8 in v2.5.7).

---

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

### 90k Baseline (v2.3, Databricks)

Measured on 32-core Databricks VM with MKL ILP64, 90k synthetic samples × 90k SNPs.

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
| Eigendecomp (54%) | LAPACK dsyevd — O(n³) | Single call, irreducible. 90k at 32 cores ≈ 3,100s |
| Kinship (25%) | jlinalg DGEMM (chunked) | Multi-threaded BLAS matrix multiply |
| LMM Association (21%) | C extension + OpenMP (golden section per SNP) | Rotation is a single dgemm per chunk (utg_t, DGEMM TRANSA) |

### What v1.4 Did Not Change

- **Wall-clock time**: Eigendecomp, kinship, and LMM times are unchanged from Phase 19 baseline. JAMMA was already operating at the hardware-limited floor for CPU eigendecomposition.
- **Thread configuration**: MKL was already running at 32 threads on Databricks. The thread-pinning code (`_pin_blas_threads`) was a no-op because MKL loads during `import jax` before pinning runs. v1.4 formalized this into `blas_threads()` context managers but the runtime behavior is identical.

### Configuration Guide

| Scale | Samples | RAM Required | MKL Build | Reference |
|-------|---------|-------------|-----------|-----------|
| Small | ≤10k | 8 GB | Any | |
| Medium | 10–50k | 64 GB | LP64 or ILP64 | |
| Large | 50–100k | 256 GB | ILP64 required | 85k validated (v1.4) |
| XLarge | 100–125k | 768 GB | ILP64 required | 125k validated (v4.2.0), peak ~560 GB |

RAM requirements are for the full pipeline (kinship + eigendecomp + LMM). Eigendecomp is the memory peak: K matrix (n²×8 bytes) + eigenvectors (n²×8 bytes) must coexist. At 125k this is ~252 GB + 252 GB = 504 GB; the process peaked at 381 GB RSS with 768 GB physical. Scaling beyond 125k on 768 GB is not feasible — at 150k the eigendecomp alone would require ~720 GB (DSYEVR), leaving nothing for LMM.

Note: with early sample filtering (phenotype missingness), `n` in these formulae is
the number of valid samples (non-missing phenotype), which may be smaller than the
BED file sample count.

### CPU Device Sharding (JAX Backend Only)

The JAX backend uses JAX `NamedSharding` to partition SNP batches across virtual CPU
devices, parallelising the per-SNP REML grid search and golden-section
refinement. The default production backend is NumPy+C (auto-selected);
use `--backend jax` to enable JAX sharding.

**Current optimisation target: Intel x86_64 Linux** (Databricks / HPC).
The heuristics are calibrated for MKL-backed numpy on Intel Xeon hardware.
Other platforms (ARM, Apple Silicon, AMD) work correctly but may benefit
from manual tuning via `JAMMA_JAX_DEVICES` and `JAMMA_BLAS_THREADS`.
GPU acceleration (`use_gpu=True`) is functional but not yet tuned for
production workloads.

**Auto-configuration:**

| Setting | Default | Override |
| ------- | ------- | -------- |
| JAX devices | `max(1, physical_cores // 2)` | `JAMMA_JAX_DEVICES` |
| BLAS threads | `physical_cores // n_devices` | `JAMMA_BLAS_THREADS` |

#### Benchmark: JAX optimisation time

Hardware: Azure E96ds_v6 (Intel Xeon Platinum 8573C, 48 physical /
96 logical cores, 672 GB RAM). numpy 2.4.2 with MKL ILP64, JAX 0.9.0,
Python 3.12, Databricks Runtime 16.4 LTS with Docker (jamma-dbr image).

| Devices | Mouse (1.4K×11K) | 5K×50K     | 10K×100K    | 20K×100K    |
| ------- | ---------------- | ---------- | ----------- | ----------- |
| 1       | 3.38s            | 54.4s      | 65.4s       | 93.7s       |
| 8       | 0.84s            | 12.0s      | 34.5s       | 67.7s       |
| 16      | **0.79s**        | **8.0s**   | **28.8s**   | **40.6s**   |
| 32      | 0.79s            | 8.3s       | 28.7s       | 55.0s       |

Peak speedup is generally at `physical_cores // 2`. Beyond that, XLA
cross-device coordination overhead typically outweighs the per-device
compute savings.

See [USER_GUIDE.md](USER_GUIDE.md) for installation instructions and [GEMMA_DIVERGENCES.md](GEMMA_DIVERGENCES.md) for documented divergences from GEMMA.

### Test Suite

Full test suite passing. Kinship tolerance aligned from 1e-10 to 1e-8 in v2.5.7 to match EQUIVALENCE.md bounds. All other tolerance constants in `src/jamma/validation/tolerances.py` unchanged from v1.3.

---
*Last updated: 2026-03-20*
