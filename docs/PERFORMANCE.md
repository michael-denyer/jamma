# Performance Summary

## v2.5 — 125k Scale Validation

v2.5 validated JAMMA at 125,632 samples on 91,586 real SNPs with full GEMMA equivalence. Key fixes: eigendecomp threading (v2.5.4), matrix writer disk space (v2.5.2), LMM rotation threading (v2.5.6).

### 125k Real Data Benchmark (v2.5.5, Databricks)

Hardware: Azure E96ds_v6 (Intel Xeon Platinum 8573C, 48 physical / 96 logical cores, 672 GB RAM). numpy 2.4.2 with MKL ILP64, JAX 0.9.0, Python 3.12, Databricks Runtime 16.4 LTS.

| Phase | Time | % of Total |
|-------|------|-----------|
| SNP statistics | 103s (2 min) | 1% |
| Kinship compute | 2,011s (34 min) | 16% |
| Kinship write | 547s (9 min) | 4% |
| Eigendecomp | 8,465s (2h 21m) | 69% |
| LMM association | 1,131s (19 min) | 9% |
| **Total** | **~12,257s (3h 24m)** | **100%** |

LMM timing breakdown (from per-phase accumulators):

| LMM Sub-phase | Time | Notes |
|---------------|------|-------|
| U.T @ G rotation | 797s | 48 threads, 41 chunks × ~19s/chunk |
| JAX compute | 329s | Grid REML + golden section per SNP |
| Result write | 5s | Pre-allocated numpy arrays |

### 125k Validation: JAMMA vs GEMMA

| Metric | Result |
|--------|--------|
| **Kinship Spearman rho** | 1.00000000 |
| Kinship max abs diff | 5.00e-11 |
| Kinship mean abs diff | 1.24e-12 |
| Kinship Frobenius relative | 1.45e-10 |
| **Association Spearman rho (-log10 p)** | 1.000000 |
| Significance agree (p < 0.05) | 91,586/91,586 (100%) |
| Significance agree (p < 5e-8) | 91,586/91,586 (100%) |
| Effect direction agreement | 100.0% |
| Max relative p-value diff | 9.66e-04 |

### Scaling from 90k to 125k

Comparison against the v2.3 baseline (90k samples, 32-core Databricks VM). Hardware and sample counts differ, so this is not a like-for-like comparison — it shows how wall time scales with both sample size and hardware.

| Phase | v2.3 (90k, 32 cores) | v2.5 (125k, 48 cores) |
|-------|------|-----------|
| Kinship | 1,440s (24 min) | 2,011s (34 min) |
| Eigendecomp | 3,114s (52 min) | 8,465s (2h 21m) |
| LMM | 1,211s (20 min) | 1,131s (19 min) |
| **Total** | **5,764s (96 min)** | **11,607s (3h 14m)** |

Eigendecomp dominates the increase: O(n³) scaling from 90k→125k is ~2.7×, plus memory bandwidth saturation at 125k (the 126 GB eigenvector matrices exceed L3 cache). LMM is faster despite more samples, likely due to the 48-core machine and the v2.5.6 rotation threading fix.

---

## C Extension LMM Acceleration (NumPy Backend)

The NumPy backend includes an optional C extension (`_lmm_accel.c`) with OpenMP parallelism
that replaces the Python loop over SNPs for Wald test computation. The extension uses a
workspace API (pre-allocated per-thread buffers) and SoA Uab layout with invariant precompute.

### C Extension vs JAX Backend (E96ds_v6, 48 cores, synthetic 95k SNPs)

| Scale    | C ext LMM | JAX LMM | C ext Speedup |
|----------|-----------|---------|---------------|
| 5k×95k   | 20.0s     | 142.9s  | **7.1x**      |
| 20k×95k  | 90.3s     | 193.9s  | **2.1x**      |
| 50k×95k  | 273.4s    | 530.8s  | **1.9x**      |
| 75k×95k  | 484.3s    | 867.6s  | **1.8x**      |

The C extension wins at all scales. At small n, JAX's per-SNP overhead (Python Brent loop +
XLA dispatch) dominates. At large n, UT@G rotation (identical DGEMM in both) dominates and
compute ratios converge to ~3x.

### C Extension Scaling (LMM compute only, 95k SNPs)

| Scale    | UT@G Rotation | Compute | LMM Total | RSS     |
|----------|---------------|---------|-----------|---------|
| 5k×95k   | 2.9s          | 10.2s   | 20.0s     | 7.4 GB  |
| 20k×95k  | 24.6s         | 42.9s   | 90.3s     | 17.4 GB |
| 50k×95k  | 118.4s        | 101.9s  | 273.4s    | 45.7 GB |
| 75k×95k  | 254.0s        | 150.2s  | 484.3s    | 80.2 GB |

Compute scales O(n_samples). Rotation scales O(n² × n_snps) and dominates at 50k+.
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

Full test suite passing (`uv run pytest tests/ -x`). Tolerance constants in `src/jamma/validation/tolerances.py` unchanged from v1.3 (kinship tolerance aligned to 1e-8 in v2.5.7).

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
| Kinship (25%) | JAX-batched dgemm | Already JIT-compiled matrix multiply |
| LMM Association (21%) | JAX JIT + golden section per SNP | Rotation is a single dgemm per chunk |

### What v1.4 Did Not Change

- **Wall-clock time**: Eigendecomp, kinship, and LMM times are unchanged from Phase 19 baseline. JAMMA was already operating at the hardware-limited floor for CPU eigendecomposition.
- **Thread configuration**: MKL was already running at 32 threads on Databricks. The thread-pinning code (`_pin_blas_threads`) was a no-op because MKL loads during `import jax` before pinning runs. v1.4 formalized this into `blas_threads()` context managers but the runtime behavior is identical.

### Configuration Guide

| Scale | Samples | RAM Required | MKL Build | Reference |
|-------|---------|-------------|-----------|-----------|
| Small | ≤10k | 8 GB | Any | |
| Medium | 10–50k | 64 GB | LP64 or ILP64 | |
| Large | 50–100k | 256 GB | ILP64 required | 85k validated (v1.4) |
| XLarge | 100–125k | 672 GB+ | ILP64 required | 125k validated (v2.5) |

### CPU Device Sharding

JAMMA uses JAX `NamedSharding` to partition SNP batches across virtual CPU
devices, parallelising the per-SNP REML grid search and golden-section
refinement.

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
*Last updated: 2026-03-01*
