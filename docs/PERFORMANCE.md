# Performance Summary

## v2.10 — 125k Scale (Latest)

v2.10.1 at 125,632 samples on 91,586 real SNPs. **~9x faster than GEMMA** (3h 4m vs ~27h). 8% faster than v2.5 thanks to rotation-compute overlap (Phase 54) and C extension improvements. Perfect GEMMA equivalence.

**Note**: GEMMA was compiled with default OpenBLAS, not MKL. Building GEMMA against ILP64 MKL is non-trivial (requires Makefile patches and ILP64 linking for matrices >46k). The comparison reflects typical deployment: GEMMA as-distributed vs JAMMA with ILP64 numpy-mkl.

### 125k Real Data Benchmark (v2.10.1, Databricks)

Hardware: Azure E96ds_v6 (Intel Xeon Platinum 8573C, 48 physical / 96 logical cores, 672 GB RAM). numpy 2.4.2 with MKL ILP64, JAX, Python 3.12, Databricks Runtime 16.4 LTS.

| Phase | Time | % of Total |
|-------|------|-----------|
| Kinship compute | 2,047s (34 min) | 19% |
| Kinship write | 556s (9 min) | 5% |
| Eigen + LMM | 8,437s (2h 21m) | 76% |
| **Total** | **~11,040s (3h 4m)** | **100%** |

Throughput: 11 SNPs/sec (eigen+LMM), 8.3 SNPs/sec end-to-end.

### 125k Validation: JAMMA vs GEMMA

| Metric | Result |
|--------|--------|
| **Kinship Spearman rho** | 1.00000000 |
| Kinship max abs diff | 5.00e-11 |
| Kinship mean abs diff | 1.24e-12 |
| Kinship max relative diff | 8.49e-07 |
| Kinship Frobenius relative | 1.45e-10 |
| **Association Spearman rho (-log10 p)** | 1.000000 |
| Significance agree (p < 0.05) | 91,586/91,586 (100%) |
| Significance agree (p < 5e-8) | 91,586/91,586 (100%) |
| Effect direction agreement | 100.0% |
| Max relative p-value diff | 9.66e-04 |

### Progression: 125k benchmarks across versions

All runs on the same hardware (E96ds_v6) and dataset (125,632 × 91,586).

| Phase | v2.10.1 (latest) | v2.5.6 (1 dev) | v2.5.6 (24 dev) | v2.10 vs v2.5 (1 dev) |
|-------|-----------------|----------------|-----------------|----------------------|
| Kinship compute | 2,047s | 2,068s | 2,011s | -1% |
| Kinship write | 556s | 575s | 547s | -3% |
| Eigen+LMM | 8,437s | 9,365s | 9,699s | **-10%** |
| **Pipeline total** | **11,040s** | **12,008s** | **12,257s** | **-8%** |

v2.10.1 is 968s (16 min) faster than the best prior run. Improvement is in the eigen+LMM phase — kinship is unchanged (same DGEMM). Multi-device sharding (24 dev) was net negative due to eigendecomp regression (+23%); v2.10 runs on 1 device.

### Scaling from 90k to 125k

| Phase | v2.3 (90k, 32 cores) | v2.10 (125k, 48 cores) |
|-------|------|-----------|
| Kinship compute | 1,440s (24 min) | 2,047s (34 min) |
| Kinship write | — | 556s (9 min) |
| Eigen+LMM | 4,325s (72 min) | 8,437s (2h 21m) |
| **Total** | **5,764s (96 min)** | **11,040s (3h 4m)** |

Eigendecomp dominates the increase: O(n³) scaling from 90k→125k is ~2.7×, plus memory bandwidth saturation at 125k (the 126 GB eigenvector matrices exceed L3 cache).

### Full Pipeline Scaling (v2.10.1, 95k SNPs, 48 cores)

| Phase | 50k×95k | 75k×95k |
|-------|---------|---------|
| Kinship compute | 289s (5 min) | 561s (9 min) |
| Eigendecomp (DSYEVD) | 495s (8 min) | 1,380s (23 min) |
| LMM (C ext) | 448s (7 min) | 611s (10 min) |
| **Total (C ext)** | **1,240s (21 min)** | **2,576s (43 min)** |
| LMM (JAX) | 502s (8 min) | 693s (12 min) |
| **Total (JAX)** | **1,299s (22 min)** | **2,650s (44 min)** |

Eigendecomp scales O(n³): 495s at 50k → 1,380s at 75k (2.8× for 1.5× samples). LMM scales roughly O(n²) due to rotation dominance.

---

## C Extension LMM Acceleration (NumPy Backend)

The NumPy backend includes an optional C extension (`_lmm_accel.c`) with OpenMP parallelism
that replaces the Python loop over SNPs for Wald test computation. The extension uses a
workspace API (pre-allocated per-thread buffers) and SoA Uab layout with invariant precompute.

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
| 5k×95k   | 19.8s     | 31.8s   | **1.6x**      |
| 20k×95k  | 94.0s     | 145.0s  | **1.5x**      |
| 50k×95k  | 448.0s    | 502.0s  | **1.1x**      |
| 75k×95k  | 611.0s    | 693.0s  | **1.1x**      |

The C extension wins at all scales but the gap narrows at large n where UT@G rotation
(identical DGEMM in both backends) dominates. At small SNP counts (5k), the speedup is
clearer because compute is a larger fraction of total time. At 95k SNPs with large samples,
rotation dominates both backends and ratios converge to ~1.1x.

### C Extension Scaling (LMM timing breakdown, 95k SNPs)

| Scale    | UT@G Rotation | Compute  | LMM Total | RSS      |
|----------|---------------|----------|-----------|----------|
| 50k×95k  | 148.7s        | 218.0s   | 446.1s    | 53.4 GB  |
| 75k×95k  | 274.6s        | 220.9s   | 609.0s    | 93.4 GB  |

### C Extension Scaling (LMM timing breakdown, 5k SNPs)

| Scale    | UT@G Rotation | Compute  | LMM Total | RSS      |
|----------|---------------|----------|-----------|----------|
| 5k×5k    | 0.18s         | 0.51s    | 1.1s      | 1.0 GB   |
| 20k×5k   | 1.17s         | 1.71s    | 4.0s      | 5.7 GB   |
| 50k×5k   | 6.23s         | 5.23s    | 14.4s     | 24.6 GB  |
| 75k×5k   | 13.59s        | 8.81s    | 26.9s     | 51.4 GB  |

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
*Last updated: 2026-03-04*
