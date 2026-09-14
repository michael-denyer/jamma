# Performance Summary

## Matrix text output, 2026-09-14

The default `%.10g`/tab writer uses C++17 conversion with ordered Python
threads. It retains `np.savetxt` for small inputs and the process writer for
custom formats or unavailable native support.

This speeds explicit text exports. The normal pipeline retains computed
kinship in memory unless `save_kinship` is requested; saved matrices default
to binary `.npy`.

Measured on the same 18-core Mac, Python 3.12.13 and NumPy 2.5.1, at revision
`cca9799d`. Medians of five interleaved runs, using 18 workers for both paths:

| Matrix | Process writer | Native writer | Speedup |
|--------|----------------|---------------|---------|
| Mouse, 1,940 × 1,940 | 320ms | 13.6ms | 23.5x |
| 5,000 × 5,000 | 577ms | 65.9ms | 8.8x |
| 2,000 × 100,000 | 2.35s | 499ms | 4.7x |

The complete fresh-process mouse `-gk 1 --legacy-text` command fell from
773ms to 469ms, a 39% reduction. Every timed matrix matched `np.savetxt`
byte-for-byte; CLI outputs also had identical SHA256 digests. Validation runs
outside the timer. Timings include file creation, close, and atomic replacement,
with filesystem caching enabled and no fsync. The wide case gives the process
writer enough rows to occupy every worker.

Each thread formats at most 65,536 values per block, or one whole row when
wider. At most two output buffers per worker are in flight, each reserving
32 bytes per value. Layout or dtype conversion also happens per block. The
native path needs only the atomic output temporary file; it creates no matrix
memmap or intermediate text chunks.

Both libc++ on macOS and libstdc++ on Linux passed the 603,803-value precision
corpus, including random binary64 patterns, decimal ties, notation boundaries,
NaNs, infinities, and signed zero. Installed macOS 14-targeted and manylinux
wheels passed. Linux ASan/UBSan checks and a real SIGINT also passed. Linux
correctness was tested under x86_64 emulation; these performance measurements
are macOS results, not Databricks or a full 100,000-square matrix measurement.

```bash
uv run python scripts/bench_matrix_text.py --cases mouse square wide --cli --repetitions 5 --json /tmp/text-bench.json
uv run python scripts/smoke_test_matrix_text.py
```

[Raw repetitions, hashes, and environment](benchmarks/2026-09-14-native-text.json).

## Aligned process benchmarks, 2026-09-14

Measured 2026-09-14 on mouse_hs1940: 1,940 samples and 12,226 SNPs,
with 1,410 samples and 10,768 SNPs retained for association. Apple M5 Pro,
18 physical cores, macOS 26.6.2, Python 3.12.13, NumPy 2.5.1,
JAMMA 8.1.0 with native extensions and Accelerate-ILP64,
and GEMMA 0.98.5 in OpenBLAS and Accelerate builds. The runtime source is
revision `cca9799d`. The machine was otherwise idle; all writer benchmarks
finished before this backend comparison started. This is the local development installation, not a fresh
portable-wheel installation.

| Operation | GEMMA (OpenBLAS) | GEMMA (Accelerate) | JAMMA NumPy | JAMMA NumPy+C | JAMMA NumPy+C (stream) | C speedup | vs GEMMA (OB) | vs GEMMA (Accel) |
|-----------|-----------------|-------------------|-------------|--------------|------------------------|-----------|---------------|------------------|
| Kinship (`-gk 1`) | 1.1s | 1.2s | 835ms | 457ms | — | 1.8x | 2.4x | 2.7x |
| LMM Wald (`-lmm 1`) | 7.3s | 4.3s | 5.4s | 544ms | 599ms | 10.0x | 13.5x | 7.8x |
| LMM All (`-lmm 4`) | 13.5s | 7.6s | 7.9s | 567ms | 596ms | 13.9x | 23.8x | 13.3x |
| Full GWAS Wald (compute kinship + association) | 8.4s | 5.5s | 5.7s | 724ms | 785ms | 7.9x | 11.6x | 7.6x |
| LMM Wald+4cov (`-lmm 1 -c`) | 27.2s | 12.6s | 16.4s | 1.1s | 1.1s | 15.2x | 25.3x | 11.8x |

Best of three fresh-process runs per operation and backend, run sequentially
with backend order rotated between repetitions. This measures a warm filesystem
cache, not cold-storage performance. It includes process startup, imports,
input loading, computation and final output writing. There is no untimed JAMMA
warmup or preloaded genotype/kinship array. Ratios use unrounded times and the
faster of the JAMMA C batch and streaming backends for each operation.
The C speedup column includes startup and dispatch differences. Standalone
kinship uses the new native text formatter; its computation does not use the
LMM accelerator.

The required work determines which I/O belongs in each row:

- **Kinship:** both read PLINK and write the same text matrix format. JAMMA uses
  `--legacy-text` because a saved matrix is the requested result.
- **Association:** both read the same precomputed text kinship, PLINK and optional
  covariate files, then write association results. Kinship computation is excluded
  for both; eigendecomposition is included for both.
- **Full GWAS:** both start from PLINK and finish with association results. GEMMA
  runs `-gk 1` followed by `-lmm 1 -k`; JAMMA calls `gwas()` in a fresh Python
  process and retains kinship in memory. Avoiding intermediate kinship I/O and
  filtering samples early are workflow benefits included in the timing.
- **LOCO:** GEMMA computes each chromosome's kinship with an explicit SNP list
  excluding that chromosome, then associates only that chromosome's SNPs. JAMMA
  computes LOCO internally in one process. Both test each retained SNP once.
  GEMMA's required intermediate matrix writes and reads remain timed. Preparing
  the chromosome SNP lists is outside the timer, as is temporary-directory setup.

GEMMA 0.98.5's PLINK `CalcKin` path does not pass the `-loco`/`-ksnps`
selection to `PlinkKin`, so the benchmark uses `-snps` with an explicit
complement list for kinship. See [GEMMA's implementation](https://github.com/genetics-statistics/GEMMA/blob/v0.98.5/src/param.cpp).

| Backend | LOCO Wald | vs fastest GEMMA |
|---------|-----------|------------------|
| GEMMA (OpenBLAS) | 35.4s | 1.0x |
| GEMMA (Accelerate) | 34.0s | 1.0x |
| JAMMA NumPy+C | 3.3s | 10.4x |

The scripts reject missing/duplicate SNPs, mismatched tested SNP sets or alleles,
and effect/standard-error/p-value differences outside the existing numerical
validation tolerances. Saved kinship matrices are also compared. Validation is
outside the timing window. A failed command or comparison produces no summary
table or JSON report.

### LOCO with covariates

The LOCO row has no covariates. A chromosome-1 check with four covariates
(950 SNPs) agrees between the tools to within 3.8e-6 standard errors on every
beta, but `rs13475789` fails the 1% relative beta tolerance: GEMMA reports
`4.366448e-6`, JAMMA `4.444851e-6`, a difference of `7.8e-8` on a beta that is
1e-4 of its standard error (`0.035`, p = 0.9999). A relative tolerance is not
meaningful at zero, so `bench_loco.py --covariates` produces no table until the
beta check gains a standard-error-scaled floor.

### Reproduce

```bash
uv run python scripts/bench_all_backends.py --runs 3 --json /tmp/jamma-backends.json
uv run python scripts/bench_loco.py --runs 3 --json /tmp/jamma-loco.json
```

Run sequentially on an otherwise idle machine. Both scripts require the C
extension and auto-detect GEMMA at `~/.local/bin/gemma` and
`~/.local/bin/gemma-accelerate`. `--json` records each repetition and its exact
commands. Temporary output paths in those commands are removed after validation;
the scripts recreate equivalent directories on each invocation.

[Updated backend repetitions and input/build hashes](benchmarks/2026-09-14-native-backends.json)
record all 72 measurements. The LOCO row retains the earlier measurement at
`7b63772a` from [the preceding report](benchmarks/2026-09-14-aligned.json).
The checked-in reports retain timings and provenance; `--json` also saves the
exact commands with local paths.

### Observed variation

Minimum-to-maximum ranges across the three repetitions, not confidence
intervals. The ratios above compare minima.

| Operation | GEMMA Accelerate range (s) | JAMMA C batch range (s) |
|-----------|---------------------------|-------------------------|
| Kinship | 1.238–1.245 | 0.457–0.474 |
| Wald association | 4.266–4.284 | 0.544–0.551 |
| All-tests association | 7.564–7.628 | 0.567–0.574 |
| Full GWAS Wald | 5.490–5.548 | 0.724–0.738 |
| Wald + four covariates | 12.627–12.806 | 1.078–1.106 |
| LOCO Wald | 34.018–34.149 | 3.284–3.329 |

The earlier small-scale comparisons used different timing boundaries. Their
GEMMA ratios and LOCO comparisons are withdrawn; JAMMA version measurements
remain in [the historical record](PERFORMANCE_HISTORY.md). The historical
125k measurements below have not been rerun under this protocol and do not
establish a current, aligned speedup.

## LOCO resource ownership, 2026-09-09

Apple M5 Pro (18 cores), Accelerate-ILP64, NumPy 2.5.1, Python 3.12.13,
native OpenMP build. The full backend comparison ran sequentially, best of
three, first at `d4a3a67d` and then with the fix. Native Wald, All and
Wald+4cov took 302ms, 318ms and 868ms respectively, versus 311ms, 350ms
and 896ms before. These separate benchmark rounds do not establish a speedup.

Complete LOCO Wald runs used `run_lmm_loco` on mouse_hs1940, including
kinship and output writing, with memory checks and progress disabled. Each
worker count ran before/fixed/fixed/before in fresh processes, sequentially.
The table reports the mean of the two observations per version.

The first fix serialised eigen solves and association in bounded batches so
their BLAS scopes could not interleave:

| LOCO workers | Before | Batch barrier | Change |
|--------------|--------|---------------|--------|
| 1 | 3.275s | 3.179s | -2.9% |
| 6 | 1.409s | 1.864s | +32.3% |

The barrier was replaced on 2026-09-10 by one BLAS scope entered on the
consumer thread for the whole eigen stream. Workers never change limits and
association's scope nests inside the eigen scope, so the race cannot occur
and the solves overlap association again. Same machine, same ABBA protocol,
PR 361's head against the replacement:

| LOCO workers | Batch barrier | One scope | Change |
|--------------|---------------|-----------|--------|
| 1 | 3.330s | 3.351s | +0.6% |
| 6 | 2.113s | 1.568s | -25.8% |

Every association output was byte-identical across all sixteen runs. On
Accelerate `blas_threads` is a no-op, so neither fix changes thread counts
here; the difference is the barrier alone. On MKL and OpenBLAS, solves still
in flight while association runs inherit its rotation limit, which is
oversubscription rather than a stale restore; `JAMMA_BLAS_THREADS=cores/W`
bounds it. These timings do not predict Linux MKL/OpenBLAS performance or
large-sample memory use.

## v4.2.0 — 125k Scale (most recent full-scale benchmark)

v4.2.0 at 125,632 samples on 91,586 real SNPs. Historical wall times were 2h 29m for JAMMA and approximately 27h for GEMMA. These used different BLAS builds and have not been revalidated under the aligned benchmark protocol. 19% faster than v2.10.1 thanks to jlinalg eigendecomp and C extension LMM improvements. Eigendecomp used DSYEVR (memory-constrained fallback from DSYEVD).

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

All runs on the same hardware (E96ds_v6) and dataset (125,632 x 91,586).

| Phase | v4.2.0 | v2.10.1 | v2.5.6 (1 dev) | v4.2 vs v2.10 |
|-------|----------------|---------|----------------|---------------|
| Kinship compute | 1,591s | 2,047s | 2,068s | **-22%** |
| Eigendecomp | 6,427s | — | — | — |
| LMM (C ext) | 887s | — | — | — |
| Eigen+LMM | 7,314s | 8,437s | 9,365s | **-13%** |
| **Pipeline total** | **8,942s** | **11,040s** | **12,008s** | **-19%** |

v4.2.0 is 2,098s (35 min) faster than v2.10.1. Kinship is 22% faster (vendor BLAS dispatch improvements). Eigen+LMM is 13% faster despite using DSYEVR (memory-constrained fallback) instead of DSYEVD. LMM dropped from ~2,000s to 887s (C ext with 48 OpenMP threads).

### Scaling from 90k to 125k

| Phase | v2.3 (90k, 32 cores) | v4.2.0 (125k, 48 cores) |
|-------|------|-----------|
| Kinship compute | 1,440s (24 min) | 1,591s (27 min) |
| Eigendecomp | 3,114s (52 min) | 6,427s (1h 47m) |
| LMM | 1,211s (20 min) | 887s (15 min) |
| **Total** | **5,764s (96 min)** | **8,942s (2h 29m)** |

Eigendecomp dominates the increase: O(n^3) scaling from 90k->125k is ~2.1x (DSYEVR). LMM actually got faster at 125k than 90k was at v2.3 thanks to the C extension. The 126 GB eigenvector matrices exceed L3 cache, making eigendecomp memory-bandwidth bound.

### Full Pipeline Scaling (v4.6.1, 95k SNPs, 48 cores)

| Phase | 5k x 95k | 20k x 95k | 50k x 95k | 75k x 95k | 125k x 92k (real) |
|-------|----------|-----------|-----------|---------|-----------------|
| Kinship compute | 10s | 67s (1 min) | 284s (5 min) | 500s (8 min) | 1,591s (27 min)* |
| Eigendecomp | 1s | 44s | 516s (9 min) | 1,478s (25 min) | 6,427s (1h 47m)*+ |
| LMM (C ext) | 8s | 42s | 182s (3 min) | 362s (6 min) | 887s (15 min)* |
| **Total (C ext)** | **19s** | **155s (3 min)** | **988s (16 min)** | **2,353s (39 min)** | **8,942s (2h 29m)** |

*125k numbers from v4.2.0 (same hardware, not re-benchmarked). +125k used DSYEVR (memory-constrained fallback); all others used DSYEVD. Eigendecomp scales O(n^3): 516s at 50k -> 1,478s at 75k (2.9x for 1.5x samples). LMM scales roughly O(n^2) due to rotation dominance. v4.6.1 LMM is 14-16% faster than v4.2.0 at 50k-75k thanks to centralized jlinalg thread control and pthreads-based SNP stats.

---

## C Extension LMM Acceleration (NumPy Backend)

The NumPy backend includes an optional multi-source `_lmm_accel` C extension with OpenMP parallelism
that replaces the Python loop over SNPs for Wald test computation. The extension uses a
workspace API (pre-allocated per-thread buffers). The primary path (fused kernel) takes
utg_t in (n_snps, n_samples) layout directly from DGEMM TRANSA, computing wx/xx/xy
on-the-fly without a separate SoA Uab buffer. The SoA Uab layout with invariant precompute
is retained as a fallback when the fused C extension is unavailable. Mean imputation of
missing genotypes is done in-place on the chunk buffer (no copy), so the per-chunk memory
footprint equals the rotation output buffer only.

### C Extension Scaling (LMM timing breakdown, 95k SNPs)

| Scale    | UT@G Rotation | Compute  | LMM Total | RSS      |
|----------|---------------|----------|-----------|----------|
| 5k x 95k   | 2.8s          | 3.6s     | 11.5s     | 6.6 GB   |
| 20k x 95k  | 22.5s         | 15.2s    | 56.6s     | 27.0 GB  |
| 50k x 95k  | 125.8s        | 44.8s    | 216.8s    | 60.3 GB  |
| 75k x 95k  | 285.1s        | 61.3s    | 418.3s    | 95.1 GB  |
| 125k x 92k | 652.0s        | 93.1s    | 882.1s    | 320.6 GB |

### C Extension Scaling (LMM timing breakdown, 5k SNPs)

| Scale    | UT@G Rotation | Compute  | LMM Total | RSS      |
|----------|---------------|----------|-----------|----------|
| 5k x 5k    | 0.18s         | 0.51s    | 1.1s      | 1.0 GB   |
| 20k x 5k   | 1.17s         | 1.71s    | 4.0s      | 5.7 GB   |
| 50k x 5k   | 6.23s         | 5.23s    | 14.4s     | 24.6 GB  |
| 75k x 5k   | 13.59s        | 8.81s    | 26.9s     | 51.4 GB  |

Compute scales O(n_samples). Rotation scales O(n^2 x n_snps) and dominates at 20k+.
Both use MKL DGEMM with 48 threads.

---

## v2.0 — Production GWAS Features

v2.0 added LOCO kinship, eigendecomposition reuse, SNP filtering, HWE QC, and phenotype selection. No performance regressions from v1.4.

### New Features Performance Characteristics

| Feature | Scaling | Notes |
|---------|---------|-------|
| LOCO kinship | O(n_chr x n^2) | Linear in chromosomes; one eigendecomp per chromosome |
| Eigendecomp reuse (`-d`/`-u`) | Eliminates O(n^3) | Skips eigendecomp entirely on subsequent phenotypes |
| SNP filtering (`-snps`/`-ksnps`) | O(log n) per chunk | Searchsorted-based chunk filtering |
| HWE filtering (`-hwe`) | O(1) per SNP | Piggybacks on pass-1 streaming (no extra disk pass) |

### LOCO Scaling

LOCO analysis processes each chromosome independently: compute K_loco via streaming subtraction, eigendecompose, run LMM. Total time scales linearly with the number of chromosomes (typically 22 for human data). Each per-chromosome eigendecomp is the same O(n^3) as a standard run, so total LOCO wall time is approximately `n_chr x single_eigendecomp_time + LMM_time`.

### Test Suite

Default test suite passing (`uv run pytest tests/ -x`), which excludes `slow` and `tier2` markers per `pyproject.toml` defaults. Run `uv run pytest tests/ -x -m ""` for the full suite including slow/tier2 tests. Tolerance constants in `src/jamma/validation/tolerances.py` unchanged from v1.3 (kinship tolerance aligned to 1e-8 in v2.5.7).

---

## v1.4 — Memory Optimization and Scale Validation

v1.4 targeted memory optimization and correctness at production scale (85k+ real samples). The primary achievement is **validated GEMMA equivalence at 85,000 samples on 91,613 real SNPs** with 100% agreement on significance calls, effect directions, and SNP rankings.

### Changes Applied

| Change | Impact |
|--------|--------|
| Phase-specific LMM memory estimates | Fixed false MemoryError at 100k samples (was demanding 320GB pipeline peak when only 96GB needed) |
| Progress bar lifecycle fix | Bars complete cleanly (no hanging on final iteration) |
| Vectorized per-SNP imputation | Streaming runner imputation ~2x faster |
| Top-level `gwas()` API | Single-call Python entry point for full GWAS pipeline |
| GEMMA comparison notebook | Compare-only mode with OOM-safe kinship comparison at 85k scale |

### 90k Baseline (v2.3, Databricks)

Measured on 32-core Databricks VM with MKL ILP64, 90k synthetic samples x 90k SNPs.

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
| Eigendecomp (54%) | LAPACK dsyevd — O(n^3) | Single call, irreducible. 90k at 32 cores ~ 3,100s |
| Kinship (25%) | vendor BLAS DSYRK (chunked) | Multi-threaded BLAS matrix multiply |
| LMM Association (21%) | C extension + OpenMP (golden section per SNP) | Rotation is a single dgemm per chunk (utg_t, DGEMM TRANSA) |

### What v1.4 Did Not Change

- **Wall-clock time**: Eigendecomp, kinship, and LMM times are unchanged from the earlier baseline. JAMMA was already operating at the hardware-limited floor for CPU eigendecomposition.
- **Thread configuration**: MKL was already running at 32 threads on Databricks. v1.4 formalized thread configuration into `blas_threads()` context managers but the runtime behavior is identical.

### Configuration Guide

| Scale | Samples | RAM Required | MKL Build | Reference |
|-------|---------|-------------|-----------|-----------|
| Small | <=10k | 8 GB | Any | |
| Medium | 10-50k | 64 GB | LP64 or ILP64 | |
| Large | 50-100k | 256 GB | ILP64 required | 85k validated (v1.4) |
| XLarge | 100-125k | 768 GB | ILP64 required | 125k validated (v4.2.0), peak ~560 GB |

RAM requirements are for the full pipeline (kinship + eigendecomp + LMM). Eigendecomp is the memory peak: K matrix (n^2 x 8 bytes) + eigenvectors (n^2 x 8 bytes) must coexist. At 125k this is ~252 GB + 252 GB = 504 GB; the process peaked at 381 GB RSS with 768 GB physical. Scaling beyond 125k on 768 GB is not feasible — at 150k the eigendecomp alone would require ~720 GB (DSYEVR), leaving nothing for LMM.

Note: with early sample filtering (phenotype missingness), `n` in these formulae is
the number of valid samples (non-missing phenotype), which may be smaller than the
BED file sample count.

### Test Suite

Full test suite passing. Kinship tolerance aligned from 1e-10 to 1e-8 in v2.5.7 to match GEMMA_EQUIVALENCE.md bounds. All other tolerance constants in `src/jamma/validation/tolerances.py` unchanged from v1.3.

---

## Backend interpretation

GEMMA's OpenBLAS and Accelerate builds are reported separately. The JAMMA NumPy
column forces NumPy fallback; NumPy+C enables the native LMM and vendor BLAS
extensions. Streaming includes genotype reads within the timed process. These
measurements describe this dataset, machine and build, not a universal speedup.
