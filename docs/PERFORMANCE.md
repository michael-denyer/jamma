# Performance Summary

> **Currency note.** Headline numbers are historical benchmarks from each
> noted version. Two separate currency questions matter here, and they have
> different answers.
>
> *Small scale* is current. master at `9d33cc1` was benchmarked on
> mouse_hs1940 on 2026-09-02, on the same machine that produced the v7.2.0
> and v6.0.0 runs (sections below).
>
> *Large scale* is not. The most recent end-to-end large-scale benchmark
> (125,632 samples) is still from v4.2.0; v4.6.1 added partial scaling data at
> smaller sizes. Nothing in the v5, v6, or v7 line has been re-benchmarked at
> full scale. The kinship, eigendecomp, and LMM hot paths are unchanged from
> v4.6.1. JAX and BLIS were stripped in v5.0 (commit `663a22b`) -- the
> backend set is now `numpy` and `numpy-streaming` only, both routing
> through jlinalg with vendor LAPACK > NumPy fallback.

## master `9d33cc1` on mouse_hs1940 (current)

Measured 2026-09-02. Same machine, toolchain, and dataset as the v7.2.0 run
below: Apple M5 Pro (18 cores), Accelerate-ILP64, numpy 2.5.1, Python 3.12,
OpenMP on, GEMMA 0.98.5 in the Homebrew OpenBLAS and Apple Accelerate builds,
dev-mode build with `-march=native`. One round of best-of-3, the v7.2.0
methodology, so the same caveat applies: a delta inside a few percent is not
a measured change.

| Operation | GEMMA (OpenBLAS) | GEMMA (Accelerate) | JAMMA NumPy | JAMMA NumPy+C | JAMMA NumPy+C (stream) | C speedup | vs GEMMA (OB) | vs GEMMA (Accel) |
|-----------|-----------------|-------------------|-------------|--------------|------------------------|-----------|---------------|------------------|
| Kinship (`-gk 1`) | 1.1s | 1.2s | 196ms | 196ms | -- | 1.0x | **5.5x** | **6.3x** |
| LMM Wald (`-lmm 1`) | 7.3s | 4.2s | 2.4s | 291ms | 416ms | 8.2x | **24.9x** | **14.5x** |
| LMM All (`-lmm 4`) | 13.3s | 7.6s | 4.8s | 298ms | 400ms | 16.1x | **44.7x** | **25.3x** |
| LMM Wald+4cov (`-lmm 1 -c`) | 27.2s | 11.5s | 5.8s | 654ms | 712ms | 8.9x | **41.6x** | **17.6x** |
| LOCO Wald (`-loco`) | 2m31s | 1m20s | -- | **3.3s** | -- | -- | **~46x** | **~24x** |

### master against v7.2.0

Three changes landed between the two runs, each measured in its own PR on
this machine before merging:

- #292 gives the C kernel every physical core under Accelerate. The
  `cores // 2` halving assumed the kernel always overlaps an Accelerate GEMM;
  a single-chunk run never does, and the pipelined run measured faster with
  the full count too.
- #294 cuts a run the memory budget alone would leave below the pipeline
  threshold to 16 chunks, so rotation of chunk N+1 overlaps the kernel on
  chunk N. mouse_hs1940 previously ran as one chunk with no overlap. Plans
  that already pipelined, which is every large-scale run, are untouched.
  The cut applies only up to 10,000 samples: every extra chunk re-streams
  the eigenvector matrix through the rotation GEMM, and the kernel time the
  overlap hides shrinks relative to that GEMM as samples grow. Measured with
  `scripts/bench_large_n_stages.py --stages association` at 5,000 SNPs
  (interleaved ABBA blocks, cut versus no cut): 1,410 samples -20%, 5,000
  -6.4%, 10,000 -0.2%, 30,000 +5.6%. It also applies only where the BLAS
  cannot be throttled, which is Accelerate on macOS. With a controllable
  BLAS the pipelined plan splits the cores between rotation and compute and
  re-limits the thread pool per chunk, and on an 8-core Linux MKL node
  (Databricks `Standard_E16ds_v6`, 5 blocks) the same cut measured +22.4% on
  the mouse_hs1940 shape, every block between +14.7% and +31.5%. Linux
  therefore keeps the plan it had before the cut existed. On MKL the
  pipelined plan's thread split also moves the rotation's last bits, so two
  plans are bit-identical only under Accelerate; the runner's digest check
  reports the difference on Linux.
- #295 evaluates logdet(H) as a product of mantissas with an exact integer
  exponent instead of one scalar `log()` per sample per likelihood
  evaluation. That call was 86% of the golden-section refinement loop. The
  n_cvt=1 kernel went from 150 ms to 56 ms; the general kernel gains less
  because its Pab recursion dominates. See `GEMMA_DIVERGENCES.md` section 3
  for the measured bound.

| Operation | v7.2.0 | master | Delta |
|-----------|--------|--------|-------|
| Kinship (`-gk 1`) | 192ms | 196ms | +2.1% |
| LMM Wald (`-lmm 1`) | 439ms | 291ms | -33.7% |
| LMM All (`-lmm 4`) | 570ms | 298ms | -47.7% |
| LMM Wald+4cov (`-lmm 1 -c`) | 827ms | 654ms | -20.9% |
| LMM Wald, streaming | 551ms | 416ms | -24.5% |
| LMM All, streaming | 680ms | 400ms | -41.2% |
| LMM Wald+4cov, streaming | 939ms | 712ms | -24.2% |
| LOCO Wald (`-loco`) | 3.3s | 3.3s | 0% |

Kinship and LOCO do not reach the changed code. LOCO is 19 eigendecompositions
of a 1,410 x 1,410 matrix plus 19 short LMM passes, so the kernel gain is
below its 0.1 s reporting resolution.

**Pure-NumPy `-lmm 4` reads 4.8s against 3.5s in the v7.2.0 table.** None of
the three changes reaches the NumPy fallback, and the 4.7 to 4.8s figure
reproduced on `e1f5c71` before any of them merged, on numpy 2.5.1 and 2.5.2
and on Python 3.13 and 3.14. It is a pre-existing regression against the
July figure that has not been traced yet, and it does not touch the C path
users run by default.

## v7.2.0 on mouse_hs1940 (superseded by the master run above)

Measured 2026-07-27. Apple M5 Pro (18 cores), 69 GB RAM, macOS 26.5.2.
Accelerate-ILP64, numpy 2.5.1, Python 3.13.5, OpenMP on. GEMMA 0.98.5 in two
builds, Homebrew OpenBLAS and Apple Accelerate. Dataset: mouse_hs1940, 1,940
samples x 12,226 SNPs across 19 chromosomes; 1,410 samples survive
phenotype-missingness filtering, so the eigendecomposition is 1,410 x 1,410.
The build came from a clean worktree and carries `-march=native` from the
dev-mode compile, so these are not portable-wheel timings.

| Operation | GEMMA (OpenBLAS) | GEMMA (Accelerate) | JAMMA NumPy | JAMMA NumPy+C | JAMMA NumPy+C (stream) | C speedup | vs GEMMA (OB) | vs GEMMA (Accel) |
|-----------|-----------------|-------------------|-------------|--------------|------------------------|-----------|---------------|------------------|
| Kinship (`-gk 1`) | 1.0s | 1.2s | 192ms | 192ms | -- | 1.0x | **5.3x** | **6.3x** |
| LMM Wald (`-lmm 1`) | 7.0s | 4.3s | 2.3s | 439ms | 551ms | 5.3x | **15.9x** | **9.7x** |
| LMM All (`-lmm 4`) | 12.8s | 7.6s | 3.5s | 570ms | 680ms | 6.2x | **22.4x** | **13.3x** |
| LMM Wald+4cov (`-lmm 1 -c`) | 25.9s | 12.6s | 5.8s | 827ms | 939ms | 7.0x | **31.4x** | **15.2x** |
| LOCO Wald (`-loco`) | 2m21s | 1m22s | -- | **3.3s** | -- | -- | **~43x** | **~25x** |

**Methodology caveat.** This is one round of best-of-3, where the v6.0.0 run
below was three interleaved rounds of best-of-3. A single round cannot separate
a small regression from warm-up noise, so treat the deltas below as "no
detectable change" rather than as a measured equality.

### v7.2.0 against v6.0.0

Both JAMMA columns, batch and streaming, against the v6.0.0 figures in the next
section. The largest move is -2.2%, on streaming all-tests. That is a shade
outside the +/-2% band the v6.0.0 run called noise, and it is negative, so
nothing here reads as a regression.

| Operation | v6.0.0 | v7.2.0 | Delta |
|-----------|--------|--------|-------|
| Kinship (`-gk 1`) | 195ms | 192ms | -1.5% |
| LMM Wald (`-lmm 1`) | 430ms | 439ms | +2.1% |
| LMM All (`-lmm 4`) | 580ms | 570ms | -1.7% |
| LMM Wald+4cov (`-lmm 1 -c`) | 836ms | 827ms | -1.1% |
| LMM Wald, streaming | 541ms | 551ms | +1.8% |
| LMM All, streaming | 695ms | 680ms | -2.2% |
| LMM Wald+4cov, streaming | 945ms | 939ms | -0.6% |
| LOCO Wald (`-loco`) | 3.3s | 3.3s | 0% |

That is the expected result. The v6.0.0 to v7.2.0 diff is the `PipelineConfig`
phenotype-field consolidation, the `pipeline.py` and `loco.py` splits, and
pyrefly type work. None of it reaches the arithmetic in the hot loop.

The GEMMA control columns drifted more than JAMMA's did. GEMMA+Accelerate on
`-lmm 1 -c` went 11.4s to 12.6s and its LOCO run 1m21s to 1m22s, against an
unchanged JAMMA binary path. That is machine variation on the GEMMA side, and
it is why the "vs GEMMA (Accel)" column moved from 13.6x to 15.2x on that row
without JAMMA getting faster.

## v6.0.0 vs v5.6.0 on mouse_hs1940 (superseded by the v7.2.0 run above)

Measured 2026-07-25. This run answers one narrow question: did the v5.6.0 to
v6.0.0 changes move the LMM hot path? They did not. Every operation lands
inside run-to-run noise.

Hardware: Apple M5 Pro (18 cores), 64 GB RAM, macOS 26.5.2. Accelerate-ILP64,
numpy 2.5.1, Python 3.13.5. GEMMA 0.98.5 in two builds, Homebrew OpenBLAS and
Apple Accelerate. Dataset: mouse_hs1940, 1,940 samples x 12,226 SNPs across 19
chromosomes; 1,410 samples survive phenotype-missingness filtering, so the
eigendecomposition is 1,410 x 1,410.

Both versions were built from clean worktrees with identical compiler flags and
pinned to the same numpy, leaving JAMMA's own code as the only variable. Both
carry `-march=native` from the dev-mode compile, so these are not
portable-wheel timings.

### Version comparison (JAMMA NumPy+C)

Minimum across 3 rounds per version, each round itself a best-of-3. The rounds
were interleaved v6, v5.6.0, v6, v5.6.0, so machine drift lands on both versions
equally. GEMMA ran in every round as a fixed control and its times agreed across
versions, confirming the machine was stable.

| Operation | v5.6.0 | v6.0.0 | Delta |
|-----------|--------|--------|-------|
| Kinship (`-gk 1`) | 194ms | 195ms | +0.5% |
| LMM Wald (`-lmm 1`) | 429ms | 430ms | +0.2% |
| LMM All (`-lmm 4`) | 573ms | 580ms | +1.2% |
| LMM Wald+4cov (`-lmm 1 -c`) | 841ms | 836ms | -0.6% |
| LMM Wald, streaming | 537ms | 541ms | +0.7% |
| LMM All, streaming | 708ms | 695ms | -1.8% |
| LMM Wald+4cov, streaming | 941ms | 945ms | +0.4% |
| LOCO Wald (`-loco`) | 3.3s | 3.3s | 0% |

Nothing exceeds +/-2%, in either direction. The LOCO row is 3 interleaved rounds
of best-of-5 and returned 3.3s on both versions in every round.

This is the expected result. The v5.6.0 to v6.0.0 diff is the `LmmConfig` API
consolidation and the split of the C accelerator into separate translation
units. Neither changes the arithmetic in the hot loop.

One measurement note worth recording. An early v6 round reported 567ms for Wald,
against 435ms from that same version's best-of-1 pass. A best-of-3 cannot
legitimately be worse than a best-of-1, which marked it as a warm-up artifact
rather than a regression; the two later rounds returned 452ms and 430ms. A
single round of this benchmark is not enough to call a regression on.

**Scope limit.** 1,940 samples exercises the LMM kernels and barely touches
eigendecomposition, which is 54-72% of wall time at 90k-125k scale. This run
says nothing about large-scale performance.

### v6.0.0 vs GEMMA 0.98.5

Same runs, same methodology. Every cell is that configuration's best observed
time across the 3 rounds, and the derived columns are computed from those
minima.

| Operation | GEMMA (OpenBLAS) | GEMMA (Accelerate) | JAMMA NumPy | JAMMA NumPy+C | JAMMA NumPy+C (stream) | C speedup | vs GEMMA (OB) | vs GEMMA (Accel) |
|-----------|-----------------|-------------------|-------------|--------------|------------------------|-----------|---------------|------------------|
| Kinship (`-gk 1`) | 1.1s | 1.2s | 195ms | 195ms | -- | 1.0x | **5.6x** | **6.2x** |
| LMM Wald (`-lmm 1`) | 7.1s | 4.2s | 2.4s | 430ms | 541ms | 5.6x | **16.5x** | **9.8x** |
| LMM All (`-lmm 4`) | 13.0s | 7.4s | 3.6s | 580ms | 695ms | 6.2x | **22.4x** | **12.8x** |
| LMM Wald+4cov (`-lmm 1 -c`) | 26.9s | 11.4s | 5.8s | 836ms | 945ms | 6.9x | **32.2x** | **13.6x** |
| LOCO Wald (`-loco`) | 2m14s | 1m21s | -- | **3.3s** | -- | -- | **~41x** | **~25x** |

Kinship is pure NumPy and BLAS in both JAMMA columns, so its C speedup is 1.0x
by construction. The LOCO row comes from a separate best-of-3 invocation of
`scripts/bench_loco.py`; its speedups are rounded because JAMMA's 3.3s is
reported to 0.1s, which bounds the ratio's precision at about 1.5%.

GEMMA's per-chromosome LOCO spread was tight: 4.3s across all 19 chromosomes on
the Accelerate build, 7.1-7.2s on OpenBLAS.

### Reproducing

```bash
uv run python scripts/bench_all_backends.py --runs 3
uv run python scripts/bench_loco.py --runs 3
```

Both auto-detect GEMMA at `~/.local/bin/gemma` and `~/.local/bin/gemma-accelerate`.
Run them sequentially. Parallel execution contaminates the timings, and a single
round is not enough to separate a regression from warm-up noise.

### Superseded: Apple M2 README table

The README performance table carried these numbers before the 2026-07-25 refresh.
Hardware was an Apple M2 and the JAMMA version was not recorded, so they are kept
only as a historical reference and are not comparable to the table above.

| Operation | GEMMA (OpenBLAS) | GEMMA (Accelerate) | JAMMA NumPy | JAMMA NumPy+C | JAMMA NumPy+C (stream) |
|-----------|-----------------|-------------------|-------------|--------------|------------------------|
| Kinship (`-gk 1`) | 2.1s | 1.7s | 262ms | 262ms | -- |
| LMM Wald (`-lmm 1`) | 11.0s | 7.6s | 4.1s | 879ms | 1.1s |
| LMM All (`-lmm 4`) | 20.5s | 13.9s | 6.0s | 1.3s | 1.4s |
| LMM Wald+4cov (`-lmm 1 -c`) | 40.8s | 18.8s | 9.1s | 2.4s | 2.6s |
| LOCO Wald (`-loco`) | 3m30s | 2m26s | -- | 7.1s | -- |

---

## v4.2.0 — 125k Scale (most recent full-scale benchmark)

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

## Benchmark Methodology Notes

GEMMA (Accelerate) is GEMMA 0.98.5 compiled against Apple's Accelerate framework instead of Homebrew OpenBLAS — **1.3-2.2x faster** due to AMX-accelerated BLAS, with identical numerical results. **NumPy+C** uses a C extension with OpenMP for Wald (`-lmm 1`) — REML optimization is compute-bound and parallelizes well across SNPs. The C speedup grows with covariates because the Pab table recursion is more expensive. NumPy+C is the fastest backend at all modes including all-tests (`-lmm 4`) with this small scale run. **NumPy+C (stream)** reads genotypes from disk in chunks — slightly slower than batch, but the production code path for large datasets that don't fit in memory. Kinship is always pure NumPy/BLAS. The LOCO speedup has two further sources: (1) JAMMA computes per-chromosome LOCO kinship via streaming and tests only that chromosome's SNPs, while GEMMA `-loco` tests *all* SNPs against each LOCO kinship (19x redundant work on 19 chromosomes); (2) JAMMA runs all chromosomes in a single process, avoiding 19 cold-start overheads.

---
Document last updated: *2026-07-27* (v7.2.0 benchmarked on mouse_hs1940 against
the v6.0.0 figures; no measurable delta. No full-scale re-benchmark performed --
hot paths unchanged from v4.6.1).
