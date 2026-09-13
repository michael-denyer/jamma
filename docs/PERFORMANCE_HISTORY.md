# Historical small-scale measurements

These measurements predate the aligned process benchmark introduced on 2026-09-09.
GEMMA times include CLI startup and file I/O; JAMMA batch times start after loading
inputs and omit result-file writing. They are retained to document JAMMA version
changes, not as end-to-end GEMMA comparisons. GEMMA speedup columns have been removed.
Historical LOCO comparisons are withdrawn: GEMMA received a full kinship matrix
and tested all SNPs repeatedly, so the tools did not perform the same analysis.
The remaining LOCO rows compare historical JAMMA versions only.
See [current measurements](PERFORMANCE.md).

## master `9d33cc1` on mouse_hs1940 (historical)

Measured 2026-09-02. Same machine, toolchain, and dataset as the v7.2.0 run
below: Apple M5 Pro (18 cores), Accelerate-ILP64, numpy 2.5.1, Python 3.12,
OpenMP on, GEMMA 0.98.5 in the Homebrew OpenBLAS and Apple Accelerate builds,
dev-mode build with `-march=native`. One round of best-of-3, the v7.2.0
methodology, so the same caveat applies: a delta inside a few percent is not
a measured change.

| Operation | GEMMA (OpenBLAS) | GEMMA (Accelerate) | JAMMA NumPy | JAMMA NumPy+C | JAMMA NumPy+C (stream) | C speedup |
|-----------|-----------------|-------------------|-------------|--------------|------------------------|-----------|
| Kinship (`-gk 1`) | 1.1s | 1.2s | 196ms | 196ms | -- | 1.0x |
| LMM Wald (`-lmm 1`) | 7.3s | 4.2s | 2.4s | 291ms | 416ms | 8.2x |
| LMM All (`-lmm 4`) | 13.3s | 7.6s | 4.8s | 298ms | 400ms | 16.1x |
| LMM Wald+4cov (`-lmm 1 -c`) | 27.2s | 11.5s | 5.8s | 654ms | 712ms | 8.9x |

### master against v7.2.0

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

The C path uses every physical core under Accelerate, evaluates logdet(H) as
a mantissa product with an exact exponent (`GEMMA_DIVERGENCES.md` section 3),
and, under Accelerate only, cuts an input the memory budget leaves in fewer
than 8 chunks to 16 chunks up to 10,000 samples, so genotype rotation
overlaps the kernel. The cut is platform- and size-dependent. Measured with
`scripts/bench_large_n_stages.py --stages association` at 5,000 SNPs,
interleaved ABBA blocks, cut against no cut:

| Platform | Samples x SNPs | Blocks | Cut vs no cut |
|----------|----------------|--------|---------------|
| Apple M5 Pro, 18 cores, Accelerate | 1,410 x 12,226 | best-of-3 | -20% |
| Apple M5 Pro | 5,000 x 5,000 | 3 | -6.4% |
| Apple M5 Pro | 10,000 x 5,000 | 4 | -0.2% |
| Apple M5 Pro | 30,000 x 5,000 | 3 | +5.6% |
| Linux `Standard_E16ds_v6`, 8 cores, MKL | 1,410 x 12,226 | 5 | +22.4% |

Under MKL the pipelined plan's thread split also changes the rotation GEMM's
last bits, so two chunk plans are bit-identical only under Accelerate.

At 125,000 samples x 5,000 SNPs on `Standard_E96ds_v6` (48 physical cores,
MKL ILP64), master runs the association pass 1.8% faster than the
pre-tuning tree (52.0 s against 52.9 s, 2 interleaved blocks): the
log-determinant gain at a scale where rotation dominates.

**Pure-NumPy `-lmm 4` is 4.8s.** The v7.2.0 table's 3.5s ran part of mode 4
through the C extension; with every C route disabled
(`JAMMA_FORCE_NUMPY_FALLBACK=1`) that commit measures 4.7s.

## v7.2.0 on mouse_hs1940 (superseded by the master run above)

Measured 2026-07-27. Apple M5 Pro (18 cores), 69 GB RAM, macOS 26.5.2.
Accelerate-ILP64, numpy 2.5.1, Python 3.13.5, OpenMP on. GEMMA 0.98.5 in two
builds, Homebrew OpenBLAS and Apple Accelerate. Dataset: mouse_hs1940, 1,940
samples x 12,226 SNPs across 19 chromosomes; 1,410 samples survive
phenotype-missingness filtering, so the eigendecomposition is 1,410 x 1,410.
The build came from a clean worktree and carries `-march=native` from the
dev-mode compile, so these are not portable-wheel timings.

| Operation | GEMMA (OpenBLAS) | GEMMA (Accelerate) | JAMMA NumPy | JAMMA NumPy+C | JAMMA NumPy+C (stream) | C speedup |
|-----------|-----------------|-------------------|-------------|--------------|------------------------|-----------|
| Kinship (`-gk 1`) | 1.0s | 1.2s | 192ms | 192ms | -- | 1.0x |
| LMM Wald (`-lmm 1`) | 7.0s | 4.3s | 2.3s | 439ms | 551ms | 5.3x |
| LMM All (`-lmm 4`) | 12.8s | 7.6s | 3.5s* | 570ms | 680ms | 6.2x |
| LMM Wald+4cov (`-lmm 1 -c`) | 25.9s | 12.6s | 5.8s | 827ms | 939ms | 7.0x |

*Ran part of mode 4 through the C extension; the fully pure-NumPy time on
this commit is 4.7s.

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

| Operation | GEMMA (OpenBLAS) | GEMMA (Accelerate) | JAMMA NumPy | JAMMA NumPy+C | JAMMA NumPy+C (stream) | C speedup |
|-----------|-----------------|-------------------|-------------|--------------|------------------------|-----------|
| Kinship (`-gk 1`) | 1.1s | 1.2s | 195ms | 195ms | -- | 1.0x |
| LMM Wald (`-lmm 1`) | 7.1s | 4.2s | 2.4s | 430ms | 541ms | 5.6x |
| LMM All (`-lmm 4`) | 13.0s | 7.4s | 3.6s | 580ms | 695ms | 6.2x |
| LMM Wald+4cov (`-lmm 1 -c`) | 26.9s | 11.4s | 5.8s | 836ms | 945ms | 6.9x |

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

---
