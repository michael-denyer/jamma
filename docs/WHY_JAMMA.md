# Why JAMMA? Key Differentiators from GEMMA

JAMMA delivers the same statistical results as GEMMA while solving practical problems that make GEMMA difficult to use at scale.

## Quick Comparison

| Feature | GEMMA | JAMMA |
|---------|-------|-------|
| **OOM Handling** | Silent crash (OS kill) | Pre-flight check with clear error |
| **Large-Scale** | Requires manual tuning | Streaming I/O, pre-flight memory checks (>100k requires ILP64) |
| **Speed** | 1x baseline | Up to 10x faster (C extension + vendor BLAS) |
| **Installation** | C++ compilation required | `pip install jamma` |
| **Error Messages** | Segfault or cryptic | Clear, actionable |
| **Numerical Results** | Reference | Equivalent ([proof](GEMMA_EQUIVALENCE.md)) |
| **Sample Filtering** | Kinship always n_samples x n_samples | Kinship at n_valid x n_valid when samples are dropped |

---

## 1. Memory Safety: Fail Fast, Not Silent Crash

### The GEMMA Problem

GEMMA loads everything into memory and lets the OS handle failure:

```bash
$ gemma -bfile large_study -gk 1
# ... runs for 20 minutes ...
Killed
```

No warning. No error message. Just `Killed` from the OOM killer. You've lost 20 minutes of compute time and have no idea why.

### The JAMMA Solution

JAMMA checks memory requirements BEFORE allocation:

```bash
$ jamma -bfile large_study -gk 1
MemoryError: Eigendecomposition requires 640.0 GB but only 512.0 GB available.
  Kinship matrix: 640.0 GB (n=200000 samples)
  Eigendecomp workspace: ~2x kinship

Suggestion: Use a larger instance or streaming mode.
```

**Key features:**

- Pre-flight memory estimation before any large allocation
- Clear breakdown of where memory goes
- Actionable suggestions for resolution
- RSS logging at workflow boundaries for debugging
- Early sample filtering: when samples are dropped due to phenotype or covariate missingness, kinship is accumulated at (n_valid x n_valid) size directly — the full (n_samples x n_samples) matrix is never allocated

---

## 2. Scale: Large Samples Without Manual Tuning

### The GEMMA Problem

GEMMA requires the full n x p genotype matrix in memory. For 90k samples x 90k SNPs:

- Genotype matrix: ~32 GB
- Kinship matrix: ~65 GB
- Eigendecomposition workspace: ~130 GB peak

Studies over 100k samples require ILP64 BLAS and 512 GB+ RAM due to O(n^3) eigendecomposition memory.

### The JAMMA Solution

JAMMA streams data from disk, never materializing the full matrix:

```python
# Kinship computed in chunks - never loads full genotype matrix
kinship = compute_kinship_streaming("large_study", chunk_size=10000)

# LMM also streams - only kinship (n^2) kept in memory
results = run_lmm_association_numpy_streaming(
    "large_study", phenotypes, kinship, chunk_size=5000
)
```

**Memory profile:**

- Peak is eigendecomposition: n^2 x 8 bytes x ~2 (K + workspace)
- Genotype chunks: chunk_size x n x 8 bytes (transient)
- Results written incrementally to disk (no accumulation)

---

## 3. Speed: C Extension Acceleration

### Benchmark (mouse_hs1940: 1,940 samples x 12,226 SNPs, Apple M2, GEMMA 0.98.5)

| Operation          | GEMMA 0.98.5 | JAMMA (NumPy+C) | Speedup |
|--------------------|--------------|------------------|---------|
| Kinship (`-gk 1`)  | 2.1s         | 1.7s             | ~1.2x   |
| LMM (`-lmm 1`)     | 11.3s        | 5.3s             | **2.1x** |
| **Total**          | **13.4s**    | **7.0s**         | **1.9x** |

Kinship is BLAS-bound (both use OpenBLAS/Accelerate matmul) so times are similar. The LMM speedup comes from the OpenMP-parallelized C extension for batch SNP processing.

### At Scale: 125k Samples (Databricks E96ds_v6, 48 cores, ILP64 MKL)

| Pipeline                      | GEMMA 0.98.5 | JAMMA v4.2.0 | Speedup |
|-------------------------------|--------------|---------------|---------|
| Full GWAS (125,632 x 91,586) | ~27 hours    | 2h 29m        | **~10x** |

**Caveat**: GEMMA was compiled with default OpenBLAS, not MKL. Building GEMMA against MKL is non-trivial (requires patching the Makefile and linking against ILP64 MKL for matrices >46k) and we did not attempt it. The comparison reflects typical deployment: GEMMA as-distributed vs JAMMA with ILP64 numpy-mkl. The speedup would be smaller with an MKL-linked GEMMA, though the batch-parallel LMM architecture would still provide a significant advantage.

### Why Faster?

The key insight: **GEMMA loops over SNPs sequentially; JAMMA processes SNPs in parallel batches.**

| Aspect | GEMMA | JAMMA |
| ------ | ----- | ----- |
| SNP loop | Sequential C++ `for` loop | Batch parallel via C extension + OpenMP |
| Per-SNP overhead | Function call + memory allocation | Pre-allocated workspace (zero alloc per SNP) |
| BLAS utilization | Many small matmuls | Few large batched matmuls |
| Memory access | Row-by-row, cache-unfriendly | Contiguous, cache-optimized |

**Detailed breakdown:**

1. **Batch vectorization**: JAMMA's C extension processes all SNPs in a chunk as a single batched operation with OpenMP thread parallelism. GEMMA's C++ loop processes one SNP at a time — even with multithreaded BLAS for individual matrix operations, the outer loop is serial.

2. **Workspace API**: The C extension pre-allocates per-thread buffers (Pab, workspace arrays) once per chunk. GEMMA allocates and frees per-SNP buffers in its inner loop.

3. **Efficient Pab computation**: The cumulative Uab/Pab structure is computed once per covariate set, then broadcast across SNPs.

The real difference is algorithm design: **data-parallel batch processing vs sequential-with-parallel-primitives**.

---

## 4. Installation: No C++ Compilation

### GEMMA Installation

```bash
# Hope you have the right BLAS/LAPACK versions
git clone https://github.com/genetics-statistics/GEMMA
cd GEMMA
make
# ... 50 lines of compiler errors about GSL ...
```

### JAMMA Installation

```bash
pip install jamma
```

That's it. Pure Python with an optional C extension (auto-compiled on first use) handles the numerical heavy lifting.

---

## 5. Error Handling: Clear, Not Cryptic

### GEMMA Errors

```text
Segmentation fault (core dumped)
```

or

```text
ERROR: error! number of columns in the kinship matrix
```

### JAMMA Errors

```text
ValueError: Covariate file row 15, column 3: cannot parse 'NA' as numeric
  Hint: Use 'NA' (case-sensitive) for missing values

MemoryError: LMM association requires 45.2 GB but only 32.0 GB available.
  Eigendecomp: 25.0 GB (already loaded)
  Genotype chunks: 12.0 GB (chunk_size=50000)
  Result buffer: 8.2 GB

  Suggestion: Reduce chunk_size to 25000 or use streaming mode.
```

Every error includes:

- What went wrong
- Where it happened
- How to fix it

---

## 6. Numerical Equivalence: Same Science

Despite all improvements, JAMMA produces **identical scientific conclusions** to GEMMA:

| Metric | Tolerance | Validation |
|--------|-----------|------------|
| Kinship matrix | < 1e-8 relative | CI test on every commit |
| Beta coefficients | < 1e-2 relative | GEMMA fixture comparison |
| P-values (Wald/Score) | < 1e-4 relative | GEMMA fixture comparison |
| P-values (LRT) | < 5e-3 relative | MLE subtraction amplification |
| Lambda (REML) | < 5e-5 relative | Optimizer tolerance gap (JAMMA golden section vs GEMMA Brent) |
| Significance calls | 100% agreement | All thresholds (0.05, 0.01, 5e-8) |
| Effect directions | 100% agreement | Sign of beta |
| SNP rankings | Identical | Spearman correlation = 1.0 |

See [GEMMA_EQUIVALENCE.md](GEMMA_EQUIVALENCE.md) for the formal error propagation analysis.

The goal is a **drop-in replacement**: same CLI, same output format, same scientific results.

---

## 7. Modern Python Ecosystem

### Debugging & Profiling

```python
# RSS logging at workflow boundaries
from jamma.utils.logging import log_rss_memory

log_rss_memory("kinship", "before")  # Logs current RSS in GB
kinship = compute_centered_kinship(genotypes)
log_rss_memory("kinship", "after")
```

### Memory Estimation API

```python
from jamma.core.memory import estimate_workflow_memory

# Before starting a big job
estimate = estimate_workflow_memory(n_samples=200_000, n_snps=95_000)
print(f"Peak: {estimate.total_gb:.1f}GB")
print(f"Available: {estimate.available_gb:.1f}GB")
print(f"Will fit: {estimate.sufficient}")
```

### Type Safety

Full type annotations throughout. IDE autocomplete works. Mypy catches bugs.

---

## 8. Modern Development Approach

JAMMA applies contemporary software engineering practices that GEMMA (written in 2012) predates:

### Modern Tooling

| Aspect | GEMMA (2012) | JAMMA (2026) |
|--------|--------------|--------------|
| Package manager | Manual Makefile | uv/pip with lockfile |
| Linting | None | ruff (fast, comprehensive) |
| Formatting | Manual | ruff-format (deterministic) |
| Testing | Ad-hoc | pytest with property-based tests |
| CI | Travis (deprecated) | GitHub Actions |
| Documentation | LaTeX manual | Markdown with live examples |

### Code Quality

```python
# Type hints for all public APIs
def run_lmm_association_numpy(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray,
    snp_info: list[dict],
    *,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    lmm_mode: int = 1,  # 1=Wald, 2=LRT, 3=Score, 4=All
) -> LmmRunResult: ...

# Dataclasses for structured returns
@dataclass
class AssocResult:
    chr: str
    rs: str
    ps: int
    n_miss: int
    allele1: str
    allele0: str
    af: float
    beta: float
    se: float
    logl_H1: float | None = None   # Wald/All
    l_remle: float | None = None    # Wald/All
    p_wald: float | None = None     # Wald/All
    p_score: float | None = None    # Score/All
    l_mle: float | None = None      # LRT/All
    p_lrt: float | None = None      # LRT/All
```

### Testing Philosophy

- **Property-based tests**: Hypothesis generates edge cases automatically
- **Tier system**: Fast unit tests (CI) vs slow validation tests (nightly)
- **GEMMA fixtures**: Automated comparison against reference implementation
- **Randomized test order**: Catches hidden test dependencies

### Dependency Management

```toml
# pyproject.toml - single source of truth
[project]
dependencies = [
    "bed-reader>=1.0.0",
    "numpy>=2.0.0",
    "psutil>=5.9.0",
    "threadpoolctl>=3.0.0",
    "click>=8.0.0",
    "loguru>=0.7.0",
    "progressbar2>=4.2.0",
]

[tool.ruff]
line-length = 88
```

### Observable Operations

Every long-running operation can be monitored:

```python
# Progress logging (streaming runners)
results = run_lmm_association_numpy_streaming(
    bed_path, phenotypes, kinship,
    show_progress=True,  # Progress bar + RSS logging
)

# Memory estimation before commitment
from jamma.core.memory import estimate_lmm_memory

estimate = estimate_lmm_memory(n_samples, n_snps)
if not estimate.sufficient:
    raise MemoryError(f"Need {estimate.total_gb:.1f}GB")
```

---

## When to Use GEMMA Instead

JAMMA is not always the right choice:

1. **Multivariate LMM (mvLMM)**: GEMMA-only for now (planned for a future JAMMA release)
2. **Extreme validation requirements**: When you need bit-exact GEMMA output
3. **Air-gapped systems**: Where pip install isn't an option

---

## Summary

| Concern | GEMMA | JAMMA |
|---------|-------|-------|
| Crashes at scale | Silent OOM | Pre-flight checks |
| Large samples | Manual tuning | Automatic streaming (>100k requires ILP64) |
| Speed | Baseline | Up to 10x faster |
| Installation | C++ build | pip install |
| Errors | Cryptic | Actionable |
| Results | Reference | Equivalent |

JAMMA is GEMMA reimagined for modern Python workflows: same statistical rigor, better developer experience.
