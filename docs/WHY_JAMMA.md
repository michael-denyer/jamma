# Why JAMMA? Key Differentiators from GEMMA

JAMMA delivers the same statistical results as GEMMA while solving practical problems that make GEMMA difficult to use at scale.

## Quick Comparison

| Feature | GEMMA | JAMMA |
|---------|-------|-------|
| **OOM Handling** | Silent crash (OS kill) | Pre-flight check with clear error |
| **Large-Scale** | Requires manual tuning | Streaming I/O, pre-flight memory checks (>100k requires ILP64) |
| **Speed** | 1x baseline | See [aligned benchmarks](PERFORMANCE.md) for measured workflow timings |
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
Error: Insufficient memory for kinship accumulation (peak: 372.0GB). Need 372.0GB (+10.0GB margin = 382.0GB), but only 256.0GB available. Use --no-check-memory to override, or use a machine with more RAM.
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
dataset = GenotypeDataset.open_plink(Path("large_study"))
kinship = compute_kinship_streaming(dataset, chunk_size=10000)

# LMM also streams - only kinship (n^2) kept in memory
results = run_lmm_association_numpy_streaming(
    dataset, phenotypes, kinship, chunk_size=5000
)
```

**Memory profile:**

- Peak is eigendecomposition: n^2 x 8 bytes x ~2 (K + workspace)
- Genotype chunks: chunk_size x n x 8 bytes (transient)
- Results written incrementally to disk (no accumulation)

---

## 3. Speed: C Extension Acceleration

### Measured workflows

The [performance report](PERFORMANCE.md) compares fresh processes on the same
inputs and checks the resulting SNP sets and numerical results. It reports
standalone kinship, association with precomputed kinship, full GWAS and LOCO.
Full GWAS lets JAMMA retain kinship in memory, so its avoided intermediate I/O
counts toward the measured benefit.

The older CLI-versus-preloaded-runner speedup figures have been withdrawn.
The historical 125k timings used different BLAS builds and have not been rerun
under the aligned protocol; they are not a current speedup claim.

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
Error: Covariate file row 15, column 3: cannot parse 'n/a' as numeric (use 'NA' for missing)

Error: Estimated memory (45.2GB) for numpy-streaming exceeds budget (32.0GB). Use --no-check-memory to override.
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
from jamma.core.memory_snapshot import log_memory_snapshot

log_memory_snapshot("kinship:before")  # Logs RSS + free RAM
kinship = compute_kinship_streaming(GenotypeDataset.open_plink(bfile))
log_memory_snapshot("kinship:after")
```

### Memory Estimation API

```python
from jamma.core.memory import available_ram_gb, fits
from jamma.genotype.dataset import GenotypeEncoding
from jamma.lmm.association_plan import plan_association

# Before starting a big job
quote = plan_association(
    200_000, 95_000, backend="numpy-streaming",
    genotype_encoding=GenotypeEncoding.HARD_CALLS,
).price(eigen=None)
print(f"Peak: {quote.total_peak_gb:.1f}GB")
print(f"Available: {available_ram_gb():.1f}GB")
print(f"Will fit: {fits(quote.total_peak_gb, available_ram_gb())}")
```

### Type Safety

Full type annotations throughout, so IDE autocomplete works. [pyrefly](https://pyrefly.org)
type-checks `src`, `tests`, and `scripts` on every commit and in CI, and the gate is
absolute: the project sits at zero errors with no baseline file to hide behind.

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
    kinship: np.ndarray | None,
    snp_info: Sequence[SnpInfoRecord] | SnpMeta,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    config: LmmConfig = DEFAULT_LMM_CONFIG,  # lmm_mode, maf/miss thresholds
    output_path: Path | None = None,
    hwe_threshold: float = 0.0,
    max_chunk_size: int | None = None,
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
    beta: float = float("nan")
    se: float = float("nan")
    logl_H1: float | None = None   # Wald/LRT/All
    l_remle: float | None = None    # Wald/All
    p_wald: float | None = None     # Wald/All
    p_score: float | None = None    # Score/All
    l_mle: float | None = None      # LRT/All
    p_lrt: float | None = None      # LRT/All
```

### Testing Philosophy

- **Property-based tests**: Hypothesis generates edge cases automatically
- **Tier system**: Fast unit tests (every PR) vs slow validation tests (after merge to master)
- **GEMMA fixtures**: Automated comparison against reference implementation
- **Randomized test order**: Catches hidden test dependencies

### Dependency Management

```toml
# pyproject.toml - single source of truth
[project]
dependencies = [
    "bed-reader>=1.0.0",
    "numpy>=2.4.6",
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
    dataset, phenotypes, kinship,
    config=LmmConfig(show_progress=True),  # Progress bar + RSS logging
)

# Memory estimation before commitment
from jamma.core.memory import available_ram_gb, require
from jamma.genotype.dataset import GenotypeEncoding
from jamma.lmm.association_plan import plan_association

require(
    plan_association(
        n_samples, n_snps, genotype_encoding=GenotypeEncoding.HARD_CALLS
    ).price(eigen=None).association_gb,
    available_ram_gb(),
    "LMM",
)
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
| Speed | Baseline | See [aligned benchmarks](PERFORMANCE.md) |
| Installation | C++ build | pip install |
| Errors | Cryptic | Actionable |
| Results | Reference | Equivalent |

JAMMA is GEMMA reimagined for modern Python workflows: same statistical rigor, better developer experience.
