# Why JAMMA? Key Differentiators from GEMMA

JAMMA delivers the same statistical results as GEMMA while solving practical problems that make GEMMA difficult to use at scale.

## Quick Comparison

| Feature | GEMMA | JAMMA |
|---------|-------|-------|
| **OOM Handling** | Silent crash (OS kill) | Pre-flight check with clear error |
| **Large-Scale** | Requires manual tuning | Streaming I/O, pre-flight memory checks (>100k requires ILP64) |
| **Speed** | 1x baseline | 4-7x faster (JAX) |
| **Installation** | C++ compilation required | `pip install jamma` |
| **Error Messages** | Segfault or cryptic | Clear, actionable |
| **Numerical Results** | Reference | Equivalent ([proof](EQUIVALENCE.md)) |

---

## 1. Memory Safety: Fail Fast, Not Silent Crash

### The GEMMA Problem

GEMMA loads everything into memory and lets the OS handle failure:

```
$ gemma -bfile large_study -gk 1
# ... runs for 20 minutes ...
Killed
```

No warning. No error message. Just `Killed` from the OOM killer. You've lost 20 minutes of compute time and have no idea why.

### The JAMMA Solution

JAMMA checks memory requirements BEFORE allocation:

```
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

---

## 2. Scale: Large Samples Without Manual Tuning

### The GEMMA Problem

GEMMA requires the full n×p genotype matrix in memory. For 90k samples × 90k SNPs:

- Genotype matrix: ~32 GB
- Kinship matrix: ~65 GB
- Eigendecomposition workspace: ~130 GB peak

Studies over 100k samples require ILP64 BLAS and 512 GB+ RAM due to O(n³) eigendecomposition memory.

### The JAMMA Solution

JAMMA streams data from disk, never materializing the full matrix:

```python
# Kinship computed in chunks - never loads full genotype matrix
kinship = compute_kinship_streaming("large_study", chunk_size=10000)

# LMM also streams - only kinship (n²) kept in memory
results = run_lmm_association_streaming(
    "large_study", phenotypes, kinship, chunk_size=5000
)
```

**Memory profile:**
- Peak is eigendecomposition: n² × 8 bytes × ~2 (K + workspace)
- Genotype chunks: chunk_size × n × 8 bytes (transient)
- Results written incrementally to disk (no accumulation)

---

## 3. Speed: JAX Acceleration

### Benchmark (mouse_hs1940: 1,940 samples × 12,226 SNPs)

| Operation | GEMMA | JAMMA | Speedup |
|-----------|-------|-------|---------|
| Kinship (`-gk 1`) | 6.5s | 0.9s | **7.1x** |
| LMM (`-lmm 1`) | 19.5s | 4.7s | **4.2x** |
| **Total** | 26.0s | 5.6s | **4.6x** |

### Why Faster?

The key insight: **GEMMA loops over SNPs sequentially; JAMMA processes all SNPs in parallel.**

| Aspect | GEMMA | JAMMA |
| ------ | ----- | ----- |
| SNP loop | Sequential C++ `for` loop | Batch parallel via `jax.vmap` |
| Per-SNP overhead | Function call + memory allocation | Zero (fused kernel) |
| BLAS utilization | Many small matmuls | Few large batched matmuls |
| Memory access | Row-by-row, cache-unfriendly | Contiguous, cache-optimized |

**Detailed breakdown:**

1. **Batch vectorization**: JAMMA uses `jax.vmap` to process all SNPs as a single batched operation. GEMMA's C++ loop processes one SNP at a time—even with multithreaded BLAS for individual matrix operations, the outer loop is serial.

2. **JIT fusion**: JAX's XLA compiler fuses the entire Pab → beta → Wald chain into one compiled kernel. GEMMA makes separate BLAS calls with memory round-trips between each step.

3. **Shared array residency**: Eigenvalues and eigenvectors stay on-device (CPU cache or GPU memory) throughout the SNP loop. No repeated loading from RAM.

4. **Efficient Pab computation**: The cumulative Uab/Pab structure is computed once per covariate set, then broadcast across SNPs.

The "C++ vs Python" framing is misleading—JAX compiles to XLA which generates machine code comparable to optimized C++. The real difference is algorithm design: **data-parallel vs sequential-with-parallel-primitives**.

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

That's it. Pure Python with JAX handles the numerical heavy lifting.

---

## 5. Error Handling: Clear, Not Cryptic

### GEMMA Errors

```
Segmentation fault (core dumped)
```

or

```
ERROR: error! number of columns in the kinship matrix
```

### JAMMA Errors

```
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
| Lambda (REML) | < 5e-5 relative | Brent tolerance propagation |
| Significance calls | 100% agreement | All thresholds (0.05, 0.01, 5e-8) |
| Effect directions | 100% agreement | Sign of beta |
| SNP rankings | Identical | Spearman correlation = 1.0 |

See [EQUIVALENCE.md](EQUIVALENCE.md) for the formal error propagation analysis.

The goal is a **drop-in replacement**: same CLI, same output format, same scientific results.

---

## 7. Modern Python Ecosystem

### Debugging & Profiling

```python
# RSS logging at workflow boundaries
from jamma.utils.logging import log_rss_memory

log_rss_memory("kinship", "before")  # Logs current RSS in GB
kinship = compute_kinship(genotypes)
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
def run_lmm_association(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray,
    snp_info: list[SnpInfo],
    *,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
) -> list[AssocResult]: ...

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
    logl_H1: float
    l_remle: float
    p_wald: float
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
    "jax>=0.8.0",
    "numpy>=2.0,<2.4",
    "bed-reader>=1.0.0",
]

[tool.pytest.ini_options]
addopts = "-n auto"  # Parallel by default

[tool.ruff]
line-length = 88
```

### Observable Operations

Every long-running operation can be monitored:

```python
# Progress logging (streaming runners)
results = run_lmm_association_streaming(
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

1. **Multivariate LMM (mvLMM)**: GEMMA-only for now (planned for JAMMA v2)
2. **Extreme validation requirements**: When you need bit-exact GEMMA output
3. **Air-gapped systems**: Where pip install isn't an option

---

## Summary

| Concern | GEMMA | JAMMA |
|---------|-------|-------|
| Crashes at scale | Silent OOM | Pre-flight checks |
| Large samples | Manual tuning | Automatic streaming (>100k requires ILP64) |
| Speed | Baseline | 4-7x faster |
| Installation | C++ build | pip install |
| Errors | Cryptic | Actionable |
| Results | Reference | Equivalent |

JAMMA is GEMMA reimagined for modern Python workflows: same statistical rigor, better developer experience.
