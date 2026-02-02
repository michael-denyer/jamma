# Why JAMMA? Key Differentiators from GEMMA

JAMMA delivers the same statistical results as GEMMA while solving practical problems that make GEMMA difficult to use at scale.

## Quick Comparison

| Feature | GEMMA | JAMMA |
|---------|-------|-------|
| **OOM Handling** | Silent crash (OS kill) | Pre-flight check with clear error |
| **200k Samples** | Requires manual tuning | Works out of the box (streaming) |
| **Speed** | 1x baseline | 4-7x faster (JAX) |
| **Installation** | C++ compilation required | `pip install jamma` |
| **Error Messages** | Segfault or cryptic | Clear, actionable |
| **Numerical Results** | Reference | Equivalent (beta 1e-6, p 1e-8) |

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

## 2. Scale: 200k+ Samples Without Manual Tuning

### The GEMMA Problem

GEMMA requires the full n×p genotype matrix in memory. For 200k samples × 95k SNPs:

- Genotype matrix: ~76 GB
- Kinship matrix: ~320 GB
- Eigendecomposition workspace: ~640 GB peak

Most cloud VMs can't handle this without careful manual tuning.

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

1. **XLA JIT compilation**: Matrix operations compiled to optimized machine code
2. **Vectorized SNP processing**: Batch SNPs through GPU-friendly operations
3. **Shared array residency**: Eigenvalues, eigenvectors kept on device throughout
4. **Efficient Pab computation**: Cumulative structure avoids redundant calculation

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
| Beta coefficients | < 1e-6 relative | GEMMA fixture comparison |
| P-values | < 1e-8 relative | GEMMA fixture comparison |
| Significance calls | 100% agreement | All thresholds (0.05, 0.01, 5e-8) |
| Effect directions | 100% agreement | Sign of beta |
| SNP rankings | Identical | Spearman correlation = 1.0 |

The goal is a **drop-in replacement**: same CLI, same output format, same scientific results.

---

## 7. Modern Python Ecosystem

### Debugging & Profiling

```python
# RSS logging at workflow boundaries
from jamma.utils.logging import log_rss_memory

log_rss_memory("before kinship")  # Logs current RSS in GB
kinship = compute_kinship(genotypes)
log_rss_memory("after kinship")
```

### Memory Estimation API

```python
from jamma.core.memory import estimate_workflow_memory, estimate_lmm_memory

# Before starting a big job
estimate = estimate_workflow_memory(n_samples=200_000, n_snps=95_000)
print(f"Peak: {estimate.peak_gb:.1f} GB")
print(f"Available: {estimate.available_gb:.1f} GB")
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
# Progress logging
results = run_lmm_association(
    genotypes, phenotypes, kinship, snp_info,
    show_progress=True,  # RSS logging at boundaries
)

# Memory estimation before commitment
estimate = estimate_lmm_memory(n_samples, n_snps, chunk_size)
if not estimate.sufficient:
    raise MemoryError(estimate.error_message)
```

---

## When to Use GEMMA Instead

JAMMA is not always the right choice:

1. **Multivariate LMM (mvLMM)**: GEMMA-only for now (planned for JAMMA v2)
2. **Score/LRT tests**: GEMMA-only for now (planned for JAMMA v1.1)
3. **Extreme validation requirements**: When you need bit-exact GEMMA output
4. **Air-gapped systems**: Where pip install isn't an option

---

## Summary

| Concern | GEMMA | JAMMA |
|---------|-------|-------|
| Crashes at scale | Silent OOM | Pre-flight checks |
| 200k samples | Manual tuning | Automatic streaming |
| Speed | Baseline | 4-7x faster |
| Installation | C++ build | pip install |
| Errors | Cryptic | Actionable |
| Results | Reference | Equivalent |

JAMMA is GEMMA reimagined for modern Python workflows: same statistical rigor, better developer experience.
