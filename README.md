<p align="center">
  <img src="logos/JAMMA_Banner_logo.png" alt="JAMMA" width="600">
</p>

<p align="center">
  <a href="https://github.com/michael-denyer/jamma/actions/workflows/ci.yml"><img src="https://github.com/michael-denyer/jamma/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/jamma/"><img src="https://img.shields.io/pypi/v/jamma.svg" alt="PyPI"></a>
  <a href="https://bioconda.github.io/recipes/jamma/README.html"><img src="https://img.shields.io/conda/vn/bioconda/jamma.svg" alt="Bioconda"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/jax-ml/jax"><img src="https://img.shields.io/badge/JAX-accelerated-9cf.svg" alt="JAX"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-1.26+-013243.svg?logo=numpy" alt="NumPy"></a>
  <a href="https://hypothesis.readthedocs.io/"><img src="https://img.shields.io/badge/tested%20with-Hypothesis-blue.svg" alt="Hypothesis"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPL%203.0-blue.svg" alt="License: GPL-3.0"></a>
</p>

**JAX-Accelerated Mixed Model Association** — A modern Python reimplementation of [GEMMA](https://github.com/genetics-statistics/GEMMA) for genome-wide association studies (GWAS).

- **GEMMA-compatible**: Drop-in replacement with identical CLI flags and output formats
- **Numerical equivalence**: Validated against GEMMA on 85k real samples (91,613 SNPs) — 100% significance agreement, 100% effect direction agreement
- **Fast**: Up to 7x faster than GEMMA on kinship and 4x faster on LMM association
- **Memory-safe**: Pre-flight memory checks prevent OOM crashes before allocation
- **Pure Python**: JAX + NumPy stack, no C++ compilation required

## Installation

```bash
pip install jamma
```

Or with uv:

```bash
uv add jamma
```

## Quick Start

```bash
# Compute kinship matrix (centered relatedness)
jamma -o output gk -bfile data/my_study -gk 1

# Run LMM association (Wald test)
jamma -o results lmm -bfile data/my_study -k output/output.cXX.txt -lmm 1
```

Output files match GEMMA format exactly:

- `output.cXX.txt` — Kinship matrix
- `results.assoc.txt` — Association results (chr, rs, ps, n_miss, allele1, allele0, af, beta, se, logl_H1, l_remle, p_wald)
- `results.log.txt` — Run log

## Python API

### One-call GWAS (recommended)

```python
from jamma import gwas

# Full pipeline: load data → kinship → eigendecomp → LMM → results
result = gwas("data/my_study", kinship_file="data/kinship.cXX.txt")
print(f"Tested {result.n_snps_tested} SNPs in {result.timing['total_s']:.1f}s")

# Compute kinship from scratch and save it
result = gwas("data/my_study", save_kinship=True, output_dir="output")

# With covariates and LRT test
result = gwas("data/my_study", kinship_file="k.txt", covariate_file="covars.txt", lmm_mode=2)
```

### Low-level API

```python
from jamma.io import load_plink_binary
from jamma.kinship import compute_centered_kinship
from jamma.lmm import run_lmm_association_streaming
from jamma.lmm.eigen import eigendecompose_kinship

# Load PLINK data
data = load_plink_binary("data/my_study")

# Compute kinship
kinship = compute_centered_kinship(data.genotypes)

# Eigendecompose for LMM
eigenvalues, eigenvectors = eigendecompose_kinship(kinship)

# Run association (streaming from disk)
results = run_lmm_association_streaming(
    bed_path="data/my_study",
    phenotypes=phenotypes,
    kinship=kinship,
    chunk_size=5000,
)
```

## Memory Safety

Unlike GEMMA, JAMMA includes pre-flight memory checks that prevent out-of-memory crashes:

```python
from jamma.core.memory import estimate_workflow_memory

# Check memory requirements BEFORE loading data
estimate = estimate_workflow_memory(n_samples=200_000, n_snps=95_000)
print(f"Peak memory: {estimate.total_gb:.1f}GB")
print(f"Available: {estimate.available_gb:.1f}GB")
print(f"Sufficient: {estimate.sufficient}")
```

**Key features:**

- Pre-flight checks before large allocations (eigendecomposition, genotype loading)
- RSS memory logging at workflow boundaries
- Incremental result writing (no memory accumulation)
- Safe chunk size defaults with hard caps

GEMMA will silently OOM and get killed by the OS. JAMMA fails fast with clear error messages.

## Performance

Benchmark on mouse_hs1940 (1,940 samples × 12,226 SNPs):

| Operation          | GEMMA | JAMMA | Speedup  |
|--------------------|-------|-------|----------|
| Kinship (`-gk 1`)  | 6.5s  | 0.9s  | **7.1x** |
| LMM (`-lmm 1`)     | 19.5s | 4.7s  | **4.2x** |
| **Total**          | 26.0s | 5.6s  | **4.6x** |

## Supported Features

### Current

- [x] Kinship matrix computation (`-gk 1`)
- [x] Univariate LMM Wald test (`-lmm 1`)
- [x] Likelihood ratio test (`-lmm 2`)
- [x] Score test (`-lmm 3`)
- [x] All tests mode (`-lmm 4`)
- [x] Pre-computed kinship input (`-k`)
- [x] Covariate support (`-c`)
- [x] PLINK binary format (`.bed/.bim/.fam`)
- [x] Streaming I/O for 200k+ samples
- [x] JAX acceleration (CPU/GPU)
- [x] Pre-flight memory checks (fail-fast before OOM)
- [x] RSS memory logging at workflow boundaries
- [x] Incremental result writing

### Planned

- [ ] Multivariate LMM (mvLMM)

## Documentation

- [Why JAMMA?](docs/WHY_JAMMA.md) — Key differentiators from GEMMA
- [User Guide](docs/USER_GUIDE.md) — Installation, usage examples, CLI reference
- [Code Map](docs/CODEMAP.md) — Architecture diagrams and source navigation
- [Equivalence Proof](docs/EQUIVALENCE.md) — Mathematical proofs and empirical validation against GEMMA
- [GEMMA Divergences](docs/GEMMA_DIVERGENCES.md) — Known differences from GEMMA
- [Performance](docs/PERFORMANCE.md) — Bottleneck analysis, scale validation, configuration guide

## Requirements

- Python 3.11+
- JAX 0.8.0+
- NumPy 1.26+

## License

GPL-3.0 (same as GEMMA)
