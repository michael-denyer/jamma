# JAMMA

[![CI](https://github.com/michael-denyer/jamma/actions/workflows/ci.yml/badge.svg)](https://github.com/michael-denyer/jamma/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/jamma.svg)](https://pypi.org/project/jamma/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**JAX-Accelerated Mixed Model Association** — A modern Python reimplementation of [GEMMA](https://github.com/genetics-statistics/GEMMA) for genome-wide association studies (GWAS).

## Features

- **GEMMA-compatible**: Drop-in replacement with identical CLI flags and output formats
- **Numerical equivalence**: Results match GEMMA — identical significance calls, effect directions, and SNP rankings
- **Fast**: 7x faster than GEMMA on kinship, 4x faster on LMM association
- **Scalable**: Streaming I/O handles 200k+ samples without OOM
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

```python
from jamma.io import load_plink_binary
from jamma.kinship import compute_centered_kinship
from jamma.lmm import run_lmm_association, eigendecompose_kinship

# Load PLINK data
data = load_plink_binary("data/my_study")

# Compute kinship
kinship = compute_centered_kinship(data.genotypes)

# Eigendecompose for LMM
eigenvalues, eigenvectors = eigendecompose_kinship(kinship)

# Run association
results = run_lmm_association(
    genotypes=data.genotypes,
    phenotypes=phenotypes,
    kinship=kinship,
    snp_info=snp_info,
)
```

## Streaming for Large Datasets

For datasets that don't fit in memory:

```python
from jamma.kinship import compute_kinship_streaming
from jamma.lmm import run_lmm_association_streaming

# Kinship: streams genotypes from disk in chunks
kinship = compute_kinship_streaming("data/large_study", chunk_size=10000)

# LMM: streams SNPs for association testing
results = run_lmm_association_streaming(
    bed_path="data/large_study",
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
print(f"Peak memory: {estimate.peak_gb:.1f}GB")
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
- [Mathematical Equivalence](docs/MATHEMATICAL_EQUIVALENCE.md) — Validation against GEMMA
- [Formal Proof](docs/FORMAL_PROOF.md) — Mathematical proof of algorithmic equivalence
- [GEMMA Divergences](docs/GEMMA_DIVERGENCES.md) — Known differences from GEMMA

## Requirements

- Python 3.11+
- JAX 0.8.0+
- NumPy 2.0+ (< 2.4)

## License

GPL-3.0 (same as GEMMA)
