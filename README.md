<p align="center">
  <a href="https://github.com/michael-denyer/jamma/actions/workflows/ci.yml"><img src="https://github.com/michael-denyer/jamma/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/jamma/"><img src="https://img.shields.io/pypi/v/jamma.svg?color=orange" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://github.com/jax-ml/jax"><img src="https://img.shields.io/badge/JAX-accelerated-7B68EE.svg" alt="JAX"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-2.0+-013243.svg?logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://hypothesis.readthedocs.io/"><img src="https://img.shields.io/badge/tested%20with-Hypothesis-BD1C2B.svg" alt="Hypothesis"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPL%203.0-green.svg" alt="License: GPL-3.0"></a>
  <a href="https://buymeacoffee.com/codenyer"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/michael-denyer/jamma/master/logos/JAMMA_Large_Logo_v2.png" alt="JAMMA" width="500">
</p>

**Fast Mixed Model Association** — A modern Python reimplementation of [GEMMA](https://github.com/genetics-statistics/GEMMA) for genome-wide association studies (GWAS).

- **GEMMA-compatible**: Drop-in replacement with identical CLI flags and output formats
- **Numerical equivalence**: Validated against GEMMA — 100% significance agreement, 100% effect direction agreement
- **Fast**: Up to 11x faster than GEMMA on kinship and 6x faster on LMM association
- **Memory-safe**: Pre-flight memory checks prevent OOM crashes before allocation
- **Cross-platform**: Runs on Linux, macOS, and Windows — NumPy backend works everywhere, JAX backend adds GPU acceleration
- **Pure Python**: NumPy + optional JAX stack, no C++ compilation required
- **Large-scale ready**: Optional [numpy-mkl ILP64](https://github.com/michael-denyer/numpy-mkl) wheels (numpy 2.4.2) for >46k sample eigendecomposition

## Installation

```bash
# Base install (NumPy backend — works on all platforms)
pip install jamma

# With JAX acceleration (Linux, ARM Mac, Windows CPU)
pip install jamma[jax]
```

Or with uv:

```bash
uv add jamma        # NumPy backend
uv add jamma[jax]   # With JAX acceleration
```

### Platform Support

| Platform | `pip install jamma` | `pip install jamma[jax]` | Notes |
|----------|---------------------|--------------------------|-------|
| Linux x86_64 | NumPy backend | JAX (CPU/GPU) | Full support; ILP64 for >46k samples |
| ARM Mac (M1+) | NumPy backend | JAX (CPU) | Full support |
| Intel Mac | NumPy backend | Not available | JAX 0.5.0+ dropped Intel Mac |
| Windows | NumPy backend | JAX (CPU) | JAX CPU via jaxlib wheels |

The backend is auto-detected: if JAX is installed, it's used automatically.
Force a specific backend with `--backend numpy` or `--backend jax`.

## Quick Start

```bash
# Compute kinship matrix (centered relatedness)
jamma -gk 1 -bfile data/my_study -o output

# Run LMM association (Wald test)
jamma -lmm 1 -bfile data/my_study -k output/output.cXX.txt -o results
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

# LOCO analysis (leave-one-chromosome-out)
result = gwas("data/my_study", loco=True)

# Multi-phenotype with eigendecomp reuse
result = gwas("data/my_study", write_eigen=True, phenotype_column=1)
result = gwas("data/my_study", eigenvalue_file="output/result.eigenD.txt",
              eigenvector_file="output/result.eigenU.txt", phenotype_column=2)

# SNP filtering
result = gwas("data/my_study", kinship_file="k.txt", snps_file="snps.txt", hwe=0.001)
```

### Low-level API (JAX backend)

```python
import numpy as np

from jamma.io import load_plink_binary
from jamma.kinship import compute_centered_kinship
from jamma.lmm import run_lmm_association_streaming
from jamma.lmm.eigen import eigendecompose_kinship

# Load PLINK data and phenotypes
data = load_plink_binary("data/my_study")
phenotypes = np.loadtxt("data/my_study.pheno")  # loaded separately from .fam or phenotype file

# Compute kinship and eigendecompose (treat kinship as consumed after this)
kinship = compute_centered_kinship(data.genotypes)
eigenvalues, eigenvectors = eigendecompose_kinship(kinship)

# Run association (streaming from disk)
results, n_tested = run_lmm_association_streaming(
    bed_path="data/my_study",
    phenotypes=phenotypes,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    chunk_size=5000,
)
```

### Low-level API (NumPy backend)

```python
import numpy as np

from jamma.io import load_plink_binary
from jamma.kinship import compute_centered_kinship
from jamma.lmm import run_lmm_association_numpy
from jamma.lmm.eigen import eigendecompose_kinship

data = load_plink_binary("data/my_study")
kinship = compute_centered_kinship(data.genotypes)
eigenvalues, eigenvectors = eigendecompose_kinship(kinship)

results, n_tested = run_lmm_association_numpy(
    genotypes=data.genotypes,
    phenotypes=np.loadtxt("data/my_study.pheno"),
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    lmm_mode=1,
    output_path="output/results.assoc.txt",
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

Benchmark on mouse_hs1940 (1,940 samples × 12,226 SNPs), Apple M2:

| Operation          | GEMMA  | JAMMA | Speedup   |
|--------------------|--------|-------|-----------|
| Kinship (`-gk 1`)  | 26.5s  | 2.4s  | **11.0x** |
| LMM (`-lmm 1`)     | 27.6s  | 4.5s  | **6.1x**  |
| **Total**          | 54.1s  | 6.9s  | **7.8x**  |

## Supported Features

### Current

- [x] Kinship matrix computation — centered (`-gk 1`) and standardized (`-gk 2`)
- [x] Univariate LMM Wald test (`-lmm 1`)
- [x] Likelihood ratio test (`-lmm 2`)
- [x] Score test (`-lmm 3`)
- [x] All tests mode (`-lmm 4`)
- [x] LOCO kinship — leave-one-chromosome-out analysis (`-loco`)
- [x] Eigendecomposition reuse — multi-phenotype workflows (`-d`/`-u`/`-eigen`)
- [x] Phenotype column selection (`-n`)
- [x] SNP subset selection for association and kinship (`-snps`/`-ksnps`)
- [x] HWE QC filtering (`-hwe`)
- [x] Pre-computed kinship input (`-k`)
- [x] Covariate support (`-c`)
- [x] PLINK binary format (`.bed/.bim/.fam`) with input dimension validation
- [x] Large-scale streaming I/O (>100k samples via [numpy-mkl ILP64](https://github.com/michael-denyer/numpy-mkl) — numpy 2.4.2)
- [x] JAX acceleration (CPU/GPU) with automatic CPU device sharding
- [x] XLA profiling traces (`--profile-dir`) for TensorBoard/Perfetto
- [x] Lambda optimization bounds (`-lmin`/`-lmax`)
- [x] Individual weights for kinship (`-widv`)
- [x] Categorical covariates with one-hot encoding (`-cat`)
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
- [Contributing](CONTRIBUTING.md) — Development setup, testing, and PR guidelines
- [Changelog](CHANGELOG.md) — Version history

## Requirements

- Python 3.11+
- NumPy 2.0+
- JAX 0.5.0+ (optional, for GPU acceleration: `pip install jamma[jax]`)

## License

GPL-3.0 (same as GEMMA)
