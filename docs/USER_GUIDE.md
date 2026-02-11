# JAMMA User Guide

## Installation

### From PyPI

```bash
pip install jamma
```

### From Source

```bash
git clone https://github.com/michael-denyer/jamma.git
cd jamma
uv sync
```

### GPU Support

For GPU acceleration, install JAX with CUDA support:

```bash
pip install jax[cuda12]
```

## Input Data Format

JAMMA uses PLINK binary format (`.bed`, `.bim`, `.fam` files):

```text
my_study.bed   # Binary genotype data
my_study.bim   # SNP information
my_study.fam   # Sample information
```

## Commands

### Kinship Matrix Computation (`gk`)

Compute genetic relatedness matrix from genotype data:

```bash
jamma -o kinship -outdir output gk -bfile data/my_study -gk 1
```

**Options:**

- `-bfile PATH` — PLINK binary file prefix (required)
- `-gk MODE` — Kinship type: 1 = centered (default), 2 = standardized
- `-maf FLOAT` — MAF threshold (default: 0.0, no filter)
- `-miss FLOAT` — Missing rate threshold (default: 1.0, no filter)
- `-o PREFIX` — Output file prefix
- `-outdir DIR` — Output directory

**Note:** Monomorphic SNPs (variance = 0) are always filtered to match GEMMA behavior.

**Output:**

- `output/kinship.cXX.txt` — Kinship matrix (GEMMA format)
- `output/kinship.log.txt` — Run log

### LMM Association Testing (`lmm`)

Run univariate linear mixed model association tests:

```bash
jamma -o assoc -outdir output lmm \
  -bfile data/my_study \
  -k output/kinship.cXX.txt \
  -lmm 1
```

**With covariates:**

```bash
jamma -o assoc -outdir output lmm \
  -bfile data/my_study \
  -k output/kinship.cXX.txt \
  -c covariates.txt \
  -lmm 1
```

**Options:**

- `-bfile PATH` — PLINK binary file prefix (required)
- `-k PATH` — Kinship matrix file (required)
- `-lmm MODE` — Test type: 1 = Wald (default), 2 = LRT, 3 = Score, 4 = All
- `-c PATH` — Covariate file (GEMMA format: whitespace-delimited, first column should be intercept)
- `-maf FLOAT` — MAF threshold (default: 0.01)
- `-miss FLOAT` — Missing rate threshold (default: 0.05)
- `--mem-budget GB` — Memory budget in GB (default: available - 10%)
- `--no-check-memory` — Disable pre-flight memory checks

**Note:** Monomorphic SNPs (variance = 0) are always filtered to match GEMMA behavior.

**Output:**

- `output/assoc.assoc.txt` — Association results
- `output/assoc.log.txt` — Run log

## Output Format

### Association Results (`.assoc.txt`)

Tab-separated file with columns:

| Column | Description |
| ------ | ----------- |
| `chr` | Chromosome |
| `rs` | SNP identifier |
| `ps` | Position |
| `n_miss` | Number of missing genotypes |
| `allele1` | Effect allele |
| `allele0` | Reference allele |
| `af` | Allele frequency |
| `beta` | Effect size |
| `se` | Standard error |
| `logl_H1` | Log-likelihood under H1 |
| `l_remle` | REML estimate of lambda |
| `p_wald` | Wald test p-value |

### Kinship Matrix (`.cXX.txt`)

Space-separated N×N matrix where N is the number of samples. Compatible with GEMMA format.

## Python API

### One-call GWAS (recommended)

The simplest way to run a complete GWAS from Python:

```python
from jamma import gwas

# With pre-computed kinship
result = gwas("data/my_study", kinship_file="data/kinship.cXX.txt")
print(f"Tested {result.n_snps_tested} SNPs in {result.timing['total_s']:.1f}s")

# Compute kinship from scratch, save it for reuse
result = gwas("data/my_study", save_kinship=True, output_dir="output")

# With covariates
result = gwas(
    "data/my_study",
    kinship_file="k.txt",
    covariate_file="covars.txt",
    lmm_mode=2,  # LRT test
)
```

`gwas()` handles the full pipeline: load data, compute or load kinship,
eigendecompose, run LMM association, and write results. Returns a `GWASResult`
with timing breakdown and summary stats.

### Low-level API

For more control, use the component functions directly:

#### Kinship Computation

```python
from jamma.io import load_plink_binary
from jamma.kinship import compute_centered_kinship

# Load genotypes
data = load_plink_binary("data/my_study")

# Compute kinship
K = compute_centered_kinship(data.genotypes)
```

#### LMM Association

```python
from jamma.lmm import run_lmm_association_streaming

# Streaming runner (genotypes from disk, never loads full matrix)
results = run_lmm_association_streaming(
    bed_path="data/my_study",
    phenotypes=phenotypes,
    kinship=K,
    chunk_size=10_000,
)
```

## Large-Scale Eigendecomposition (>46k samples)

JAMMA's LMM requires eigendecomposition of the N×N kinship matrix. The default
numpy stack uses LP64 BLAS (32-bit integers), which overflows at ~46k samples
(46k × 46k = 2.1 billion elements > int32 max).

### NumPy with MKL ILP64 (Linux)

For datasets with >46k samples on Linux, install numpy built against
Intel MKL with 64-bit integer support (ILP64). Pre-built wheels are
available for numpy 2.4.2 (Python 3.11–3.14, Linux and Windows):

```bash
# Install ILP64 numpy 2.4.2 wheel
pip install numpy \
  --extra-index-url https://michael-denyer.github.io/numpy-mkl \
  --force-reinstall --upgrade

# CRITICAL: Install jamma without dependencies to avoid overwriting ILP64 numpy
pip install jamma --no-deps
```

> **Note:** scipy does not support ILP64 — it hardcodes `ilp64=False` in
> `get_lapack_funcs()` ([scipy#23351](https://github.com/scipy/scipy/issues/23351)).
> JAMMA uses `numpy.linalg.eigh` which correctly uses ILP64 when numpy is built
> with ILP64 MKL.

**Verify ILP64 is active:**

```python
import numpy as np
cfg = np.show_config(mode="dicts")
blas = cfg["Build Dependencies"]["blas"]
print(f"BLAS: {blas['name']}")           # Should show: mkl
print(f"Symbol suffix: {blas.get('symbol suffix', 'none')}")  # Should show: _64
```

**Testing the ILP64 build:**

```bash
# Run JAMMA's validation suite to confirm equivalence
uv run pytest tests/test_kinship_validation.py tests/test_lmm_validation.py -v

# Quick eigendecomposition sanity check
python -c "
import numpy as np
n = 50000  # Exceeds LP64 limit
K = np.random.randn(n, 100) @ np.random.randn(100, n)
K = (K + K.T) / 2
vals, vecs = np.linalg.eigh(K)
print(f'Eigendecomposition of {n}x{n} matrix: OK')
print(f'Top eigenvalue: {vals[-1]:.2f}')
"
```

### MKL License Note

MKL is distributed under the [Intel Simplified Software License (ISSL)](https://www.intel.com/content/www/us/en/developer/articles/tool/onemkl-license-faq.html),
which permits free redistribution with no royalty fees. However, the ISSL is **not
an open source license** — it restricts reverse engineering and decompilation, and
is not GPL-compatible.

This does not affect JAMMA itself (GPL-3.0). JAMMA calls numpy APIs (BSD
licensed) and has no direct dependency on MKL. Users who install MKL-backed numpy
wheels do so as a separate, optional runtime choice. Users requiring a pure
GPL/FOSS stack can use standard numpy with OpenBLAS (the default), which works
for datasets up to ~46k samples.

### Alternative Approaches for >46k Samples

If MKL ILP64 is not available:

1. **GPU eigendecomposition**: cuSOLVER on NVIDIA GPUs uses different integer interfaces
2. **Approximate methods**: Randomized SVD or truncated eigendecomposition
3. **Sample subsetting**: Use ~40k representative samples for kinship computation

## Performance Tips

1. **Use JAX runner** for large datasets (>1000 samples)
2. **Enable GPU** with `use_gpu=True` if available
3. **Batch processing**: JAMMA automatically batches kinship computation
4. **Memory**: For very large datasets, consider sample subsetting

## Validation

JAMMA results match GEMMA within validated tolerances:

- Kinship matrices: < 1e-8 relative difference
- P-values (Wald/Score): < 1e-4 relative difference
- P-values (LRT): < 5e-3 relative difference (MLE subtraction amplification)
- Beta coefficients: < 1e-2 relative difference (lambda propagation)
- Log-likelihood (REML): < 1e-6 relative difference
- Log-likelihood (MLE/logl_H1): < 5e-3 relative difference on real data
- Significance calls: 100% agreement at all thresholds
- Effect directions and SNP rankings: identical

### Optimizer Divergence on Weak-Signal SNPs

GEMMA uses Brent's method for lambda optimization; JAMMA uses grid search followed by
golden section refinement. Both converge to within 1e-5 of the true optimum for
strong-signal SNPs. However, weak-signal SNPs — where the optimization landscape is
flat and lambda converges near the lower bound (1e-5) — can produce slightly different
optima between the two methods. This propagates to per-SNP MLE log-likelihood (logl_H1)
with up to ~0.14% relative difference on real datasets (observed on mouse_hs1940 at
SNP index 596 of 10768). The quantities that drive scientific conclusions (p-values,
effect directions, significance rankings) are unaffected.

See [EQUIVALENCE.md](EQUIVALENCE.md) for empirical validation and formal error
propagation analysis.

## Memory Safety

JAMMA includes pre-flight memory checks that fail fast before OOM instead of crashing silently.

### Pre-flight Checks

By default, JAMMA estimates memory requirements before large allocations:

```bash
# Check memory estimate without running
jamma lmm -bfile data/large_study -k kinship.cXX.txt --mem-budget 64
```

If the estimate exceeds available memory, you'll get a clear error:

```text
MemoryError: LMM requires ~128.5GB but only 64.0GB available
  Breakdown: kinship=74.5GB, eigendecomp=37.0GB, association=17.0GB
```

### Controlling Memory Behavior

```bash
# Set explicit memory budget (GB)
jamma lmm ... --mem-budget 128

# Disable checks (use at your own risk)
jamma lmm ... --no-check-memory
```

### Programmatic Memory Estimation

```python
from jamma.core.memory import estimate_workflow_memory, estimate_lmm_memory

# Full pipeline estimate (before starting anything)
full = estimate_workflow_memory(n_samples=200_000, n_snps=95_000)
print(f"Full pipeline peak: {full.total_gb:.1f}GB")
print(f"Eigendecomp workspace: {full.eigendecomp_workspace_gb:.1f}GB")
print(f"Available: {full.available_gb:.1f}GB")
print(f"Sufficient: {full.sufficient}")

# LMM-only estimate (after eigendecomp is done, kinship freed)
lmm = estimate_lmm_memory(n_samples=200_000, n_snps=95_000)
print(f"LMM phase: {lmm.total_gb:.1f}GB")
```

## Troubleshooting

### JAX not using GPU

Check JAX backend:

```python
import jax
print(jax.devices())  # Should show GPU if available
```

### Memory errors on large datasets

JAMMA runs a pre-flight memory check before kinship and eigendecomposition. The
check estimates peak memory (dominated by eigendecomposition: K + U + workspace)
and applies a 50% safety margin to account for JAX temporary arrays.

**Approximate sample limits by machine size:**

| Machine RAM | ~Available | Max samples |
|-------------|------------|-------------|
| 512GB       | 490GB      | ~142k       |
| 256GB       | 240GB      | ~100k       |
| 128GB       | 120GB      | ~70k        |
| 64GB        | 58GB       | ~49k        |
| 32GB        | 28GB       | ~34k        |
| 16GB        | 14GB       | ~24k        |

These limits assume the streaming pipeline (CLI default). Actual limits depend on
available memory at runtime — other processes reduce headroom.

**If the pre-flight check rejects your run:**

1. **Free memory** from other processes or previous runs
2. **Use `--no-check-memory`** to bypass the check (at your own risk):

   ```bash
   jamma gk --no-check-memory -g data/study
   jamma lmm --no-check-memory -g data/study -p pheno.txt -k kinship.txt
   ```

3. **Reduce chunk size** for lower per-batch memory:

   ```bash
   jamma lmm ... --chunk-size 1000
   ```

### Results differ from GEMMA

Small numerical differences (< 1e-5) are expected due to different optimization algorithms. Scientific conclusions (significance, rankings) should be identical. If you see larger differences, please open an issue.
