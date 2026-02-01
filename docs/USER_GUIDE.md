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

```
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
- `-o PREFIX` — Output file prefix
- `-outdir DIR` — Output directory

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

**Options:**
- `-bfile PATH` — PLINK binary file prefix (required)
- `-k PATH` — Kinship matrix file (required)
- `-lmm MODE` — Test type: 1 = Wald test (default)
- `-maf FLOAT` — MAF threshold (default: 0.01)
- `-miss FLOAT` — Missing rate threshold (default: 0.05)

**Output:**
- `output/assoc.assoc.txt` — Association results
- `output/assoc.log.txt` — Run log

## Output Format

### Association Results (`.assoc.txt`)

Tab-separated file with columns:

| Column | Description |
|--------|-------------|
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

### Kinship Computation

```python
from jamma.io import load_plink_binary
from jamma.kinship import compute_centered_kinship

# Load genotypes
data = load_plink_binary("data/my_study")

# Compute kinship
K = compute_centered_kinship(data.genotypes)
```

### LMM Association

```python
from jamma.lmm import run_lmm_association
from jamma.lmm.runner_jax import run_lmm_association_jax

# NumPy runner (supports covariates)
results = run_lmm_association(
    genotypes=data.genotypes,
    phenotypes=phenotypes,
    kinship=K,
    snp_info=snp_info,
)

# JAX runner (faster, intercept-only)
results = run_lmm_association_jax(
    genotypes=data.genotypes,
    phenotypes=phenotypes,
    kinship=K,
    snp_info=snp_info,
    use_gpu=True,  # Enable GPU acceleration
)
```

## Performance Tips

1. **Use JAX runner** for large datasets (>1000 samples)
2. **Enable GPU** with `use_gpu=True` if available
3. **Batch processing**: JAMMA automatically batches kinship computation
4. **Memory**: For very large datasets, consider sample subsetting

## Validation

JAMMA results match GEMMA within floating-point tolerance:

- Kinship matrices: < 1e-8 relative difference
- P-values: < 5e-5 relative difference
- Significance calls: 100% agreement at all thresholds

See [MATHEMATICAL_EQUIVALENCE.md](MATHEMATICAL_EQUIVALENCE.md) for details.

## Troubleshooting

### JAX not using GPU

Check JAX backend:

```python
import jax
print(jax.devices())  # Should show GPU if available
```

### Memory errors on large datasets

Reduce batch size or use sample subsetting:

```python
# Process in chunks
for chunk in np.array_split(range(n_snps), 10):
    results.extend(run_lmm_association(..., snp_indices=chunk))
```

### Results differ from GEMMA

Small numerical differences (< 1e-5) are expected due to different optimization algorithms. Scientific conclusions (significance, rankings) should be identical. If you see larger differences, please open an issue.
