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
- `-lmm MODE` — Test type: 1 = Wald test (default)
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

## Memory Safety

JAMMA includes pre-flight memory checks that fail fast before OOM instead of crashing silently.

### Pre-flight Checks

By default, JAMMA estimates memory requirements before large allocations:

```bash
# Check memory estimate without running
jamma lmm -bfile data/large_study -k kinship.cXX.txt --mem-budget 64
```

If the estimate exceeds available memory, you'll get a clear error:

```
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
from jamma.lmm.memory import estimate_lmm_memory

estimate = estimate_lmm_memory(
    n_samples=200_000,
    n_snps=95_000,
    n_covariates=5,
    has_kinship=True,  # Using pre-computed kinship
)

print(f"Peak memory: {estimate.peak_gb:.1f}GB")
print(f"Eigendecomp: {estimate.eigendecomp_gb:.1f}GB")
print(f"Association: {estimate.association_gb:.1f}GB")
```

## Troubleshooting

### JAX not using GPU

Check JAX backend:

```python
import jax
print(jax.devices())  # Should show GPU if available
```

### Memory errors on large datasets

JAMMA's pre-flight checks should catch most OOM issues. If you still hit memory limits:

1. **Check the estimate first:**
   ```bash
   jamma lmm ... --mem-budget 999  # Will show actual requirement
   ```

2. **Use streaming for very large datasets:**
   ```python
   from jamma.lmm import run_lmm_association_streaming

   results = run_lmm_association_streaming(
       bed_path="data/large_study",
       phenotypes=phenotypes,
       kinship=kinship,
       chunk_size=5000,  # Process 5000 SNPs at a time
       output_path="results.assoc.txt",  # Write incrementally
   )
   ```

3. **Reduce chunk size** if memory is tight:
   ```bash
   # Smaller chunks use less memory but are slower
   jamma lmm ... --chunk-size 1000
   ```

### Results differ from GEMMA

Small numerical differences (< 1e-5) are expected due to different optimization algorithms. Scientific conclusions (significance, rankings) should be identical. If you see larger differences, please open an issue.
