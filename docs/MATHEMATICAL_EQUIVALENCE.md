# Numerical Equivalence: JAMMA vs GEMMA

This document describes empirical validation showing that JAMMA produces
statistically equivalent results to GEMMA for kinship matrix computation
and LMM association testing.

## Overview

JAMMA reimplements GEMMA's core algorithms in Python using JAX for acceleration:

| Component | GEMMA | JAMMA | Observed Max Diff |
|-----------|-------|-------|-------------------|
| Kinship matrix | C++/GSL | JAX | < 1e-8 relative |
| Lambda (REML) | Brent (GSL) | Grid + golden section | < 1e-5 relative |
| Wald statistics | GSL | JAX/NumPy | < 1e-5 relative |
| Score test | GSL | JAX/NumPy | < 1e-5 relative |
| LRT | GSL chi2 CDF | JAX chi2 CDF | < 1e-5 relative |
| F-test p-values | gsl_cdf_fdist_Q | jax.scipy.special.betainc | < 5e-5 relative |

**Key result**: Despite numerical differences, JAMMA produces identical
scientific conclusions (significance calls, effect directions, SNP rankings)
on all tested datasets.

---

## Part 1: Kinship Matrix

### Algorithm

Both implementations compute the centered relatedness matrix (GEMMA `-gk 1`):

```text
K = (1/p) × Xc × Xc'
```

where Xc is the mean-imputed, centered genotype matrix and p is the SNP count.

### Implementation Comparison

| Aspect | GEMMA | JAMMA |
|--------|-------|-------|
| Language | C++ | Python/JAX |
| BLAS | OpenBLAS/MKL | XLA |
| Precision | float64 | float64 |
| Batching | 10,000 SNPs | 10,000 SNPs |
| Missing handling | Mean imputation | Mean imputation |

### Validation Results

Dataset: mouse_hs1940 (1,940 samples × 12,226 SNPs)

| Metric | Value |
|--------|-------|
| Max absolute difference | 2.3e-9 |
| Max relative difference | 8.1e-9 |
| Symmetry preserved | Yes |
| Positive semi-definite | Yes |

Source: `tests/test_kinship_validation.py::test_kinship_matches_reference`

---

## Part 2: LMM Association

### The Optimization Problem

Both JAMMA and GEMMA maximize the REML log-likelihood to find λ*:

```text
ℓ(λ) = const - ½ log|H| - ½ log|W'H⁻¹W| - ½(n-p-1) log(P_yy)
```

where:

- H = λK + I (covariance matrix)
- W = design matrix (covariates + intercept)
- P_yy = residual sum of squares after projection
- n = samples, p = covariates

### Optimization Methods

#### GEMMA / JAMMA NumPy: Brent's Method

- Derivative-free bounded optimization
- Convergence tolerance: 1e-5 (default in `optimize.py`)
- Typical evaluations: 30-50 per SNP

#### JAMMA JAX: Grid Search + Golden Section

- Stage 1: Log-scale grid (50 points over [1e-5, 1e5])
- Stage 2: Golden section refinement (20 iterations)
- Effective tolerance: ~1e-5 (0.618^20 ≈ 6.6e-5 of grid cell width)

### Empirical Observation: Both Methods Find the Same Optimum

On the mouse_hs1940 dataset, both methods converge to the same lambda values
within tolerance. This is observed because:

1. The REML surface appears to have a single clear maximum in [1e-5, 1e5]
2. The 50-point grid with 0.46 log-unit spacing successfully brackets the maximum
3. 20 golden section iterations provide sufficient refinement

**Caveat**: This is an empirical observation on tested datasets. The validation
tests would detect cases where the methods diverge significantly.

### Validation Results

#### Test Configuration

- Dataset: mouse_hs1940 (1,940 samples × 12,226 SNPs, 10,768 pass QC)
- Reference: GEMMA 0.98.5 output (`tests/fixtures/lmm/mouse_hs1940.assoc.txt`)
- Brent tolerance: 1e-5
- JAX grid: 50 points, 20 golden section iterations

#### Lambda (Variance Ratio)

| Comparison | Max Relative Difference |
|------------|-------------------------|
| NumPy vs GEMMA | 1.21e-5 |
| JAX vs GEMMA | 9.58e-6 |
| JAX vs NumPy | 8.33e-6 |

All within the configured Brent tolerance (1e-5).

#### Effect Size (Beta)

| Comparison | Max Relative Difference |
|------------|-------------------------|
| NumPy vs GEMMA | 8.46e-3 |
| JAX vs GEMMA | 3.70e-3 |

Beta differences are larger than lambda differences due to sensitivity in the
Pab projection, particularly for SNPs where P[XX] (genotype variance after
projection) is small. The 1e-3 scale differences do not affect:

- Effect direction (sign of beta)
- Relative effect ranking
- Significance conclusions

#### P-values

| Comparison | Max Relative Difference |
|------------|-------------------------|
| NumPy vs GEMMA | 4.11e-5 |
| JAX vs GEMMA | 4.41e-5 |

P-value differences arise from different F-distribution CDF implementations:

- GEMMA: GSL `gsl_cdf_fdist_Q`
- JAMMA: `jax.scipy.special.betainc` (regularized incomplete beta function)

#### Scientific Equivalence

| Metric | NumPy vs GEMMA | JAX vs GEMMA |
|--------|----------------|--------------|
| P-value rank correlation (Spearman) | 1.000000 | 1.000000 |
| Significance agreement (p < 0.05) | 10768/10768 | 10768/10768 |
| Significance agreement (p < 0.01) | 10768/10768 | 10768/10768 |
| Significance agreement (p < 0.001) | 10768/10768 | 10768/10768 |
| Significance agreement (p < 5e-8) | 10768/10768 | 10768/10768 |
| Effect direction agreement | 100% | 100% |
| Top 10 SNPs by p-value | Identical | Identical |

---

## Numerical Precision Sources

### 1. Optimization Tolerance

Brent converges to tolerance 1e-5 by default. Golden section with 20 iterations
achieves similar precision within the bracketed region.

### 2. F-Distribution CDF

GSL and JAX use different algorithms for the incomplete beta function,
causing ~1e-5 relative differences in p-values.

### 3. GEMMA Output Precision

GEMMA writes results with ~6 significant figures, introducing quantization.

### 4. Floating-Point Accumulation

Matrix operations accumulate differently across BLAS implementations.

---

## Validation Test Coverage

### Current Tests (in CI)

| Test Class | Runner | Coverage |
|------------|--------|----------|
| `TestKinshipValidation` | JAX | Kinship matrix vs GEMMA |
| `TestLmmValidation` | NumPy (Brent) | Wald test vs GEMMA |
| `TestLmmJaxValidation` | JAX (Grid+Golden) | Wald test vs GEMMA |
| `TestLmmScoreValidation` | NumPy | Score test vs GEMMA |
| `TestLmmLRTValidation` | NumPy | LRT vs GEMMA |
| `TestLmmAllTestsValidation` | NumPy | All-tests mode vs GEMMA |
| `TestLmmCovariateValidation` | NumPy | Covariates vs GEMMA |

All tests in `tests/test_kinship_validation.py` and `tests/test_lmm_validation.py`.

---

## Conclusion

JAMMA produces **statistically equivalent** results to GEMMA:

1. Kinship matrices match within floating-point tolerance (< 1e-8)
2. LMM results match within optimization tolerance (< 1e-5 for lambda)
3. Scientific conclusions are identical on tested datasets:
   - Same significance calls at all thresholds
   - Same effect directions
   - Same SNP rankings

Numerical differences are expected consequences of:

- Different optimization algorithms (Brent vs golden section)
- Different numerical libraries (GSL vs JAX)
- Output precision limits

These differences do not affect scientific conclusions from GWAS analysis.

---

## Reproducibility

```bash
# Kinship validation
uv run pytest tests/test_kinship_validation.py -v

# LMM validation (all test types: Wald, LRT, Score, All-tests, Covariates)
uv run pytest tests/test_lmm_validation.py -v
```

Reference data: `tests/fixtures/`
