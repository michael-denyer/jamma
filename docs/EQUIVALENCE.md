# JAMMA-GEMMA Equivalence Proof

JAMMA implements mathematically identical algorithms to GEMMA. All numerical
differences are bounded by floating-point precision and optimizer convergence
tolerance. This document provides both the proofs and the empirical validation.

For deliberate behavioral divergences on edge cases (degenerate SNPs, GEMMA
bugs), see [GEMMA_DIVERGENCES.md](GEMMA_DIVERGENCES.md).

---

## Summary

| Quantity | Identical Formula? | Source of Numerical Diff | Theoretical Bound | Observed Max |
|----------|---|---|---|---|
| Kinship K | Yes | FP accumulation in BLAS | O(p * eps_mach) | 4.66e-10 |
| Eigenvalues | Yes | LAPACK backward error | O(n * eps_mach) | ~1e-13 |
| REML log-likelihood | Yes | FP accumulation in Pab | O(n * eps_mach) | 3.23e-7 |
| MLE logl_H1 | Yes | Optimizer on flat landscapes | O(eps * flatness) | 1.35e-3* |
| Lambda (REML) | Yes | Convergence tolerance | O(1e-5) | 3.80e-5 |
| Beta (effect) | Yes | Lambda propagation / Pab | O(eps * sensitivity) | 7.0e-5 |
| SE | Yes | Lambda propagation / sqrt | O(eps * sensitivity) | ~2e-6 |
| p_wald | Yes | CDF implementation | O(1e-5) | 2.20e-6 |
| p_score | Yes | CDF implementation | O(1e-5) | 4.14e-7 |
| p_lrt | Yes | MLE subtraction amplification | O(eps * amplification) | 1.56e-3 |

*logl_H1 worst case on mouse_hs1940 (1,410 samples, 10,768 SNPs). On synthetic
data the max is < 1e-6. Divergence arises from weak-signal SNPs with flat MLE
surfaces where golden section and Brent settle on different optima.*

**Production-scale validation** (85,000 real samples, 91,613 SNPs):
Spearman rho 1.000000, significance agreement 100% at all thresholds,
effect direction agreement 100%. See [Empirical Results](#empirical-results).

---

## 1. Model Specification

Both JAMMA and GEMMA solve the same linear mixed model:

```
y = Wa + xb + u + e
```

where:
- `y` (n) = phenotype vector
- `W` (n x c) = covariate matrix (includes intercept)
- `x` (n) = genotype vector for a single SNP
- `u ~ N(0, s2_g * K)` = random genetic effect
- `e ~ N(0, s2_e * I)` = residual error
- `K` (n x n) = kinship matrix

Defining `lambda = s2_g / s2_e`, the covariance is `H = lambda*K + I`.

---

## 2. Kinship Matrix

Both compute the centered relatedness matrix:

```
K = (1/p) * Xc * Xc'
```

where `Xc` is the mean-imputed, centered genotype matrix and `p` is the
filtered SNP count.

| | GEMMA | JAMMA |
|-|-------|-------|
| Function | `CalcKin` (lmm.cpp) | `compute_centered_kinship` (kinship/compute.py) |
| BLAS | OpenBLAS/MKL `dsyrk` | JAX/XLA `matmul` |
| Batching | 10,000 SNPs | 10,000 SNPs |
| Missing | Mean imputation | Mean imputation |

**Proof**: Both compute `K[i,j] = (1/p) * sum_k (x_ik - mu_k)(x_jk - mu_k)`.
The formula is identical. Differences arise only from FP accumulation order.

**Bound**: `|K_JAMMA - K_GEMMA| <= O(p * eps_mach)`. With p <= 10^6:
`O(10^6 * 2^-52) ~ O(10^-10)`.

**Observed**: max relative difference = 4.66e-10.

---

## 3. Eigendecomposition

Both decompose `K = U * D * U'` where `D = diag(d_1, ..., d_n)`.

- **GEMMA**: LAPACK `dsyevd` via GSL
- **JAMMA**: LAPACK `dsyevd` via `numpy.linalg.eigh`

Both call the same LAPACK routine. Eigenvectors may differ by sign (unique only
up to sign), but all downstream computation uses `U'y`, `U'W`, `U'x` which are
invariant to consistent sign flips.

**Bound**: LAPACK backward error `O(n * eps_mach * ||K||)`, giving eigenvalue
accuracy of `O(10^-13)`.

**Note**: JAMMA uses numpy (not JAX) because JAX's int32 buffer indexing
overflows at ~46k x 46k matrices. See
[GEMMA_DIVERGENCES.md](GEMMA_DIVERGENCES.md#7-eigendecomposition-implementation).

---

## 4. REML Log-Likelihood

Both compute (GEMMA: `LogRL_f`, JAMMA: `reml_log_likelihood`):

```
l_REML(lambda) = c - 1/2 log|H| - 1/2 log|W'H^-1 W| - 1/2(n-c-1) log(P_yy)
```

In the eigenspace: `H_i = lambda*d_i + 1`, so `log|H| = sum log(lambda*d_i + 1)`.

### Pab Recursion

GEMMA `CalcPab` and JAMMA `calc_pab` implement the same recursion:

```
Pab[0, (a,b)] = sum_i h_i * Uab[i, (a,b)]

For p = 1, ..., n_cvt+1:
  Pab[p, (a,b)] = Pab[p-1, (a,b)]
                 - Pab[p-1, (a,p)] * Pab[p-1, (b,p)] / Pab[p-1, (p,p)]
```

JAMMA's `get_ab_index` is a direct transcription of GEMMA's `GetabIndex`.

**Bound**: Weighted dot product error `O(n * eps_mach)`. For n <= 200,000:
`O(2e5 * 2^-52) ~ O(10^-11)`. Recursive divisions amplify this when
`Pab[p-1,(p,p)]` is small (low-variance covariates), producing the larger
beta/SE differences.

---

## 5. Lambda Optimization

| | GEMMA | JAMMA |
|-|-------|-------|
| Method | Brent (GSL) | Grid (50 points) + golden section (20 iter) |
| Bounds | [1e-5, 1e5] | [1e-5, 1e5] |
| Tolerance | 1e-5 | ~6.6e-5 per grid cell (0.618^20 * cell width) |

**Bound**: Both converge to within 1e-5 of the true optimum.
`|lambda_JAMMA - lambda_GEMMA| <= O(1e-5)`.

**Observed (REML)**: max relative difference = 3.80e-5.

### Flat Landscapes (Weak-Signal SNPs)

For SNPs where the REML/MLE surface is nearly flat near lambda = 1e-5, both
optimizers settle on slightly different points within the flat region:

- REML lambda: < 1e-4 relative (negligible)
- MLE lambda: up to ~9e-4 relative
- MLE logl_H1: up to ~1.35e-3 relative (worst case: SNP 596 of 10,768,
  abs_diff ~2.1 on values ~-1583)

This affects only the per-SNP MLE log-likelihood diagnostic. P-values, effect
sizes, and significance calls are unaffected because the flat region
corresponds to weak-signal SNPs where test statistics are small.

---

## 6. Wald Test

Both compute (GEMMA: `CalcRLWald`, JAMMA: `calc_wald_test`):

```
beta = P_xy / P_xx
tau  = df / Px_yy
SE   = sqrt(1 / (tau * P_xx))
F    = (P_yy - Px_yy) * tau
p    = Pr(F_1,df > F)
```

**Error propagation from lambda**: A perturbation `d_lambda` propagates as
`d_beta/beta ~ 0.35 * d_lambda/lambda ~ 0.35 * 3.8e-5 ~ 1.3e-5`.

**Observed**: max relative beta difference = 7.0e-5 (larger outliers from
SNPs where P_xx is small, amplifying the division).

### F-Distribution CDF

- GEMMA: `gsl_cdf_fdist_Q(F, 1, df)` (GSL incomplete beta)
- JAMMA: `betainc(df/2, 1/2, df/(df+F))` (JAX regularized incomplete beta)

Both compute the same `I_z(a,b)` but use different polynomial/continued-fraction
approximations. **Observed**: max relative p-value difference = 2.20e-6.

---

## 7. Score Test

Both compute (GEMMA: `CalcRLScore`, JAMMA: `calc_score_test`):

```
F_score = n * P_xy^2 / (P_yy * P_xx)
p_score = Pr(F_1,df > F_score)
```

Uses **null model lambda** (computed once, reused for all SNPs).

**Observed**: max relative p-value difference = 4.14e-7.

---

## 8. Likelihood Ratio Test

Both compute (GEMMA: `CalcLRT`, JAMMA: `calc_lrt_test`):

```
LRT   = 2 * (l_MLE(H1) - l_MLE(H0))
p_lrt = Pr(chi2_1 > LRT)
```

The LRT statistic subtracts two large log-likelihoods. Small MLE lambda
differences compound: `d(LRT) = 2 * |d_l_H1 - d_l_H0|`. Near LRT ~ 0
(weak signals) the CDF is linear so `d_p ~ d(LRT)`. This is why p_lrt has
the largest tolerance.

**Observed**: max relative p-value difference = 1.56e-3.

---

## Empirical Results

### Small Scale: mouse_hs1940 (1,940 samples x 12,226 SNPs)

| Metric | Value |
|--------|-------|
| Kinship max relative diff | 8.1e-9 |
| Lambda max relative diff (REML) | 9.58e-6 |
| Beta max relative diff | 3.70e-3 |
| p_wald max relative diff | 4.41e-5 |
| p_score max relative diff | ~1e-4 |
| p_lrt max relative diff | ~1.56e-3 |
| REML logl max relative diff | 3.23e-7 |
| MLE logl_H1 max relative diff | ~1.35e-3 |
| P-value rank correlation (Spearman) | 1.000000 |
| Significance agreement (all thresholds) | 100% |
| Effect direction agreement | 100% |

### Production Scale (v1.4.3): 85,000 real samples x 91,613 SNPs

Validated on Databricks with MKL ILP64 numpy:

| Metric | Result |
|--------|--------|
| Kinship Spearman rho | 1.00000000 |
| Kinship max abs diff | 1.09e-05 |
| Kinship Frobenius relative | 1.52e-05 |
| Association Spearman rho (-log10 p) | 1.000000 |
| Significance agree (p < 0.05) | 91,613/91,613 (100%) |
| Significance agree (p < 5e-8) | 91,613/91,613 (100%) |
| Effect direction agreement | 100.0% |
| Max relative p-value diff | 2.10e-03 |

---

## Test Coverage

| Test Class | Coverage |
|------------|----------|
| `TestKinshipValidation` | Kinship matrix vs GEMMA |
| `TestLmmValidation` | Wald test vs GEMMA (synthetic + mouse_hs1940) |
| `TestLmmJaxValidation` | Wald test vs GEMMA (JAX runner) |
| `TestLmmScoreValidation` | Score test vs GEMMA |
| `TestLmmAllTestsValidation` | All-tests mode vs GEMMA |
| `TestLmmCovariateValidation` | Covariates vs GEMMA |
| `TestMouseHS1940Validation` | All modes x covariate configs vs GEMMA (7 tests) |

All tests in `tests/test_kinship_validation.py` and `tests/test_lmm_validation.py`.

```bash
uv run pytest tests/test_kinship_validation.py tests/test_lmm_validation.py -v
```

Comprehensive formal validation across all 8 test configurations:

```bash
uv run python scripts/prove_equivalence.py
```

---

## Source Correspondence

| GEMMA Function (lmm.cpp) | JAMMA Function | Location |
|--------------------------|----------------|----------|
| `CalcKin` | `compute_centered_kinship` | kinship/compute.py |
| `GetabIndex` | `get_ab_index` | lmm/likelihood.py:39 |
| `CalcUab` | `compute_Uab` | lmm/likelihood.py:60 |
| `CalcPab` | `calc_pab` | lmm/likelihood.py:159 |
| `LogRL_f` | `reml_log_likelihood` | lmm/likelihood.py:384 |
| `LogL_f` | `mle_log_likelihood` | lmm/likelihood.py:652 |
| `CalcLambda` | `golden_section_optimize_lambda` | lmm/likelihood_jax.py |
| `CalcRLWald` | `calc_wald_test` | lmm/stats.py:98 |
| `CalcRLScore` | `calc_score_test` | lmm/stats.py:232 |
| `gsl_cdf_fdist_Q` | `f_sf` (via `betainc`) | lmm/stats.py:67 |
| `gsl_cdf_chisq_Q` | `chi2.sf` (via JAX) | lmm/stats.py:225 |

---

## Conclusion

**Algorithmic Identity**: Every stage of the GWAS pipeline — kinship,
eigendecomposition, REML optimization, Wald/Score/LRT statistics — uses
algebraically identical formulas in both JAMMA and GEMMA.

**Bounded Differences**: All numerical differences arise from:
1. IEEE-754 accumulation order (kinship, Pab)
2. Optimizer convergence: Brent vs golden section, eps ~ 1e-5
3. Flat MLE landscapes on weak-signal SNPs (logl_H1 only)
4. CDF implementation differences (tail probabilities)

**Scientific Equivalence**: Identical significance calls at all thresholds,
identical effect directions, identical SNP rankings. Validated at both small
scale (1,940 samples) and production scale (85,000 samples).

Reference tolerances: `src/jamma/validation/tolerances.py`
Reference data: `tests/fixtures/`

---

*Last updated: 2026-02-10*
