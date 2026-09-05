# JAMMA-GEMMA formulas and validation evidence

JAMMA implements GEMMA's statistical formulas with different numerical kernels
and optimizers. This document records formula derivations, empirical comparisons,
and the conditioning limits of those comparisons.

See [Mathematical validation](MATHEMATICAL_VALIDATION.md) for the declared cases,
reproduction commands and untested configurations. Production comparisons below
apply to their listed software versions and settings.

For deliberate behavioral divergences on edge cases (degenerate SNPs, GEMMA
bugs), see [GEMMA_DIVERGENCES.md](GEMMA_DIVERGENCES.md).

---

## Summary

| Quantity | Identical Formula? | Source of Numerical Diff | Theoretical Bound | Observed Max |
|----------|---|---|---|---|
| Kinship K | Yes | FP accumulation in BLAS | O(p * eps_mach) | 4.66e-10 |
| Eigenvalues | Yes | LAPACK backward error | O(n * eps_mach) | ~1e-13 |
| REML log-likelihood | Yes | FP accumulation in Pab | O(n * eps_mach) | 3.23e-7 |
| MLE logl_H1 | Yes | MLE optimization / accumulation | Conditioning dependent | See correction below |
| Lambda (REML) | Yes | Optimization / score conditioning | Conditioning dependent | See section 5 |
| Beta (effect) | Yes | Lambda propagation / Pab | O(eps * sensitivity) | 7.0e-5 |
| SE | Yes | Lambda propagation / sqrt | O(eps * sensitivity) | ~2e-6 |
| p_wald | Yes | CDF implementation | O(1e-5) | 2.20e-6 |
| p_score | Yes | CDF implementation | O(1e-5) | 4.14e-7 |
| p_lrt | Yes | MLE subtraction amplification | O(eps * amplification) | 1.56e-3 |

Modes 2 and 4 report the alternative MLE likelihood used by LRT as `logl_H1`.
Mode 1 reports normalized REML likelihood. Independent dense-oracle tests check
these output contracts.

**Production measurements for v2.5** (125,632 real samples, 91,586 SNPs):
Spearman rho 1.000000, significance agreement 100% at all thresholds,
effect direction agreement 100%. See [Empirical Results](#empirical-results).

---

## 1. Model Specification

Both JAMMA and GEMMA solve the same linear mixed model:

```text
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

```text
K = (1/p) * Xc * Xc'
```

where `Xc` is the mean-imputed, centered genotype matrix and `p` is the
filtered SNP count.

| | GEMMA | JAMMA |
|-|-------|-------|
| Function | `CalcKin` (lmm.cpp) | `compute_kinship_streaming` (kinship/stream.py) |
| BLAS | OpenBLAS/MKL `dsyrk` | Vendor BLAS `dsyrk` (via jlinalg) |
| Batching | 10,000 SNPs | 10,000 SNPs |
| Missing | Mean imputation | Mean imputation |

**Proof**: Both compute `K[i,j] = (1/p) * sum_k (x_ik - mu_k)(x_jk - mu_k)`.
The formula is identical. Differences arise only from FP accumulation order.

**Bound**: `|K_JAMMA - K_GEMMA| <= O(p * eps_mach)`. With p <= 10^6:
`O(10^6 * 2^-52) ~ O(10^-10)`.

**Early row selection**: `filter_sample_indices` determines the population used
for SNP filtering; `valid_indices` selects the output rows. The LMM pipeline
filters on analysed samples and imputes and centres the selected genotype columns
over all samples, independently of whether kinship is saved. For a fixed SNP set
and transformed genotype matrix, selecting rows before symmetric accumulation
produces the corresponding principal submatrix of full kinship. The analysed
matrix is then centred before weighting and eigendecomposition. This preserves
the smaller `(n_valid, n_valid)` allocation without changing preprocessing or
widening the kinship tolerance.

**Observed**: max relative difference = 4.66e-10.

---

## 3. Eigendecomposition

Both decompose `K = U * D * U'` where `D = diag(d_1, ..., d_n)`.

- **GEMMA**: LAPACK `dsyevd` via GSL
- **JAMMA**: LAPACK `dsyevd`/`dsyevr` via `jlinalg.eigh` (vendor BLAS dispatch)

Both call the same LAPACK routines. JAMMA defaults to DSYEVD (faster, O(N²)
workspace) and falls back to DSYEVR (slower, O(N) workspace) when DSYEVD won't
fit in memory. Both drivers produce equivalent results within LAPACK backward
error bounds. Eigenvectors may differ by sign (unique only up to sign), but all
downstream computation uses `U'y`, `U'W`, `U'x` which are invariant to
consistent sign flips.

**Bound**: LAPACK backward error `O(n * eps_mach * ||K||)`, giving eigenvalue
accuracy of `O(10^-13)`.

**Note**: JAMMA uses jlinalg for vendor LAPACK dispatch because it supports
ILP64 for large matrices (>46k x 46k). See
[GEMMA_DIVERGENCES.md](GEMMA_DIVERGENCES.md#7-eigendecomposition-implementation).

---

## 4. REML Log-Likelihood

Both compute (GEMMA: `LogRL_f`, JAMMA: `reml_log_likelihood`):

```text
l_REML(lambda) = c - 1/2 log|H| - 1/2 log|W'H^-1 W| - 1/2(n-c-1) log(P_yy)
```

In the eigenspace: `H_i = lambda*d_i + 1`, so `log|H| = sum log(lambda*d_i + 1)`.

The pure-NumPy path evaluates that sum term by term. The C accelerator
evaluates the same quantity as a product of mantissas with an exact integer
exponent (`_lmm_logdet.h`), calling `log()` once per evaluation instead of
once per sample. Both carry `O(n * eps_mach)` rounding. Measured against a
40-digit reference on the 1,940 mouse_hs1940 eigenvalues, the relative error
of each form lies between 1e-18 and 1.1e-12 across lambda from 1e-5 to 1e5,
and neither is uniformly smaller; the two forms agree to 2.1e-14 relative.
See [GEMMA_DIVERGENCES.md](GEMMA_DIVERGENCES.md#3-reml-logdet-computation).

### Pab Recursion

GEMMA `CalcPab` and JAMMA `calc_pab` implement the same recursion:

```text
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
| Method | Brent (GSL) | 50-point grid + 20 golden steps; analytic-score refinement for interior REML peaks |
| Bounds | [1e-5, 1e5] | [1e-5, 1e5] |
| Comparison tolerance | | `ToleranceConfig.lambda_rtol=2e-5` |

Golden-section bracket shrinkage does not guarantee lambda accuracy when the
objective is flat enough for rounding to change the retained interval. The
September 2026 audit confirmed this with eight real-data peaks and an
independent 80-digit dense REML calculation. Increasing the iteration count
alone did not recover their stationary points.

The REML optimizer now differentiates weighted cross-products directly, uses
compensated reductions, and applies the Schur-complement chain rule. One
Newton step is accepted only with negative curvature, a candidate inside the
original coarse bracket, and a smaller absolute score. This remains vectorized
across SNPs in NumPy. MLE retains its existing golden-section optimizer.

The eight committed reference roots are independently reproducible with
`scripts/verify_reml_precision_oracle.py`. Tests compare generic NumPy, split
NumPy, and native C at `5e-6` relative tolerance. These cases establish a
regression contract, not a universal error bound for arbitrarily ill-conditioned
inputs. Neither the production tolerance nor the existing real-data test
tolerances were widened.

---

## 6. Wald Test

Both compute the same formula (GEMMA: `CalcRLWald`; JAMMA's production path is
the vectorized `batch_calc_wald_stats_from_pab_numpy` in `lmm/stats.py` and
the C kernels, with `calc_wald_test` in `tests/reference/stats.py` kept as a
scalar reference for tests):

```text
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
- JAMMA: `betainc(df/2, 1/2, df/(df+F))` (Cephes regularized incomplete beta)

Both compute the same `I_z(a,b)` but use different polynomial/continued-fraction
approximations. **Observed**: max relative p-value difference = 2.20e-6.

---

## 7. Score Test

Both compute the same formula (GEMMA: `CalcRLScore`; JAMMA's production path is
the vectorized `batch_calc_score_stats_numpy` in `lmm/stats.py` and the C
kernels, with `calc_score_test` in `tests/reference/stats.py` kept as a scalar
reference for tests):

```text
F_score = n * P_xy^2 / (P_yy * P_xx)
p_score = Pr(F_1,df > F_score)
```

Uses **null model lambda** (computed once, reused for all SNPs).

**Observed**: max relative p-value difference = 4.14e-7.

---

## 8. Likelihood Ratio Test

Both compute the same formula (GEMMA: `CalcLRT`; JAMMA's production path is
`_batch_lrt_pvalues_numpy` in `lmm/stats.py` and the C kernels, with
`calc_lrt_test` in `tests/reference/stats.py` kept as a scalar reference for
tests):

```text
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

### Production Scale (v2.5): 125,632 real samples x 91,586 SNPs

Validated on Databricks (Azure E96ds_v6, 48 physical cores, 672 GB RAM) with
MKL ILP64 numpy 2.4.2:

| Metric | Result |
|--------|--------|
| Kinship Spearman rho | 1.00000000 |
| Kinship max abs diff | 5.00e-11 |
| Kinship mean abs diff | 1.24e-12 |
| Kinship Frobenius relative | 1.45e-10 |
| Association Spearman rho (-log10 p) | 1.000000 |
| Significance agree (p < 0.05) | 91,586/91,586 (100%) |
| Significance agree (p < 5e-8) | 91,586/91,586 (100%) |
| Effect direction agreement | 100.0% |
| Max relative p-value diff | 9.66e-04 |

The 125k validation shows tighter kinship tolerances than 85k (5e-11 vs 1e-05
max abs diff), likely reflecting identical BLAS libraries on the 125k run
versus mixed OpenBLAS/MKL at 85k.

---

## Test Coverage

| Test Location | Coverage |
|---------------|----------|
| `tests/test_kinship_validation.py::TestKinshipValidation` | Kinship matrix vs GEMMA |
| `tests/test_runner_numpy.py` (tier0/tier1/tier2) | Wald/Score/LRT vs GEMMA (synthetic + mouse_hs1940) |
| `tests/test_numpy_streaming.py::TestNumpyStreamingGemmaParity` (tier1) | Streaming runner vs GEMMA (all modes + covariates) |
| `tests/lmm_accel/` (tier0/tier1/tier2) | C extension Wald+covariate vs GEMMA |

Run kinship validation:

```bash
uv run pytest tests/test_kinship_validation.py -v
```

Run all GEMMA parity tests (spans tier0/tier1/tier2):

```bash
# -m '' overrides the default tier filter from addopts
uv run pytest tests/test_runner_numpy.py tests/test_numpy_streaming.py tests/lmm_accel/ -v -n0 -m ''
```

Comprehensive formal validation across all 8 test configurations:

```bash
uv run python scripts/demonstrate_equivalence.py
```

---

## Source Correspondence

| GEMMA Function (lmm.cpp) | JAMMA Function | Location |
|--------------------------|----------------|----------|
| `CalcKin` | `compute_kinship_streaming` | kinship/stream.py |
| `GetabIndex` | `get_ab_index` | lmm/pab.py |
| `CalcUab` | `compute_Uab` | lmm/pab.py |
| `CalcPab` | `calc_pab` | lmm/pab.py |
| `LogRL_f` | `reml_log_likelihood` | lmm/likelihood.py |
| `LogL_f` | `mle_log_likelihood` | lmm/likelihood.py |
| `CalcLambda` | `golden_section_optimize_lambda_numpy` | lmm/likelihood_numpy.py |
| `CalcRLWald` | `batch_calc_wald_stats_from_pab_numpy` (production); `calc_wald_test` (scalar reference, tests only) | lmm/stats.py; tests/reference/stats.py |
| `CalcRLScore` | `batch_calc_score_stats_numpy` (production); `calc_score_test` (scalar reference, tests only) | lmm/stats.py; tests/reference/stats.py |
| `CalcLRT` | `_batch_lrt_pvalues_numpy` (production); `calc_lrt_test` (scalar reference, tests only) | lmm/stats.py; tests/reference/stats.py |
| `gsl_cdf_fdist_Q` | `f_sf` (via `betainc`, scalar reference, tests only); `_f_to_pvalue` (production) | tests/reference/stats.py; lmm/stats.py |
| `gsl_cdf_chisq_Q` | `chi2_sf` / `chi2_sf_batch` (erfc) | lmm/special.py |

---

## Scope of the evidence

The derivations describe the intended statistical formulas. Finite comparisons
establish agreement only for their recorded datasets, settings and backends.
They do not establish identical rankings or significance calls for every input.

Current field tolerances remain in `src/jamma/validation/tolerances.py`.
See [Mathematical validation](MATHEMATICAL_VALIDATION.md) for the declared matrix,
negative controls, backend evidence and remaining gaps, including the weighted
fixture's default-refinement MLE lambda discrepancy.
