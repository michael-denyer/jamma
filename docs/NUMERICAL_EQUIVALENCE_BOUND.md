# Numerical Equivalence Bound: JAMMA vs GEMMA (Full Pipeline)

This document states a formal, end-to-end **numerical equivalence bound**
between JAMMA and GEMMA for the complete GWAS pipeline. It complements
`docs/FORMAL_PROOF.md` (exact-algebra equivalence) and focuses on the
**magnitude of numerical differences** under floating-point arithmetic.

The goal is a *stated bound* that is rigorous but data-dependent: constants
depend on condition numbers and problem scale, which vary by dataset.

---

## Scope

The bound covers:
1. Kinship matrix construction.
2. Eigendecomposition of the kinship matrix.
3. REML optimization for `λ` (variance ratio).
4. Association tests (Wald, score, LRT).
5. P-values (F/χ² distributions).

The statement below assumes **identical inputs** and **float64** arithmetic.

---

## Notation

Let:
- `G ∈ R^{n×p}` be the genotype matrix with missing values.
- `Xc` be the mean-imputed, centered genotype matrix.
- `K = (1/p) Xc Xcᵀ` be the kinship matrix.
- `K = U D Uᵀ` be its eigendecomposition (`D = diag(d_i)`).
- `H(λ) = λK + I` be the covariance matrix.
- `W ∈ R^{n×c}` be covariates (including intercept), `x ∈ R^n` a SNP vector.
- `P(λ) = H^{-1} − H^{-1}W (WᵀH^{-1}W)^{-1} WᵀH^{-1}`.
- `β, se, T` denote effect, standard error, and test statistics (Wald/score/LRT).
- `p` denotes p-values.

Define:
- `ε = 2^{-52} ≈ 2.22e−16` (IEEE-754 float64 machine epsilon).
- `γ_k = kε / (1 − kε)` (Higham rounding error factor).
- `κ(·)` denotes condition number in the spectral norm.

We write `ΔX = X_JAMMA − X_GEMMA` for differences in outputs.

---

## Assumptions

1. **Identical preprocessing**: missing imputation, centering, and SNP filtering
   are identical, so both pipelines start from the same `Xc`.
2. **Float64 everywhere**: JAX is configured with `jax_enable_x64 = True`.
3. **Stable BLAS/LAPACK**: matrix multiplication and eigendecomposition are
   backward stable in the standard IEEE-754 model.
4. **Well-conditioned eigenspace**: `K` is PSD and the eigenvalue gaps are
   nonzero (or small) so eigenvectors are not arbitrarily ill-conditioned.
5. **REML concavity**: the REML log-likelihood is strictly concave in `log λ`
   over the search interval, with curvature bounded below by `m > 0`.
6. **Bounded test inputs**: `P_xx > 0`, `Px_yy > 0`, and denominators in test
   statistics are bounded away from zero.
7. **CDF stability**: the F/χ² CDFs used are Lipschitz in the relevant range.

If any assumption fails, equivalence can still hold, but the stated bound may
not be meaningful. See `docs/GEMMA_DIVERGENCES.md` for known edge-case behavior.

---

## Theorem (End-to-End Numerical Equivalence Bound)

Under the assumptions above, the full pipeline outputs produced by JAMMA and
GEMMA satisfy:

```
||ΔK||_F        ≤ C_K · γ_p · ||Xc||_F^2 / p
||ΔD||_2        ≤ C_E · ε · ||K||_2
||ΔU||_2        ≤ C_U · ε · ||K||_2 / gap(K)
|Δλ|            ≤ τ_opt + L_λ (||ΔK||_2 + ||ΔU||_2 + ||ΔD||_2)
|Δβ|, |Δse|, |ΔT| ≤ L_T (||ΔK||_2 + |Δλ| + ||ΔU||_2 + ||ΔD||_2)
|Δp|            ≤ L_CDF · |ΔT| + δ_CDF
```

where:
- `C_K, C_E, C_U, L_λ, L_T, L_CDF` are data-dependent constants determined by
  condition numbers and operator norms.
- `gap(K)` is the minimum eigenvalue separation.
- `τ_opt` is the optimizer stopping tolerance (Brent or golden-section).
- `δ_CDF` is the worst-case difference between CDF implementations.

This bound is *explicit but data-dependent*: it is stated in terms of norms,
eigenvalue gaps, and optimizer tolerances, all of which can be computed or
estimated for a given dataset.

---

## Proof Sketch (By Pipeline Stage)

### 1. Kinship matrix

Each entry of `K` is a length-`p` dot product. For IEEE-754 arithmetic,
the standard dot-product bound applies:

```
|fl(xᵀy) − xᵀy| ≤ γ_p · ||x||_2 ||y||_2
```

Summing across rows yields the Frobenius bound:

```
||ΔK||_F ≤ C_K · γ_p · ||Xc||_F^2 / p
```

with `C_K` capturing batching and symmetric accumulation effects.

### 2. Eigendecomposition

LAPACK symmetric eigensolvers are backward stable, so the computed
eigendecomposition satisfies:

```
K + E = Û D̂ Ûᵀ,   ||E||_2 ≤ C_E · ε · ||K||_2
```

Eigenvector error depends on spectral gaps:

```
||ΔU||_2 ≤ C_U · ε · ||K||_2 / gap(K)
```

This is the standard Davis–Kahan type perturbation bound.

### 3. REML optimization

Let `ℓ(λ)` be the REML log-likelihood and assume strict concavity in `log λ`
with curvature `m > 0`. If the optimization terminates with a bracket width
or tolerance `τ_opt`, then:

```
|λ̂ − λ*| ≤ τ_opt + L_λ (||ΔK||_2 + ||ΔU||_2 + ||ΔD||_2)
```

where `L_λ` depends on the Lipschitz constants of `∂ℓ/∂λ` with respect to its
matrix arguments. This captures both algorithmic tolerance and propagated
numerical perturbations.

### 4. Association statistics

Wald, score, and LRT statistics are smooth functions of
`(K, U, D, λ, W, x, y)` under the assumption that denominators are bounded
away from zero. A first-order perturbation bound yields:

```
|ΔT| ≤ L_T (||ΔK||_2 + |Δλ| + ||ΔU||_2 + ||ΔD||_2)
```

and similar bounds for `β` and `se`.

### 5. P-values

Let `F` be the test statistic and `CDF` the corresponding distribution
function. If `CDF` is Lipschitz on the relevant domain:

```
|Δp| ≤ L_CDF · |ΔF| + δ_CDF
```

where `δ_CDF` captures algorithmic differences between GEMMA’s GSL CDF and
JAX’s `betainc`/`chi2` evaluation.

---

## Composition (Single Pipeline Bound)

Define the linear-algebra propagation term:

```
B_lin = L_T (||ΔK||_2 + ||ΔU||_2 + ||ΔD||_2 + τ_opt)
```

Then the p-value bound is:

```
|Δp| ≤ L_CDF · B_lin + δ_CDF
```

This makes explicit that the bound is dominated by:
1. **Optimizer tolerance** (`τ_opt`), and
2. **CDF implementation differences** (`δ_CDF`),
when the linear algebra is well-conditioned.

---

## Practical Interpretation

- `τ_opt` is configured at approximately `1e-5` for Brent, and the JAX
  grid + golden-section search is designed to reach a comparable tolerance.
- `δ_CDF` can be measured empirically; see
  `docs/MATHEMATICAL_EQUIVALENCE.md` for observed differences.
- If `κ(K)` and `gap(K)` are favorable, the linear algebra perturbation terms
  are typically **smaller than `τ_opt`** and the bound is driven by
  optimization tolerance and CDF differences.

This is the strongest *formal* statement we can make without enforcing
dataset-specific condition-number constraints.

---

## What This Does Not Claim

1. It does not claim exact bitwise agreement.
2. It does not claim equivalence for ill-conditioned or degenerate inputs.
3. It does not override documented edge-case divergences.

For empirical validation on real datasets, see `docs/MATHEMATICAL_EQUIVALENCE.md`.
