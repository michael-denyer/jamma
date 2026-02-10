# JAMMA vs GEMMA: Documented Divergences

JAMMA aims for **numerical equivalence** with GEMMA on well-formed inputs, but makes deliberate deviations for robustness in edge cases. This document catalogs each divergence with rationale.

## Philosophy

GEMMA is the **reference implementation**, not the specification. Where GEMMA has bugs or undefined behavior, JAMMA chooses correctness over bug-compatibility. All divergences affect only degenerate/edge cases that should not occur in real GWAS data.

**Validation approach**: JAMMA passes GEMMA validation tests on real-world data within documented tolerances. Divergences manifest only in synthetic edge cases.

---

## 1. `safe_sqrt` Behavior

### GEMMA (mathfunc.cpp:122-131)
```c++
double safe_sqrt(const double d) {
  double d1 = d;
  if (fabs(d < 0.001))    // BUG: evaluates (d < 0.001) as bool, then fabs(0 or 1)
    d1 = fabs(d);         // effectively ALWAYS applies abs()
  if (d1 < 0.0)
    return nan("");
  return sqrt(d1);
}
```

**Bug**: `fabs(d < 0.001)` evaluates the comparison `d < 0.001` as a boolean (0 or 1), then takes `fabs()` of that result. Since `fabs(0)=0` and `fabs(1)=1`, the condition is effectively always true-ish. Result: `safe_sqrt(-5.0)` returns `sqrt(5.0) = 2.236`.

### JAMMA (stats.py:15-36)
```python
def _safe_sqrt(d: float) -> float:
    if abs(d) < 0.001:
        d = abs(d)
    if d < 0.0:
        return float("nan")
    return np.sqrt(d)
```

**Behavior**: Only applies `abs()` for values in `(-0.001, 0.001)`. Large negatives return NaN.

### Divergence Impact
| Input | GEMMA | JAMMA |
|-------|-------|-------|
| `safe_sqrt(4.0)` | 2.0 | 2.0 |
| `safe_sqrt(-0.0001)` | 0.01 | 0.01 |
| `safe_sqrt(-5.0)` | **2.236** | **NaN** |

### Rationale
- Large negative variance values indicate a bug in upstream computation, not a recoverable condition
- Returning `sqrt(abs(x))` silently masks errors
- NaN propagation surfaces problems for investigation

### When This Matters
Only when `1/(tau * P_xx)` is large and negative due to:
- Degenerate SNPs (P_xx ≈ 0)
- Numerical instability in projection

---

## 2. Wald Test Guards (P_xx, Px_yy)

### GEMMA (lmm.cpp:1153-1161)
```c++
beta = P_xy / P_xx;
double tau = (double)df / Px_yy;
se = safe_sqrt(1.0 / (tau * P_xx));
p_wald = gsl_cdf_fdist_Q((P_yy - Px_yy) * tau, 1.0, df);
```

**Behavior**: No guards. Division by zero produces `inf` or `NaN` depending on numerator.

### JAMMA (stats.py:137-155)
```python
if P_xx <= 0.0:
    return float("nan"), float("nan"), float("nan")

if Px_yy >= 0.0 and Px_yy < 1e-8:
    Px_yy = 1e-8
```

**Behavior**:
- P_xx ≤ 0: Return NaN for all stats (SNP has no variance)
- Px_yy clamping: Prevent division by near-zero residual variance

### Divergence Impact
| Condition | GEMMA | JAMMA |
|-----------|-------|-------|
| P_xx = 0 (constant SNP) | beta=NaN, se=inf, p=NaN | beta=NaN, se=NaN, p=NaN |
| Px_yy = 1e-12 | tau=1e12, se≈0 | tau=1e8, se finite |

### Rationale
- Constant SNPs (P_xx = 0) have no genetic variance to test
- Consistent NaN is more useful than mixed inf/NaN
- Px_yy clamping prevents numerical overflow in downstream calculations

### When This Matters
- Monomorphic SNPs (all samples have same genotype)
- SNPs with MAF below filtering threshold that slipped through
- Numerical edge cases from projection

---

## 3. REML logdet Computation

### GEMMA (lmm.cpp:835)
```c++
logdet_h += safe_log(fabs(d));
```

### JAMMA (likelihood.py)
```python
logdet_h = np.sum(np.log(np.abs(v_temp)))
```

### Status: **ALIGNED**
Both use `log(abs(v))` to handle potential negative eigenvalues from non-PSD kinship matrices.

---

## 4. Monomorphic SNP Detection

### GEMMA (gemma.cpp:2377-2392)

```c++
// In PlinkKin() - count-based detection
int n_total = 0;
for (size_t i = 0; i < n_rows; i++) {
    if (x[i] != MISSING) {
        n_total++;
        // ... accumulate sums
    }
}
// Check for polymorphism via counts
if (n_total == 0 || n_aa == n_total || n_bb == n_total) {
    flag_poly = false;  // Monomorphic
}
```

**Behavior**: Count genotype classes (AA, AB, BB) and flag as monomorphic if only one class exists.

### JAMMA (kinship/compute.py, lmm/runner_jax.py)

```python
# Variance-based detection
col_vars = np.nanvar(genotypes, axis=0)
is_polymorphic = col_vars > 0
```

**Behavior**: Compute variance and flag as monomorphic if variance == 0.

### Status: **Equivalent Results, Different Method**

Both approaches correctly identify monomorphic SNPs:

- GEMMA: Count-based (n_aa == n_total or n_bb == n_total)
- JAMMA: Variance-based (var == 0)

For biallelic SNPs with values {0, 1, 2}, both methods produce identical classification:

- Variance == 0 ⟺ all values are equal ⟺ only one genotype class exists

The variance-based approach is simpler and equally robust. A single-sample GWAS where this might differ is biologically meaningless anyway.

---

## 5. JAX Path: Covariate Support

### GEMMA

Supports arbitrary covariates (n_cvt >= 1).

### JAMMA JAX Path (runner_jax.py)

Supports arbitrary covariates (n_cvt >= 1) since v1.2 (Phase 11).

### Status: **Aligned**

The JAX runner generalizes Uab/Pab shapes from hardcoded n_cvt=1 to arbitrary
n_cvt. All LMM modes (Wald, LRT, Score, all-tests) work with covariates.

---

## 6. Lambda Optimization: Brent vs Golden Section

**GEMMA:** Brent's method (GSL `gsl_min_fminimizer_brent`) with bounds
[1e-5, 1e5] and tolerance 1e-5.

**JAMMA (since v1.2):** Grid search (50 log-spaced points) + golden section
refinement (20 iterations) with effective tolerance ~6.6e-5 per grid cell.

### Divergence on Flat Optimization Landscapes

Both methods converge to within 1e-5 of the true optimum for strong-signal
SNPs. However, for weak-signal SNPs where the optimization surface is nearly
flat (lambda converging near the lower bound 1e-5), Brent and golden section
can settle on slightly different points within the flat region:

- **REML lambda**: negligible impact (< 1e-4 relative)
- **MLE lambda**: up to ~9e-4 relative on mouse_hs1940
- **MLE logl_H1**: up to ~1.35e-3 relative (worst case: SNP 596 of 10768,
  abs_diff ~2.1 on values ~-1583)

This only affects the per-SNP MLE log-likelihood diagnostic (logl_H1).
P-values, effect sizes, and significance calls are identical because the flat
region corresponds to weak-signal SNPs where test statistics are small.

JAMMA warns when lambda converges at the lower bound:

```text
RuntimeWarning: Lambda converged at lower bound (1.00e-05 ~ 1.00e-05).
True optimum may be below search range.
```

**Validation note:** Lambda values at the boundary (< 1e-4) are excluded from
tolerance comparisons since relative error is inflated when dividing by small
numbers.

---

## 7. Eigendecomposition Implementation

GEMMA uses GSL (GNU Scientific Library) for eigendecomposition. JAMMA uses `numpy.linalg.eigh` (LAPACK) instead of JAX's `jnp.linalg.eigh`.

**Rationale:** JAX uses int32 buffer indexing internally, which overflows at ~2.1 billion elements (~46k × 46k matrix). For 200k+ sample GWAS, the kinship matrix has 40+ billion elements, causing:

```text
JaxRuntimeError: INVALID_ARGUMENT: Buffer Definition Event:
Value (=5000300001) exceeds the maximum representable value of the desired type
```

numpy's LAPACK binding supports large matrices without this limitation.

**Performance:** numpy's LAPACK-based eigh is highly optimized (multi-threaded, vectorized). The eigendecomposition is O(n³) and runs once per dataset, so it's not the performance bottleneck. The JAX-accelerated SNP processing dominates runtime for large datasets.

---

## Summary Table

| Feature | GEMMA Behavior | JAMMA Behavior | Impact |
|---------|---------------|----------------|--------|
| safe_sqrt(-5.0) | sqrt(5.0) | NaN | Edge case only |
| P_xx = 0 | inf/NaN mix | NaN | Degenerate SNPs |
| Px_yy clamping | None | 1e-8 floor | Numerical stability |
| logdet with neg eigenvalues | log(abs(v)) | log(abs(v)) | Aligned |
| Monomorphic detection | Count-based | Variance-based | Aligned (equivalent) |
| JAX covariates | n_cvt >= 1 | n_cvt >= 1 | Aligned (since v1.2) |
| Lambda optimizer | Brent | Golden section | logl_H1 ~1.35e-3 on flat landscapes |
| Eigendecomp library | GSL | numpy LAPACK | 200k+ sample support |

---

## Validation Strategy

1. **Real-world data**: JAMMA matches GEMMA within tolerance on actual GWAS datasets
2. **Edge case tests**: `tests/test_hypothesis.py` verifies JAMMA's robust behavior
3. **No silent failures**: Divergences produce NaN, not silently wrong values

### Production-Scale Validation (v1.4.3)

JAMMA vs GEMMA validated on 85,000 real samples × 91,613 SNPs (Databricks, MKL ILP64):

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

*Last updated: 2026-02-10*
