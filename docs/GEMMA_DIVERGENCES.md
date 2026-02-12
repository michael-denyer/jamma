# JAMMA vs GEMMA: Documented Divergences

JAMMA aims for **numerical equivalence** with GEMMA on well-formed inputs, but makes deliberate deviations for robustness in edge cases. This document catalogs each divergence with rationale.

For mathematical proofs of equivalence and empirical validation results, see
[EQUIVALENCE.md](EQUIVALENCE.md).

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

GEMMA uses Brent's method; JAMMA uses grid search + golden section. Both
converge to within 1e-5 for strong signals. On flat landscapes (weak-signal
SNPs) they settle on slightly different optima — affects only the MLE
log-likelihood diagnostic, not p-values or effect sizes.

See [EQUIVALENCE.md § Lambda Optimization](EQUIVALENCE.md#5-lambda-optimization) for full analysis and error bounds.

---

## 7. Eigendecomposition Implementation

GEMMA uses GSL (GNU Scientific Library) for eigendecomposition. JAMMA uses `numpy.linalg.eigh` (LAPACK) instead of JAX's `jnp.linalg.eigh`.

**Rationale:** JAX uses int32 buffer indexing internally, which overflows at ~2.1 billion elements (~46k × 46k matrix). For large-sample GWAS (50k+), the kinship matrix exceeds this limit, causing:

```text
JaxRuntimeError: INVALID_ARGUMENT: Buffer Definition Event:
Value (=5000300001) exceeds the maximum representable value of the desired type
```

numpy's LAPACK binding supports large matrices without this limitation.

**Performance:** numpy's LAPACK-based eigh is highly optimized (multi-threaded, vectorized). The eigendecomposition is O(n³) and runs once per dataset, so it's not the performance bottleneck. The JAX-accelerated SNP processing dominates runtime for large datasets.

---

## 8. HWE Test Implementation

### GEMMA

Uses the **Wigginton exact test** — a permutation-based exact test for Hardy-Weinberg equilibrium that is accurate at all sample sizes, including small cohorts.

### JAMMA

Uses a **chi-squared goodness-of-fit test** (df=1) computed via JAX's `jax.scipy.stats.chi2.sf`. The chi-squared test compares observed genotype counts to expected counts under HWE.

### Divergence Impact

| Sample Size | Divergence |
|-------------|------------|
| n > 100 | Negligible — both tests agree on filtering decisions |
| n = 30–100 | Slight p-value differences, rare disagreements near threshold |
| n < 30 | More significant differences — Wigginton exact test is more accurate |

### Rationale

The chi-squared test avoids a scipy dependency (Wigginton's exact test requires `scipy.stats`). For JAMMA's target use case (large-scale GWAS with thousands to hundreds of thousands of samples), the chi-squared approximation is indistinguishable from the exact test.

---

## 9. LOCO Kinship Computation

### GEMMA

Materializes **all** per-chromosome LOCO kinship matrices simultaneously, requiring `n_chr × n² × 8` bytes of memory.

### JAMMA

Uses **streaming subtraction**: computes the full kinship matrix K once, then derives each chromosome's LOCO kinship as `K_loco_c = (p × K - K_c) / (p - p_c)` one at a time, where K_c is the contribution of chromosome c's SNPs and p_c is the SNP count for that chromosome.

### Divergence Impact

| Aspect | GEMMA | JAMMA |
|--------|-------|-------|
| Math | Same formula | Same formula |
| Memory | O(n_chr × n²) | O(n²) |
| I/O | One pass per chromosome | Two passes total (full K + per-chr subtraction) |

### Rationale

The streaming approach produces mathematically identical LOCO kinship matrices while using constant memory (one K_loco at a time). This is critical for large-sample GWAS where materializing 22 copies of an n×n matrix is infeasible.

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
| Lambda optimizer | Brent | Golden section | See [EQUIVALENCE.md](EQUIVALENCE.md#5-lambda-optimization) |
| Eigendecomp library | GSL | numpy LAPACK | Large-sample support (ILP64) |
| HWE test | Wigginton exact | Chi-squared (df=1) | Identical for large n |
| LOCO kinship | Materialized all | Streaming subtraction | Same math, lower memory |

---

## GEMMA Features Not Implemented

JAMMA targets full parity on the **univariate LMM workflow** — the core GWAS pipeline
that the vast majority of GEMMA users rely on. The following GEMMA features are
deliberately out of scope, either because better alternatives exist or because they
represent niche use cases.

### Not planned

| GEMMA Feature | Flag | Rationale |
|---------------|------|-----------|
| BSLMM (Bayesian sparse LMM) | `-bslmm 1/2/3` | Entirely different model class (MCMC-based). Rarely used in modern GWAS — polygenic risk score methods (LDpred2, PRS-CS) have largely replaced BSLMM for prediction. Would require a separate codebase. |
| Linear model (no random effects) | `-lm 1/2/3/4` | Trivially available via statsmodels or scipy. No kinship matrix needed — not an LMM. |
| R² LD filtering | `-r2` | LD pruning is standard upstream QC done with PLINK (`--indep-pairwise`). Not an association testing concern. |
| Debug/legacy flags | `-debug`, `-legacy`, `-strict`, `-issue` | Internal GEMMA development flags, not user-facing functionality. |
| BSLMM MCMC parameters | `-w`, `-s`, `-seed`, `-rpace`, `-wpace`, `-hmin/max`, `-rmin/max`, `-pmin/max`, `-smin/max` | Only relevant if BSLMM were implemented. |
| Pace output | `-pace` | JAMMA uses progress bars with ETA instead. |

---

## Validation Strategy

1. **Real-world data**: JAMMA matches GEMMA within tolerance on actual GWAS datasets
2. **Edge case tests**: `tests/test_hypothesis.py` verifies JAMMA's robust behavior
3. **No silent failures**: Divergences produce NaN, not silently wrong values

For full empirical validation results (small-scale and production-scale), see
[EQUIVALENCE.md § Empirical Results](EQUIVALENCE.md#empirical-results).

---

*Last updated: 2026-02-12*
