# JAMMA vs GEMMA: Documented Divergences

JAMMA aims for **numerical equivalence** with GEMMA on well-formed inputs, but makes deliberate deviations for robustness in edge cases. This document catalogs each divergence with rationale.

For mathematical proofs of equivalence and empirical validation results, see
[GEMMA_EQUIVALENCE.md](GEMMA_EQUIVALENCE.md).

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

### JAMMA (`calc_wald_test` in stats.py)

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

### JAMMA (kinship/compute.py)

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

## 5. Covariate Support

### GEMMA

Supports arbitrary covariates (n_cvt >= 1).

### JAMMA

Supports arbitrary covariates (n_cvt >= 1) since v1.2.

### Status: **Aligned**

All LMM modes (Wald, LRT, Score, all-tests) work with covariates.

---

## 6. Lambda Optimization: Brent vs Golden Section

### GEMMA

Uses **Brent's method** (GSL `gsl_min_fminimizer_brent`) — a hybrid algorithm
combining inverse quadratic interpolation with golden section fallback. Variable
iteration count per SNP; serial execution.

### JAMMA (likelihood_numpy.py)

Uses **grid search (50 log-spaced points) + golden section refinement (20
iterations)**. All SNPs in a chunk are optimized simultaneously in lockstep
(same bracket operation per iteration, vectorized across the SNP batch).

### Unimodality Assumption

Both methods assume the REML/MLE log-likelihood is **unimodal** in log-lambda
space. This is empirically true for standard GWAS data (hundreds to hundreds of
thousands of samples, intercept + typical covariates) but is not mathematically
guaranteed. Theoretical scenarios where multimodality could arise — very small
samples (<50), extreme covariate collinearity, or lambda bounds spanning >15
orders of magnitude — are outside JAMMA's practical use case.

The golden section method has no mechanism to detect or recover from
multimodality. Brent's method also assumes unimodality but is somewhat more
robust to flat regions due to its inverse quadratic interpolation step.

**Small-sample warning.** When fewer than 50 samples enter the LMM (after
phenotype and covariate filtering), JAMMA emits a warning via
`jamma.lmm.runner.warn_if_small_sample()`. LMM-based GWAS has insufficient
statistical power below this scale regardless of optimizer, and this is
precisely the regime where the unimodality assumption above is most likely to
fail. The warning fires once per run from `PipelineRunner`.

### Why Not Brent?

Brent's method is inherently serial: each SNP follows a different convergence
path with a variable number of iterations. Golden section processes all SNPs in
lockstep, enabling batch vectorization across the entire SNP chunk. Replacing
golden section with Brent would require either scalar per-SNP optimization
(destroying batch vectorization, ~100x slower) or padded vectorized Brent
(wasteful, complex state machine that diverges per SNP).

### Convergence

After grid bracketing to ±1 cell, 20 golden section iterations reduce the
bracket by `0.618^20 ≈ 6.6e-5` of the cell width. With a 50-point log-spaced
grid over `[1e-5, 1e5]`, that bracket shrinkage gives a relative lambda
tolerance bounded by `~6.6e-5`. GEMMA's Brent uses `1e-5` per its GSL
configuration. The two converge to the same optimum to within `O(5e-5)`
relative on unimodal REML surfaces -- max observed is `3.80e-5` on
mouse_hs1940. The validation tolerance config (`validation/tolerances.py`)
sets `lambda_rtol = 2e-5` for synthetic data and `5e-5` for real-data
parity tests.

### Boundary Diagnostic

When the grid search maximum falls at the first or last grid point, the bracket
may not contain the true optimum. JAMMA tracks this via
`count_lambda_boundary_hits()` in `results.py` and emits a warning:

```text
Lambda bound convergence: 42 SNPs at l_min=1.0e-05
```

This corresponds to weak-signal SNPs where lambda converges at the optimization
bound — normal behavior that also occurs in GEMMA.

### Divergence Impact

| Signal Strength | Lambda Divergence | Effect on Results |
|-----------------|-------------------|-------------------|
| Strong signal | < 1e-4 relative | Negligible |
| Moderate signal | < 1e-4 relative | Negligible |
| Weak signal (flat MLE surface) | up to ~1.35e-3 relative on mouse_hs1940 | Affects only MLE logl_H1 diagnostic |

P-values, effect sizes, and significance calls are unaffected. The flat region
corresponds to weak-signal SNPs where test statistics are small regardless of
the exact lambda.

See [GEMMA_EQUIVALENCE.md § Lambda Optimization](GEMMA_EQUIVALENCE.md#5-lambda-optimization)
for full error bounds and empirical validation results.

---

## 7. Eigendecomposition Implementation

GEMMA uses GSL (GNU Scientific Library) for eigendecomposition (always DSYEVD).
JAMMA uses `jlinalg.eigh` which dispatches to vendor DSYEVD/DSYEVR via the
jlinalg C layer, with a NumPy fallback when no vendor LAPACK is available.

**DSYEVD vs DSYEVR:** JAMMA defaults to DSYEVD (faster, O(N^2) workspace) and
falls back to DSYEVR (slower, O(N) workspace) when DSYEVD won't fit in memory.
GEMMA always uses DSYEVD. Both LAPACK drivers produce equivalent results within
backward error bounds (`O(n * eps_mach * ||K||)`). The DSYEVR fallback can
increase the maximum sample count by ~40% for a given machine size.

**ILP64 requirement:** For large-sample GWAS (50k+), the kinship matrix exceeds
the int32 element limit (~2.1 billion elements at ~46k x 46k). jlinalg
dispatches to vendor LAPACK (ILP64 when available) and supports large matrices
without this limitation.

**Performance:** Vendor LAPACK eigh is highly optimized (multi-threaded,
vectorized). The eigendecomposition is O(n^3) and runs once per dataset. The
C extension batch SNP processing dominates runtime for large datasets.

---

## 8. HWE Test Implementation

### GEMMA

Uses the **Wigginton exact test** — a permutation-based exact test for Hardy-Weinberg equilibrium that is accurate at all sample sizes, including small cohorts.

### JAMMA

Uses a **chi-squared goodness-of-fit test** (df=1) computed via `math.erfc` (stdlib) vectorized over SNPs — no scipy dependency. The chi-squared test compares observed genotype counts to expected counts under HWE. Implementation is in `core/snp_filter.py:compute_hwe_pvalues`.

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

Uses **streaming subtraction**: computes the full kinship matrix K once, then derives each chromosome's LOCO kinship as `K_loco_c = (p × K - p_c × K_c) / (p - p_c)` one at a time, where K_c is the normalized kinship for chromosome c's SNPs and p_c is the SNP count for that chromosome. (Equivalently, `(S_full - S_chr) / (p - p_c)` where S are unnormalized outer-product sums.)

### Divergence Impact

| Aspect | GEMMA | JAMMA |
|--------|-------|-------|
| Math | Same formula | Same formula |
| Memory | O(n_chr × n²) | O(n²) |
| I/O | One pass per chromosome | Two passes total (full K + per-chr subtraction) |

### Rationale

The streaming approach produces mathematically identical LOCO kinship matrices while using constant memory (one K_loco at a time). This is critical for large-sample GWAS where materializing 22 copies of an n×n matrix is infeasible.

The streaming approach produces mathematically identical LOCO kinship matrices while requiring only constant memory (one K_loco buffer at a time).

---

## 10. LOCO with External Kinship

### GEMMA

GEMMA's `-loco` flag does **not** apply LOCO adjustment to an externally provided
kinship matrix (`-k`). When both `-loco` and `-k` are specified, GEMMA uses the
full kinship matrix unchanged for all chromosomes — the `-loco` flag is silently
ignored for the kinship component. This means users who pre-compute kinship
externally and then run `gemma -lmm 1 -loco -k K.txt` get standard (non-LOCO)
association results despite requesting LOCO mode.

### JAMMA

JAMMA makes `-loco` and `-k` mutually exclusive. LOCO mode always computes
kinship internally via streaming subtraction, ensuring correct per-chromosome
LOCO kinship. There is no silent fallback to non-LOCO behavior.

### Divergence Impact

| Scenario             | GEMMA                                     | JAMMA                            |
|----------------------|-------------------------------------------|----------------------------------|
| `-loco` without `-k` | Computes LOCO kinship internally          | Computes LOCO kinship internally |
| `-loco` with `-k`    | Silently uses full K (no LOCO adjustment) | Rejects with clear error         |

### Rationale

Using the full kinship matrix in LOCO mode defeats the purpose of LOCO analysis
(eliminating proximal contamination). Making this an error prevents users from
unknowingly running non-LOCO analysis.

### Validation Approach

JAMMA's LOCO integration tests use a two-step validation: (1) JAMMA computes
per-chromosome LOCO kinship matrices, (2) GEMMA runs standard LMM with each
LOCO kinship as external `-k` input. This validates that JAMMA's LOCO kinship
formula produces results numerically equivalent to GEMMA's LMM expectations.
See [`tests/test_loco_numpy.py`](../tests/test_loco_numpy.py), which compares
against the committed GEMMA reference output in
`tests/fixtures/gemma_loco/gemma_loco_chr{1,2,3}.assoc.txt`. The PLINK binary
those were generated from is too large to commit, so the fixture's `.bed` ships
out of band and the comparisons skip with "gemma_loco fixture not available"
when it is absent. Regenerate it with
[`scripts/generate_loco_fixtures.sh`](../scripts/generate_loco_fixtures.sh).

---

## 11. Default File Format: Binary .npy vs Text

GEMMA writes kinship matrices as space-separated text (`.cXX.txt`) and eigendecomposition
as line-separated text (`.eigenD.txt`, `.eigenU.txt`). No binary format option.

JAMMA defaults to binary NumPy format (`.cXX.npy`, `.eigenD.npy`, `.eigenU.npy`). The
`--legacy-text` flag produces GEMMA-compatible text files. When `--legacy-text` is
used for eigen files, JAMMA also writes `.npy` sidecar files for faster subsequent reads.

JAMMA reads GEMMA's text formats natively. Pass GEMMA-produced `.cXX.txt` directly
to `-k`, and `.eigenD.txt`/`.eigenU.txt` to `-d`/`-u`. No conversion needed.

When a `.txt` path is passed, JAMMA checks for a `.npy` sibling with the same stem.
If the `.npy` exists and is at least as recent (by mtime) as the `.txt`, the binary
file is loaded instead. If the `.txt` is newer (e.g., regenerated by GEMMA), the
stale `.npy` is ignored and text is parsed.

| Aspect | GEMMA | JAMMA |
|--------|-------|-------|
| Default kinship format | `.cXX.txt` (text) | `.cXX.npy` (binary) |
| Default eigen format | `.eigenD.txt`, `.eigenU.txt` | `.eigenD.npy`, `.eigenU.npy` |
| Reads other format | N/A (text only) | Auto-detects both formats |
| `--legacy-text` | N/A | Produces GEMMA-compatible text |

Binary `.npy` is 10-100x faster for I/O at scale and ~50% smaller on disk. GEMMA
text compatibility is preserved via `--legacy-text` and the auto-detecting reader.

---

## 12. Early Sample Filtering in Kinship Computation

### GEMMA

GEMMA computes kinship over all samples in the PLINK `.fam` file, regardless of
phenotype or covariate missingness. Sample exclusion is applied downstream during
LMM — the kinship matrix is always n_samples × n_samples.

### JAMMA

When `save_kinship=False` and some samples have missing phenotype or covariates,
JAMMA passes `valid_indices` to `compute_kinship_streaming`, which subsets each
genotype chunk to the valid rows before accumulation. The resulting kinship matrix
has shape (n_valid, n_valid) and is never expanded to n_samples × n_samples.

When `save_kinship=True`, JAMMA computes the full n_samples × n_samples kinship
so the saved file is reusable across phenotype masks. In that case behavior matches
GEMMA.

### Divergence Impact

| Condition | GEMMA | JAMMA |
|-----------|-------|-------|
| All samples valid | n × n | n × n |
| Some missing, save_kinship=False | n × n (masked later) | n_valid × n_valid |
| Some missing, save_kinship=True | n × n | n × n |

The kinship values themselves are identical — only the matrix dimensions differ in
the `save_kinship=False` path. This is a memory optimization, not a numerical change.

---

## 13. LOCO `--legacy-text` Support (Resolved)

This was previously a divergence: `--legacy-text` was honored on the standard
code path but ignored on the LOCO path, so `--loco --legacy-text --write-eigen`
silently produced binary `.npy` artifacts instead of the GEMMA-compatible
`.cXX.txt` / `.eigenD.txt` / `.eigenU.txt` files the user asked for.

**Fixed.** `run_lmm_loco()` now accepts a `legacy_text` parameter and threads it
through the per-chromosome eigen-cache lookup (`_find_loco_eigen_cache`), the
kinship save (filename suffix + `write_kinship_matrix`), and the eigen write
(`write_eigen_files`). `PipelineRunner._run_loco` forwards
`config.legacy_text`, so `--loco --legacy-text` now writes GEMMA text artifacts
on the LOCO path identically to the standard path. As with the non-LOCO path,
text mode writes the `.txt` files plus `.npy` sidecars for fast reload.

---

## Summary Table

| Feature | GEMMA Behavior | JAMMA Behavior | Impact |
|---------|---------------|----------------|--------|
| safe_sqrt(-5.0) | sqrt(5.0) | NaN | Edge case only |
| P_xx = 0 | inf/NaN mix | NaN | Degenerate SNPs |
| Px_yy clamping | None | 1e-8 floor | Numerical stability |
| logdet with neg eigenvalues | log(abs(v)) | log(abs(v)) | Aligned |
| Monomorphic detection | Count-based | Variance-based | Aligned (equivalent) |
| Covariates | n_cvt >= 1 | n_cvt >= 1 | Aligned (since v1.2) |
| Lambda optimizer | Brent (serial) | Golden section (batch vectorized, assumes unimodal) | < 1e-4 relative; see §6 |
| Eigendecomp library | GSL (DSYEVD only) | jlinalg vendor dispatch (DSYEVD/DSYEVR) | Large-sample support (ILP64), DSYEVR fallback |
| HWE test | Wigginton exact | Chi-squared (df=1) | Identical for large n |
| LOCO kinship | Materialized all | Streaming subtraction | Same math, lower memory |
| LOCO + external kinship | Silently uses full K | Rejects (mutual exclusion) | Correctness guard |
| Default file format | Text (`.cXX.txt`, `.eigenD.txt`) | Binary `.npy` (`--legacy-text` for text) | GEMMA files read natively |
| Early sample filtering | Kinship always n × n | Kinship at n_valid × n_valid when save_kinship=False | Memory saving only; values identical |
| LOCO + `--legacy-text` | N/A (GEMMA has no LOCO) | Honored on LOCO path — writes `.cXX.txt` / `.eigenD.txt` / `.eigenU.txt` (see §13) | Parity with standard path |

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
| BIMBAM format input | `-g`, `-p`, `-a` | JAMMA uses PLINK binary format exclusively. BIMBAM is a legacy text format (~10-50x larger than PLINK binary) with no QC tooling. Convert with `plink --import-dosage file.mean.genotype --fam file.fam --make-bed --out output`. |

---

## Validation Strategy

1. **Real-world data**: JAMMA matches GEMMA within tolerance on actual GWAS datasets
2. **Edge case tests**: `tests/test_hypothesis.py` verifies JAMMA's robust behavior
3. **No silent failures**: Divergences produce NaN, not silently wrong values

For full empirical validation results (small-scale and production-scale), see
[GEMMA_EQUIVALENCE.md § Empirical Results](GEMMA_EQUIVALENCE.md#empirical-results).
