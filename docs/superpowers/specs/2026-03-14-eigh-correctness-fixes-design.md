# Phase 80 eigh Correctness & Performance Fixes

## Problem

Phase 80 delivered a working eigh implementation, but with significant deviations from the plan:

1. **dsytrd**: Unblocked (dsytd2-style) instead of blocked DLATRD + dsyr2k
2. **dormtr**: Per-reflector rank-1 updates instead of WY blocked via dgemm
3. **dstedc D&C merge**: Broken — wrong deflation threshold, naive secular eigenvector formula, excessive memory allocation per recursion level
4. **DSTEDC_BASE=2000**: Hides the broken D&C by routing everything to QR base case

The implementation passes all tests because QR is correct, but the D&C path (the entire point of "divide-and-conquer") is dead code with known bugs.

## Scope

Fix all four C source files. No changes to jlinalg.h, pymodule.c, or `__init__.py`.

## Design

### 1. dsytrd.c — Blocked DLATRD + dsyr2k

Replace unblocked algorithm with LAPACK's blocked scheme:

**Algorithm:**
```
for j = 0 to N-2 step NB:
    nb_actual = min(NB, N-1-j)
    DLATRD(A[j:N, j:N], nb_actual) → V[m×nb], W[m×nb], d[nb], e[nb], tau[nb]
    if j + nb_actual < N-1:
        dsyr2k(A_trail, V_trail, W_trail)   # rank-2k trailing update
    tail: unblocked dsytd2 for remaining columns (no trailing update)
```

**DLATRD panel factorization** (within each NB-column block):
For each column i within the panel:
1. `dlarfg` on the sub-diagonal to get reflector v_i, tau_i
2. `dsymv`: p = tau * A_trail * v_i (symmetric matrix-vector product)
3. Correct for previously applied reflectors: p -= tau * V * (W^T * v) + tau * W * (V^T * v)
4. `alpha2 = (tau/2) * p^T * v`
5. `w = p - alpha2 * v`
6. Store v_i in V, w_i in W

**dsymv**: Implemented as a static helper within dsytrd.c. Reads the full trailing submatrix using symmetry (lower triangle stored, access both via index arithmetic). Not added to the dispatch table — internal only.

**NB=64**: Matches the plan. Tunable constant.

### 2. dormtr.c — WY blocked via dgemm

Replace per-reflector rank-1 updates with DLARFT + DLARFB:

**Algorithm:**
```
for j = last_block down to 0 step NB:
    nb_actual = min(NB, N-1-j)
    DLARFT(A, tau, j, nb_actual) → T[nb×nb]  (upper triangular)
    DLARFB(V, T, C[j+1:N, :])                (block reflector application)
```

**DLARFT** (form triangular factor T):
T encodes H_0 * H_1 * ... * H_{nb-1} = I - V * T * V^T.
```
T[0,0] = tau[0]
for i = 1..nb-1:
    T[0:i, i] = -tau[i] * T[0:i, 0:i] * V^T[0:i, :] * v_i
    T[i, i] = tau[i]
```
T is small (NB×NB = 64×64), computed with direct loops.

**DLARFB** (apply block reflector):
C = (I - V * T * V^T) * C:
1. W = V^T * C[j+1:N, :] → dgemm(V^T, C, W), shape nb×M
2. W = T * W → triangular multiply (loops, T is 64×64)
3. C[j+1:N, :] -= V * W → dgemm(V, W, temp) then subtract

### 3. dstedc.c — Fix D&C merge

**Lower DSTEDC_BASE to 25.**

**Deflation (LAPACK DLAED2-style):**
- Type-a: `rho * |z[i]| < 8 * eps * max(|d[i]|, |z[i]|)` — tests whether the component contributes meaningfully
- Type-b: `|d[i] - d[j]| < 8 * eps * max(|d[i]|, |d[j]|)` — local relative tolerance, not global ||T||

**z vector normalization:**
After extracting boundary rows from Q_L and Q_R:
```c
for (j = 0; j < n; j++) z_vec[j] /= sqrt(2.0);
rho = fabs(2.0 * e[m-1]);
```

**Secular eigenvector formula (dlaed3 product formula):**
Replace naive `q[k] = z[k] / (d[k] - lambda)` with:
```c
for each eigenvalue lambda[i]:
    for k = 0..n-1:
        num = 1.0;  // running product
        den = 1.0;
        for j = 0..n-1, j != k:
            num *= (lambda[j] - d[k]);  // use NEW eigenvalues in numerator
            den *= (d[j] - d[k]);       // use OLD eigenvalues in denominator
        q[k] = z[k] * sqrt(abs(num / den));
    // fix sign of q from secular equation gradient
    normalize q
```
This avoids the `1/(d[k]-lambda)` singularity entirely.

**Memory — top-level allocation:**
`jlinalg_dstedc_c` allocates one N×N workspace buffer and passes it through recursion. `merge_rank1` receives workspace pointer instead of malloc/free per call. Additional O(N) work arrays (d_defl, z_defl, permutation, etc.) allocated once at top level.

### 4. eigh.c — Documentation only

Update header comment to reflect actual memory model:
- Input K: N×N (overwritten)
- Output U: N×N (eigenvectors)
- dstedc workspace: N×N (allocated internally by dstedc)
- O(N) vectors: d, e, tau, plus dstedc work arrays
- Total: 3N² + O(N)

### Tests

Existing tests should pass without tolerance changes for synthetic data. The mouse_hs1940 test (N=1940) will now exercise the D&C path. If the real-data test needs tolerance adjustment, investigate the root cause first — don't blindly relax.

### Files Modified

| File | Change |
|------|--------|
| `src/jamma/jlinalg/src/dsytrd.c` | Rewrite: blocked DLATRD + dsyr2k |
| `src/jamma/jlinalg/src/dormtr.c` | Rewrite: WY blocked via DLARFT/DLARFB + dgemm |
| `src/jamma/jlinalg/src/dstedc.c` | Fix: deflation, z normalization, product formula, memory |
| `src/jamma/jlinalg/src/eigh.c` | Documentation update only |
| `tests/test_jlinalg_eigh.py` | Potential tolerance adjustments for real-data test |

### Files NOT Modified

| File | Reason |
|------|--------|
| `jlinalg.h` | Public API unchanged |
| `pymodule.c` | Python wrapper unchanged |
| `__init__.py` | Fallback unchanged |
| `hatch_build.py` | Build flags unchanged |
