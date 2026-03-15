# eigh Correctness & Performance Fixes

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all four LAPACK eigendecomposition C source files to match LAPACK's algorithms — blocked dsytrd, blocked dormtr, correct D&C merge in dstedc.

**Architecture:** Four independent file rewrites. dsytrd gets DLATRD panel factorization + dsyr2k trailing update. dormtr gets WY blocked application via DLARFT/DLARFB + dgemm. dstedc gets correct deflation, z-vector normalization, dlaed3 product formula for secular eigenvectors, and top-level workspace allocation. eigh gets documentation fixes only.

**Tech Stack:** C11, numpy C API, jblas BLAS primitives (dgemm, dsyr2k, ddot, daxpy, dscal, dgemv)

**Spec:** `docs/superpowers/specs/2026-03-14-eigh-correctness-fixes-design.md`

**Test commands:**
- Fast eigh tests: `uv run pytest tests/test_jblas_eigh.py -x -n0 -v -k "not slow and not mouse"`
- Slow eigh tests: `uv run pytest tests/test_jblas_eigh.py -x -n0 -v`
- Full suite: `uv run pytest tests/ -x`

**CRITICAL RULES:**
- NEVER run `git add --force` or `git add -f`
- NEVER stage files in .gitignore (.planning/, .claude/, .beads/, CLAUDE.md)
- Read CLAUDE.md before starting work — follow project conventions
- Use `uv run pytest tests/ -x` for test verification (picks up `-n 3` from addopts)
- LAPACK source files compile with `-O2` only (no `-ffast-math`). Do NOT add optimization flags.

---

## Chunk 1: dsytrd.c — Blocked DLATRD + dsyr2k

### Task 1: Rewrite dsytrd.c with blocked algorithm

This is a full rewrite of `src/jamma/jblas/src/dsytrd.c`. The file currently implements an unblocked dsytd2-style algorithm. Replace with LAPACK's blocked scheme: DLATRD panel factorization + dsyr2k trailing update.

**Files:**
- Rewrite: `src/jamma/jblas/src/dsytrd.c`
- Test: `tests/test_jblas_eigh.py` (existing tests validate correctness)

**Available BLAS primitives** (from `jblas.h`, use via `jblas_dispatch.*` or direct `jblas_*_c` calls):
- `jblas_dispatch.ddot(n, x, 1, y, 1)` — dot product
- `jblas_dispatch.daxpy(n, alpha, x, 1, y, 1)` — y += alpha*x
- `jblas_dispatch.dscal(n, alpha, x, 1)` — x *= alpha
- `jblas_dispatch.dgemv(m, n, A, x, y)` — y = A*x (row-major, no transpose)
- `jblas_dgemm_c(M, N, K, A, lda, B, ldb, C, ldc, transa, transb)` — C = op(A)*op(B)
- `jblas_dsyr2k_c(N, K, A, lda, B, ldb, C, ldc)` — C -= A*B^T + B*A^T

**Key design decisions:**
- NB=64 (block size constant, tunable)
- `dsymv_lower` is a static helper within this file (not in dispatch table). It computes `y = alpha * A * x` where A is symmetric, lower triangle stored in row-major. Must handle the full trailing submatrix using symmetry: for row i, col k, access `A[max(i,k)*lda + min(i,k)]` for lower, `A[min(i,k)*lda + max(i,k)]` for upper — but since A is stored with both triangles updated by the rank-2k step, we can just read A directly in the trailing submatrix.
- `dlarfg` is preserved from current code (it's correct)
- The rank-2k trailing update uses `jblas_dsyr2k_c` which does `C -= A*B^T + B*A^T`. The DLATRD algorithm produces V and W such that `A_trail -= V*W^T + W*V^T`, which matches dsyr2k's sign convention exactly.

- [ ] **Step 1: Read the current dsytrd.c and understand the structure**

Read `src/jamma/jblas/src/dsytrd.c`. Note: `dlarfg` (lines 54-81) is correct and will be preserved. Everything else gets rewritten.

- [ ] **Step 2: Write the new dsytrd.c**

Replace the entire file with the blocked implementation. The structure is:

```c
/**
 * dsytrd.c — Blocked Householder tridiagonalization for jblas.
 *
 * Implements jblas_dsytrd_c: reduces a symmetric N x N matrix A (stored
 * row-major, lower triangle used) to tridiagonal form T via orthogonal
 * similarity A = Q T Q^T.
 *
 * Algorithm: Blocked DLATRD panel factorization + dsyr2k trailing update.
 *
 * For each NB-column panel (j = 0 to N-2 step NB):
 *   1. DLATRD: Factor NB columns, producing V[m x nb] and W[m x nb].
 *   2. dsyr2k: A_trail -= V_trail * W_trail^T + W_trail * V_trail^T.
 *   The last panel (or panels smaller than NB) uses unblocked factorization
 *   with no trailing update.
 *
 * DLATRD panel (for each column i within the NB block):
 *   1. dlarfg on A[j+i+1:N, j+i] to get reflector v_i, tau_i.
 *   2. dsymv: p = tau * A_trail * v_i (symmetric matrix-vector product).
 *   3. Correct for previously applied reflectors within this panel:
 *        p -= tau * V[:, 0:i] * (W[:, 0:i]^T * v)
 *        p -= tau * W[:, 0:i] * (V[:, 0:i]^T * v)
 *      These are dgemv calls (small: i columns at most NB).
 *   4. alpha2 = (tau/2) * dot(p, v).
 *   5. w = p - alpha2 * v.
 *   6. Store v in V[:, i], w in W[:, i].
 *
 * dsymv_lower (static helper):
 *   Computes y = alpha * A * x for symmetric A where both triangles of the
 *   trailing submatrix are valid (dsyr2k updates both). Simple row-major
 *   matrix-vector product reading A directly.
 *
 * On exit:
 *   d[i]   = diagonal element i  (i = 0..N-1)
 *   e[i]   = off-diagonal element i  (i = 0..N-2)
 *   tau[i] = Householder scalar for reflector i  (i = 0..N-2)
 *   Lower triangle of A holds the Householder vectors (LAPACK dsytrd convention).
 *
 * Memory:
 *   Workspace: V[N x NB] + W[N x NB] = 2*N*NB doubles, allocated once.
 *   No N x N temporary buffers.
 */
```

Key implementation details:

**V/W layout**: `m_panel x nb_alloc` row-major. `V[k * nb_alloc + i]` = row k, column i. Column i has stride `nb_alloc`. This layout is chosen so that the trailing portion `V + nb * nb_alloc` is a contiguous row-major `m_trail x nb` submatrix for dsyr2k.

**dsymv_lower**: Symmetric matrix-vector product for lower-triangle storage. Within the DLATRD panel, A hasn't been updated by the current panel's rank-2k yet (that happens after DLATRD completes), so dsymv must use the lower-triangle-with-symmetry access pattern. Takes a stride parameter for x (columns of V have stride `nb_alloc`).

**V/W population**: After dlarfg, the Householder vector is stored in column j+i of A with stride lda (one element per row, non-contiguous). Copy v into V[:,i] immediately — this is what LAPACK's DLATRD does. Then use V[:,i] (stride nb_alloc) for all subsequent BLAS operations.

**Trailing dsyr2k update**: After DLATRD completes nb columns, `V + nb * nb_alloc` is the trailing portion as a contiguous `m_trail x nb` row-major submatrix. `jblas_dsyr2k_c(m_trail, nb, V_trail, nb_alloc, W_trail, nb_alloc, A_trail, lda)` does `A_trail -= V_trail * W_trail^T + W_trail * V_trail^T`.

Here is the complete rewrite. Note this is ~300 lines replacing the current ~190 lines:

```c
/**
 * dsytrd.c — Blocked Householder tridiagonalization for jblas.
 * [header comment as above]
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include "jblas.h"

#define NB_DSYTRD 64

/* dlarfg — preserved from current implementation, unchanged */
static void dlarfg(npy_intp n, double *alpha, double *x, npy_intp incx,
                   double *tau) { /* ... same as current ... */ }

/* dsymv_lower — symmetric matrix-vector product for lower-triangle storage.
 * y = alpha * A * x, where A[n x n] is symmetric, lower triangle stored.
 * x has stride incx, y is contiguous (stride 1). */
static void dsymv_lower(npy_intp n, double alpha,
                         const double *A, npy_intp lda,
                         const double *x, npy_intp incx,
                         double *y)
{
    for (npy_intp i = 0; i < n; i++) {
        double s = 0.0;
        for (npy_intp j = 0; j <= i; j++)
            s += A[i * lda + j] * x[j * incx];
        for (npy_intp j = i + 1; j < n; j++)
            s += A[j * lda + i] * x[j * incx];
        y[i] = alpha * s;
    }
}

/* dlatrd_panel — Factor nb columns of the symmetric matrix, producing V and W.
 *
 * A:   N x N symmetric matrix (row-major, lower triangle), modified in place.
 *      On exit, Householder vectors stored in columns j..j+nb-1 below diagonal.
 * j:   Starting column index in A.
 * nb:  Number of columns to factor.
 * N:   Full matrix dimension.
 * d:   Diagonal output (d[j..j+nb-1] written).
 * e:   Off-diagonal output (e[j..j+nb-1] written).
 * tau: Householder scalars (tau[j..j+nb-1] written).
 * V:   m_panel x nb_alloc workspace (row-major). V[:, i] = i-th Householder vector.
 * W:   m_panel x nb_alloc workspace (row-major). W[:, i] = i-th update vector.
 * nb_alloc: Leading dimension of V and W (>= nb).
 * p:   Scratch vector of length m_panel.
 */
static void dlatrd_panel(double *A, npy_intp lda, npy_intp N,
                          npy_intp j, npy_intp nb,
                          double *d, double *e, double *tau,
                          double *V, double *W, npy_intp nb_alloc,
                          double *p)
{
    npy_intp m_panel = N - j - 1;  /* total trailing rows for this panel */

    for (npy_intp i = 0; i < nb; i++) {
        npy_intp col = j + i;          /* absolute column index */
        npy_intp m   = N - col - 1;    /* trailing size for this column */
        npy_intp off = m_panel - m;    /* offset into V/W for this column's v */

        /* Record diagonal */
        d[col] = A[col * lda + col];

        if (m <= 0) {
            tau[col] = 0.0;
            continue;
        }

        /* Generate Householder reflector from A[col+1:N, col] */
        double alpha_val = A[(col + 1) * lda + col];
        double *x_tail = (m > 1) ? &A[(col + 2) * lda + col] : NULL;
        dlarfg(m, &alpha_val, x_tail, lda, &tau[col]);
        e[col] = alpha_val;
        A[(col + 1) * lda + col] = alpha_val;

        if (tau[col] == 0.0) {
            /* No reflection — zero out V and W columns */
            for (npy_intp k = 0; k < m_panel; k++) {
                V[k * nb_alloc + i] = 0.0;
                W[k * nb_alloc + i] = 0.0;
            }
            continue;
        }

        /* Copy v into V[:, i] — rows 0..off-1 are zero, then v[0]=1, v[1..] from A */
        for (npy_intp k = 0; k < off; k++)
            V[k * nb_alloc + i] = 0.0;
        V[off * nb_alloc + i] = 1.0;
        for (npy_intp k = 1; k < m; k++)
            V[(off + k) * nb_alloc + i] = A[(col + 1 + k) * lda + col];

        /* Temporarily set A[col+1, col] = 1.0 for dsymv */
        double saved_e = A[(col + 1) * lda + col];
        A[(col + 1) * lda + col] = 1.0;

        /* p = tau * A_trail * v
         * A_trail = A[col+1:N, col+1:N], size m x m
         * v = A[col+1:N, col], stride lda */
        dsymv_lower(m, tau[col],
                     A + (col + 1) * lda + (col + 1), lda,
                     A + (col + 1) * lda + col, lda,
                     p + off);

        /* Restore A[col+1, col] */
        A[(col + 1) * lda + col] = saved_e;

        /* Zero the leading part of p */
        for (npy_intp k = 0; k < off; k++)
            p[k] = 0.0;

        /* Correct for previously applied reflectors (i > 0):
         * p -= tau * V[:, 0:i] * (W[:, 0:i]^T * v) + tau * W[:, 0:i] * (V[:, 0:i]^T * v)
         *
         * Let v_col = V[:, i] (already stored).
         * dot1[j] = sum_k W[k, j] * V[k, i]  for j = 0..i-1  (W^T * v)
         * dot2[j] = sum_k V[k, j] * V[k, i]  for j = 0..i-1  (V^T * v)
         * p -= tau * V[:, 0:i] * dot1 + tau * W[:, 0:i] * dot2
         */
        if (i > 0) {
            double t = tau[col];
            /* Compute dot products: use V/W columns (stride nb_alloc) */
            for (npy_intp prev = 0; prev < i; prev++) {
                double d1 = 0.0, d2 = 0.0;
                for (npy_intp k = 0; k < m_panel; k++) {
                    d1 += W[k * nb_alloc + prev] * V[k * nb_alloc + i];
                    d2 += V[k * nb_alloc + prev] * V[k * nb_alloc + i];
                }
                /* p -= t * d1 * V[:, prev] + t * d2 * W[:, prev] */
                for (npy_intp k = 0; k < m_panel; k++) {
                    p[k] -= t * (d1 * V[k * nb_alloc + prev]
                               + d2 * W[k * nb_alloc + prev]);
                }
            }
        }

        /* alpha2 = (tau/2) * dot(p, v) */
        double dot_pv = 0.0;
        for (npy_intp k = 0; k < m_panel; k++)
            dot_pv += p[k] * V[k * nb_alloc + i];
        double alpha2 = (tau[col] / 2.0) * dot_pv;

        /* w = p - alpha2 * v */
        for (npy_intp k = 0; k < m_panel; k++)
            W[k * nb_alloc + i] = p[k] - alpha2 * V[k * nb_alloc + i];
    }
}

/* jblas_dsytrd_c — public API */
int jblas_dsytrd_c(npy_intp N, double *A, npy_intp lda,
                   double *d, double *e, double *tau)
{
    if (N <= 0) return 0;
    if (N == 1) { d[0] = A[0]; return 0; }

    memset(tau, 0, (size_t)(N - 1) * sizeof(double));

    npy_intp m_panel = N - 1;  /* max trailing size */
    npy_intp nb_alloc = NB_DSYTRD;

    /* Allocate V[m_panel x nb_alloc], W[m_panel x nb_alloc], p[m_panel] */
    double *V = (double *)calloc((size_t)m_panel * (size_t)nb_alloc, sizeof(double));
    double *W = (double *)calloc((size_t)m_panel * (size_t)nb_alloc, sizeof(double));
    double *p = (double *)malloc((size_t)m_panel * sizeof(double));
    if (!V || !W || !p) {
        free(V); free(W); free(p);
        return -1;
    }

    for (npy_intp j = 0; j < N - 1; j += NB_DSYTRD) {
        npy_intp nb = (N - 1 - j < NB_DSYTRD) ? (N - 1 - j) : NB_DSYTRD;
        npy_intp m_cur = N - j - 1;

        /* Zero V and W for this panel */
        memset(V, 0, (size_t)m_panel * (size_t)nb_alloc * sizeof(double));
        memset(W, 0, (size_t)m_panel * (size_t)nb_alloc * sizeof(double));

        /* DLATRD: factor nb columns */
        dlatrd_panel(A, lda, N, j, nb, d, e, tau, V, W, nb_alloc, p);

        /* Trailing dsyr2k update: A_trail -= V_trail * W_trail^T + W_trail * V_trail^T
         * Only if there are rows beyond this panel. */
        npy_intp m_trail = N - j - nb - 1;
        if (m_trail > 0) {
            /* V_trail starts at row nb within V, W similarly.
             * V_trail[m_trail x nb] with lda = nb_alloc.
             * A_trail = A[j+nb+1 : N, j+nb+1 : N], size m_trail x m_trail. */
            /* BUT WAIT: V's "trailing" portion has offset. V was allocated for
             * m_panel rows. The first row of V corresponds to row j+1 in A.
             * Row nb of V corresponds to row j+nb+1 in A = first row of trailing.
             * So V_trail = V + nb * nb_alloc, with m_trail rows, nb columns. */
            jblas_dsyr2k_c(m_trail, nb,
                           V + nb * nb_alloc, nb_alloc,
                           W + nb * nb_alloc, nb_alloc,
                           A + (j + nb + 1) * lda + (j + nb + 1), lda);
        }
    }

    /* Capture final diagonal (may not have been set if N-1 was exactly at panel boundary) */
    d[N - 1] = A[(N - 1) * lda + (N - 1)];

    free(V);
    free(W);
    free(p);
    return 0;
}
```

**Important edge case**: When `nb < NB_DSYTRD` (last panel), there's no trailing update needed because there are no rows beyond the panel. The `m_trail > 0` guard handles this.

**Important edge case**: For N=2, m_panel=1, nb=1. DLATRD processes 1 column, no trailing update. This is equivalent to the unblocked case.

- [ ] **Step 3: Rebuild and run tests**

```bash
uv run pip install -e . --no-build-isolation 2>&1 | tail -5
uv run pytest tests/test_jblas_eigh.py -x -n0 -v -k "not slow and not mouse"
```

Expected: All fast eigh tests pass. If any fail, debug the dsytrd implementation (most likely: V/W indexing, dsymv symmetry access, or rank-2k trailing offset).

- [ ] **Step 4: Run slow tests including mouse_hs1940**

```bash
uv run pytest tests/test_jblas_eigh.py -x -n0 -v
```

Expected: All tests pass. The mouse_hs1940 test (N=1940) exercises the blocked path (30 panels of 64 + 1 tail of 14).

- [ ] **Step 5: Commit**

```bash
git add src/jamma/jblas/src/dsytrd.c
git commit -m "fix(80): dsytrd — blocked DLATRD + dsyr2k trailing update

Replace unblocked dsytd2-style algorithm with LAPACK's blocked scheme.
NB=64 panel factorization via DLATRD, trailing rank-2k update via
jblas_dsyr2k_c. Static dsymv_lower helper for symmetric matrix-vector
product within panel factorization."
```

---

## Chunk 2: dormtr.c — WY blocked via DLARFT/DLARFB + dgemm

### Task 2: Rewrite dormtr.c with WY blocked algorithm

Full rewrite of `src/jamma/jblas/src/dormtr.c`. Currently applies per-reflector rank-1 updates. Replace with DLARFT (form triangular T factor) + DLARFB (block reflector application via dgemm).

**Files:**
- Rewrite: `src/jamma/jblas/src/dormtr.c`
- Test: `tests/test_jblas_eigh.py` (existing tests)

**Key design decisions:**
- NB must match dsytrd's NB (64). Use same `#define NB_DORMTR 64`.
- Process reflectors in blocks of NB from right to left (last block first).
- DLARFT forms the NB x NB upper triangular T factor.
- DLARFB applies: C = (I - V T V^T) C, decomposed into dgemm calls.
- The block size for dgemm calls is nb x M (where M = number of columns in C = N for eigenvectors). The V matrix is vlen x nb. These are not square — dgemm handles rectangular.

**DLARFT** (form T for reflectors j..j+nb-1):
```
T is upper triangular, nb x nb.
T[0,0] = tau[j]
for i = 1..nb-1:
    /* T[0:i, i] = -tau[j+i] * T[0:i, 0:i] * V[0:i]^T * v_i */
    /* v_i is column i of V (the i-th reflector in this block) */
    step 1: z[k] = sum_row V[row, k] * V[row, i]  for k = 0..i-1  (V^T * v_i)
    step 2: T[0:i, i] = -tau[j+i] * T[0:i, 0:i] * z  (triangular matrix-vector)
    T[i, i] = tau[j+i]
```

**DLARFB** (apply block reflector to C):
```
C = (I - V T V^T) C
  = C - V T (V^T C)

step 1: W = V^T @ C[j+1:N, :]     — dgemm(V^T, C, W), W is nb x M
step 2: W = T @ W                   — triangular multiply (nb x nb * nb x M)
step 3: C[j+1:N, :] -= V @ W       — dgemm(V, W, temp) then subtract
```

For step 3, since `jblas_dgemm_c` computes C = A*B (overwriting C, no alpha/beta), we cannot use dgemm directly for the subtract. Instead, loop directly — nb rank-1 updates, each touching vlen x M elements, with sequential memory access for good cache behavior. No extra N^2 memory needed.

- [ ] **Step 1: Write the new dormtr.c**

```c
/**
 * dormtr.c — Blocked Householder back-transformation for jblas.
 *
 * Implements jblas_dormtr_c: applies Q (from dsytrd's Householder vectors
 * stored in the lower triangle of A) to the eigenvector matrix C:
 *   C = Q @ C
 *
 * Algorithm: DLARFT (form triangular T factor) + DLARFB (block application).
 * Processes reflectors in blocks of NB from right to left.
 *
 * DLARFT: Forms upper triangular T[nb x nb] encoding the product
 *   H_j * H_{j+1} * ... * H_{j+nb-1} = I - V * T * V^T
 *
 * DLARFB: Applies (I - V * T * V^T) * C:
 *   1. W = V^T @ C[j+1:N, :]     (nb x M)
 *   2. W = T @ W                   (triangular multiply)
 *   3. C[j+1:N, :] -= V @ W       (rank-nb update via loops)
 *
 * Memory: T[NB x NB] + W[NB x M] + z[NB] scratch. No N x N temporaries.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <numpy/arrayobject.h>
#include "jblas.h"

#define NB_DORMTR 64

/* dlarft — Form upper triangular T for a block of nb Householder reflectors.
 *
 * V: vlen x nb matrix (row-major, stride nb_alloc). V[:, i] is reflector i.
 *    V[0, i] = 1 for i-th reflector (leading element).
 * tau: Householder scalars, length nb.
 * T: nb x nb output (row-major, stride nb_alloc). Upper triangular.
 * z: scratch vector of length nb.
 */
static void dlarft(npy_intp vlen, npy_intp nb,
                    const double *V, npy_intp nb_alloc,
                    const double *tau,
                    double *T, npy_intp T_stride,
                    double *z)
{
    memset(T, 0, (size_t)nb * (size_t)T_stride * sizeof(double));

    for (npy_intp i = 0; i < nb; i++) {
        if (tau[i] == 0.0) {
            T[i * T_stride + i] = 0.0;
            continue;
        }
        T[i * T_stride + i] = tau[i];

        if (i == 0) continue;

        /* z = V[:, 0:i]^T * V[:, i]  (i dot products) */
        for (npy_intp k = 0; k < i; k++) {
            double dot = 0.0;
            for (npy_intp r = 0; r < vlen; r++)
                dot += V[r * nb_alloc + k] * V[r * nb_alloc + i];
            z[k] = dot;
        }

        /* T[0:i, i] = -tau[i] * T[0:i, 0:i] * z
         * T is upper triangular, so T[0:i, 0:i] * z is a triangular matvec. */
        for (npy_intp r = 0; r < i; r++) {
            double s = 0.0;
            for (npy_intp c = r; c < i; c++)  /* upper triangular: c >= r */
                s += T[r * T_stride + c] * z[c];
            T[r * T_stride + i] = -tau[i] * s;
        }
    }
}

int jblas_dormtr_c(npy_intp N, npy_intp M,
                   const double *A, npy_intp lda, const double *tau,
                   double *C, npy_intp ldc)
{
    if (N <= 1 || M <= 0)
        return 0;

    npy_intp nb_alloc = NB_DORMTR;

    /* Allocate workspace: T[nb x nb], W[nb x M], V_block[vlen x nb], z[nb] */
    double *T_buf = (double *)malloc((size_t)nb_alloc * (size_t)nb_alloc * sizeof(double));
    double *W     = (double *)malloc((size_t)nb_alloc * (size_t)M * sizeof(double));
    npy_intp max_vlen = N - 1;
    double *V_block = (double *)malloc((size_t)max_vlen * (size_t)nb_alloc * sizeof(double));
    double *z     = (double *)malloc((size_t)nb_alloc * sizeof(double));
    if (!T_buf || !W || !V_block || !z) {
        free(T_buf); free(W); free(V_block); free(z);
        return -1;
    }

    /* Process reflectors in blocks from right to left.
     * Reflectors are indexed 0..N-2 (n_ref = N-1 total).
     * Blocks: [0..63], [64..127], ..., [last_start..N-2].
     * Process in reverse: last block first. */
    npy_intp n_ref = N - 1;  /* total number of reflectors */
    npy_intp j_start = ((n_ref - 1) / NB_DORMTR) * NB_DORMTR;  /* start of last block */
    for (; j_start >= 0; j_start -= NB_DORMTR) {
        npy_intp nb = n_ref - j_start;
        if (nb > NB_DORMTR) nb = NB_DORMTR;

        npy_intp vlen = N - j_start - 1;  /* length of reflectors in this block */

        /* Build V_block[vlen x nb]: column i is reflector j_start + i.
         * v[0] = 1 (implicit), v[k] = A[(j_start+i+1+k)*lda + (j_start+i)] for k >= 1.
         *
         * But reflectors within a block have different starting rows:
         * reflector j_start+i has v[0]=1 at row j_start+i+1, with trailing elements
         * in rows j_start+i+2..N-1. In V_block, column i represents reflector j_start+i.
         * The vector has (N - j_start - i - 1) elements starting at V_block row i.
         * Rows 0..i-1 of column i are zero.
         *
         * V_block[r, i]:
         *   r < i:  0
         *   r == i: 1 (implicit v[0])
         *   r > i:  A[(j_start + 1 + r) * lda + (j_start + i)]
         */
        memset(V_block, 0, (size_t)vlen * (size_t)nb_alloc * sizeof(double));
        for (npy_intp i = 0; i < nb; i++) {
            V_block[i * nb_alloc + i] = 1.0;
            for (npy_intp r = i + 1; r < vlen; r++)
                V_block[r * nb_alloc + i] = A[(j_start + 1 + r) * lda + (j_start + i)];
        }

        /* DLARFT: form T[nb x nb] */
        dlarft(vlen, nb, V_block, nb_alloc, tau + j_start, T_buf, nb_alloc, z);

        /* DLARFB: C = (I - V T V^T) C
         * Step 1: W = V^T @ C[j_start+1:N, :]
         *   W[i, c] = sum_r V_block[r, i] * C[(j_start+1+r)*ldc + c]
         *   W is nb x M. */
        memset(W, 0, (size_t)nb * (size_t)M * sizeof(double));
        for (npy_intp i = 0; i < nb; i++) {
            for (npy_intp r = 0; r < vlen; r++) {
                double v_ri = V_block[r * nb_alloc + i];
                if (v_ri == 0.0) continue;
                const double *C_row = C + (j_start + 1 + r) * ldc;
                double *W_row = W + i * M;
                for (npy_intp c = 0; c < M; c++)
                    W_row[c] += v_ri * C_row[c];
            }
        }

        /* Step 2: W = T @ W  (T is nb x nb upper triangular, W is nb x M)
         * Process top-to-bottom. This is safe in-place: row i reads from
         * rows >= i (T is upper triangular), and we've only modified rows < i. */
        for (npy_intp i = 0; i < nb; i++) {
            for (npy_intp c = 0; c < M; c++) {
                double s = 0.0;
                for (npy_intp k = i; k < nb; k++)
                    s += T_buf[i * nb_alloc + k] * W[k * M + c];
                W[i * M + c] = s;
            }
        }

        /* Step 3: C[j_start+1:N, :] -= V @ W
         * For each reflector column i, add -V[:, i] outer W[i, :] */
        for (npy_intp i = 0; i < nb; i++) {
            const double *W_row = W + i * M;
            for (npy_intp r = 0; r < vlen; r++) {
                double v_ri = V_block[r * nb_alloc + i];
                if (v_ri == 0.0) continue;
                double *C_row = C + (j_start + 1 + r) * ldc;
                for (npy_intp c = 0; c < M; c++)
                    C_row[c] -= v_ri * W_row[c];
            }
        }

        if (j_start == 0) break;
    }

    free(T_buf);
    free(W);
    free(V_block);
    free(z);
    return 0;
}
```

- [ ] **Step 2: Rebuild and run fast tests**

```bash
uv run pip install -e . --no-build-isolation 2>&1 | tail -5
uv run pytest tests/test_jblas_eigh.py -x -n0 -v -k "not slow and not mouse"
```

Expected: All fast tests pass. If not, debug dormtr (most likely: V_block indexing, DLARFT T computation, or DLARFB step 2 in-place correctness).

- [ ] **Step 3: Run full eigh tests**

```bash
uv run pytest tests/test_jblas_eigh.py -x -n0 -v
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/jamma/jblas/src/dormtr.c
git commit -m "fix(80): dormtr — WY blocked via DLARFT/DLARFB

Replace per-reflector rank-1 updates with blocked WY algorithm.
DLARFT forms upper triangular T[NB x NB] encoding the block
reflector product. DLARFB applies (I - V*T*V^T)*C via matrix
products. NB=64 matching dsytrd."
```

---

## Chunk 3: dstedc.c — Fix D&C merge path

### Task 3: Fix dstedc divide-and-conquer merge

This is the most complex task. Fix four issues in `src/jamma/jblas/src/dstedc.c`:
1. Lower DSTEDC_BASE from 2000 to 25
2. Fix deflation threshold to LAPACK-style local tolerance
3. Add z-vector 1/sqrt(2) normalization
4. Replace naive secular eigenvector formula with dlaed3 product formula
5. Top-level workspace allocation (single N x N buffer passed through recursion)

**Files:**
- Modify: `src/jamma/jblas/src/dstedc.c`
- Test: `tests/test_jblas_eigh.py` (existing tests)

**Key design decisions:**

**DSTEDC_BASE = 25**: LAPACK's threshold. QR iteration is O(N^2) per sweep with O(N) sweeps worst case = O(N^3). For N=25 this is fast. D&C is O(N^2 log N) overall.

**Deflation (LAPACK DLAED2 convention):**
- Before deflation, sort the merged d array and apply the same permutation to z and Z columns. This ensures the secular solver receives sorted poles.
- Type-a (tiny z component): `rho * |z[i]|^2 < tol` where `tol = 8 * eps * max(|d_max|, rho * z_norm_sq)`. The test is whether the component's contribution to the secular equation is negligible. Actually LAPACK DLAED2 uses: `rho * |z[i]| <= tol` where `tol = 8 * eps * (max eigenvalue estimate)`. Let me use the simpler form: `|z[i]| < 8 * eps * norm` where norm = `max(|d[i]|, max(|z|))`.
- Type-b (close eigenvalues): `|d[i] - d[j]| < 8 * eps * max(|d[i]|, |d[j]|)`. Local relative threshold.

**z-vector normalization:**
```c
/* After extracting boundary rows from Q_L and Q_R: */
double inv_sqrt2 = 1.0 / sqrt(2.0);
for (j = 0; j < n; j++) z_vec[j] *= inv_sqrt2;
rho = fabs(2.0 * e[m-1]);
```

**dlaed3 product formula for secular eigenvectors:**
For the i-th secular eigenvalue λ_i, the k-th component of the eigenvector is:
```c
q[k] = z[k] * sqrt(abs(
    prod_{j=0, j!=k}^{n_nd-1} (lambda[j] - d[k]) /
    prod_{j=0, j!=k}^{n_nd-1} (d[j] - d[k])
));
```
The sign of q[k] is determined by sgn(z[k]) * sgn(product of (lambda[i] - d[k]) / (d[j] - d[k]) for j != k). In practice, compute the products as running products, using logs to avoid overflow:

Actually, for numerical stability, compute the ratio term-by-term:
```c
for (k = 0; k < n_nd; k++) {
    double prod = 1.0;
    for (j = 0; j < n_nd; j++) {
        if (j == k) continue;
        prod *= (lambda[j] - d[k]) / (d[j] - d[k]);
    }
    q[k] = z[k] * sqrt(fabs(prod));
}
```
This is O(n_nd^2) per eigenvalue, O(n_nd^3) total — same as the naive formula but numerically stable because both numerator and denominator products involve differences of similar magnitude.

**Memory — top-level allocation:**
`jblas_dstedc_c` allocates:
- `work`: N x N doubles (single workspace buffer for merge)
- `iwork`: 5*N npy_intp (permutation + scratch indices)
- These are passed to `dstedc_recurse` and `merge_rank1` via parameters.

`merge_rank1` uses the workspace for Q_sec (N x N) and reuses portions for Q_nd, etc. Since merge calls at different recursion levels don't overlap (the recursion is depth-first), a single buffer suffices.

- [ ] **Step 1: Modify dstedc.c — update DSTEDC_BASE and function signatures**

Change `#define DSTEDC_BASE 2000` to `#define DSTEDC_BASE 25`.

Add workspace parameters to `dstedc_recurse` and `merge_rank1`:
```c
static int dstedc_recurse(npy_intp n, double *d, double *e,
                          double *Z, npy_intp ldz,
                          double *work, npy_intp lwork,
                          npy_intp *iwork);

static int merge_rank1(npy_intp n, npy_intp m,
                       double *d, double *z_vec, double rho,
                       double *Z, npy_intp ldz,
                       double *work, npy_intp lwork,
                       npy_intp *iwork);
```

- [ ] **Step 2: Fix z-vector construction in dstedc_recurse**

In the `dstedc_recurse` function, after the two recursive calls, add 1/sqrt(2) normalization and rho adjustment:

```c
/* Build z vector from post-recursion eigenvectors */
double *z_vec = /* ... same extraction as before ... */;

/* Normalize z by 1/sqrt(2) and adjust rho (LAPACK DLAED1 convention) */
double inv_sqrt2 = 1.0 / sqrt(2.0);
for (npy_intp j = 0; j < n; j++)
    z_vec[j] *= inv_sqrt2;
rho = fabs(2.0 * rho);  /* was fabs(e[m-1]), now 2x */
```

Wait — `rho` was set to `fabs(e[m-1])` before the recursive calls. After normalization by 1/sqrt(2), we need rho = 2 * |e[m-1]| so that `rho * z^2` = `2*|e[m-1]| * (z_orig/sqrt(2))^2` = `|e[m-1]| * z_orig^2` — same product. So `rho = 2.0 * fabs(e[m-1])`.

Actually let me re-derive. The rank-1 decomposition is:
```
T = diag(T_L_bar, T_R_bar) + e[m-1] * (e_m e_{m+1}^T + e_{m+1} e_m^T)
```
where `e[m-1]` is the connecting off-diagonal (can be negative). After eigenvector transformation:
```
T_tilde = D + rho * z * z^T
```
where `rho = e[m-1]` (signed), `z = [last_row(Q_L), first_row(Q_R)]`.

If we define `z_new = z / sqrt(2)`, then `rho_new = 2*rho` preserves `rho_new * z_new * z_new^T = 2*rho * z*z^T/2 = rho * z * z^T`.

But `rho` can be negative. LAPACK uses `rho = sign(e[m-1]) * 2 * |e[m-1]|` and if rho < 0, negates z and uses |rho|. Let me match this:

```c
double rho_raw = e[m - 1];  /* signed off-diagonal */
double rho = 2.0 * fabs(rho_raw);
double sign_rho = (rho_raw >= 0.0) ? 1.0 : -1.0;
double inv_sqrt2 = 1.0 / sqrt(2.0);
for (npy_intp j = 0; j < n; j++)
    z_vec[j] *= inv_sqrt2 * sign_rho;
/* Now rho > 0, secular equation: 1 + rho * sum z[k]^2 / (d[k] - lambda) = 0 */
```

This ensures rho > 0 for the secular solver, matching LAPACK's convention.

Also update the diagonal adjustment: `d[m-1] -= |e[m-1]|` and `d[m] -= |e[m-1]|` (using the original unsigned rho). This is unchanged from current code.

- [ ] **Step 3: Fix deflation in merge_rank1**

Replace global threshold with local:
```c
/* Type-a deflation: tiny z component.
 * Tests whether pole k's contribution rho*z[k]^2/(d[k]-lambda) is negligible.
 * Threshold: rho * z[k]^2 < 8*eps * |d[k]| (dimensionally consistent). */
for (npy_intp k = 0; k < n; k++) {
    if (rho * z_defl[k] * z_defl[k] <= 8.0 * EPS * fmax(fabs(d[k]), rho * z_defl[k] * z_defl[k])) {
        defl[k] = 1;
    }
}

/* Type-b deflation: close eigenvalues */
for (npy_intp k = 0; k < n - 1; k++) {
    if (defl[k]) continue;
    for (npy_intp j = k + 1; j < n; j++) {
        if (defl[j]) continue;
        double tol_kj = 8.0 * EPS * fmax(fabs(d_defl[k]), fabs(d_defl[j]));
        if (fabs(d_defl[k] - d_defl[j]) <= tol_kj) {
            /* ... Givens rotation same as before ... */
        }
    }
}
```

- [ ] **Step 4: Replace secular eigenvector formula with dlaed3 product formula**

Replace the naive formula in the "Step 5" section of merge_rank1:
```c
/* Step 5: dlaed3 product formula for secular eigenvectors.
 * For eigenvalue lambda[i], eigenvector component k is:
 *   q[k] = z[k] * sqrt(|prod_{j!=k} (lambda[j] - d[k]) / (d[j] - d[k])|)
 * Sign: sgn(q[k]) = sgn(z[k]) * sgn(prod_{j!=k} (lambda[j] - d[k]) / (d[j] - d[k]))
 */
for (npy_intp i = 0; i < n_nd; i++) {
    double norm2 = 0.0;
    for (npy_intp k = 0; k < n_nd; k++) {
        /* Compute product ratio term-by-term for numerical stability */
        double prod = 1.0;
        for (npy_intp j = 0; j < n_nd; j++) {
            if (j == k) continue;
            double num = lam_nd[j] - d_nd[k];
            double den = d_nd[j] - d_nd[k];
            if (fabs(den) < 1e-300)
                den = (den >= 0.0) ? 1e-300 : -1e-300;
            prod *= num / den;
        }
        /* Sign comes from sgn(z[k]) alone. A negative product is routine
         * for interior eigenvalues (not degenerate) — take sqrt of |prod|.
         * No additional sign correction needed. */
        double val = z_nd[k] * sqrt(fabs(prod));
        Q_nd[k * n_nd + i] = val;
        norm2 += val * val;
    }
    /* Normalize */
    double norm = sqrt(norm2);
    if (norm > 0.0) {
        for (npy_intp k = 0; k < n_nd; k++)
            Q_nd[k * n_nd + i] /= norm;
    }
}
```

- [ ] **Step 5: Implement top-level workspace allocation**

Modify `jblas_dstedc_c` to allocate workspace and pass to `dstedc_recurse`:
```c
int jblas_dstedc_c(npy_intp N, double *d, double *e,
                   double *Z, npy_intp ldz)
{
    if (N <= 0 || N == 1) return 0;

    /* Initialize Z to identity */
    memset(Z, 0, (size_t)N * (size_t)ldz * sizeof(double));
    for (npy_intp k = 0; k < N; k++)
        Z[k * ldz + k] = 1.0;

    /* Allocate workspace: N*N for merge buffer + 5*N for index arrays */
    npy_intp lwork = N * N;
    double *work = (double *)malloc((size_t)lwork * sizeof(double));
    npy_intp *iwork = (npy_intp *)malloc(5 * (size_t)N * sizeof(npy_intp));
    if (!work || !iwork) {
        free(work); free(iwork);
        return -1;
    }

    int ret = dstedc_recurse(N, d, e, Z, ldz, work, lwork, iwork);

    free(work);
    free(iwork);

    if (ret == 0)
        sort_eig(d, Z, ldz, N);
    return ret;
}
```

Modify `merge_rank1` to use the passed workspace instead of malloc:
- `Q_sec = work` (size n*n, fits within lwork = N*N since n <= N)
- Other small arrays (d_defl, z_defl, d_new, defl, nondfl, etc.) carved from iwork and dynamically allocated O(N) arrays.

Actually, the workspace partitioning within merge_rank1 is tricky because we need Q_sec (n*n), Q_nd (n_nd*n_nd), Q_nd_full (n*n_nd), and Z_new (n*ldz) at various points. Some of these can reuse the same memory since they don't overlap temporally:

1. Phase 1 (deflation): needs d_defl[n], z_defl[n], defl[n] — O(N) from iwork
2. Phase 2 (secular solve): needs d_nd[n_nd], z_nd[n_nd], lam_nd[n_nd] — O(N) extra
3. Phase 3 (eigenvectors): needs Q_nd[n_nd * n_nd] — can use first n_nd*n_nd of work
4. Phase 4 (assemble Q_sec): needs Q_sec[n * n] = work, Q_nd_full[n * n_nd] — overlap with Q_nd
5. Phase 5 (back-transform): Z_new[n * ldz] — can reuse Q_sec memory after Q_sec is consumed

This is feasible but complex. Simpler approach: keep merge_rank1's malloc/free for the O(N) arrays (d_defl, z_defl, lam_nd, etc. — these are O(N) and cheap), but use the passed workspace for the O(N^2) arrays (Q_sec, Z_new).

```c
/* Q_sec and Z_new from workspace (both n*n, used at different times) */
double *Q_sec = work;          /* n*n, used until step 7 */
/* After Q_sec is consumed in step 7, reuse for Z_new */
double *Z_new = work;          /* n*ldz, reused after Q_sec is done */
/* WAIT: step 7 does Z_new = Z @ Q_sec via dgemm. Can't read Q_sec and write Z_new
 * if they overlap. So we need separate buffers. */
```

OK, we need two N*N buffers for the merge step (Q_sec and Z_new). But we only allocated one N*N workspace. Options:
1. Allocate 2*N*N workspace at top level
2. Malloc Z_new inside merge_rank1 (it's used once, freed immediately)
3. Use Q_sec in-place: instead of Z_new = Z @ Q_sec, compute Z = Z @ Q_sec in-place using the columns of Q_sec. This requires processing column-by-column with a temp vector.

Option 2 is simplest and the malloc is once per merge (not per recursion depth — D&C has O(log N) levels, so O(log N) merges).

Let's use option 2: pass one N*N workspace for Q_sec, malloc Z_new locally.

- [ ] **Step 6: Rebuild and run tests**

```bash
uv run pip install -e . --no-build-isolation 2>&1 | tail -5
uv run pytest tests/test_jblas_eigh.py -x -n0 -v -k "not slow and not mouse"
```

This is the critical step. With DSTEDC_BASE=25, the D&C path is now live for all N > 25 tests (N=31, 63, 64, 65, 71, 72, 73, 100, 127, 128, 129, 200). If any fail, the merge implementation has a bug.

Expected failure modes if something is wrong:
- Reconstruction > 1e-14: eigenvector formula or back-transform bug
- Orthogonality > 1e-14: eigenvector normalization or Q_sec assembly bug
- Wrong eigenvalues: deflation or secular solver bug

- [ ] **Step 7: Run slow tests**

```bash
uv run pytest tests/test_jblas_eigh.py -x -n0 -v
```

Expected: All pass. Mouse_hs1940 (N=1940) exercises D&C with ~78 levels of recursion.

If mouse_hs1940 tolerances need adjustment: investigate the root cause (e.g., near-degenerate eigenvalues in the kinship matrix causing deflation sensitivity). Only adjust tolerances if the root cause is understood and the new tolerance is justified.

- [ ] **Step 8: Run full project test suite**

```bash
uv run pytest tests/ -x
```

Expected: All 2208+ tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/jamma/jblas/src/dstedc.c
git commit -m "fix(80): dstedc — correct D&C merge with LAPACK-style deflation

Lower DSTEDC_BASE from 2000 to 25, enabling divide-and-conquer for
all practical sizes. Fix deflation to use local relative threshold
instead of global ||T||. Add 1/sqrt(2) z-vector normalization with
2*rho adjustment (LAPACK DLAED1 convention). Replace naive secular
eigenvector formula with dlaed3 product formula for numerical
stability. Top-level workspace allocation (single N*N buffer)
replaces per-merge malloc/free."
```

---

## Chunk 4: eigh.c documentation + final verification

### Task 4: Update eigh.c documentation and run final verification

**Files:**
- Modify: `src/jamma/jblas/src/eigh.c` (documentation only)
- Test: full test suite

- [ ] **Step 1: Update eigh.c header comment**

Change the Memory section of the header comment from:
```c
 * Memory:
 *   Workspace: d[N], e[N], tau[N-1] — allocated and freed here.
 *   K is overwritten with the Householder vectors from dsytrd.
 *   dstedc owns its own internal N x N merge buffer (malloc/free inside).
 *   eigh does NOT allocate the merge buffer — it belongs to dstedc.
```
to:
```c
 * Memory:
 *   Workspace: d[N], e[N], tau[N-1] — allocated and freed here.
 *   K is overwritten with the Householder vectors from dsytrd.
 *   dstedc allocates one N x N merge workspace + O(N) scratch internally.
 *   dormtr allocates NB x M + NB x NB + vlen x NB workspace internally.
 *
 *   Total peak memory: K(N^2) + U(N^2) + dstedc_work(N^2) + O(N*NB) = 3N^2 + O(N).
 *   dsytrd workspace: 2*N*NB (V and W panels) — freed before dstedc runs.
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest tests/ -x
```

Expected: All tests pass.

- [ ] **Step 3: Run benchmarks to verify no performance regression**

```bash
uv run pytest tests/test_jblas_eigh.py -x -n0 -v --benchmark-only -m benchmark 2>/dev/null || echo "No benchmark tests — skip"
```

If no benchmark tests exist, run a quick manual timing:
```bash
uv run python -c "
import numpy as np
import time
from jamma.jblas import eigh

rng = np.random.default_rng(42)
for N in [100, 500, 1000]:
    A = rng.standard_normal((N, N))
    K = A @ A.T / N
    start = time.perf_counter()
    w, v = eigh(K.copy())
    elapsed = time.perf_counter() - start
    recon = np.linalg.norm(K - v @ np.diag(w) @ v.T, 'fro') / np.linalg.norm(K, 'fro')
    orth = np.linalg.norm(v.T @ v - np.eye(N), 'fro')
    print(f'N={N:4d}: {elapsed:.3f}s  recon={recon:.2e}  orth={orth:.2e}')
"
```

Expected: Reconstruction < 1e-14, orthogonality < 1e-14 for all sizes.

- [ ] **Step 4: Commit documentation update**

```bash
git add src/jamma/jblas/src/eigh.c
git commit -m "docs(80): eigh — update memory model documentation

Document actual peak memory: 3N^2 + O(N), comprising input K,
output U, and dstedc's internal merge workspace."
```

- [ ] **Step 5: Push all commits**

```bash
git push
```
