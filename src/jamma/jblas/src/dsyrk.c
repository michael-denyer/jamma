/**
 * dsyrk.c — Symmetric rank-k update for jblas.
 *
 * Implements C = X @ X.T using three-level Goto/BLIS blocking with PC-outer
 * nesting (differs from dgemm.c's JC-outer nesting; PC-outer is natural here
 * because both A and B panels derive from X, sharing the KC-deep pass).
 * Lower-triangle tile skipping saves ~50% computation.
 * After all KC passes complete, mirrors the lower triangle to fill the upper
 * triangle.
 *
 * Loop structure (PC-outer with JC hoisted above IC for thread safety):
 *
 *   PC (K in KC-deep blocks):
 *     JC (N in NC-wide column panels):
 *       Pack B panel from X.T for this (PC, JC) pair (shared jblas_packed_B).
 *       IC (N in MC-tall row panels, OpenMP parallel):
 *         Diagonal skip at IC level: skip if JC panel above diagonal for IC.
 *         Pack A panel from X for this IC slice (per-thread packed_A).
 *         Microkernel loop: MR x NR tiles with per-tile diagonal skip.
 *
 * Diagonal skip:
 *   - JC level: if jc > ic + mc_actual - 1, skip (entire JC panel is above
 *     the diagonal for this IC panel).
 *   - Tile level: if col_abs > row_abs + mr_tile - 1, skip (tile is entirely
 *     above diagonal).
 *
 * Workspace:
 *   Uses jblas_packed_B (KC * NC doubles) — never overflows because JC loop
 *   partitions N into NC-wide panels (Pitfall 1 prevention).
 *   Uses jblas_packed_A (per-thread MC * KC doubles).
 *
 * Thread safety:
 *   Acquires jblas_dgemm_mutex (shared with dgemm.c and dsyr2k.c) to
 *   serialise access to jblas_packed_B across concurrent callers.
 *
 * Mirror step:
 *   After all loops complete (mutex released), copies lower to upper:
 *   C[j*ldc + i] = C[i*ldc + j] for all j < i.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numpy/arrayobject.h>
#include "jblas.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------------------------------------------------------------------------
 * Utility macros (matching dgemm.c)
 * ---------------------------------------------------------------------------
 */
#define MIN(a, b)       ((a) < (b) ? (a) : (b))
#define CEIL_DIV(a, b)  (((a) + (b) - 1) / (b))
#define MAX_MR  8
#define MAX_NR  8

/* ---------------------------------------------------------------------------
 * _dsyrk_core — Core symmetric rank-k loop with explicit workspace.
 *
 * Computes the lower triangle of C = X @ X.T using three-level Goto/BLIS
 * blocking.  Does NOT zero C or mirror — caller handles those.
 *
 * packed_A, packed_B, n_threads: caller-owned workspace buffers.
 * ---------------------------------------------------------------------------
 */
static void _dsyrk_core(npy_intp N, npy_intp K,
                         const double *X, npy_intp ldx,
                         double *C, npy_intp ldc,
                         double *packed_A, double *packed_B,
                         int n_threads)
{
    int MR = JBLAS_MR;
    int NR = JBLAS_NR;
    int KC = JBLAS_KC;
    int MC = JBLAS_MC;
    int NC = JBLAS_NC;

    /* PC outer loop: partition K into KC-deep blocks */
    for (npy_intp pc = 0; pc < K; pc += KC) {
        npy_intp kc_actual = MIN(KC, K - pc);

        for (npy_intp jc = 0; jc < N; jc += NC) {
            npy_intp nc_actual = MIN(NC, N - jc);

            jblas_pack_B(X + jc * ldx + pc, ldx,
                         kc_actual, nc_actual, packed_B, NR, 1);

            npy_intp n_nr_strips = CEIL_DIV(nc_actual, NR);

#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
            for (npy_intp ic = 0; ic < N; ic += MC) {
                npy_intp mc_actual = MIN(MC, N - ic);

                if (jc > ic + mc_actual - 1)
                    continue;

#ifdef _OPENMP
                int tid = omp_get_thread_num();
                if (tid >= n_threads) {
                    fprintf(stderr,
                        "FATAL: OpenMP thread %d exceeds allocated workspace "
                        "for %d threads\n", tid, n_threads);
                    abort();
                }
#else
                int tid = 0;
#endif
                double *packed_A_ptr = packed_A +
                    (size_t)tid * (size_t)MC * (size_t)KC;

                jblas_pack_A(X + ic * ldx + pc, ldx,
                             mc_actual, kc_actual, packed_A_ptr, MR, 0);

                npy_intp n_mr_strips = CEIL_DIV(mc_actual, MR);

                for (npy_intp jr_s = 0; jr_s < n_nr_strips; jr_s++) {
                    npy_intp jr      = jr_s * NR;
                    npy_intp nr_tile = MIN(NR, nc_actual - jr);
                    npy_intp col_abs = jc + jr;

                    const double *pB_strip = packed_B +
                        (size_t)jr_s * (size_t)kc_actual * (size_t)NR;

                    for (npy_intp ir_s = 0; ir_s < n_mr_strips; ir_s++) {
                        npy_intp ir      = ir_s * MR;
                        npy_intp mr_tile = MIN(MR, mc_actual - ir);
                        npy_intp row_abs = ic + ir;

                        if (col_abs > row_abs + mr_tile - 1)
                            continue;

                        const double *pA_strip = packed_A_ptr +
                            (size_t)ir_s * (size_t)kc_actual * (size_t)MR;

                        double *C_tile = C + row_abs * ldc + col_abs;

                        if (mr_tile == MR && nr_tile == NR) {
                            jblas_dgemm_microkernel(kc_actual, pA_strip, pB_strip,
                                                    C_tile, ldc);
                        } else {
                            double scratch[MAX_MR * MAX_NR];
                            memset(scratch, 0, sizeof(scratch));

                            jblas_dgemm_microkernel(kc_actual, pA_strip, pB_strip,
                                                    scratch, NR);

                            for (npy_intp r = 0; r < mr_tile; r++) {
                                for (npy_intp c = 0; c < nr_tile; c++) {
                                    C_tile[r * ldc + c] += scratch[r * NR + c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * jblas_dsyrk_c — Symmetric rank-k update: C = X @ X.T (lower then mirror)
 *
 * N    : number of rows/columns of C and rows of X.
 * K    : number of columns of X.
 * X    : input matrix, row-major, shape (N, K), leading dimension ldx.
 * ldx  : leading dimension of X (>= K).
 * C    : output matrix, row-major, shape (N, N), leading dimension ldc.
 * ldc  : leading dimension of C (>= N).
 * ---------------------------------------------------------------------------
 */
void jblas_dsyrk_c(npy_intp N, npy_intp K,
                   const double *X, npy_intp ldx,
                   double *C, npy_intp ldc)
{
    /* Guard: negative dimensions are programming errors */
    if (N < 0 || K < 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_c: negative dimension N=%ld K=%ld\n",
            (long)N, (long)K);
        abort();
    }

    /* Zero-initialize C (entire N x N output) */
    for (npy_intp i = 0; i < N; i++) {
        memset(C + i * ldc, 0, (size_t)N * sizeof(double));
    }

    if (N == 0 || K == 0)
        return;

    if (!jblas_packed_A || !jblas_packed_B) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_c called but workspace not allocated "
            "(jblas_dgemm_init() failed or was never called)\n");
        abort();
    }

    int lock_err = pthread_mutex_lock(&jblas_dgemm_mutex);
    if (lock_err != 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_c: pthread_mutex_lock failed (errno=%d)\n",
            lock_err);
        abort();
    }

    _dsyrk_core(N, K, X, ldx, C, ldc,
                jblas_packed_A, jblas_packed_B, jblas_n_threads);

    int unlock_err = pthread_mutex_unlock(&jblas_dgemm_mutex);
    if (unlock_err != 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_c: pthread_mutex_unlock failed (errno=%d)\n",
            unlock_err);
        abort();
    }

    /* Mirror lower triangle to upper triangle. */
    for (npy_intp i = 0; i < N; i++) {
        for (npy_intp j = i + 1; j < N; j++) {
            C[i * ldc + j] = C[j * ldc + i];
        }
    }
}

/* ---------------------------------------------------------------------------
 * jblas_dsyrk_ws — Workspace-explicit symmetric rank-k update (no mutex).
 *
 * Same computation as jblas_dsyrk_c but uses caller-owned workspace.
 * Safe for concurrent use (e.g. inside eigensolver).
 * ---------------------------------------------------------------------------
 */
void jblas_dsyrk_ws(npy_intp N, npy_intp K,
                     const double *X, npy_intp ldx,
                     double *C, npy_intp ldc,
                     jblas_workspace_t *ws)
{
    if (N < 0 || K < 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_ws: negative dimension N=%ld K=%ld\n",
            (long)N, (long)K);
        abort();
    }

    for (npy_intp i = 0; i < N; i++) {
        memset(C + i * ldc, 0, (size_t)N * sizeof(double));
    }

    if (N == 0 || K == 0)
        return;

    if (!ws || !ws->packed_A || !ws->packed_B) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_ws called with NULL workspace\n");
        abort();
    }

    _dsyrk_core(N, K, X, ldx, C, ldc,
                ws->packed_A, ws->packed_B, ws->n_threads);

    /* Mirror lower triangle to upper triangle. */
    for (npy_intp i = 0; i < N; i++) {
        for (npy_intp j = i + 1; j < N; j++) {
            C[i * ldc + j] = C[j * ldc + i];
        }
    }
}

/* ---------------------------------------------------------------------------
 * jblas_dsyrk_lower_c — Symmetric rank-k update: C = X @ X.T (lower only)
 *
 * Identical to jblas_dsyrk_c but:
 *   1. Only zeroes the lower triangle of C (not the full matrix).
 *   2. Skips the mirror step — upper triangle is NOT filled.
 *
 * This saves O(N^2) wasted writes for callers that only read the lower
 * triangle (e.g. eigensolver-internal paths, kinship computation).
 *
 * N    : number of rows/columns of C and rows of X.
 * K    : number of columns of X.
 * X    : input matrix, row-major, shape (N, K), leading dimension ldx.
 * ldx  : leading dimension of X (>= K).
 * C    : output matrix, row-major, shape (N, N), leading dimension ldc.
 * ldc  : leading dimension of C (>= N).
 * ---------------------------------------------------------------------------
 */
void jblas_dsyrk_lower_c(npy_intp N, npy_intp K,
                          const double *X, npy_intp ldx,
                          double *C, npy_intp ldc)
{
    /* Guard: negative dimensions are programming errors */
    if (N < 0 || K < 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_lower_c: negative dimension N=%ld K=%ld\n",
            (long)N, (long)K);
        abort();
    }

    /* Zero-initialize lower triangle of C only */
    for (npy_intp i = 0; i < N; i++) {
        memset(C + i * ldc, 0, (size_t)(i + 1) * sizeof(double));
    }

    if (N == 0 || K == 0)
        return;

    if (!jblas_packed_A || !jblas_packed_B) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_lower_c called but workspace not allocated "
            "(jblas_dgemm_init() failed or was never called)\n");
        abort();
    }

    int lock_err = pthread_mutex_lock(&jblas_dgemm_mutex);
    if (lock_err != 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_lower_c: pthread_mutex_lock failed (errno=%d)\n",
            lock_err);
        abort();
    }

    _dsyrk_core(N, K, X, ldx, C, ldc,
                jblas_packed_A, jblas_packed_B, jblas_n_threads);

    int unlock_err = pthread_mutex_unlock(&jblas_dgemm_mutex);
    if (unlock_err != 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_lower_c: pthread_mutex_unlock failed (errno=%d)\n",
            unlock_err);
        abort();
    }

    /* No mirror step — only the lower triangle is valid. */
}

/* ---------------------------------------------------------------------------
 * jblas_dsyrk_lower_ws — Workspace-explicit lower-only rank-k update (no mutex).
 *
 * Same as jblas_dsyrk_lower_c but uses caller-owned workspace.
 * ---------------------------------------------------------------------------
 */
void jblas_dsyrk_lower_ws(npy_intp N, npy_intp K,
                            const double *X, npy_intp ldx,
                            double *C, npy_intp ldc,
                            jblas_workspace_t *ws)
{
    if (N < 0 || K < 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_lower_ws: negative dimension N=%ld K=%ld\n",
            (long)N, (long)K);
        abort();
    }

    for (npy_intp i = 0; i < N; i++) {
        memset(C + i * ldc, 0, (size_t)(i + 1) * sizeof(double));
    }

    if (N == 0 || K == 0)
        return;

    if (!ws || !ws->packed_A || !ws->packed_B) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_lower_ws called with NULL workspace\n");
        abort();
    }

    _dsyrk_core(N, K, X, ldx, C, ldc,
                ws->packed_A, ws->packed_B, ws->n_threads);

    /* No mirror step — only the lower triangle is valid. */
}
