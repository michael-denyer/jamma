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

    int MR = JBLAS_MR;
    int NR = JBLAS_NR;
    int KC = JBLAS_KC;
    int MC = JBLAS_MC;
    int NC = JBLAS_NC;

    /* PC outer loop: partition K into KC-deep blocks */
    for (npy_intp pc = 0; pc < K; pc += KC) {
        npy_intp kc_actual = MIN(KC, K - pc);

        /* JC loop: partition N into NC-wide column panels.
         * Hoisted outside IC to pack B once per (pc, jc) pair before the
         * OpenMP parallel region — prevents data races on jblas_packed_B. */
        for (npy_intp jc = 0; jc < N; jc += NC) {
            npy_intp nc_actual = MIN(NC, N - jc);

            /* Pack B panel: X rows [jc, jc+nc_actual), columns [pc, pc+kc_actual)
             * trans=1 reads X.T: B[k, j] = X[jc+j, pc+k] = X[(jc+j)*ldx + (pc+k)]
             * Pointer starts at X row jc, column pc. */
            jblas_pack_B(X + jc * ldx + pc, ldx,
                         kc_actual, nc_actual, jblas_packed_B, NR, 1);

            npy_intp n_nr_strips = CEIL_DIV(nc_actual, NR);

            /* IC loop (OpenMP parallel): partition N into MC-tall row panels.
             * Clamped to jblas_n_threads to prevent OOB packed_A access. */
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(jblas_n_threads)
#endif
            for (npy_intp ic = 0; ic < N; ic += MC) {
                npy_intp mc_actual = MIN(MC, N - ic);

                /* Diagonal skip at IC level:
                 * If the entire IC row panel is below the diagonal for this
                 * JC column panel, skip.  The last column in the JC panel is
                 * jc + nc_actual - 1.  If ic > that index, all rows in
                 * [ic, ic+mc_actual) are below the last JC column — but that
                 * means lower triangle, which we DO want.  Instead, skip if
                 * jc > ic + mc_actual - 1 (entire JC panel is above diagonal
                 * for this IC row panel). */
                if (jc > ic + mc_actual - 1)
                    continue;

#ifdef _OPENMP
                int tid = omp_get_thread_num();
                if (tid >= jblas_n_threads) {
                    fprintf(stderr,
                        "FATAL: OpenMP thread %d exceeds allocated workspace "
                        "for %d threads\n", tid, jblas_n_threads);
                    abort();
                }
#else
                int tid = 0;
#endif
                double *packed_A_ptr = jblas_packed_A +
                    (size_t)tid * (size_t)MC * (size_t)KC;

                /* Pack A panel: X rows [ic, ic+mc_actual), columns [pc, pc+kc_actual)
                 * trans=0 reads X in row-major order. */
                jblas_pack_A(X + ic * ldx + pc, ldx,
                             mc_actual, kc_actual, packed_A_ptr, MR, 0);

                npy_intp n_mr_strips = CEIL_DIV(mc_actual, MR);

                /* JR loop: NR strips within the JC column panel */
                for (npy_intp jr_s = 0; jr_s < n_nr_strips; jr_s++) {
                    npy_intp jr      = jr_s * NR;
                    npy_intp nr_tile = MIN(NR, nc_actual - jr);
                    npy_intp col_abs = jc + jr;  /* absolute column of this NR strip */

                    /* Packed B strip pointer — jr is relative to JC panel start,
                     * so jr_s indexes directly (no subtraction needed). */
                    const double *pB_strip = jblas_packed_B +
                        (size_t)jr_s * (size_t)kc_actual * (size_t)NR;

                    /* IR loop: MR strips within the IC row panel */
                    for (npy_intp ir_s = 0; ir_s < n_mr_strips; ir_s++) {
                        npy_intp ir      = ir_s * MR;
                        npy_intp mr_tile = MIN(MR, mc_actual - ir);
                        npy_intp row_abs = ic + ir;  /* absolute row of this MR strip */

                        /* Diagonal skip at tile level:
                         * Skip if all columns of this tile are above the last
                         * row of the MR strip, i.e., col_abs > row_abs + mr_tile - 1. */
                        if (col_abs > row_abs + mr_tile - 1)
                            continue;

                        /* Packed A strip pointer */
                        const double *pA_strip = packed_A_ptr +
                            (size_t)ir_s * (size_t)kc_actual * (size_t)MR;

                        /* Target C tile: C[row_abs, col_abs] */
                        double *C_tile = C + row_abs * ldc + col_abs;

                        if (mr_tile == MR && nr_tile == NR) {
                            /* Full tile: microkernel accumulates directly to C */
                            jblas_dgemm_microkernel(kc_actual, pA_strip, pB_strip,
                                                    C_tile, ldc);
                        } else {
                            /* Tail tile: accumulate into scratch, add valid
                             * portion to C. */
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

    int unlock_err = pthread_mutex_unlock(&jblas_dgemm_mutex);
    if (unlock_err != 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_c: pthread_mutex_unlock failed (errno=%d)\n",
            unlock_err);
        abort();
    }

    /* Mirror lower triangle to upper triangle.
     * In row-major, element (i, j) is at C[i*ldc + j].
     * The lower triangle (i >= j) was accumulated above.
     * Copy lower to upper: for each i < j, set C[i*ldc + j] = C[j*ldc + i].
     *   C[j*ldc + i]  — row j, col i, j > i → lower triangle (source)
     *   C[i*ldc + j]  — row i, col j, j > i → upper triangle (dest)   */
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
                         kc_actual, nc_actual, jblas_packed_B, NR, 1);

            npy_intp n_nr_strips = CEIL_DIV(nc_actual, NR);

#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(jblas_n_threads)
#endif
            for (npy_intp ic = 0; ic < N; ic += MC) {
                npy_intp mc_actual = MIN(MC, N - ic);

                /* Diagonal skip: skip if JC panel entirely above diagonal */
                if (jc > ic + mc_actual - 1)
                    continue;

#ifdef _OPENMP
                int tid = omp_get_thread_num();
                if (tid >= jblas_n_threads) {
                    fprintf(stderr,
                        "FATAL: OpenMP thread %d exceeds allocated workspace "
                        "for %d threads\n", tid, jblas_n_threads);
                    abort();
                }
#else
                int tid = 0;
#endif
                double *packed_A_ptr = jblas_packed_A +
                    (size_t)tid * (size_t)MC * (size_t)KC;

                jblas_pack_A(X + ic * ldx + pc, ldx,
                             mc_actual, kc_actual, packed_A_ptr, MR, 0);

                npy_intp n_mr_strips = CEIL_DIV(mc_actual, MR);

                for (npy_intp jr_s = 0; jr_s < n_nr_strips; jr_s++) {
                    npy_intp jr      = jr_s * NR;
                    npy_intp nr_tile = MIN(NR, nc_actual - jr);
                    npy_intp col_abs = jc + jr;

                    const double *pB_strip = jblas_packed_B +
                        (size_t)jr_s * (size_t)kc_actual * (size_t)NR;

                    for (npy_intp ir_s = 0; ir_s < n_mr_strips; ir_s++) {
                        npy_intp ir      = ir_s * MR;
                        npy_intp mr_tile = MIN(MR, mc_actual - ir);
                        npy_intp row_abs = ic + ir;

                        /* Diagonal skip at tile level */
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

    int unlock_err = pthread_mutex_unlock(&jblas_dgemm_mutex);
    if (unlock_err != 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyrk_lower_c: pthread_mutex_unlock failed (errno=%d)\n",
            unlock_err);
        abort();
    }

    /* No mirror step — only the lower triangle is valid. */
}
