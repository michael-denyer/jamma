/**
 * dsyr2k.c — Symmetric rank-2k update for jblas.
 *
 * Implements C -= A @ B.T + B @ A.T on all elements of C.
 *
 * Design: Two-pass approach via a static helper _dsyr2k_half.
 *
 *   jblas_dsyr2k_c(N, K, A, lda, B, ldb, C, ldc):
 *     _dsyr2k_half(N, K, A, lda, B, ldb, C, ldc)  -- subtracts A @ B.T
 *     _dsyr2k_half(N, K, B, ldb, A, lda, C, ldc)  -- subtracts B @ A.T
 *
 * _dsyr2k_half loop structure (PC-outer with JC hoisted for thread safety):
 *
 *   PC (K in KC-deep blocks):
 *     JC (N in NC-wide column panels):
 *       Pack Q.T panel for this (PC, JC) pair (shared jblas_packed_B).
 *       IC (N in MC-tall row panels, OpenMP parallel):
 *         Pack P panel for this IC slice (per-thread packed_A).
 *         Microkernel loop: MR x NR tiles.
 *         Subtract microkernel result from C (alpha = -1).
 *
 * Alpha = -1 implementation:
 *   After packing P into packed_A, the buffer is negated in-place so the
 *   microkernel computes C += (-P) * Q.T = C -= P * Q.T.  This lets full
 *   tiles (MR x NR) write directly to C via the microkernel fast path
 *   (same as dgemm), avoiding the scratch buffer overhead.  Tail tiles
 *   still use a stack-allocated scratch buffer (MAX_MR * MAX_NR = 512 bytes).
 *   The negation cost is O(MC*KC) per IC iteration — negligible vs the
 *   O(MC*NC*KC) microkernel work.
 *
 * Full-matrix update:
 *   Both triangles of C are updated (no diagonal skip), which is required
 *   for correctness when C is not symmetric.  The result matches the NumPy
 *   fallback: result = C - A @ B.T - B @ A.T for the full matrix.
 *   dsytrd callers that read only the lower triangle will still get the
 *   correct lower-triangle values.
 *
 * Workspace:
 *   Uses jblas_packed_B (KC * NC doubles) — bounded by JC partitioning.
 *   Uses jblas_packed_A (per-thread MC * KC doubles).
 *
 * Thread safety:
 *   Acquires jblas_dgemm_mutex (shared with dgemm.c and dsyrk.c) for
 *   the entire two-pass computation.
 *
 * Caller responsibility:
 *   C must be initialised by the caller before calling jblas_dsyr2k_c.
 *   This function only subtracts from C, it does not zero-initialise it.
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
 * Utility macros (matching dgemm.c and dsyrk.c)
 * ---------------------------------------------------------------------------
 */
#define MIN(a, b)       ((a) < (b) ? (a) : (b))
#define CEIL_DIV(a, b)  (((a) + (b) - 1) / (b))
#define MAX_MR  8
#define MAX_NR  8

/* ---------------------------------------------------------------------------
 * _dsyr2k_half — Compute C -= P @ Q.T on all elements of C.
 *
 * Helper for jblas_dsyr2k_c.  Called twice:
 *   once with P=A, Q=B  (subtracts A @ B.T)
 *   once with P=B, Q=A  (subtracts B @ A.T)
 *
 * Assumes jblas_dgemm_mutex is already held by the caller.
 * Does NOT zero-initialize C — the caller is responsible for initialisation.
 *
 * Full-matrix update (both triangles): all (IC, JC) panel combinations are
 * computed.  This ensures correctness when C is not symmetric, matching the
 * NumPy fallback contract.
 *
 * N    : number of rows/columns of C, and rows of P and Q.
 * K    : number of columns of P and Q.
 * P    : left factor, row-major, shape (N, K), leading dimension ldp.
 * ldp  : leading dimension of P.
 * Q    : right factor, row-major, shape (N, K), leading dimension ldq.
 * ldq  : leading dimension of Q.
 * C    : in/out matrix, row-major, shape (N, N), leading dimension ldc.
 * ldc  : leading dimension of C.
 * ---------------------------------------------------------------------------
 */
static void _dsyr2k_half(npy_intp N, npy_intp K,
                         const double *P, npy_intp ldp,
                         const double *Q, npy_intp ldq,
                         double *C, npy_intp ldc)
{
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

            /* Pack Q.T panel: Q rows [jc, jc+nc_actual), cols [pc, pc+kc_actual).
             * trans=1 reads Q.T: B[k, j] = Q[jc+j, pc+k].
             * Pointer starts at Q row jc, column pc. */
            jblas_pack_B(Q + jc * ldq + pc, ldq,
                         kc_actual, nc_actual, jblas_packed_B, NR, 1);

            npy_intp n_nr_strips = CEIL_DIV(nc_actual, NR);

            /* IC loop (OpenMP parallel): partition N into MC-tall row panels.
             * Clamped to jblas_n_threads to prevent OOB packed_A access. */
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(jblas_n_threads)
#endif
            for (npy_intp ic = 0; ic < N; ic += MC) {
                npy_intp mc_actual = MIN(MC, N - ic);

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

                /* Pack P panel: P rows [ic, ic+mc_actual), cols [pc, pc+kc_actual).
                 * trans=0 reads P in row-major order. */
                jblas_pack_A(P + ic * ldp + pc, ldp,
                             mc_actual, kc_actual, packed_A_ptr, MR, 0);

                /* Negate packed_A so the microkernel computes C += (-P) * Q.T,
                 * i.e. C -= P * Q.T.  This lets full tiles write directly to C
                 * (same fast path as dgemm) instead of routing through a scratch
                 * buffer.  Cost: O(MC*KC) per IC iteration — negligible vs the
                 * O(MC*NC*KC) microkernel work (factor of NC ≈ 4096 cheaper). */
                npy_intp n_packed = CEIL_DIV(mc_actual, MR) * (npy_intp)MR * kc_actual;
                for (npy_intp i = 0; i < n_packed; i++)
                    packed_A_ptr[i] = -packed_A_ptr[i];

                npy_intp n_mr_strips = CEIL_DIV(mc_actual, MR);

                /* JR loop: NR strips within the JC column panel */
                for (npy_intp jr_s = 0; jr_s < n_nr_strips; jr_s++) {
                    npy_intp jr      = jr_s * NR;
                    npy_intp nr_tile = MIN(NR, nc_actual - jr);

                    const double *pB_strip = jblas_packed_B +
                        (size_t)jr_s * (size_t)kc_actual * (size_t)NR;

                    /* IR loop: MR strips within the IC row panel */
                    for (npy_intp ir_s = 0; ir_s < n_mr_strips; ir_s++) {
                        npy_intp ir      = ir_s * MR;
                        npy_intp mr_tile = MIN(MR, mc_actual - ir);

                        const double *pA_strip = packed_A_ptr +
                            (size_t)ir_s * (size_t)kc_actual * (size_t)MR;

                        /* Target C tile: C[ic+ir, jc+jr] */
                        double *C_tile = C + (ic + ir) * ldc + (jc + jr);

                        if (mr_tile == MR && nr_tile == NR) {
                            /* Full tile: microkernel writes directly to C.
                             * packed_A is negated, so this accumulates C -= P*Q.T. */
                            jblas_dgemm_microkernel(kc_actual, pA_strip, pB_strip,
                                                    C_tile, ldc);
                        } else {
                            /* Tail tile: accumulate into scratch, add valid
                             * portion to C (packed_A is already negated). */
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
 * jblas_dsyr2k_c — Symmetric rank-2k update: C -= A @ B.T + B @ A.T
 *
 * Updates all elements of C (full-matrix update, both triangles).
 * No diagonal skip or mirror step is performed.
 *
 * N    : number of rows/columns of C, and rows of A and B.
 * K    : number of columns of A and B.
 * A    : first factor, row-major, shape (N, K), leading dimension lda.
 * lda  : leading dimension of A (>= K).
 * B    : second factor, row-major, shape (N, K), leading dimension ldb.
 * ldb  : leading dimension of B (>= K).
 * C    : in/out matrix, row-major, shape (N, N), leading dimension ldc.
 * ldc  : leading dimension of C (>= N).
 *
 * The caller must initialise C before calling this function.
 * ---------------------------------------------------------------------------
 */
void jblas_dsyr2k_c(npy_intp N, npy_intp K,
                    const double *A, npy_intp lda,
                    const double *B, npy_intp ldb,
                    double *C, npy_intp ldc)
{
    /* Guard: negative dimensions are programming errors */
    if (N < 0 || K < 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyr2k_c: negative dimension N=%ld K=%ld\n",
            (long)N, (long)K);
        abort();
    }

    if (N == 0 || K == 0)
        return;

    if (!jblas_packed_A || !jblas_packed_B) {
        fprintf(stderr,
            "FATAL: jblas_dsyr2k_c called but workspace not allocated "
            "(jblas_dgemm_init() failed or was never called)\n");
        abort();
    }

    int lock_err = pthread_mutex_lock(&jblas_dgemm_mutex);
    if (lock_err != 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyr2k_c: pthread_mutex_lock failed (errno=%d)\n",
            lock_err);
        abort();
    }

    /* Pass 1: C -= A @ B.T */
    _dsyr2k_half(N, K, A, lda, B, ldb, C, ldc);

    /* Pass 2: C -= B @ A.T */
    _dsyr2k_half(N, K, B, ldb, A, lda, C, ldc);

    int unlock_err = pthread_mutex_unlock(&jblas_dgemm_mutex);
    if (unlock_err != 0) {
        fprintf(stderr,
            "FATAL: jblas_dsyr2k_c: pthread_mutex_unlock failed (errno=%d)\n",
            unlock_err);
        abort();
    }
}
