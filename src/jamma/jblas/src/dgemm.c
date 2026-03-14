/**
 * dgemm.c — Goto/BLIS three-level blocking dgemm for jblas.
 *
 * Implements the full DGEMM (C = op(A) * op(B)) using the Goto/BLIS
 * three-level blocking loop:
 *
 *   Outer (JC):  Partition N into NC-wide column panels of B.
 *   Middle (PC): Partition K into KC-deep blocks (pack B panel once).
 *   Inner (IC):  Partition M into MC-tall row panels of A (parallel over threads).
 *     Microkernel loop: MR x NR tiles.
 *
 * Blocking parameters and workspace buffers are set by jblas_dgemm_init(),
 * which is called from jblas_init() in platform.c after ISA detection.
 *
 * Thread safety:
 *   - A pthread mutex serialises concurrent callers (jblas_packed_B is a single
 *     shared buffer written outside the OpenMP region).
 *   - jblas_packed_A is per-thread: each thread computes its own packed_A
 *     offset from omp_get_thread_num().
 *   - OpenMP parallel-for over the IC loop.
 *   - The parallel-for is clamped to jblas_n_threads (the thread count at init
 *     time) to prevent OOB packed_A access if omp_set_num_threads() increases
 *     the count after workspace allocation.
 *
 * Parallelism limitation:
 *   Only the IC loop is parallelised, capping useful thread count at
 *   ceil(M / MC).  With AVX2 MC=72, any M < 72 runs single-threaded.
 *   This is acceptable for JAMMA's primary use (kinship: M = N = n_samples).
 *   JC-loop parallelism would require per-thread packed_B buffers.
 *
 * Transpose support:
 *   transa=1 means A is stored transposed in memory; we adjust the pointer
 *   arithmetic so pack_A reads from A^T rather than A.
 *   transb=1 similarly for B.
 *
 * Tail handling:
 *   - pack_A zero-pads the tail row strip to a multiple of MR.
 *   - pack_B zero-pads the tail column strip to a multiple of NR.
 *   - Tail microkernel tiles write to a stack-allocated MR x NR scratch buffer,
 *     then copy only the valid (mr_tail x nr_tail) subblock back to C.
 *
 * Boundary cases handled:
 *   - M=0, N=0, or K=0: C is zero-initialised, then early return.
 *   - mc_actual, kc_actual, nr_actual < MR/KC/NR: zero-padding handles these.
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

#include <pthread.h>

/* Mutex protecting workspace buffers from concurrent callers.
 * With the GIL released (Py_BEGIN_ALLOW_THREADS in pymodule.c), two Python
 * threads can enter jblas_dgemm_c simultaneously.  Both packed_B (shared across
 * all threads) and packed_A (per-thread within one call, but shared across
 * concurrent calls) would race without serialisation.
 * Internal OpenMP parallelism still works (it parallelises the IC loop).
 * Non-static so dsyrk.c and dsyr2k.c can share the same lock (they all use
 * the shared jblas_packed_B workspace). */
pthread_mutex_t jblas_dgemm_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ---------------------------------------------------------------------------
 * Utility macros
 * ---------------------------------------------------------------------------
 */
#define MIN(a, b)       ((a) < (b) ? (a) : (b))
#define CEIL_DIV(a, b)  (((a) + (b) - 1) / (b))
/* Round x up to the next multiple of a (a must be a power of 2) */
#define ALIGN_UP(x, a)  (((x) + (a) - 1) & ~((size_t)(a) - 1))
#define MAX_MR  8   /* largest MR across all ISAs */
#define MAX_NR  8   /* largest NR across all ISAs */

/* ---------------------------------------------------------------------------
 * Global dgemm state — set once by jblas_dgemm_init(), then read-only
 * ---------------------------------------------------------------------------
 */
int JBLAS_MR = 4;
int JBLAS_NR = 4;
int JBLAS_KC = 128;
int JBLAS_MC = 32;
int JBLAS_NC = 1024;

double *jblas_packed_A = NULL;
double *jblas_packed_B = NULL;
int     jblas_n_threads = 1;

/* Microkernel function pointer — set to generic here; overwritten by
 * platform.c after ISA detection. */
jblas_dgemm_micro_fn jblas_dgemm_microkernel = jblas_dgemm_micro_generic;

/* ---------------------------------------------------------------------------
 * jblas_dgemm_init — Validate blocking invariants and allocate packing workspace.
 *
 * Called from jblas_init() in platform.c after ISA detection.
 * ISA-specific blocking params are set in platform.c before this is called,
 * so this function uses the already-set JBLAS_MR/NR/KC/MC/NC values.
 *
 * Returns 0 on success, -1 on allocation failure.
 * ---------------------------------------------------------------------------
 */
int jblas_dgemm_init(void) {
    /* Guard: MR/NR must fit in the stack-allocated scratch buffer used for
     * tail tiles (MAX_MR * MAX_NR doubles).  A future ISA with larger tile
     * sizes must bump MAX_MR/MAX_NR accordingly. */
    if (JBLAS_MR > MAX_MR || JBLAS_NR > MAX_NR) {
        fprintf(stderr,
            "FATAL: JBLAS_MR=%d > MAX_MR=%d or JBLAS_NR=%d > MAX_NR=%d\n",
            JBLAS_MR, MAX_MR, JBLAS_NR, MAX_NR);
        return -1;
    }

    /* Buffer safety: packing functions write ceil(NC/NR)*NR*KC and
     * ceil(MC/MR)*MR*KC doubles respectively.  These only equal NC*KC
     * and MC*KC when the divisibility invariants hold. */
    if (JBLAS_NC % JBLAS_NR != 0 || JBLAS_MC % JBLAS_MR != 0) {
        fprintf(stderr,
            "FATAL: blocking invariant violated: NC=%d must be divisible by NR=%d, "
            "MC=%d must be divisible by MR=%d\n",
            JBLAS_NC, JBLAS_NR, JBLAS_MC, JBLAS_MR);
        return -1;
    }

#ifdef _OPENMP
    jblas_n_threads = omp_get_max_threads();
    if (jblas_n_threads < 1) jblas_n_threads = 1;
#else
    jblas_n_threads = 1;
#endif

    /* packed_B: KC * NC doubles — shared across threads.
     * ALIGN_UP satisfies C11 aligned_alloc requirement: size % alignment == 0. */
    size_t b_bytes = ALIGN_UP(
        (size_t)JBLAS_KC * (size_t)JBLAS_NC * sizeof(double), 64);
    jblas_packed_B = (double *)aligned_alloc(64, b_bytes);
    if (!jblas_packed_B) {
        fprintf(stderr, "jblas: aligned_alloc failed for packed_B (%zu bytes)\n", b_bytes);
        return -1;
    }

    /* packed_A: n_threads * MC * KC doubles — per-thread slice */
    size_t a_bytes = ALIGN_UP(
        (size_t)jblas_n_threads * (size_t)JBLAS_MC * (size_t)JBLAS_KC * sizeof(double), 64);
    jblas_packed_A = (double *)aligned_alloc(64, a_bytes);
    if (!jblas_packed_A) {
        fprintf(stderr, "jblas: aligned_alloc failed for packed_A (%zu bytes)\n", a_bytes);
        free(jblas_packed_B);
        jblas_packed_B = NULL;
        return -1;
    }

    /* Redundant with the file-scope initializer of jblas_dgemm_microkernel
     * above — kept for defensive clarity.  platform.c overwrites with the
     * ISA-specific microkernel after this returns. */
    jblas_dgemm_microkernel = jblas_dgemm_micro_generic;

    return 0;
}

/* ---------------------------------------------------------------------------
 * jblas_dgemm_cleanup — Free workspace buffers.
 *
 * Safe to call on NULL pointers. Called from module cleanup if needed.
 * ---------------------------------------------------------------------------
 */
void jblas_dgemm_cleanup(void) {
    free(jblas_packed_A);
    free(jblas_packed_B);
    jblas_packed_A = NULL;
    jblas_packed_B = NULL;
}

/* ---------------------------------------------------------------------------
 * jblas_pack_A — Pack a mc x kc submatrix of A into MR-wide column strips.
 *
 * Layout in packed buffer:
 *   Strips are grouped as ceil(mc/mr) strips of MR rows each.
 *   Within each strip: [k=0..kc)[row=0..MR) stored contiguously.
 *   I.e., packed[strip * kc * MR + k * MR + r].
 *
 * Tail strip (mc % MR != 0): copy tail rows, zero-pad remaining rows to MR.
 *
 * trans=0: A[i,p] = A_base[i*lda + p]  (A is row-major, no transpose)
 * trans=1: A^T[i,p] = A_base[p*lda + i] (A is column-major / transposed)
 *
 * Parameters:
 *   A       — pointer to the top-left of the A panel (already offset for pc/ic)
 *   lda     — leading dimension of A (stride between rows, in doubles)
 *   mc      — number of rows in this panel
 *   kc      — number of columns in this panel
 *   packed  — output buffer, pre-allocated for ceil(mc/mr)*kc*mr doubles
 *   mr      — tile height (JBLAS_MR)
 *   trans   — 0=no transpose, 1=transpose
 * ---------------------------------------------------------------------------
 */
void jblas_pack_A(const double *A, npy_intp lda,
                  npy_intp mc, npy_intp kc, double *packed,
                  int mr, int trans)
{
    npy_intp n_full   = mc / mr;          /* number of full MR-row strips */
    npy_intp tail     = mc % mr;          /* leftover rows in last strip  */
    npy_intp n_strips = n_full + (tail > 0 ? 1 : 0);

    for (npy_intp s = 0; s < n_strips; s++) {
        npy_intp row_base = s * mr;       /* first row index of this strip */
        npy_intp rows     = (s < n_full) ? mr : tail; /* actual rows in strip */
        double *dst       = packed + s * (npy_intp)mr * kc;

        for (npy_intp k = 0; k < kc; k++) {
            double *col = dst + k * mr;
            for (npy_intp r = 0; r < rows; r++) {
                npy_intp row = row_base + r;
                if (!trans) {
                    col[r] = A[row * lda + k];
                } else {
                    /* trans: A[p, i] stored as A[k*lda + row] */
                    col[r] = A[k * lda + row];
                }
            }
            /* Zero-pad tail rows */
            for (npy_intp r = rows; r < mr; r++) {
                col[r] = 0.0;
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * jblas_pack_B — Pack a kc x nr subpanel of B into NR-wide column strips.
 *
 * Layout in packed buffer:
 *   Strips are grouped as ceil(nr/NR) strips of NR columns each.
 *   Within each strip: [k=0..kc)[col=0..NR) stored contiguously.
 *   I.e., packed[strip * kc * NR + k * NR + c].
 *
 * Tail strip (nr % NR != 0): copy tail cols, zero-pad remaining to NR.
 *
 * trans=0: B[p,j] = B_base[p*ldb + j]  (B is row-major, no transpose)
 * trans=1: B^T[p,j] = B_base[j*ldb + p] (B is column-major / transposed)
 *
 * Parameters:
 *   B        — pointer to top-left of B panel (already offset for jc/pc)
 *   ldb      — leading dimension of B (stride between rows, in doubles)
 *   kc       — number of rows in this panel (depth)
 *   nr       — number of columns in this panel (may be < NR for tail)
 *   packed   — output buffer, pre-allocated for kc*ceil(nr/nr_param)*nr_param doubles
 *   nr_param — tile width (JBLAS_NR)
 *   trans    — 0=no transpose, 1=transpose
 * ---------------------------------------------------------------------------
 */
void jblas_pack_B(const double *B, npy_intp ldb,
                  npy_intp kc, npy_intp nr, double *packed,
                  int nr_param, int trans)
{
    npy_intp n_full   = nr / nr_param;
    npy_intp tail     = nr % nr_param;
    npy_intp n_strips = n_full + (tail > 0 ? 1 : 0);

    for (npy_intp s = 0; s < n_strips; s++) {
        npy_intp col_base = s * nr_param;
        npy_intp cols     = (s < n_full) ? nr_param : tail;
        double *dst       = packed + s * kc * (npy_intp)nr_param;

        for (npy_intp k = 0; k < kc; k++) {
            double *row = dst + k * nr_param;
            for (npy_intp c = 0; c < cols; c++) {
                npy_intp col = col_base + c;
                if (!trans) {
                    row[c] = B[k * ldb + col];
                } else {
                    /* trans: B[j, p] stored as B[col*ldb + k] */
                    row[c] = B[col * ldb + k];
                }
            }
            /* Zero-pad tail columns */
            for (npy_intp c = cols; c < nr_param; c++) {
                row[c] = 0.0;
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * jblas_dgemm_c — Three-level blocking DGEMM.
 *
 * Computes C = op(A) * op(B) where:
 *   op(A) is M x K,  op(B) is K x N,  C is M x N.
 *   transa=0: op(A)=A, transa=1: op(A)=A^T
 *   transb=0: op(B)=B, transb=1: op(B)=B^T
 *
 * All arrays are row-major.  lda, ldb, ldc are leading dimensions (column
 * count of the physical stored matrix, regardless of transpose).
 *
 * Note: The physical dimensions of A in memory are:
 *   transa=0: A is M x K (lda >= K)
 *   transa=1: A is K x M (lda >= M)
 * And for B:
 *   transb=0: B is K x N (ldb >= N)
 *   transb=1: B is N x K (ldb >= K)
 * ---------------------------------------------------------------------------
 */
void jblas_dgemm_c(npy_intp M, npy_intp N, npy_intp K,
                   const double *A, npy_intp lda,
                   const double *B, npy_intp ldb,
                   double *C, npy_intp ldc,
                   int transa, int transb)
{
    /* Zero-initialize C first so every early-return path leaves C zeroed,
     * not full of garbage.  Negative M/N are rejected below, but a negative
     * M with positive ldc would be an overrun — guard that first. */
    if (M < 0 || N < 0 || K < 0) {
        fprintf(stderr, "FATAL: jblas_dgemm_c: negative dimension M=%ld N=%ld K=%ld\n",
                (long)M, (long)N, (long)K);
        abort();  /* programming error — unreachable from Python API */
    }

    /* Zero-initialize C */
    for (npy_intp i = 0; i < M; i++) {
        memset(C + i * ldc, 0, (size_t)N * sizeof(double));
    }

    if (M == 0 || N == 0 || K == 0)
        return;

    if (!jblas_packed_A || !jblas_packed_B) {
        fprintf(stderr,
            "FATAL: jblas_dgemm_c called but workspace not allocated "
            "(jblas_dgemm_init() failed or was never called)\n");
        abort();  /* programming error — py_dgemm guards this; C callers must too */
    }

    int lock_err = pthread_mutex_lock(&jblas_dgemm_mutex);
    if (lock_err != 0) {
        fprintf(stderr,
            "FATAL: jblas_dgemm_c: pthread_mutex_lock failed (errno=%d)\n",
            lock_err);
        abort();
    }

    int MR = JBLAS_MR;
    int NR = JBLAS_NR;
    int KC = JBLAS_KC;
    int MC = JBLAS_MC;
    int NC = JBLAS_NC;

    /* Outer loop: partition N into NC-wide panels */
    for (npy_intp jc = 0; jc < N; jc += NC) {
        npy_intp nr_actual = MIN(NC, N - jc);

        /* Middle loop: partition K into KC-deep blocks */
        for (npy_intp pc = 0; pc < K; pc += KC) {
            npy_intp kc_actual = MIN(KC, K - pc);

            /* Pack B panel: kc_actual x nr_actual → jblas_packed_B
             * B pointer offset depends on transpose:
             *   transb=0: B physical row pc, column jc → B + pc*ldb + jc
             *   transb=1: B physical row jc, column pc → B + jc*ldb + pc */
            const double *B_panel = transb ? (B + jc * ldb + pc)
                                           : (B + pc * ldb + jc);
            jblas_pack_B(B_panel, ldb, kc_actual, nr_actual,
                         jblas_packed_B, NR, transb);

            /* Inner loop: partition M into MC-tall row panels (OpenMP parallel).
             * Clamp to jblas_n_threads to prevent OOB packed_A access if
             * omp_set_num_threads() was called after jblas_dgemm_init(). */
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(jblas_n_threads)
#endif
            for (npy_intp ic = 0; ic < M; ic += MC) {
                npy_intp mc_actual = MIN(MC, M - ic);

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

                /* Pack A panel: mc_actual x kc_actual → packed_A_ptr
                 * A pointer offset depends on transpose:
                 *   transa=0: A physical row ic, column pc → A + ic*lda + pc
                 *   transa=1: A physical row pc, column ic → A + pc*lda + ic */
                const double *A_panel = transa ? (A + pc * lda + ic)
                                               : (A + ic * lda + pc);
                jblas_pack_A(A_panel, lda, mc_actual, kc_actual,
                             packed_A_ptr, MR, transa);

                /* Microkernel loop: MR x NR tiles within mc_actual x nr_actual */
                npy_intp n_mr_strips = CEIL_DIV(mc_actual, MR);
                npy_intp n_nr_strips = CEIL_DIV(nr_actual, NR);

                for (npy_intp jr_s = 0; jr_s < n_nr_strips; jr_s++) {
                    npy_intp jr      = jr_s * NR;
                    npy_intp nr_tile = MIN(NR, nr_actual - jr);

                    /* Pointer into packed B for this NR strip */
                    const double *pB_strip = jblas_packed_B +
                        (size_t)jr_s * (size_t)kc_actual * (size_t)NR;

                    for (npy_intp ir_s = 0; ir_s < n_mr_strips; ir_s++) {
                        npy_intp ir      = ir_s * MR;
                        npy_intp mr_tile = MIN(MR, mc_actual - ir);

                        /* Pointer into packed A for this MR strip */
                        const double *pA_strip = packed_A_ptr +
                            (size_t)ir_s * (size_t)kc_actual * (size_t)MR;

                        /* Target C tile: C[ic+ir, jc+jr] */
                        double *C_tile = C + (ic + ir) * ldc + (jc + jr);

                        if (mr_tile == MR && nr_tile == NR) {
                            /* Full tile: microkernel writes directly to C */
                            jblas_dgemm_microkernel(kc_actual, pA_strip, pB_strip,
                                                    C_tile, ldc);
                        } else {
                            /* Tail tile: accumulate into scratch, add valid portion to C */
                            double scratch[MAX_MR * MAX_NR];
                            memset(scratch, 0, sizeof(scratch));

                            jblas_dgemm_microkernel(kc_actual, pA_strip, pB_strip,
                                                    scratch, NR);

                            /* Copy valid mr_tile x nr_tile subblock from scratch to C */
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
            "FATAL: jblas_dgemm_c: pthread_mutex_unlock failed (errno=%d)\n",
            unlock_err);
        abort();
    }
}

/* ---------------------------------------------------------------------------
 * _dgemm_dispatch — Simplified no-transpose wrapper matching jblas_dgemm_fn
 * signature for the dispatch table (jblas_dispatch.dgemm).
 *
 * Assumes row-major layout with natural leading dimensions (lda=K, ldb=N,
 * ldc=N).  Always passes transa=transb=0 (no transpose).
 *
 * Note: py_dgemm bypasses this wrapper and calls jblas_dgemm_c directly
 * with the caller's transpose flags and physical leading dimensions.
 * ---------------------------------------------------------------------------
 */
static void _dgemm_dispatch(npy_intp m, npy_intp n, npy_intp k,
                            const double *A, const double *B, double *C)
{
    /* lda = k (A is m x k row-major), ldb = n (B is k x n row-major),
     * ldc = n (C is m x n row-major). */
    jblas_dgemm_c(m, n, k, A, k, B, n, C, n, 0, 0);
}

/* Expose for platform.c to assign to jblas_dispatch.dgemm */
jblas_dgemm_fn jblas_dgemm_dispatch_fn = _dgemm_dispatch;
