/**
 * dgemm_avx2.c — AVX2/FMA DGEMM microkernel for jlinalg.
 *
 * Implements the 6x8 microkernel jlinalg_dgemm_micro_avx2.  This file must be
 * compiled with -mavx2 -mfma.  All entry points are guarded with
 * __attribute__((target("avx2,fma"))) as a belt-and-suspenders measure.
 *
 * Register allocation:
 *   12 YMM accumulator registers: acc[row][col_group]
 *     row:       0..5  (MR=6 rows)
 *     col_group: 0..1  (NR=8 cols / 4 doubles per YMM = 2 groups)
 *   2 YMM B registers: b0 (cols 0-3), b1 (cols 4-7)
 *   1 YMM A temporary: each a[r] is broadcast-loaded, used for 2 FMAs,
 *     then dead (compiler reuses the register across rows).
 *   Peak live YMM count: 12 accumulators + 2 B + 1 A = 15 (fits in 16 YMM).
 *
 * Operation per k-step:
 *   b0 = packed_B[k*NR + 0..3]   (YMM load)
 *   b1 = packed_B[k*NR + 4..7]   (YMM load)
 *   For each row r (0..5):
 *     a[r] = broadcast(packed_A[k*MR + r])
 *     acc[r][0] = fma(a[r], b0, acc[r][0])
 *     acc[r][1] = fma(a[r], b1, acc[r][1])
 *
 * After k-loop: load C, add accumulators, store.
 *
 * vzeroupper requirement:
 *   vzeroupper MUST be called before every return from an AVX function to
 *   avoid AVX-SSE transition penalties.  This file has a single linear exit
 *   path so there is exactly one vzeroupper site.
 *
 * MR=6, NR=8 are the blocking parameters set by jlinalg_init() in platform.c
 * when AVX2 is detected; they are read from the JLINALG_MR / JLINALG_NR globals
 * (which equal 6 and 8 on the AVX2 path).
 */

#if defined(__x86_64__)

#include <immintrin.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "jlinalg.h"

/**
 * jlinalg_dgemm_micro_avx2 — AVX2+FMA 6x8 DGEMM microkernel.
 *
 * Updates a 6x8 tile of C:
 *   C[0..5, 0..7] += packed_A[0..kc, 0..5] * packed_B[0..kc, 0..7]
 *
 * where packed_A and packed_B are in k-major panel format
 * (packed_A[k*6 + r], packed_B[k*8 + c]).
 *
 * Args:
 *   kc:       Depth (number of k iterations).
 *   packed_A: Pre-packed A strip, kc*6 doubles (k-major, 6 rows).
 *   packed_B: Pre-packed B strip, kc*8 doubles (k-major, 8 cols).
 *   C:        6x8 output tile, row-major, leading dimension ldc.
 *   ldc:      Leading dimension of C (column stride in full C matrix).
 */
__attribute__((target("avx2,fma")))
void jlinalg_dgemm_micro_avx2(npy_intp kc,
    const double * restrict packed_A,
    const double * restrict packed_B,
    double * restrict C, npy_intp ldc)
{
    /* 12 accumulators: acc_r0 = cols 0-3 for row r, acc_r1 = cols 4-7 */
    __m256d acc_00 = _mm256_setzero_pd();
    __m256d acc_01 = _mm256_setzero_pd();
    __m256d acc_10 = _mm256_setzero_pd();
    __m256d acc_11 = _mm256_setzero_pd();
    __m256d acc_20 = _mm256_setzero_pd();
    __m256d acc_21 = _mm256_setzero_pd();
    __m256d acc_30 = _mm256_setzero_pd();
    __m256d acc_31 = _mm256_setzero_pd();
    __m256d acc_40 = _mm256_setzero_pd();
    __m256d acc_41 = _mm256_setzero_pd();
    __m256d acc_50 = _mm256_setzero_pd();
    __m256d acc_51 = _mm256_setzero_pd();

    const double *pA = packed_A;
    const double *pB = packed_B;

    for (npy_intp k = 0; k < kc; k++, pA += 6, pB += 8) {
        /* Load 8 B values (NR=8) into 2 YMM registers */
        __m256d b0 = _mm256_loadu_pd(pB + 0);
        __m256d b1 = _mm256_loadu_pd(pB + 4);

        /* Broadcast A element for each row and accumulate */
        __m256d a0 = _mm256_set1_pd(pA[0]);
        acc_00 = _mm256_fmadd_pd(a0, b0, acc_00);
        acc_01 = _mm256_fmadd_pd(a0, b1, acc_01);

        __m256d a1 = _mm256_set1_pd(pA[1]);
        acc_10 = _mm256_fmadd_pd(a1, b0, acc_10);
        acc_11 = _mm256_fmadd_pd(a1, b1, acc_11);

        __m256d a2 = _mm256_set1_pd(pA[2]);
        acc_20 = _mm256_fmadd_pd(a2, b0, acc_20);
        acc_21 = _mm256_fmadd_pd(a2, b1, acc_21);

        __m256d a3 = _mm256_set1_pd(pA[3]);
        acc_30 = _mm256_fmadd_pd(a3, b0, acc_30);
        acc_31 = _mm256_fmadd_pd(a3, b1, acc_31);

        __m256d a4 = _mm256_set1_pd(pA[4]);
        acc_40 = _mm256_fmadd_pd(a4, b0, acc_40);
        acc_41 = _mm256_fmadd_pd(a4, b1, acc_41);

        __m256d a5 = _mm256_set1_pd(pA[5]);
        acc_50 = _mm256_fmadd_pd(a5, b0, acc_50);
        acc_51 = _mm256_fmadd_pd(a5, b1, acc_51);
    }

    /* Store: load existing C, add accumulators, store back */
    _mm256_storeu_pd(C + 0 * ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 0 * ldc + 0), acc_00));
    _mm256_storeu_pd(C + 0 * ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 0 * ldc + 4), acc_01));

    _mm256_storeu_pd(C + 1 * ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 1 * ldc + 0), acc_10));
    _mm256_storeu_pd(C + 1 * ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 1 * ldc + 4), acc_11));

    _mm256_storeu_pd(C + 2 * ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 2 * ldc + 0), acc_20));
    _mm256_storeu_pd(C + 2 * ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 2 * ldc + 4), acc_21));

    _mm256_storeu_pd(C + 3 * ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 3 * ldc + 0), acc_30));
    _mm256_storeu_pd(C + 3 * ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 3 * ldc + 4), acc_31));

    _mm256_storeu_pd(C + 4 * ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 4 * ldc + 0), acc_40));
    _mm256_storeu_pd(C + 4 * ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 4 * ldc + 4), acc_41));

    _mm256_storeu_pd(C + 5 * ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 5 * ldc + 0), acc_50));
    _mm256_storeu_pd(C + 5 * ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 5 * ldc + 4), acc_51));

    /* CRITICAL: vzeroupper must precede every return from an AVX function.
     * Single exit point ensures this is always executed. */
    _mm256_zeroupper();
}

#endif /* __x86_64__ */
