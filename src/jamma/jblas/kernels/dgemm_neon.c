/**
 * dgemm_neon.c — NEON (AArch64) DGEMM microkernel for jblas.
 *
 * Implements the 8x4 microkernel jblas_dgemm_micro_neon.  Compiled without
 * special flags on aarch64 since NEON is architecturally mandatory.
 *
 * Register allocation:
 *   16 Q-register accumulators: acc[row][col_group]
 *     row:       0..7  (MR=8 rows)
 *     col_group: 0..1  (NR=4 cols / 2 doubles per Q-register = 2 groups)
 *   2 Q-registers for B: b0 (cols 0-1), b1 (cols 2-3)
 *   1 Q-register for A: a = vdupq_n_f64(packed_A[k*8+r]), reused across rows
 *   Peak live: 16 acc + 2 B + 1 A = 19 Q-registers (fits in 32)
 *
 * Operation per k-step:
 *   b0 = vld1q_f64(packed_B + k*4 + 0)   (2 doubles)
 *   b1 = vld1q_f64(packed_B + k*4 + 2)   (2 doubles)
 *   For each row r (0..7):
 *     a[r] = vdupq_n_f64(packed_A[k*8 + r])
 *     acc[r][0] = vfmaq_f64(acc[r][0], a[r], b0)
 *     acc[r][1] = vfmaq_f64(acc[r][1], a[r], b1)
 *
 * After k-loop: load-add-store for each of 8 rows x 2 column groups.
 *
 * No vzeroupper equivalent needed on ARM (no YMM-SSE transition hazard).
 *
 * MR=8, NR=4 are the blocking parameters set by jblas_init() in platform.c
 * when NEON is detected.
 * Entire translation unit is guarded by #if defined(__aarch64__) so the
 * file compiles as an empty translation unit on x86_64 builds.
 */

#if defined(__aarch64__)

#include <arm_neon.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "jblas.h"

/**
 * jblas_dgemm_micro_neon — NEON 8x4 DGEMM microkernel.
 *
 * Updates an 8x4 tile of C:
 *   C[0..7, 0..3] += packed_A[0..kc, 0..7] * packed_B[0..kc, 0..3]
 *
 * Args:
 *   kc:       Depth (number of k iterations).
 *   packed_A: Pre-packed A strip, kc*8 doubles (k-major, 8 rows).
 *   packed_B: Pre-packed B strip, kc*4 doubles (k-major, 4 cols).
 *   C:        8x4 output tile, row-major, leading dimension ldc.
 *   ldc:      Leading dimension of C (column stride in full C matrix).
 */
void jblas_dgemm_micro_neon(npy_intp kc,
    const double * restrict packed_A,
    const double * restrict packed_B,
    double * restrict C, npy_intp ldc)
{
    /* 16 accumulators: acc[row][col_group], 2 doubles per Q-register */
    float64x2_t acc[8][2];
    for (int r = 0; r < 8; r++) {
        acc[r][0] = vdupq_n_f64(0.0);
        acc[r][1] = vdupq_n_f64(0.0);
    }

    const double *pA = packed_A;
    const double *pB = packed_B;

    for (npy_intp k = 0; k < kc; k++, pA += 8, pB += 4) {
        /* Load 4 B values (NR=4) into 2 Q-registers */
        float64x2_t b0 = vld1q_f64(pB + 0);
        float64x2_t b1 = vld1q_f64(pB + 2);

        /* Broadcast A element for each row and accumulate */
        for (int r = 0; r < 8; r++) {
            float64x2_t a = vdupq_n_f64(pA[r]);
            acc[r][0] = vfmaq_f64(acc[r][0], a, b0);
            acc[r][1] = vfmaq_f64(acc[r][1], a, b1);
        }
    }

    /* Store: load existing C, add accumulators, store back */
    for (int r = 0; r < 8; r++) {
        vst1q_f64(C + r * ldc + 0,
                  vaddq_f64(vld1q_f64(C + r * ldc + 0), acc[r][0]));
        vst1q_f64(C + r * ldc + 2,
                  vaddq_f64(vld1q_f64(C + r * ldc + 2), acc[r][1]));
    }
}

#else /* !__aarch64__ — empty translation unit; platform.c only references
       jblas_dgemm_micro_neon inside an __aarch64__ guard. */

#endif /* __aarch64__ */
