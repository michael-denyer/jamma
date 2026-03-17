/**
 * dgemm_generic.c — Scalar (generic) DGEMM microkernel for jlinalg.
 *
 * Implements jlinalg_dgemm_micro_generic: a portable C fallback microkernel
 * that operates on pre-packed A and B panels produced by jlinalg_pack_A /
 * jlinalg_pack_B.  Correct but slow — serves as the always-available baseline
 * for non-SIMD builds.
 *
 * Layout contract (from dgemm.c):
 *   packed_A: MR-wide column strips — packed_A[k * MR + r]
 *   packed_B: NR-wide row strips    — packed_B[k * NR + c]
 *   C:        row-major tile        — C[r * ldc + c]
 *
 * Operation: C[r,c] += sum_{k=0}^{kc-1} packed_A[k*MR+r] * packed_B[k*NR+c]
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "jlinalg.h"

/**
 * jlinalg_dgemm_micro_generic — Scalar MR x NR microkernel.
 *
 * Updates C (MR x NR tile, row-major with leading dimension ldc) by
 * accumulating the outer product of packed_A column strip and packed_B
 * row strip over kc depth steps.
 *
 * Args:
 *   kc:       depth (number of k iterations).
 *   packed_A: packed A strip, kc * MR doubles.
 *   packed_B: packed B strip, kc * NR doubles.
 *   C:        MR x NR output tile, row-major, leading dimension ldc.
 *   ldc:      leading dimension of C (column stride in the full C matrix).
 */
void jlinalg_dgemm_micro_generic(npy_intp kc,
    const double * restrict packed_A,
    const double * restrict packed_B,
    double * restrict C, npy_intp ldc)
{
    int MR = JLINALG_MR;
    int NR = JLINALG_NR;

    for (npy_intp k = 0; k < kc; k++) {
        const double *a = packed_A + k * MR;
        const double *b = packed_B + k * NR;
        for (int r = 0; r < MR; r++) {
            double a_val = a[r];
            for (int c = 0; c < NR; c++) {
                C[r * ldc + c] += a_val * b[c];
            }
        }
    }
}
