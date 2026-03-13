/**
 * jblas.h — Public C API for the JAMMA BLAS compute layer.
 *
 * Declares the ISA dispatch table, jblas_init(), and function signatures for
 * Level 1/2 BLAS primitives (ddot, dnrm2, daxpy, dscal, dgemv) plus Level 3
 * (dgemm, stub until C implementation is added).
 *
 * ABI version bump required if any function signature or struct layout changes.
 */

#pragma once

#include <stddef.h>             /* size_t */
#include <numpy/arrayobject.h>  /* npy_intp */

/* Bump this constant whenever the public ABI changes (new fields in
 * jblas_dispatch_t, changed function signatures, etc.). pymodule.c exposes
 * this as a Python-level integer so callers can guard against ABI mismatches. */
#define JBLAS_ABI_VERSION 1

/* ---------------------------------------------------------------------------
 * Function-pointer typedefs for ISA-dispatched microkernels
 * ---------------------------------------------------------------------------
 * incx / incy are stride arguments (always 1 in the current Python API;
 * retained for BLAS compatibility and future strided-array support).
 */

typedef double (*jblas_ddot_fn)(
    npy_intp n,
    const double *x, int incx,
    const double *y, int incy);

typedef double (*jblas_dnrm2_fn)(
    npy_intp n,
    const double *x, int incx);

typedef void (*jblas_daxpy_fn)(
    npy_intp n,
    double alpha,
    const double *x, int incx,
    double       *y, int incy);

typedef void (*jblas_dscal_fn)(
    npy_intp n,
    double alpha,
    double *x, int incx);

/* dgemv: y = A*x (no alpha/beta/transpose for the internal primitive).
 * A is row-major (m x n), x is length-n, y is length-m output. */
typedef void (*jblas_dgemv_fn)(
    npy_intp m, npy_intp n,
    const double *A,
    const double *x,
    double       *y);

/* dgemm: C = A*B.  A is (m x k), B is (k x n), C is (m x n). Row-major. */
typedef void (*jblas_dgemm_fn)(
    npy_intp m, npy_intp n, npy_intp k,
    const double *A,
    const double *B,
    double       *C);

/* ---------------------------------------------------------------------------
 * Global dispatch table (set once by jblas_init, then read-only)
 * ---------------------------------------------------------------------------
 */

typedef struct {
    jblas_ddot_fn  ddot;
    jblas_dnrm2_fn dnrm2;
    jblas_daxpy_fn daxpy;
    jblas_dscal_fn dscal;
    jblas_dgemv_fn dgemv;
    jblas_dgemm_fn dgemm;
} jblas_dispatch_t;

extern jblas_dispatch_t jblas_dispatch;

/* ---------------------------------------------------------------------------
 * Initialisation and introspection
 * ---------------------------------------------------------------------------
 */

/**
 * jblas_init — Detect ISA and populate jblas_dispatch with the best available
 * microkernel pointers.  Guard with a static flag so it is idempotent across
 * repeated calls (e.g. multiprocessing child-process re-import).
 *
 * Returns: 0 on success, -1 on failure (currently always succeeds because the
 * generic fallback is always available).
 */
int jblas_init(void);

/**
 * jblas_isa_name — Return the active ISA as a C string.
 *
 * Returns: "AVX2", "NEON", or "generic" (never NULL).
 */
const char *jblas_isa_name(void);

/* ---------------------------------------------------------------------------
 * Generic (portable C) microkernel declarations
 * ---------------------------------------------------------------------------
 * These are always compiled and linked; they are used when no SIMD path is
 * available or as the tail handler for non-multiple-of-SIMD-width inputs.
 */

double jblas_ddot_generic(
    npy_intp n,
    const double *x, int incx,
    const double *y, int incy);

double jblas_dnrm2_generic(
    npy_intp n,
    const double *x, int incx);

void jblas_daxpy_generic(
    npy_intp n,
    double alpha,
    const double *x, int incx,
    double       *y, int incy);

void jblas_dscal_generic(
    npy_intp n,
    double alpha,
    double *x, int incx);

void jblas_dgemv_generic(
    npy_intp m, npy_intp n,
    const double *A,
    const double *x,
    double       *y);

/* ---------------------------------------------------------------------------
 * x86-64 AVX2 microkernel declarations
 * ---------------------------------------------------------------------------
 * These translation units are compiled with -mavx2 -mfma.  They must only be
 * called after jblas_init() has confirmed AVX2 support via CPUID.
 */

#if defined(__x86_64__)

double jblas_ddot_avx2(
    npy_intp n,
    const double *x, int incx,
    const double *y, int incy);

double jblas_dnrm2_avx2(
    npy_intp n,
    const double *x, int incx);

void jblas_daxpy_avx2(
    npy_intp n,
    double alpha,
    const double *x, int incx,
    double       *y, int incy);

void jblas_dscal_avx2(
    npy_intp n,
    double alpha,
    double *x, int incx);

void jblas_dgemv_avx2(
    npy_intp m, npy_intp n,
    const double *A,
    const double *x,
    double       *y);

#endif /* __x86_64__ */

/* ---------------------------------------------------------------------------
 * AArch64 NEON microkernel declarations — DISABLED
 * ---------------------------------------------------------------------------
 * No .c file provides definitions yet.  Uncomment when NEON implementations
 * are added and wire into jblas_dispatch in platform.c.  Until then, aarch64
 * dispatches to the generic (portable C) kernels.
 */

#if 0  /* Enable when NEON implementations exist */
#if defined(__aarch64__)

double jblas_ddot_neon(
    npy_intp n,
    const double *x, int incx,
    const double *y, int incy);

double jblas_dnrm2_neon(
    npy_intp n,
    const double *x, int incx);

void jblas_daxpy_neon(
    npy_intp n,
    double alpha,
    const double *x, int incx,
    double       *y, int incy);

void jblas_dscal_neon(
    npy_intp n,
    double alpha,
    double *x, int incx);

void jblas_dgemv_neon(
    npy_intp m, npy_intp n,
    const double *A,
    const double *x,
    double       *y);

#endif /* __aarch64__ */
#endif /* NEON disabled */
