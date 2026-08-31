/** Private state shared by BLAS discovery and vendor operations. */

#pragma once

#include "jlinalg.h"

typedef struct {
    int found;
    int is_ilp64;
    int has_lapack;
    int has_dsyrk;
    const char *name;
    void *handle;
    jlinalg_dgemm_ilp64_fn dgemm_ilp64;
    jlinalg_cblas_dgemm_ilp64_fn cblas_dgemm_ilp64;
    jlinalg_cblas_dsyrk_ilp64_fn cblas_dsyrk_ilp64;
    jlinalg_dsyrk_ilp64_fn dsyrk_ilp64;
    jlinalg_dsyevd_ilp64_fn dsyevd_ilp64;
    jlinalg_lapacke_dsyevd_ilp64_fn lapacke_dsyevd_ilp64;
    int has_lapacke_dsyevd;
    jlinalg_dsyevr_ilp64_fn dsyevr_ilp64;
    int has_dsyevr;
} blas_candidate_t;

/** Borrow the selected backend. Discovery retains ownership. */
const blas_candidate_t *blas_dispatch_active(void);

/** Whether vendor dgemm survived discovery and the environment veto. */
int blas_dispatch_has_vendor_dgemm(void);
