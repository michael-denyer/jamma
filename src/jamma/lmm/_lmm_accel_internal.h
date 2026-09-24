/* Internal Python entry points shared with the _lmm_accel module table. */

#ifndef JAMMA_LMM_ACCEL_INTERNAL_H
#define JAMMA_LMM_ACCEL_INTERNAL_H

#include "_lmm_support.h"

/* Bytes one workspace keeps live, by lifetime. persistent covers the native
 * buffers the creator allocates and the Python arrays the run retains beside
 * them; per_thread is held once per creator thread. */
typedef struct {
    size_t persistent;
    size_t per_thread;
} workspace_bytes_t;

/* Validated creator inputs both families build from. The arrays are owned
 * references the caller releases; a family INCREFs what it keeps. */
typedef struct {
    PyArrayObject *eigenvalues;
    PyArrayObject *uab_inv;       /* (n_inv, n_samples) */
    PyArrayObject *UtW;           /* (n_samples, n_cvt) */
    PyArrayObject *Uty;
    PyArrayObject *hi_eval_null;  /* NULL unless tests.score */
    int n_samples, n_cvt, n_grid, n_refine;
    int n_threads;                /* thread capacity, >= 1 */
    double l_min, l_max, logl_H0;
    lmm_tests_t tests;
} workspace_inputs_t;

/* Each family derives these from the same layout its creator allocates from. */
workspace_bytes_t ncvt1_workspace_bytes(int n_samples, int n_grid,
                                        lmm_tests_t tests);
workspace_bytes_t general_workspace_bytes(int n_cvt, int n_samples, int n_grid,
                                          lmm_tests_t tests);

/* A workspace capsule, or NULL with PyErr set. */
PyObject *ncvt1_create_workspace(const workspace_inputs_t *in);
PyObject *general_create_workspace(const workspace_inputs_t *in);

/* Each returns 0 when the capsule belongs to the other family, else 1 with
 * *out filled (*result is NULL with PyErr set on a failed compute). */
int ncvt1_capsule_bytes(PyObject *capsule, workspace_bytes_t *out);
int general_capsule_bytes(PyObject *capsule, workspace_bytes_t *out);
int ncvt1_compute_chunk(PyObject *capsule, PyObject *utg_t, int n_threads,
                        PyObject **result);
int general_compute_chunk(PyObject *capsule, PyObject *utg_t, int n_threads,
                          PyObject **result);

/* BGEN layout-2 probability decoding (_lmm_accel_bgen.c). */
PyObject *decode_bgen_probabilities_c(PyObject *self, PyObject *args,
                                      PyObject *kwargs);

#endif /* JAMMA_LMM_ACCEL_INTERNAL_H */
