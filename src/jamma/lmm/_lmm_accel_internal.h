/* Internal Python entry points shared with the _lmm_accel module table. */

#ifndef JAMMA_LMM_ACCEL_INTERNAL_H
#define JAMMA_LMM_ACCEL_INTERNAL_H

#include "_lmm_support.h"

/* Bytes one workspace keeps live, by lifetime. persistent covers the native
 * buffers the creator allocates and the Python arrays the run retains beside
 * them; per_thread is held by the workspace once per creator thread;
 * transient_per_thread is allocated by each compute call once per thread. */
typedef struct {
    size_t persistent;
    size_t per_thread;
    size_t transient_per_thread;
} workspace_bytes_t;

/* Each family derives these from the same layout its creator allocates from. */
workspace_bytes_t ncvt1_workspace_bytes(int n_samples, int n_grid,
                                        lmm_tests_t tests);
workspace_bytes_t general_workspace_bytes(int n_cvt, int n_samples, int n_grid,
                                          lmm_tests_t tests);
/* The bytes a created workspace's layout holds. 1 and *out filled when the
 * capsule belongs to the family, else 0. */
int ncvt1_capsule_bytes(PyObject *capsule, workspace_bytes_t *out);
int general_capsule_bytes(PyObject *capsule, workspace_bytes_t *out);

PyObject *create_workspace_ncvt1_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *compute_lmm_chunk_ncvt1_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *create_workspace_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *compute_lmm_chunk_fused_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs);

#endif /* JAMMA_LMM_ACCEL_INTERNAL_H */
