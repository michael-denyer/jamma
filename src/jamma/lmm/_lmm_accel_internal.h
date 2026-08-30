/* Internal Python entry points shared with the _lmm_accel module table. */

#ifndef JAMMA_LMM_ACCEL_INTERNAL_H
#define JAMMA_LMM_ACCEL_INTERNAL_H

#include "_lmm_support.h"

PyObject *create_workspace_ncvt1_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *compute_lmm_chunk_ncvt1_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *create_workspace_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *compute_lmm_chunk_fused_general_c_py(
    PyObject *self, PyObject *args, PyObject *kwargs);

#endif /* JAMMA_LMM_ACCEL_INTERNAL_H */
