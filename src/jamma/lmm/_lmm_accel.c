/* _lmm_accel module registration and NumPy C-API ownership. */

#include "_lmm_accel_internal.h"

#include <stdint.h>
#include <stdlib.h>

/* Bump when function signatures or array layout expectations change. */
#define ABI_VERSION 20

/* -------------------------------------------------------------------------
 * _get_aligned_alloc_test_ptr
 *
 * Debug function: verify aligned_alloc returns 32-byte-aligned pointers.
 * Returns the pointer value as a Python int for assertion in tests.
 * ------------------------------------------------------------------------- */
static PyObject *_get_aligned_alloc_test_ptr(PyObject *self, PyObject *args)
{
    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) return NULL;
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "n must be positive");
        return NULL;
    }
    double *p = alloc_aligned_doubles((size_t)n);
    if (!p) return PyErr_NoMemory();
    uintptr_t addr = (uintptr_t)p;
    free(p);
    return PyLong_FromUnsignedLongLong((unsigned long long)addr);
}

/* Pure sizing companion to the workspace creators. Array payloads are split
 * by lifetime so Python can price before any workspace allocation occurs. */
static PyObject *workspace_sizes_c(PyObject *self, PyObject *args)
{
    int n_samples, n_cvt, n_grid, lmm_mode, n_threads;
    (void)self;
    if (!PyArg_ParseTuple(args, "iiiii", &n_samples, &n_cvt, &n_grid,
                          &lmm_mode, &n_threads)) return NULL;
    if (n_samples < 1 || n_cvt < 1 || n_cvt > 100 || n_grid < 2 ||
        lmm_mode < 1 || lmm_mode > 4 || n_threads < 1) {
        PyErr_SetString(PyExc_ValueError, "invalid workspace sizing dimensions");
        return NULL;
    }
    size_t rows_check = (size_t)n_cvt + 2;
    size_t index_check = ((size_t)n_cvt + 3) * rows_check / 2;
    size_t largest_factor = index_check > (size_t)n_grid
        ? index_check : (size_t)n_grid;
    if ((size_t)n_samples > SIZE_MAX / largest_factor / sizeof(double) / 16 ||
        (size_t)n_threads > SIZE_MAX / largest_factor /
            (size_t)n_samples / sizeof(double) / 4) {
        PyErr_SetString(PyExc_OverflowError, "workspace dimensions overflow size_t");
        return NULL;
    }

    size_t persistent = 0, per_thread = 0, transient_per_thread = 0;
    int output_columns = lmm_mode == 1 ? 5 : lmm_mode == 2 ? 2 :
                         lmm_mode == 3 ? 3 : 8;
    if (n_cvt == 1) {
        /* RunInvariants retains eigenvalues, UtW, Uty, Hi_eval_null, w and
         * the three invariant Uab rows while the capsule borrows the arrays. */
        persistent = (size_t)8 * n_samples * sizeof(double);
        if (lmm_mode != 3) {
            persistent += (size_t)n_grid * 6 * sizeof(double);
            persistent += aligned_double_bytes(grid_doubles(n_samples, n_grid));
        }
        if (lmm_mode == 3 || lmm_mode == 4)
            persistent += aligned_double_bytes((size_t)n_samples);
        if (lmm_mode == 3)
            persistent += (size_t)2 * aligned_double_bytes((size_t)n_samples);
        int scratch_arrays = lmm_mode == 1 ? 3 : lmm_mode == 3 ? 0 : 4;
        transient_per_thread = (size_t)scratch_arrays *
            (aligned_double_bytes((size_t)n_samples) + sizeof(double *));
    } else {
        size_t rows = (size_t)n_cvt + 2;
        size_t index = ((size_t)n_cvt + 3) * rows / 2;
        size_t var = rows, inv = index - var;
        /* Two fixed-size reduction buffers in the fresh likelihood and score
         * kernels. They coexist on each worker stack, regardless of n_cvt. */
        transient_per_thread = (size_t)2 * MAX_N_INDEX * sizeof(double);
        /* Native-owned eigenvalue and UtW-transpose payloads coexist with
         * RunInvariants' original eigenvalues, UtW, Uty and Hi_eval_null.
         * The invariant SoA is retained by reference and counted once. */
        persistent = ((size_t)4 * n_samples +
                      (size_t)2 * n_cvt * n_samples + inv * n_samples) *
                     sizeof(double);
        persistent += aligned_double_bytes(grid_doubles(n_samples, n_grid));
        persistent += ((size_t)n_grid * (inv + 2) + inv) * sizeof(double);
        if (lmm_mode == 3 || lmm_mode == 4)
            persistent += aligned_double_bytes((size_t)n_samples) +
                          inv * sizeof(double);
        persistent += pab_transport_peak_bytes(n_cvt);
        persistent += pab_python_conservative_bytes(n_cvt);
        per_thread = (general_scratch_doubles(n_samples, (int)rows) +
                      general_pab_doubles((int)rows, (int)index) + index) *
                     sizeof(double);
        if (lmm_mode == 1 || lmm_mode == 4)
            per_thread += general_pab_doubles((int)rows, (int)index)
                          * sizeof(double);
        if (lmm_mode == 2 || lmm_mode == 4)
            per_thread += general_lrt_thread_doubles(n_samples, (int)index) *
                          sizeof(double);
    }
    /* Covers PyArray headers, capsules and allocator metadata. Pab transport
     * arrays, including the temporary raw entry copy, are counted above. */
    persistent += 1024 * 1024;
    return Py_BuildValue("(KKKK)", (unsigned long long)persistent,
                         (unsigned long long)per_thread,
                         (unsigned long long)transient_per_thread,
                         (unsigned long long)output_columns * sizeof(double));
}

/* -------------------------------------------------------------------------
 * Module definition
 * ------------------------------------------------------------------------- */

#ifdef JAMMA_SENTINEL_UB
/* Sanitizer sentinel: deliberately reads 1 byte past a 4-byte
 * heap allocation. Under -fsanitize=address this MUST abort with a
 * heap-buffer-overflow trace pointing at this source line. Without ASAN,
 * returns garbage from past the buffer end. Compile with
 * -DJAMMA_SENTINEL_UB to enable; the asan-sentinel-meta-test workflow
 * job sets that macro and asserts the workflow exits non-zero with the
 * expected ASAN frame. Do NOT enable in any other build path.
 */
static PyObject *jamma_sentinel_oob(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    char *buf = (char *)malloc(4);
    if (!buf) {
        PyErr_NoMemory();
        return NULL;
    }
    /* 1-byte heap OOB — ASAN must catch this. */
    char x = buf[5];
    free(buf);
    return PyLong_FromLong((long)x);
}
#endif

/* =========================================================================
 * MODULE REGISTRATION — methods[], PyModuleDef, PyInit__lmm_accel
 *
 * Every exported entry point is named here. Implementations live in the
 * n_cvt=1 and general-family translation units; this file owns only module
 * registration and the shared NumPy C-API pointer.
 * ========================================================================= */

static PyMethodDef methods[] = {
    {
        "workspace_sizes_c", workspace_sizes_c, METH_VARARGS,
        "Return conservative persistent, per-thread, transient-thread and per-SNP bytes."
    },
    {
        "create_workspace_ncvt1_c",
        (PyCFunction)create_workspace_ncvt1_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Create the per-run n_cvt=1 workspace for one lmm_mode.\n"
        "\n"
        "Holds w/Uty for on-the-fly Uab computation, the lambda grid and its\n"
        "invariant dot products, and the null-model block the mode needs.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:   (n_samples,) float64\n"
        "    uab_invariant: (3, n_samples) float64 — SoA [ww, wy, yy]\n"
        "    w:             (n_samples,) float64 — UtW[:,0]\n"
        "    Uty:           (n_samples,) float64 — rotated phenotype\n"
        "    n_samples:     int\n"
        "    l_min:         float\n"
        "    l_max:         float\n"
        "    n_grid:        int\n"
        "    n_refine:      int\n"
        "    lmm_mode:      int, keyword-only — 1 Wald, 2 LRT, 3 Score, 4 all\n"
        "    hi_eval_null:  (n_samples,) float64 — modes 3 and 4 only\n"
        "    logl_H0:       float — modes 2 and 4 only\n"
        "\n"
        "Returns:\n"
        "    PyCapsule for compute_lmm_chunk_ncvt1_c\n"
    },
    {
        "compute_lmm_chunk_ncvt1_c",
        (PyCFunction)compute_lmm_chunk_ncvt1_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Per-chunk compute from UtG_T for any n_cvt=1 workspace, any lmm_mode.\n"
        "\n"
        "Computes wx/xx/xy on-the-fly from UtG_T and w/Uty in workspace.\n"
        "Forms the varying Uab columns from w/Uty rather than taking them\n"
        "prebuilt; the arithmetic and its order are unchanged.\n"
        "\n"
        "Dispatches on the workspace's lmm_mode to the loop that mode was\n"
        "built for: 1 REML Wald alone, 2 LRT alone, 3 Score alone, 4 all\n"
        "three in the same pass off the same coarse grid.\n"
        "\n"
        "Args:\n"
        "    workspace:  PyCapsule from create_workspace_ncvt1_c, any lmm_mode\n"
        "    utg_t:      (n_snps, n_samples) float64 — UtG.T\n"
        "    n_threads:  int\n"
        "\n"
        "Returns:\n"
        "    mode 1: dict with lambdas, logls, betas, ses, pwalds\n"
        "    mode 2: dict with logls, lambdas_mle, p_lrts\n"
        "    mode 3: dict with betas, ses, p_scores\n"
        "    mode 4: mode 1's keys plus p_scores, lambdas_mle, p_lrts\n"
        "    each value (n_snps,) float64\n"
    },
    {
        "create_workspace_general_c",
        (PyCFunction)create_workspace_general_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Create the per-run general (n_cvt >= 2) workspace for any lmm_mode.\n"
        "\n"
        "Takes the Pab table as the dict PabCTable._asdict() returns, and\n"
        "stores UtW (transposed to column-major), Uty and the varying-column\n"
        "map for on-the-fly Uab computation from UtG_T. Modes 3 and 4 also\n"
        "take hi_eval_null; modes 2 and 4 also take logl_H0.\n"
    },
    {
        "compute_lmm_chunk_fused_general_c",
        (PyCFunction)compute_lmm_chunk_fused_general_c_py,
        METH_VARARGS | METH_KEYWORDS,
        "Compute a chunk from UtG_T using a fused general workspace,\n"
        "any lmm_mode.\n"
        "\n"
        "Per-SNP varying dot products computed on-the-fly.\n"
        "Forms the varying Uab columns from UtW/Uty rather than taking them\n"
        "prebuilt; the arithmetic and its order are unchanged.\n"
        "\n"
        "Mode 1 runs REML Wald alone, 2 LRT alone, 3 Score alone, 4 all three\n"
        "in the same pass off the same coarse grid.\n"
    },
    {
        "_get_aligned_alloc_test_ptr",
        (PyCFunction)_get_aligned_alloc_test_ptr,
        METH_VARARGS,
        "Debug: return address of an aligned_alloc buffer for alignment testing."
    },
#ifdef JAMMA_SENTINEL_UB
    {
        "jamma_sentinel_oob",
        (PyCFunction)jamma_sentinel_oob,
        METH_NOARGS,
        "Sanitizer sentinel — deliberately reads past a heap "
        "allocation. Under ASAN this aborts with heap-buffer-overflow; "
        "without ASAN it returns garbage. Only compiled when "
        "-DJAMMA_SENTINEL_UB is set at build time."
    },
#endif
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_lmm_accel",
    "C extension: per-SNP REML/MLE pipelines (Wald, Score, LRT, fused mode-4) with OpenMP parallelism (n_cvt=1 + general n_cvt).",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__lmm_accel(void)
{
    import_array();  /* returns NULL on failure (NumPy Python 3 macro) */
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;

    /* ABI version — Python side checks this to detect stale .so files */
    if (PyModule_AddIntConstant(m, "ABI_VERSION", ABI_VERSION) < 0) { Py_DECREF(m); return NULL; }

    /* Expose whether this .so was compiled with OpenMP support */
#ifdef _OPENMP
    if (PyModule_AddIntConstant(m, "HAS_OPENMP", 1) < 0) { Py_DECREF(m); return NULL; }
#else
    if (PyModule_AddIntConstant(m, "HAS_OPENMP", 0) < 0) { Py_DECREF(m); return NULL; }
#endif

    return m;
}
