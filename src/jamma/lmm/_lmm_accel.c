/* _lmm_accel module registration and NumPy C-API ownership. */

#include "_lmm_accel_internal.h"

#include <stdint.h>
#include <stdlib.h>

/* Bump when function signatures or array layout expectations change. */
#define ABI_VERSION 25

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

/* Covers PyArray headers, capsules and allocator metadata beside the array
 * payloads each family layout counts. */
#define WORKSPACE_OVERHEAD_BYTES ((size_t)1024 * 1024)

static PyObject *workspace_bytes_tuple(workspace_bytes_t b)
{
    return Py_BuildValue("(KK)",
                         (unsigned long long)(b.persistent + WORKSPACE_OVERHEAD_BYTES),
                         (unsigned long long)b.per_thread);
}

/* Prices a workspace from the layout its family's creator allocates from,
 * so Python can gate a run before any allocation. */
static PyObject *workspace_sizes_c(PyObject *self, PyObject *args)
{
    int n_samples, n_cvt, n_grid, lmm_mode, n_threads;
    (void)self;
    if (!PyArg_ParseTuple(args, "iiiii", &n_samples, &n_cvt, &n_grid,
                          &lmm_mode, &n_threads)) return NULL;
    if (n_samples < 1 || n_cvt < 1 || n_cvt > MAX_N_CVT || n_grid < 2 ||
        !lmm_mode_valid(lmm_mode) || n_threads < 1) {
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
    lmm_tests_t tests = lmm_tests(lmm_mode);
    return workspace_bytes_tuple(n_cvt == 1
        ? ncvt1_workspace_bytes(n_samples, n_grid, tests)
        : general_workspace_bytes(n_cvt, n_samples, n_grid, tests));
}

/* Test hook: the bytes a created workspace's layout holds, in
 * workspace_sizes_c's form, so a test can tie the plan to the allocation. */
static PyObject *_workspace_bytes_c(PyObject *self, PyObject *capsule)
{
    workspace_bytes_t b;
    (void)self;
    if (ncvt1_capsule_bytes(capsule, &b) || general_capsule_bytes(capsule, &b))
        return workspace_bytes_tuple(b);
    PyErr_SetString(PyExc_TypeError, "expected an lmm workspace capsule");
    return NULL;
}

/* Take and validate the arrays both families read. 0, or -1 with PyErr set;
 * either way the caller releases whatever *in holds. */
static int take_workspace_arrays(workspace_inputs_t *in, PyObject *eig_obj,
                                 PyObject *uab_obj, PyObject *utw_obj,
                                 PyObject *uty_obj, PyObject *hi_obj)
{
    int n = in->n_samples;
    int n_rows = in->n_cvt + 2;
    int n_inv = (in->n_cvt + 3) * n_rows / 2 - n_rows;
    in->eigenvalues = take_vector(eig_obj, n, "eigenvalues");
    if (!in->eigenvalues || validate_eigenvalues(
            (const double *)PyArray_DATA(in->eigenvalues), n, in->l_max) < 0)
        return -1;
    in->uab_inv = take_matrix(uab_obj, n_inv, n, "uab_invariant");
    if (!in->uab_inv) return -1;
    in->UtW = take_matrix(utw_obj, n, in->n_cvt, "UtW");
    if (!in->UtW) return -1;
    in->Uty = take_vector(uty_obj, n, "Uty");
    if (!in->Uty) return -1;
    if (!in->tests.score) return 0;
    in->hi_eval_null = take_vector(hi_obj, n, "hi_eval_null");
    if (!in->hi_eval_null) return -1;
    return validate_hi_eval_null(
        (const double *)PyArray_DATA(in->hi_eval_null), n);
}

/* One creator for every covariate count: the n_cvt=1 family when n_cvt is 1,
 * else the general family. */
static PyObject *create_workspace_c(PyObject *self, PyObject *args,
                                    PyObject *kwargs)
{
    static const char *kwlist[] = {
        "eigenvalues", "uab_invariant", "UtW", "Uty",
        "n_samples", "l_min", "l_max", "n_grid", "n_refine", "n_threads",
        "n_cvt", "lmm_mode", "hi_eval_null", "logl_H0",
        NULL
    };
    PyObject *eig_obj, *uab_obj, *utw_obj, *uty_obj;
    PyObject *hi_obj = NULL, *logl_obj = NULL;
    int lmm_mode = 0;
    workspace_inputs_t in = {0};
    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOiddiiii|$iOO", (char **)kwlist,
            &eig_obj, &uab_obj, &utw_obj, &uty_obj,
            &in.n_samples, &in.l_min, &in.l_max, &in.n_grid, &in.n_refine,
            &in.n_threads, &in.n_cvt, &lmm_mode, &hi_obj, &logl_obj))
        return NULL;
    if (parse_mode_inputs(lmm_mode, &hi_obj, logl_obj, &in.tests,
                          &in.logl_H0) < 0)
        return NULL;
    if (validate_batch_params(in.n_samples, in.l_min, in.l_max, in.n_grid,
                              in.n_refine) < 0)
        return NULL;
    if (in.n_cvt < 1 || in.n_cvt > MAX_N_CVT) {
        PyErr_Format(PyExc_ValueError, "n_cvt must be 1..%d, got %d",
                     MAX_N_CVT, in.n_cvt);
        return NULL;
    }
#ifdef _OPENMP
    if (in.n_threads < 1) in.n_threads = 1;
#else
    in.n_threads = 1;
#endif

    PyObject *capsule = NULL;
    if (take_workspace_arrays(&in, eig_obj, uab_obj, utw_obj, uty_obj,
                              hi_obj) == 0)
        capsule = in.n_cvt == 1 ? ncvt1_create_workspace(&in)
                                : general_create_workspace(&in);
    Py_XDECREF(in.eigenvalues);
    Py_XDECREF(in.uab_inv);
    Py_XDECREF(in.UtW);
    Py_XDECREF(in.Uty);
    Py_XDECREF(in.hi_eval_null);
    return capsule;
}

static PyObject *compute_lmm_chunk_c(PyObject *self, PyObject *args,
                                     PyObject *kwargs)
{
    static const char *kwlist[] = {"workspace", "utg_t", "n_threads", NULL};
    PyObject *capsule, *utg_t, *result;
    int n_threads;
    (void)self;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOi", (char **)kwlist,
                                     &capsule, &utg_t, &n_threads))
        return NULL;
    if (ncvt1_compute_chunk(capsule, utg_t, n_threads, &result) ||
        general_compute_chunk(capsule, utg_t, n_threads, &result))
        return result;
    PyErr_SetString(PyExc_TypeError, "expected an lmm workspace capsule");
    return NULL;
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
 * n_cvt=1 and general-family translation units, which also own each
 * family's workspace layout; this file owns module registration, the
 * creator's, compute's and sizing query's dispatch between families, and
 * the shared NumPy C-API pointer.
 * ========================================================================= */

static PyMethodDef methods[] = {
    {
        "workspace_sizes_c", workspace_sizes_c, METH_VARARGS,
        "Return conservative persistent and per-thread bytes."
    },
    {
        "_workspace_bytes_c", _workspace_bytes_c, METH_O,
        "Return a created workspace's layout bytes in workspace_sizes_c's form."
    },
    {
        "create_workspace_c",
        (PyCFunction)create_workspace_c,
        METH_VARARGS | METH_KEYWORDS,
        "Create the per-run workspace for any n_cvt and lmm_mode.\n"
        "\n"
        "n_cvt == 1 builds the n_cvt=1 family's workspace, otherwise the\n"
        "general family's, which constructs its Pab table from n_cvt. Both\n"
        "hold UtW/Uty for on-the-fly Uab computation, the lambda grid and its\n"
        "invariant dot products, per-thread scratch for n_threads, and the\n"
        "null-model block the mode needs.\n"
        "\n"
        "Args:\n"
        "    eigenvalues:   (n_samples,) float64\n"
        "    uab_invariant: (n_inv, n_samples) float64, SoA invariant columns\n"
        "    UtW:           (n_samples, n_cvt) float64, row-major\n"
        "    Uty:           (n_samples,) float64, rotated phenotype\n"
        "    n_samples, l_min, l_max, n_grid, n_refine\n"
        "    n_threads:     int, thread capacity of every later compute call\n"
        "    n_cvt:         int, covariates including the intercept\n"
        "    lmm_mode:      int, keyword-only: 1 Wald, 2 LRT, 3 Score, 4 all\n"
        "    hi_eval_null:  (n_samples,) float64, modes 3 and 4 only\n"
        "    logl_H0:       float, modes 2 and 4 only\n"
        "\n"
        "Returns:\n"
        "    PyCapsule for compute_lmm_chunk_c\n"
    },
    {
        "compute_lmm_chunk_c",
        (PyCFunction)compute_lmm_chunk_c,
        METH_VARARGS | METH_KEYWORDS,
        "Per-chunk compute from UtG_T for any workspace.\n"
        "\n"
        "Forms the varying Uab columns from UtW/Uty on the fly and runs the\n"
        "tests the workspace was built for in one pass off one coarse grid.\n"
        "\n"
        "Args:\n"
        "    workspace:  PyCapsule from create_workspace_c\n"
        "    utg_t:      (n_snps, n_samples) float64, UtG.T\n"
        "    n_threads:  int, capped at the workspace's thread capacity\n"
        "\n"
        "Returns:\n"
        "    mode 1: dict with lambdas, logls, betas, ses, pwalds\n"
        "    mode 2: dict with logls, lambdas_mle, p_lrts\n"
        "    mode 3: dict with betas, ses, p_scores\n"
        "    mode 4: mode 1's keys plus p_scores, lambdas_mle, p_lrts\n"
        "    each value (n_snps,) float64\n"
    },
    {
        "decode_bgen_probabilities_c",
        (PyCFunction)decode_bgen_probabilities_c,
        METH_VARARGS | METH_KEYWORDS,
        "Decode a batch of BGEN v1.2 layout-2 variants, one column each.\n"
        "\n"
        "Unphased, diploid, biallelic, bit depth 1..16. Exact integer INFO\n"
        "sums need the quantised values in 16 bits, so B > 16 is rejected.\n"
        "Runs OpenMP-parallel across variants with the GIL released.\n"
        "\n"
        "Args:\n"
        "    buffers:   sequence of k bytes-like, one variant's probability\n"
        "               data each: raw, or zlib-compressed\n"
        "    uncompressed_lengths: None for raw buffers, else int64 (k,)\n"
        "               inflated length D of each zlib buffer\n"
        "    n_samples: int, the header's N\n"
        "    dosages:   out (n_samples, k) float64, F-order: (2*q11 + q12) /\n"
        "               (2**B - 1), the first allele's dosage; NaN if missing\n"
        "    q11, q12:  out (n_samples, k) uint16, F-order: stored P(11) and\n"
        "               P(12); 0 for a missing sample\n"
        "    missing:   out (n_samples, k) bool, F-order\n"
        "    bit_depth: out (k,) uint8, each variant's B\n"
        "    n_threads: int, capped at k\n"
        "    info_rows: None for every sample, else bool (n_samples,): the\n"
        "               samples the INFO sums cover\n"
        "    info_sums: None, or out int64 (k, 4) C-order: per variant, over\n"
        "               the non-missing info_rows samples, sum(2*q11 + q12),\n"
        "               sum((2*q11 + q12)**2), sum(4*q11 + q12) and the count\n"
        "\n"
        "Returns:\n"
        "    None, or (position, reason) for the first variant that failed\n"
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
