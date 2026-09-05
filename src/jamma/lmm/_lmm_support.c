/*
 * _lmm_support.c — see _lmm_support.h for what lives here and why.
 *
 * NO_IMPORT_ARRAY: _lmm_accel.c owns import_array(); this unit shares its
 * PyArray_API through PY_ARRAY_UNIQUE_SYMBOL. See the header.
 */

#define NO_IMPORT_ARRAY
#include "_lmm_support.h"

#include "_lmm_logdet.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t aligned_double_bytes(size_t n)
{
    size_t raw = n * sizeof(double);
    if (n != 0 && raw / sizeof(double) != n) return 0;
    if (raw > SIZE_MAX - 31) return 0;
    return (raw + 31) & ~(size_t)31;
}

size_t grid_doubles(int n_samples, int n_grid)
{
    return (size_t)n_samples * n_grid;
}

size_t general_scratch_doubles(int n_samples, int n_rows)
{
    return (size_t)n_rows * n_samples;
}

size_t general_pab_doubles(int n_rows, int n_index)
{
    return (size_t)n_rows * n_index;
}

size_t general_lrt_thread_doubles(int n_samples, int n_index)
{
    return (size_t)n_index * n_samples;
}

size_t pab_entry_count(int n_rows)
{
    size_t k = (size_t)n_rows - 1;
    return k * (k + 1) * (k + 2) / 6;
}

size_t pab_transport_peak_bytes(int n_cvt)
{
    size_t rows = (size_t)n_cvt + 2;
    size_t index = ((size_t)n_cvt + 3) * rows / 2;
    size_t inv = index - rows;
    size_t int_count = inv + rows + 2 * ((size_t)n_cvt + 1)
        + 2 * rows + 2 * rows;
    /* During parsing, raw stride-4 entries and their pab_entry_t copy coexist. */
    return int_count * sizeof(int)
        + 2 * pab_entry_count((int)rows) * sizeof(pab_entry_t);
}

size_t pab_python_conservative_bytes(int n_cvt)
{
    size_t entries = pab_entry_count(n_cvt + 2);
    /* Cached six-int recursion tuple and its referenced int objects. The
     * temporary flattening list additionally holds four pointers per entry. */
    return entries * (384 + 4 * sizeof(void *));
}

double *alloc_aligned_doubles(size_t n)
{
    if (n == 0) return NULL;
    size_t bytes = aligned_double_bytes(n);
    if (bytes == 0) return NULL;
    return (double *)aligned_alloc(32, bytes);
}

double **alloc_thread_scratch(int n_threads, size_t n)
{
    /* calloc'd so a partial failure leaves unfilled slots NULL. */
    double **bufs = (double **)calloc((size_t)n_threads, sizeof(double *));
    if (!bufs) return NULL;
    for (int t = 0; t < n_threads; t++) {
        bufs[t] = alloc_aligned_doubles(n);
        if (!bufs[t]) {
            for (int u = 0; u < n_threads; u++) free(bufs[u]);
            free(bufs);
            return NULL;
        }
    }
    return bufs;
}

void free_thread_scratch(double **bufs, int n_threads)
{
    if (!bufs) return;
    for (int t = 0; t < n_threads; t++) free(bufs[t]);
    free(bufs);
}

int validate_eigenvalues(const double *data, int n_samples, double l_max)
{
    for (int i = 0; i < n_samples; i++) {
        if (!isfinite(data[i])) {
            /* PyErr_Format doesn't support %g — use snprintf + %s instead */
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", data[i]);
            PyErr_Format(PyExc_ValueError,
                "eigenvalues[%d] = %s is not finite. "
                "Check kinship matrix and eigendecomposition quality.", i, buf);
            return -1;
        }
        if (l_max * data[i] + 1.0 <= 0.0) {
            /* logdet_h_lambda (_lmm_logdet.h) splits lambda * ev + 1 by its
             * bit pattern and needs it positive at every lambda up to l_max.
             * Round-off negatives of order -1e-16 pass; below -1/l_max the
             * old log(v) gave NaN and the split would give a wrong number. */
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", data[i]);
            PyErr_Format(PyExc_ValueError,
                "eigenvalues[%d] = %s makes lambda * ev + 1 non-positive at "
                "l_max. Threshold the eigendecomposition before building a "
                "workspace.", i, buf);
            return -1;
        }
    }
    return 0;
}

int validate_batch_params(int n_samples, double l_min, double l_max,
                          int n_grid, int n_refine)
{
    if (n_samples < 3) {
        PyErr_SetString(PyExc_ValueError, "n_samples must be >= 3");
        return -1;
    }
    if (!(l_min > 0.0 && l_max > l_min)) {
        PyErr_SetString(PyExc_ValueError, "Require 0 < l_min < l_max");
        return -1;
    }
    if (n_grid < 2) {
        PyErr_SetString(PyExc_ValueError, "n_grid must be >= 2");
        return -1;
    }
    if (n_refine < 1) {
        PyErr_SetString(PyExc_ValueError, "n_refine must be >= 1");
        return -1;
    }
    return 0;
}

int warn_betainc_convergence(
    const double *betas, const double *pvalues, int n_snps)
{
    int n_betainc_nan = 0;
    for (int i = 0; i < n_snps; i++) {
        if (isfinite(betas[i]) && !isfinite(pvalues[i]))
            n_betainc_nan++;
    }
    if (n_betainc_nan > 0) {
        if (PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
                "%d SNPs have NaN p-values despite finite beta/SE — "
                "betainc continued fraction did not converge "
                "(extreme F-statistics). Consider checking these SNPs manually.",
                n_betainc_nan) < 0) {
            return -1;  /* warning promoted to exception */
        }
    }
    return 0;
}

int alloc_score_output(score_output_t *out, npy_intp n_snps)
{
    npy_intp dims[1] = { n_snps };
    out->betas    = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->ses      = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->p_scores = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    if (!out->betas || !out->ses || !out->p_scores) {
        Py_XDECREF(out->betas);
        Py_XDECREF(out->ses);
        Py_XDECREF(out->p_scores);
        return -1;
    }
    return 0;
}

void decref_score_output(score_output_t *out)
{
    Py_DECREF(out->betas);
    Py_DECREF(out->ses);
    Py_DECREF(out->p_scores);
}

PyObject *build_score_result_dict(score_output_t *out)
{
    PyObject *result = PyDict_New();
    if (!result) {
        decref_score_output(out);
        return NULL;
    }

    if (PyDict_SetItemString(result, "betas",    (PyObject *)out->betas)    < 0 ||
        PyDict_SetItemString(result, "ses",      (PyObject *)out->ses)      < 0 ||
        PyDict_SetItemString(result, "p_scores", (PyObject *)out->p_scores) < 0) {
        Py_DECREF(result);
        decref_score_output(out);
        return NULL;
    }

    decref_score_output(out);
    return result;
}

int alloc_lrt_output(lrt_output_t *out, npy_intp n_snps)
{
    npy_intp dims[1] = { n_snps };
    out->lambdas_mle = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    out->p_lrts      = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    if (!out->lambdas_mle || !out->p_lrts) {
        Py_XDECREF(out->lambdas_mle);
        Py_XDECREF(out->p_lrts);
        return -1;
    }
    return 0;
}

void decref_lrt_output(lrt_output_t *out)
{
    Py_DECREF(out->lambdas_mle);
    Py_DECREF(out->p_lrts);
}

PyObject *build_lrt_result_dict(lrt_output_t *out)
{
    PyObject *result = PyDict_New();
    if (!result) {
        decref_lrt_output(out);
        return NULL;
    }

    if (PyDict_SetItemString(result, "lambdas_mle", (PyObject *)out->lambdas_mle) < 0 ||
        PyDict_SetItemString(result, "p_lrts",      (PyObject *)out->p_lrts)      < 0) {
        Py_DECREF(result);
        decref_lrt_output(out);
        return NULL;
    }

    decref_lrt_output(out);
    return result;
}

int alloc_lmm_output(lmm_output_t *out, npy_intp n_snps, int lmm_mode)
{
    npy_intp dims[1] = { n_snps };
    int do_reml  = (lmm_mode == 1 || lmm_mode == 4);
    int do_score = (lmm_mode == 3 || lmm_mode == 4);
    int do_lrt   = (lmm_mode == 2 || lmm_mode == 4);

    int ok = 1;
    if (do_reml) {
        out->lambdas = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        out->logls   = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        out->pwalds  = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        ok = ok && out->lambdas && out->logls && out->pwalds;
    }
    /* betas/ses hold Wald's beta/se (modes 1, 4) or Score's (mode 3). */
    if (do_reml || do_score) {
        out->betas = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        out->ses   = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        ok = ok && out->betas && out->ses;
    }
    if (do_score) {
        out->p_scores = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        ok = ok && out->p_scores;
    }
    if (do_lrt) {
        out->lambdas_mle = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        out->p_lrts      = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        ok = ok && out->lambdas_mle && out->p_lrts;
    }

    if (!ok) {
        decref_lmm_output(out);
        return -1;
    }
    return 0;
}

void decref_lmm_output(lmm_output_t *out)
{
    Py_XDECREF(out->lambdas);
    Py_XDECREF(out->logls);
    Py_XDECREF(out->betas);
    Py_XDECREF(out->ses);
    Py_XDECREF(out->pwalds);
    Py_XDECREF(out->p_scores);
    Py_XDECREF(out->lambdas_mle);
    Py_XDECREF(out->p_lrts);
}

PyObject *build_lmm_result_dict(lmm_output_t *out)
{
    PyObject *result = PyDict_New();
    if (!result) {
        decref_lmm_output(out);
        return NULL;
    }

    int failed = 0;
    if (out->lambdas)
        failed |= PyDict_SetItemString(result, "lambdas", (PyObject *)out->lambdas) < 0;
    if (out->logls)
        failed |= PyDict_SetItemString(result, "logls", (PyObject *)out->logls) < 0;
    if (out->betas)
        failed |= PyDict_SetItemString(result, "betas", (PyObject *)out->betas) < 0;
    if (out->ses)
        failed |= PyDict_SetItemString(result, "ses", (PyObject *)out->ses) < 0;
    if (out->pwalds)
        failed |= PyDict_SetItemString(result, "pwalds", (PyObject *)out->pwalds) < 0;
    if (out->p_scores)
        failed |= PyDict_SetItemString(result, "p_scores", (PyObject *)out->p_scores) < 0;
    if (out->lambdas_mle)
        failed |= PyDict_SetItemString(
            result, "lambdas_mle", (PyObject *)out->lambdas_mle) < 0;
    if (out->p_lrts)
        failed |= PyDict_SetItemString(result, "p_lrts", (PyObject *)out->p_lrts) < 0;

    if (failed) {
        Py_DECREF(result);
        decref_lmm_output(out);
        return NULL;
    }

    decref_lmm_output(out);
    return result;
}

int *parse_int32_array(PyObject *obj, int expected_len, const char *name)
{
    if (!PyArray_Check(obj) ||
        PyArray_NDIM((PyArrayObject *)obj) != 1 ||
        PyArray_TYPE((PyArrayObject *)obj) != NPY_INT32) {
        PyErr_Format(PyExc_TypeError,
            "%s must be a one-dimensional int32 array", name);
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if (!arr) return NULL;
    if (PyArray_SIZE(arr) != expected_len) {
        PyErr_Format(PyExc_ValueError, "%s must have %d elements", name, expected_len);
        Py_DECREF(arr);
        return NULL;
    }
    int *copy = (int *)malloc((size_t)expected_len * sizeof(int));
    if (!copy) { Py_DECREF(arr); PyErr_NoMemory(); return NULL; }
    memcpy(copy, PyArray_DATA(arr), (size_t)expected_len * sizeof(int));
    Py_DECREF(arr);
    return copy;
}

static int get_pab_index(int a, int b, int n_cvt)
{
    int cols = n_cvt + 2;
    int a1 = a < b ? a : b;
    int b1 = a < b ? b : a;
    return (2 * cols - a1 + 2) * (a1 - 1) / 2 + b1 - a1;
}

static int get_pab_int(PyObject *dict, const char *key, int *value)
{
    PyObject *obj = PyDict_GetItemString(dict, key);
    if (!obj) {
        PyErr_Format(PyExc_KeyError, "pab_table_dict missing key '%s'", key);
        return -1;
    }
    long parsed = PyLong_AsLong(obj);
    if (parsed == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
            "pab_table_dict key '%s' must be an int in range [%d, %d]",
            key, INT_MIN, INT_MAX);
        return -1;
    }
    if (parsed < INT_MIN || parsed > INT_MAX) {
        PyErr_Format(PyExc_ValueError,
            "pab_table_dict key '%s' must fit in a C int", key);
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static int reject_noncanonical_int(
    const char *key, int supplied, int expected)
{
    if (supplied == expected) return 0;
    PyErr_Format(PyExc_ValueError,
        "%s=%d does not match canonical value %d derived from n_cvt",
        key, supplied, expected);
    return -1;
}

static int reject_noncanonical_array_value(
    const char *key, int offset, int supplied, int expected)
{
    if (supplied == expected) return 0;
    PyErr_Format(PyExc_ValueError,
        "%s[%d]=%d does not match canonical value %d derived from n_cvt",
        key, offset, supplied, expected);
    return -1;
}

int parse_pab_table_from_dict(PyObject *dict, pab_table_t *t, int n_samples)
{
    int supplied_n_index, supplied_n_rows, supplied_n_inv, supplied_n_var;
    int supplied_idx_xx, supplied_idx_xy, supplied_idx_yy;

    memset(t, 0, sizeof(*t));
    if (get_pab_int(dict, "n_cvt", &t->n_cvt) < 0) return -1;
    if (t->n_cvt < 1 || t->n_cvt > MAX_N_CVT) {
        PyErr_Format(PyExc_ValueError, "n_cvt must be 1..%d, got %d", MAX_N_CVT, t->n_cvt);
        return -1;
    }

    t->n_rows = t->n_cvt + 2;
    t->n_index = (t->n_cvt + 3) * (t->n_cvt + 2) / 2;
    t->n_var = t->n_cvt + 2;
    t->n_inv = t->n_index - t->n_var;
    t->idx_xx = get_pab_index(t->n_cvt + 1, t->n_cvt + 1, t->n_cvt);
    t->idx_xy = get_pab_index(t->n_cvt + 1, t->n_cvt + 2, t->n_cvt);
    t->idx_yy = get_pab_index(t->n_cvt + 2, t->n_cvt + 2, t->n_cvt);
    t->df = n_samples - t->n_cvt - 1;

#define CHECK_CANONICAL_INT(key, field) do { \
    if (get_pab_int(dict, key, &supplied_##field) < 0 || \
        reject_noncanonical_int(key, supplied_##field, t->field) < 0) return -1; \
} while(0)
    CHECK_CANONICAL_INT("n_index", n_index);
    CHECK_CANONICAL_INT("n_rows", n_rows);
    CHECK_CANONICAL_INT("n_inv", n_inv);
    CHECK_CANONICAL_INT("n_var", n_var);
    CHECK_CANONICAL_INT("idx_xx", idx_xx);
    CHECK_CANONICAL_INT("idx_xy", idx_xy);
    CHECK_CANONICAL_INT("idx_yy", idx_yy);
#undef CHECK_CANONICAL_INT

    /* Parse array fields — free_pab_table on failure (safe: pointers NULL-init'd) */
#define GETARR(key, field, len) do { \
    PyObject *obj = PyDict_GetItemString(dict, key); \
    if (!obj) { PyErr_Format(PyExc_KeyError, "pab_table_dict missing key '%s'", key); free_pab_table(t); return -1; } \
    (field) = parse_int32_array(obj, (len), key); \
    if (!(field)) { free_pab_table(t); return -1; } \
} while(0)

    GETARR("invariant_indices", t->invariant_indices, t->n_inv);
    GETARR("varying_indices",   t->varying_indices,   t->n_var);
    GETARR("logdet_diag_rows",  t->logdet_diag_rows,  t->n_cvt + 1);
    GETARR("logdet_diag_cols",  t->logdet_diag_cols,  t->n_cvt + 1);
    GETARR("level_offsets",     t->level_offsets,      t->n_rows);
    GETARR("level_counts",      t->level_counts,       t->n_rows);
    GETARR("var_a_cols",        t->var_a_cols,         t->n_var);
    GETARR("var_b_cols",        t->var_b_cols,         t->n_var);
#undef GETARR

    for (int i = 0; i < t->n_inv; i++) {
        if (t->invariant_indices[i] < 0 || t->invariant_indices[i] >= t->n_index) {
            PyErr_Format(PyExc_ValueError,
                "invariant_indices[%d] = %d out of range [0, %d)",
                i, t->invariant_indices[i], t->n_index);
            free_pab_table(t);
            return -1;
        }
    }
    for (int i = 0; i < t->n_var; i++) {
        if (t->varying_indices[i] < 0 || t->varying_indices[i] >= t->n_index) {
            PyErr_Format(PyExc_ValueError,
                "varying_indices[%d] = %d out of range [0, %d)",
                i, t->varying_indices[i], t->n_index);
            free_pab_table(t);
            return -1;
        }
    }
    for (int d = 0; d < t->n_cvt + 1; d++) {
        if (t->logdet_diag_rows[d] < 0 || t->logdet_diag_rows[d] >= t->n_rows) {
            PyErr_Format(PyExc_ValueError,
                "logdet_diag_rows[%d] = %d out of range [0, %d)",
                d, t->logdet_diag_rows[d], t->n_rows);
            free_pab_table(t);
            return -1;
        }
        if (t->logdet_diag_cols[d] < 0 || t->logdet_diag_cols[d] >= t->n_index) {
            PyErr_Format(PyExc_ValueError,
                "logdet_diag_cols[%d] = %d out of range [0, %d)",
                d, t->logdet_diag_cols[d], t->n_index);
            free_pab_table(t);
            return -1;
        }
    }
    for (int v = 0; v < t->n_var; v++) {
        if (t->var_a_cols[v] < 0 || t->var_a_cols[v] > t->n_cvt + 1 ||
            t->var_b_cols[v] < 0 || t->var_b_cols[v] > t->n_cvt + 1) {
            PyErr_Format(PyExc_ValueError,
                "var_a_cols[%d]=%d or var_b_cols[%d]=%d out of range [0, %d]",
                v, t->var_a_cols[v], v, t->var_b_cols[v], t->n_cvt + 1);
            free_pab_table(t);
            return -1;
        }
    }

    /* Parse entries (stride-4 flat int32 array) */
    {
        PyObject *entries_obj = PyDict_GetItemString(dict, "entries");
        if (!entries_obj) {
            PyErr_SetString(PyExc_KeyError, "pab_table_dict missing key 'entries'");
            free_pab_table(t);
            return -1;
        }
        int expected_n_entries = (int)pab_entry_count(t->n_rows);
        npy_intp expected_len = (npy_intp)expected_n_entries * 4;
        t->n_entries = expected_n_entries;

        int *raw = parse_int32_array(entries_obj, (int)expected_len, "entries");
        if (!raw) { free_pab_table(t); return -1; }
        t->entries = (pab_entry_t *)malloc((size_t)t->n_entries * sizeof(pab_entry_t));
        if (!t->entries) {
            free(raw);
            PyErr_NoMemory();
            free_pab_table(t);
            return -1;
        }
        for (int i = 0; i < t->n_entries; i++) {
            t->entries[i].index_ab = raw[i * 4 + 0];
            t->entries[i].index_aw = raw[i * 4 + 1];
            t->entries[i].index_bw = raw[i * 4 + 2];
            t->entries[i].index_ww = raw[i * 4 + 3];
        }
        free(raw);

        /* Validate entry indices are in range [0, n_index) */
        for (int i = 0; i < t->n_entries; i++) {
            if (t->entries[i].index_ab < 0 || t->entries[i].index_ab >= t->n_index ||
                t->entries[i].index_aw < 0 || t->entries[i].index_aw >= t->n_index ||
                t->entries[i].index_bw < 0 || t->entries[i].index_bw >= t->n_index ||
                t->entries[i].index_ww < 0 || t->entries[i].index_ww >= t->n_index) {
                PyErr_Format(PyExc_ValueError,
                    "entries[%d] has out-of-range index (n_index=%d)", i, t->n_index);
                free_pab_table(t);
                return -1;
            }
        }

        /* Validate level_offsets/level_counts don't exceed n_entries */
        for (int p = 0; p < t->n_rows; p++) {
            if (t->level_offsets[p] < 0 ||
                t->level_counts[p] < 0 ||
                (int64_t)t->level_offsets[p] + t->level_counts[p] > t->n_entries) {
                PyErr_Format(PyExc_ValueError,
                    "level_offsets[%d]=%d + level_counts[%d]=%d exceeds n_entries=%d",
                    p, t->level_offsets[p], p, t->level_counts[p], t->n_entries);
                free_pab_table(t);
                return -1;
            }
        }
    }

    /* The kernels require the exact packed layout, not merely in-range
     * indices. Validate the authoritative Python builder's transport data
     * against the layout implied by n_cvt before exposing the workspace. */
    {
        int inv = 0, var = 0;
        int genotype_col = t->n_cvt;
        for (int a = 1; a < t->n_cvt + 3; a++) {
            for (int b = a; b < t->n_cvt + 3; b++) {
                int index = get_pab_index(a, b, t->n_cvt);
                if (a - 1 == genotype_col || b - 1 == genotype_col) {
                    if (reject_noncanonical_array_value(
                            "varying_indices", var, t->varying_indices[var], index) < 0 ||
                        reject_noncanonical_array_value(
                            "var_a_cols", var, t->var_a_cols[var], a - 1) < 0 ||
                        reject_noncanonical_array_value(
                            "var_b_cols", var, t->var_b_cols[var], b - 1) < 0) {
                        free_pab_table(t);
                        return -1;
                    }
                    var++;
                } else {
                    if (reject_noncanonical_array_value(
                            "invariant_indices", inv, t->invariant_indices[inv], index) < 0) {
                        free_pab_table(t);
                        return -1;
                    }
                    inv++;
                }
            }
        }

        int entry = 0;
        for (int p = 0; p < t->n_rows; p++) {
            int count = 0;
            if (p > 0) {
                int remaining = t->n_rows - p;
                count = remaining * (remaining + 1) / 2;
            }
            if (reject_noncanonical_array_value(
                    "level_offsets", p, t->level_offsets[p], entry) < 0 ||
                reject_noncanonical_array_value(
                    "level_counts", p, t->level_counts[p], count) < 0) {
                free_pab_table(t);
                return -1;
            }
            for (int a = p + 1; p > 0 && a < t->n_rows + 1; a++) {
                for (int b = a; b < t->n_rows + 1; b++) {
                    int expected[4] = {
                        get_pab_index(a, b, t->n_cvt),
                        get_pab_index(a, p, t->n_cvt),
                        get_pab_index(b, p, t->n_cvt),
                        get_pab_index(p, p, t->n_cvt),
                    };
                    int supplied[4] = {
                        t->entries[entry].index_ab,
                        t->entries[entry].index_aw,
                        t->entries[entry].index_bw,
                        t->entries[entry].index_ww,
                    };
                    for (int field = 0; field < 4; field++) {
                        if (reject_noncanonical_array_value(
                                "entries", entry * 4 + field,
                                supplied[field], expected[field]) < 0) {
                            free_pab_table(t);
                            return -1;
                        }
                    }
                    entry++;
                }
            }
        }

        for (int d = 0; d < t->n_cvt + 1; d++) {
            if (reject_noncanonical_array_value(
                    "logdet_diag_rows", d, t->logdet_diag_rows[d], d) < 0 ||
                reject_noncanonical_array_value(
                    "logdet_diag_cols", d, t->logdet_diag_cols[d],
                    get_pab_index(d + 1, d + 1, t->n_cvt)) < 0) {
                free_pab_table(t);
                return -1;
            }
        }
    }

    return 0;
}

void free_pab_table(pab_table_t *t)
{
    free(t->invariant_indices);
    free(t->varying_indices);
    free(t->logdet_diag_rows);
    free(t->logdet_diag_cols);
    free(t->level_offsets);
    free(t->level_counts);
    free(t->entries);
    free(t->var_a_cols);
    free(t->var_b_cols);
    memset(t, 0, sizeof(*t));
}

PyArrayObject *take_array(PyObject *obj)
{
    return (PyArrayObject *)PyArray_FROM_OTF(
        obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
}

PyArrayObject *take_vector(PyObject *obj, int n, const char *name)
{
    PyArrayObject *arr = take_array(obj);
    if (!arr) return NULL;
    if (PyArray_NDIM(arr) != 1 || PyArray_DIM(arr, 0) != n) {
        PyErr_Format(PyExc_ValueError, "%s must be shape (%d,)", name, n);
        Py_DECREF(arr);
        return NULL;
    }
    return arr;
}

PyArrayObject *take_matrix(PyObject *obj, int rows, int cols, const char *name)
{
    PyArrayObject *arr = take_array(obj);
    if (!arr) return NULL;
    if (PyArray_NDIM(arr) != 2 || PyArray_DIM(arr, 0) != rows ||
        PyArray_DIM(arr, 1) != cols) {
        PyErr_Format(PyExc_ValueError, "%s must be shape (%d, %d)",
                     name, rows, cols);
        Py_DECREF(arr);
        return NULL;
    }
    return arr;
}

PyArrayObject *take_chunk(PyObject *obj, int n_samples, int *n_snps_out)
{
    PyArrayObject *arr = take_array(obj);
    if (!arr) return NULL;
    if (PyArray_NDIM(arr) != 2 || PyArray_DIM(arr, 1) != n_samples) {
        PyErr_Format(PyExc_ValueError,
                     "utg_t must be shape (n_snps, %d)", n_samples);
        Py_DECREF(arr);
        return NULL;
    }
    npy_intp n_snps = PyArray_DIM(arr, 0);
    if (n_snps > INT_MAX) {
        PyErr_Format(PyExc_OverflowError,
                     "n_snps (%" NPY_INTP_FMT ") exceeds INT_MAX", n_snps);
        Py_DECREF(arr);
        return NULL;
    }
    *n_snps_out = (int)n_snps;
    return arr;
}

int validate_n_cvt(int n_cvt)
{
    if (n_cvt < 1 || n_cvt > MAX_N_CVT) {
        PyErr_Format(PyExc_ValueError,
                     "n_cvt must be 1..%d, got %d", MAX_N_CVT, n_cvt);
        return -1;
    }
    return 0;
}

int validate_logl_H0(double logl_H0)
{
    if (!isfinite(logl_H0)) {
        PyErr_SetString(PyExc_ValueError,
            "logl_H0 must be finite (got NaN or Inf from null model)");
        return -1;
    }
    return 0;
}

int validate_hi_eval_null(const double *hi, int n_samples)
{
    for (int i = 0; i < n_samples; i++) {
        if (!isfinite(hi[i]) || hi[i] <= 0.0) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", hi[i]);
            PyErr_Format(PyExc_ValueError,
                "Hi_eval_null[%d] = %s is not finite positive. "
                "Null model optimization may have failed.", i, buf);
            return -1;
        }
    }
    return 0;
}

void build_grid_ncvt1(int n_grid, int n_samples, double log_l_min, double step,
                      const double *eigenvalues, const double *inv_ww,
                      const double *inv_wy, const double *inv_yy,
                      double *lambda_grid, double *hi_eval_grid,
                      double *logdet_h_grid, grid_invariant_t *grid_inv)
{
    for (int g = 0; g < n_grid; g++) {
        lambda_grid[g] = exp(log_l_min + g * step);
    }
    for (int g = 0; g < n_grid; g++) {
        double lam    = lambda_grid[g];
        double *hi_row = hi_eval_grid + (size_t)g * n_samples;
        double sw = 0.0, swy = 0.0, sy = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double h = 1.0 / (lam * eigenvalues[i] + 1.0);
            hi_row[i] = h;
            sw  += h * inv_ww[i];
            swy += h * inv_wy[i];
            sy  += h * inv_yy[i];
        }
        logdet_h_grid[g] = logdet_h_lambda(eigenvalues, n_samples, lam);

        grid_inv[g].s_ww    = sw;
        grid_inv[g].s_wy    = swy;
        grid_inv[g].s_yy    = sy;
        grid_inv[g].log_s_ww = (sw > 0.0) ? log(sw) : 0.0;
    }
}
