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

double *alloc_aligned_doubles(size_t n)
{
    if (n == 0) return NULL;
    size_t raw = n * sizeof(double);
    if (raw / sizeof(double) != n) return NULL;  /* overflow check */
    size_t bytes = (raw + 31) & ~(size_t)31;
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

int validate_eigenvalues(const double *data, int n_samples)
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

int parse_pab_table_from_dict(PyObject *dict, pab_table_t *t, int n_samples)
{
    /* Read scalar integers from dict */
#define GETINT(key, field) do { \
    PyObject *v = PyDict_GetItemString(dict, key); \
    if (!v) { PyErr_Format(PyExc_KeyError, "pab_table_dict missing key '%s'", key); return -1; } \
    (field) = (int)PyLong_AsLong(v); \
    if (PyErr_Occurred()) { PyErr_Format(PyExc_TypeError, "pab_table_dict key '%s' must be int", key); return -1; } \
} while(0)

    GETINT("n_cvt",   t->n_cvt);
    GETINT("n_index", t->n_index);
    GETINT("n_rows",  t->n_rows);
    GETINT("n_inv",   t->n_inv);
    GETINT("n_var",   t->n_var);
    GETINT("idx_xx",  t->idx_xx);
    GETINT("idx_xy",  t->idx_xy);
    GETINT("idx_yy",  t->idx_yy);
#undef GETINT

    t->df = n_samples - t->n_cvt - 1;

    /* Validate basic integrity */
    if (t->n_cvt < 1 || t->n_cvt > MAX_N_CVT) {
        PyErr_Format(PyExc_ValueError, "n_cvt must be 1..%d, got %d", MAX_N_CVT, t->n_cvt);
        return -1;
    }
    if (t->n_rows < 1 || t->n_rows > MAX_N_ROWS) {
        PyErr_Format(PyExc_ValueError, "n_rows must be 1..%d, got %d", MAX_N_ROWS, t->n_rows);
        return -1;
    }
    if (t->n_inv + t->n_var != t->n_index) {
        PyErr_Format(PyExc_ValueError, "n_inv (%d) + n_var (%d) != n_index (%d)",
                     t->n_inv, t->n_var, t->n_index);
        return -1;
    }
    if (t->idx_xx < 0 || t->idx_xx >= t->n_index ||
        t->idx_xy < 0 || t->idx_xy >= t->n_index ||
        t->idx_yy < 0 || t->idx_yy >= t->n_index) {
        PyErr_Format(PyExc_ValueError,
            "idx_xx/xy/yy out of range [0, %d): got %d, %d, %d",
            t->n_index, t->idx_xx, t->idx_xy, t->idx_yy);
        return -1;
    }

    /* Initialise all pointer fields to NULL so free_pab_table is safe on partial init */
    t->invariant_indices = NULL;
    t->varying_indices   = NULL;
    t->logdet_diag_rows  = NULL;
    t->logdet_diag_cols  = NULL;
    t->level_offsets     = NULL;
    t->level_counts      = NULL;
    t->entries           = NULL;
    t->var_a_cols        = NULL;
    t->var_b_cols        = NULL;

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
        PyArrayObject *entries_arr = (PyArrayObject *)PyArray_FROM_OTF(
            entries_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
        if (!entries_arr) { free_pab_table(t); return -1; }
        int entries_len = (int)PyArray_SIZE(entries_arr);
        Py_DECREF(entries_arr);
        if (entries_len % 4 != 0) {
            PyErr_Format(PyExc_ValueError,
                "entries length (%d) not a multiple of 4", entries_len);
            free_pab_table(t);
            return -1;
        }
        t->n_entries = entries_len / 4;

        int *raw = parse_int32_array(entries_obj, entries_len, "entries");
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
