/* BGEN v1.2 layout-2 probability decoding for jamma.io.bgen.BgenReader.
 *
 * One call decodes a batch of variants, each independent, so the batch runs
 * OpenMP-parallel across variants with the GIL released. Each input buffer is
 * one variant's probability data: raw (uncompressed, or zstd already inflated
 * in Python) or zlib-compressed, inflated here with the system zlib.
 *
 * The byte layout is the BGEN v1.2 spec's "Probability data storage" for
 * Layout 2: N (u32), K (u16), Pmin (u8), Pmax (u8), one ploidy/missingness
 * byte per sample, Phased (u8), B (u8), then 2 values per unphased diploid
 * biallelic sample, P(11) then P(12), each B bits, packed little-endian from
 * the least significant bit of the first byte. Value x means x / (2^B - 1).
 */

#define NO_IMPORT_ARRAY
#include "_lmm_accel_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <zlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef enum {
    BGEN_OK = 0,
    BGEN_INFLATE,
    BGEN_TRUNCATED,
    BGEN_N_MISMATCH,
    BGEN_NOT_BIALLELIC,
    BGEN_PLOIDY,
    BGEN_PHASED,
    BGEN_BAD_PHASED_FLAG,
    BGEN_BAD_BIT_DEPTH,
    BGEN_WIDE_BIT_DEPTH,
    BGEN_SIZE,
    BGEN_SUM,
} bgen_status_t;

static const char *bgen_status_message(bgen_status_t s)
{
    switch (s) {
    case BGEN_INFLATE:
        return "zlib data does not inflate to the declared length";
    case BGEN_TRUNCATED:
        return "probability data is shorter than its 10 + N byte header";
    case BGEN_N_MISMATCH:
        return "probability data sample count differs from the header's";
    case BGEN_NOT_BIALLELIC:
        return "probability data is not biallelic (K != 2)";
    case BGEN_PLOIDY:
        return "a sample's ploidy is not 2; only diploid data is supported";
    case BGEN_PHASED:
        return "phased data is not supported";
    case BGEN_BAD_PHASED_FLAG:
        return "Phased flag is neither 0 nor 1";
    case BGEN_BAD_BIT_DEPTH:
        return "bit depth B is outside 1..32";
    case BGEN_WIDE_BIT_DEPTH:
        return "bit depth B above 16 is not supported";
    case BGEN_SIZE:
        return "probability data length differs from 10 + N + ceil(2*N*B/8)";
    case BGEN_SUM:
        return "stored P(11) + P(12) exceeds 1";
    default:
        return "unknown decode failure";
    }
}

static inline uint32_t load_le32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16)
        | ((uint32_t)p[3] << 24);
}

static inline uint32_t load_le16(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
}

/* The B-bit value starting at bit `bit` of `data` (n_bytes long), B <= 16.
 * A value spans at most 3 bytes; bytes past the end read as zero. */
static inline uint32_t load_bits(const uint8_t *data, size_t n_bytes,
                                 size_t bit, uint32_t mask)
{
    size_t byte = bit >> 3;
    uint32_t w = data[byte];
    if (byte + 1 < n_bytes) w |= (uint32_t)data[byte + 1] << 8;
    if (byte + 2 < n_bytes) w |= (uint32_t)data[byte + 2] << 16;
    return (w >> (bit & 7)) & mask;
}

/* Decode one variant's uncompressed probability data into column outputs.
 *
 * Missing samples (ploidy byte bit 7) get dosage NaN, q11 = q12 = 0 and
 * missing = 1 whatever their stored values. Every sample, missing or not,
 * must have ploidy 2: the stored value count depends on ploidy, so a fixed
 * two-values-per-sample stride is only valid when every ploidy is 2. */
static bgen_status_t decode_variant(const uint8_t *raw, size_t len, uint32_t n,
                                    double *dosage, uint16_t *q11,
                                    uint16_t *q12, npy_bool *missing,
                                    uint8_t *bit_depth)
{
    if (len < (size_t)10 + n) return BGEN_TRUNCATED;
    if (load_le32(raw) != n) return BGEN_N_MISMATCH;
    if (load_le16(raw + 4) != 2) return BGEN_NOT_BIALLELIC;

    const uint8_t *ploidy = raw + 8;
    const uint8_t phased = raw[8 + (size_t)n];
    const uint8_t bits = raw[9 + (size_t)n];
    if (phased == 1) return BGEN_PHASED;
    if (phased != 0) return BGEN_BAD_PHASED_FLAG;
    if (bits < 1 || bits > 32) return BGEN_BAD_BIT_DEPTH;
    if (bits > 16) return BGEN_WIDE_BIT_DEPTH;

    for (uint32_t i = 0; i < n; i++) {
        if ((ploidy[i] & 0x3F) != 2) return BGEN_PLOIDY;
    }
    const size_t n_bytes = ((size_t)2 * n * bits + 7) / 8;
    if (len != (size_t)10 + n + n_bytes) return BGEN_SIZE;

    const uint8_t *data = raw + 10 + (size_t)n;
    const uint32_t mask = (1u << bits) - 1u;
    const double scale = (double)mask;
    *bit_depth = bits;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t a, b;
        if (bits == 8) {
            a = data[2 * (size_t)i];
            b = data[2 * (size_t)i + 1];
        } else if (bits == 16) {
            a = load_le16(data + 4 * (size_t)i);
            b = load_le16(data + 4 * (size_t)i + 2);
        } else {
            const size_t bit = (size_t)2 * i * bits;
            a = load_bits(data, n_bytes, bit, mask);
            b = load_bits(data, n_bytes, bit + bits, mask);
        }
        if (ploidy[i] & 0x80) {
            dosage[i] = NAN;
            q11[i] = 0;
            q12[i] = 0;
            missing[i] = 1;
            continue;
        }
        if (a + b > mask) return BGEN_SUM;
        q11[i] = (uint16_t)a;
        q12[i] = (uint16_t)b;
        missing[i] = 0;
        dosage[i] = (double)(2 * a + b) / scale;
    }
    return BGEN_OK;
}

/* A writeable, Fortran-contiguous array of `type` and exactly `ndim` dims
 * (n_samples, k) or (k,). 0, or -1 with PyErr set. */
static int check_out_array(PyArrayObject *a, const char *name, int type,
                           int ndim, npy_intp rows, npy_intp cols)
{
    if (PyArray_TYPE(a) != type || PyArray_NDIM(a) != ndim ||
        !PyArray_IS_F_CONTIGUOUS(a) || !PyArray_ISWRITEABLE(a) ||
        PyArray_DIM(a, 0) != rows || (ndim == 2 && PyArray_DIM(a, 1) != cols)) {
        PyErr_Format(PyExc_ValueError,
                     "%s must be a writeable Fortran-ordered array of the "
                     "documented dtype and shape", name);
        return -1;
    }
    return 0;
}

PyObject *decode_bgen_probabilities_c(PyObject *self, PyObject *args,
                                      PyObject *kwargs)
{
    static char *kwlist[] = {"buffers", "uncompressed_lengths", "n_samples",
                             "dosages", "q11", "q12", "missing", "bit_depth",
                             "n_threads", NULL};
    PyObject *buffers_obj, *lengths_obj;
    Py_ssize_t n_samples_arg;
    PyArrayObject *dosages, *q11, *q12, *missing, *bit_depth;
    int n_threads;
    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOnO!O!O!O!O!i", kwlist, &buffers_obj, &lengths_obj,
            &n_samples_arg, &PyArray_Type, &dosages, &PyArray_Type, &q11,
            &PyArray_Type, &q12, &PyArray_Type, &missing, &PyArray_Type,
            &bit_depth, &n_threads)) {
        return NULL;
    }
    if (n_samples_arg < 1 || (unsigned long long)n_samples_arg > UINT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "n_samples must be in 1..2**32-1");
        return NULL;
    }
    const uint32_t n = (uint32_t)n_samples_arg;

    PyObject *buffers = PySequence_Fast(buffers_obj, "buffers must be a sequence");
    if (buffers == NULL) return NULL;
    const Py_ssize_t k = PySequence_Fast_GET_SIZE(buffers);

    PyObject *result = NULL;
    Py_buffer *views = NULL;
    Py_ssize_t n_views = 0;
    uint8_t **scratch = NULL;
    int n_scratch = 0;
    bgen_status_t *status = NULL;
    PyArrayObject *lengths = NULL;

    if (check_out_array(dosages, "dosages", NPY_FLOAT64, 2, n, k) < 0 ||
        check_out_array(q11, "q11", NPY_UINT16, 2, n, k) < 0 ||
        check_out_array(q12, "q12", NPY_UINT16, 2, n, k) < 0 ||
        check_out_array(missing, "missing", NPY_BOOL, 2, n, k) < 0 ||
        check_out_array(bit_depth, "bit_depth", NPY_UINT8, 1, k, 0) < 0) {
        goto done;
    }

    /* None: every buffer is raw probability data. Otherwise an int64 (k,)
     * array of the zlib buffers' inflated lengths D. */
    const int zlib_input = lengths_obj != Py_None;
    if (zlib_input) {
        lengths = (PyArrayObject *)PyArray_FROMANY(
            lengths_obj, NPY_INT64, 1, 1, NPY_ARRAY_IN_ARRAY);
        if (lengths == NULL) goto done;
        if (PyArray_DIM(lengths, 0) != k) {
            PyErr_SetString(PyExc_ValueError,
                            "uncompressed_lengths must have one entry per buffer");
            goto done;
        }
    }
    const int64_t *inflated = zlib_input
        ? (const int64_t *)PyArray_DATA(lengths) : NULL;

    views = (Py_buffer *)calloc(k > 0 ? (size_t)k : 1, sizeof(Py_buffer));
    status = (bgen_status_t *)calloc(k > 0 ? (size_t)k : 1, sizeof(bgen_status_t));
    if (views == NULL || status == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    int64_t max_inflated = 0;
    for (Py_ssize_t j = 0; j < k; j++) {
        if (PyObject_GetBuffer(PySequence_Fast_GET_ITEM(buffers, j), &views[j],
                               PyBUF_SIMPLE) < 0) {
            goto done;
        }
        n_views++;
        if (zlib_input) {
            if (inflated[j] < 0 || (uint64_t)inflated[j] > UINT32_MAX) {
                PyErr_SetString(PyExc_ValueError,
                                "uncompressed_lengths must be in 0..2**32-1");
                goto done;
            }
            if (inflated[j] > max_inflated) max_inflated = inflated[j];
        }
    }

    int threads = n_threads < 1 ? 1 : n_threads;
    if ((Py_ssize_t)threads > k) threads = k > 0 ? (int)k : 1;
#ifndef _OPENMP
    threads = 1;
#endif
    if (zlib_input) {
        scratch = (uint8_t **)calloc((size_t)threads, sizeof(uint8_t *));
        if (scratch == NULL) {
            PyErr_NoMemory();
            goto done;
        }
        n_scratch = threads;
        for (int t = 0; t < threads; t++) {
            scratch[t] = (uint8_t *)malloc(max_inflated > 0 ? (size_t)max_inflated : 1);
            if (scratch[t] == NULL) {
                PyErr_NoMemory();
                goto done;
            }
        }
    }

    double *dos_base = (double *)PyArray_DATA(dosages);
    uint16_t *q11_base = (uint16_t *)PyArray_DATA(q11);
    uint16_t *q12_base = (uint16_t *)PyArray_DATA(q12);
    npy_bool *miss_base = (npy_bool *)PyArray_DATA(missing);
    uint8_t *bits_base = (uint8_t *)PyArray_DATA(bit_depth);

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (Py_ssize_t j = 0; j < k; j++) {
        const uint8_t *raw = (const uint8_t *)views[j].buf;
        size_t len = (size_t)views[j].len;
        if (zlib_input) {
#ifdef _OPENMP
            uint8_t *dst = scratch[omp_get_thread_num()];
#else
            uint8_t *dst = scratch[0];
#endif
            uLongf dst_len = (uLongf)inflated[j];
            if (uncompress(dst, &dst_len, raw, (uLong)len) != Z_OK ||
                dst_len != (uLongf)inflated[j]) {
                status[j] = BGEN_INFLATE;
                continue;
            }
            raw = dst;
            len = (size_t)dst_len;
        }
        const size_t col = (size_t)j * n;
        status[j] = decode_variant(raw, len, n, dos_base + col, q11_base + col,
                                   q12_base + col, miss_base + col,
                                   bits_base + j);
    }
    Py_END_ALLOW_THREADS

    /* The first failing variant in batch order, so the report is the same at
     * any thread count. */
    for (Py_ssize_t j = 0; j < k; j++) {
        if (status[j] != BGEN_OK) {
            result = Py_BuildValue("(ns)", j, bgen_status_message(status[j]));
            goto done;
        }
    }
    Py_INCREF(Py_None);
    result = Py_None;

done:
    if (scratch != NULL) {
        for (int t = 0; t < n_scratch; t++) free(scratch[t]);
        free(scratch);
    }
    for (Py_ssize_t j = 0; j < n_views; j++) PyBuffer_Release(&views[j]);
    free(views);
    free(status);
    Py_XDECREF(lengths);
    Py_DECREF(buffers);
    return result;
}
