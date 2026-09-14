// Fixed %.10g/tab formatting into a caller-owned buffer. File ownership and
// scheduling stay in Python; a call borrows both buffers until the GIL returns.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <system_error>

namespace {
constexpr Py_ssize_t bytes_per_value = 32;
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "matrix text formatting requires IEEE binary64");

struct Buffer {
    Py_buffer view{};
    Buffer() = default;
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    ~Buffer() {
        if (view.obj) PyBuffer_Release(&view);
    }
};

bool overlaps(const Py_buffer &input, const Py_buffer &output) noexcept {
    const auto in = reinterpret_cast<std::uintptr_t>(input.buf);
    const auto out = reinterpret_cast<std::uintptr_t>(output.buf);
    if (input.len == 0 || output.len == 0) return false;
    return out >= in ? out - in < static_cast<std::uintptr_t>(input.len)
                     : in - out < static_cast<std::uintptr_t>(output.len);
}

PyObject *format_into(PyObject *, PyObject *args) {
    PyObject *matrix, *destination;
    if (!PyArg_ParseTuple(args, "OO!:format_into", &matrix, &PyByteArray_Type, &destination))
        return nullptr;

    Buffer input, output;
    if (PyObject_GetBuffer(matrix, &input.view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0)
        return nullptr;
    const auto &src = input.view;
    if (src.ndim != 2 || !src.shape || src.shape[0] < 0 || src.shape[1] <= 0 ||
        src.itemsize != sizeof(double) || !src.format ||
        (std::strcmp(src.format, "d") != 0 && std::strcmp(src.format, "@d") != 0 &&
         std::strcmp(src.format, "=d") != 0)) {
        PyErr_SetString(PyExc_ValueError,
                        "matrix must be a C-contiguous 2D native float64 buffer with columns");
        return nullptr;
    }
    const Py_ssize_t count = src.len / sizeof(double);
    const Py_ssize_t columns = src.shape[1];
    if (src.len < 0 || src.len % sizeof(double) != 0 || count % columns != 0 ||
        count / columns != src.shape[0]) {
        PyErr_SetString(PyExc_ValueError, "matrix buffer length does not match its shape");
        return nullptr;
    }
    if (PyObject_GetBuffer(destination, &output.view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0)
        return nullptr;
    if (output.view.len < 0 || count > output.view.len / bytes_per_value) {
        PyErr_SetString(PyExc_ValueError, "output buffer requires 32 bytes per matrix value");
        return nullptr;
    }
    if (overlaps(src, output.view)) {
        PyErr_SetString(PyExc_ValueError, "matrix and output buffers must not overlap");
        return nullptr;
    }
    if (count == 0) return PyLong_FromLong(0);

    const auto *values = static_cast<const char *>(src.buf);
    auto *begin = static_cast<char *>(output.view.buf);
    auto *current = begin;
    bool failed = false;
    Py_BEGIN_ALLOW_THREADS for (Py_ssize_t row = 0; row < src.shape[0] && !failed; ++row) {
        for (Py_ssize_t column = 0; column < columns; ++column) {
            // Contiguous buffers need not be aligned. memcpy avoids undefined
            // behavior on an ndarray backed by a byte buffer at an odd offset.
            double value;
            std::memcpy(&value, values, sizeof(value));
            values += sizeof(value);
            if (std::isnan(value)) {
                // Python's %g discards the NaN sign; libc++ preserves it.
                std::memcpy(current, "nan", 3);
                current += 3;
            } else {
                auto result = std::to_chars(current, current + bytes_per_value - 1, value,
                                            std::chars_format::general, 10);
                if (result.ec != std::errc{}) {
                    failed = true;
                    break;
                }
                current = result.ptr;
            }
            *current++ = column + 1 == columns ? '\n' : '\t';
        }
    }
    Py_END_ALLOW_THREADS if (PyErr_CheckSignals() < 0) return nullptr;
    if (failed) {
        PyErr_SetString(PyExc_RuntimeError, "%.10g conversion exceeded its buffer bound");
        return nullptr;
    }
    return PyLong_FromSsize_t(current - begin);
}

PyMethodDef methods[] = {
    {"format_into", format_into, METH_VARARGS,
     "Format native float64 matrix into a disjoint bytearray; return bytes used.\n"
     "The caller must not mutate either buffer until this call returns."},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_matrix_text",
    "Bounded matrix text conversion.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};
} // namespace

PyMODINIT_FUNC PyInit__matrix_text() {
    PyObject *result = PyModule_Create(&module);
    if (!result) return nullptr;
    if (PyModule_AddIntConstant(result, "ABI_VERSION", 1) < 0 ||
        PyModule_AddIntConstant(result, "BYTES_PER_VALUE", bytes_per_value) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
