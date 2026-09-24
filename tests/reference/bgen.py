"""NumPy BGEN v1.2 layout-2 oracle for the C decoder.

A copy of the scratch prototype decoder the design was measured with, kept
literal to the spec and deliberately independent of production: it walks the
``.bgen`` front to back from the header offset, never reading the ``.bgi``,
and unpacks B-bit values with ``np.unpackbits`` rather than byte arithmetic.
It returns the arrays ``ProbabilityBlock`` holds, for every variant in file
order, and raises ``ValueError`` wherever the C decoder reports a failure.
"""

from __future__ import annotations

import importlib
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DecodedBgen:
    """Every variant of a file; 2-D arrays are ``(n_samples, n_variants)``."""

    dosages: np.ndarray
    q11: np.ndarray
    q12: np.ndarray
    missing: np.ndarray
    bit_depth: np.ndarray


def _zstd():
    try:
        return importlib.import_module("compression.zstd")
    except ImportError:
        return importlib.import_module("backports.zstd")


def decode_probabilities(raw: bytes, n_expected: int):
    """Decode one uncompressed probability-data block.

    Returns:
        ``(dosage, q11, q12, missing, bits)`` for one variant.
    """
    n, k = struct.unpack_from("<IH", raw, 0)
    if n != n_expected:
        raise ValueError("sample count differs from the header's")
    if k != 2:
        raise ValueError("not biallelic")
    ploidy = np.frombuffer(raw, np.uint8, n, 8)
    phased, bits = raw[8 + n], raw[9 + n]
    if phased != 0:
        raise ValueError("phased or bad Phased flag")
    if not 1 <= bits <= 16:
        raise ValueError(f"bit depth {bits} unsupported")
    if np.any((ploidy & 0x3F) != 2):
        raise ValueError("ploidy is not 2")
    data = np.frombuffer(raw, np.uint8, offset=10 + n)
    if len(data) != -(-2 * n * bits // 8):
        raise ValueError("probability data length")
    missing = (ploidy & 0x80) != 0
    # Little-endian bit order: value i occupies bits [i*B, (i+1)*B).
    bitarr = np.unpackbits(data, bitorder="little")[: 2 * n * bits]
    weights = 1 << np.arange(bits, dtype=np.uint64)
    q = (bitarr.reshape(2 * n, bits).astype(np.uint64) @ weights).reshape(n, 2)
    q[missing] = 0
    mask = (1 << bits) - 1
    if np.any(q.sum(axis=1) > mask):
        raise ValueError("P(11) + P(12) exceeds 1")
    q11, q12 = q[:, 0], q[:, 1]
    dosage = (2 * q11 + q12).astype(np.float64) / float(mask)
    dosage[missing] = np.nan
    return dosage, q11.astype(np.uint16), q12.astype(np.uint16), missing, bits


def decode_file(path: Path) -> DecodedBgen:
    """Decode every variant of ``path`` sequentially from the header offset."""
    columns = []
    with open(path, "rb") as f:
        offset, header_len, m, n = struct.unpack("<4I", f.read(16))
        f.seek(header_len)
        (flags,) = struct.unpack("<I", f.read(4))
        compression, layout = flags & 3, (flags >> 2) & 0xF
        assert layout == 2
        f.seek(4 + offset)
        for _ in range(m):
            for _ in range(3):  # variant id, rsid, chromosome
                (length,) = struct.unpack("<H", f.read(2))
                f.read(length)
            _pos, k = struct.unpack("<IH", f.read(6))
            for _ in range(k):
                (length,) = struct.unpack("<I", f.read(4))
                f.read(length)
            (c,) = struct.unpack("<I", f.read(4))
            if compression == 0:
                raw = f.read(c)
            else:
                (d,) = struct.unpack("<I", f.read(4))
                blob = f.read(c - 4)
                if compression == 1:
                    raw = zlib.decompress(blob)
                else:
                    raw = _zstd().decompress(blob)
                assert len(raw) == d
            columns.append(decode_probabilities(raw, n))
    if not columns:
        empty = np.empty((n, 0))
        return DecodedBgen(
            empty,
            empty.astype(np.uint16),
            empty.astype(np.uint16),
            empty.astype(bool),
            np.empty(0, np.uint8),
        )
    dosage, q11, q12, missing, bits = zip(*columns, strict=True)
    return DecodedBgen(
        dosages=np.column_stack(dosage),
        q11=np.column_stack(q11),
        q12=np.column_stack(q12),
        missing=np.column_stack(missing),
        bit_depth=np.array(bits, dtype=np.uint8),
    )
