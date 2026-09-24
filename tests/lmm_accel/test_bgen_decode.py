"""``decode_bgen_probabilities_c`` against the NumPy BGEN oracle.

Drives the C entry point directly with probability-data buffers packed here,
so the fingerprint suite covers the decoder with deterministic inputs: the
outputs start zeroed and the recorder digests them after the call.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from tests.reference.bgen import decode_probabilities
from tests.support import requires_c

pytestmark = [pytest.mark.tier0, requires_c]

N = 37  # 2 * 37 values of an odd bit depth end mid-byte


def _pack(q11, q12, ploidy, bits: int, *, n_alleles: int = 2) -> bytes:
    """Encode one layout-2 probability block, values little-endian by bit."""
    values = np.column_stack([q11, q12]).ravel().astype(np.uint64)
    bitarr = ((values[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(
        np.uint8
    )
    data = np.packbits(bitarr.ravel(), bitorder="little").tobytes()
    head = struct.pack("<IHBB", len(q11), n_alleles, 2, 2)
    return head + bytes(ploidy) + bytes([0, bits]) + data


def _variant(rng: np.random.Generator, bits: int) -> bytes:
    mask = (1 << bits) - 1
    q11 = rng.integers(0, mask + 1, N)
    q12 = rng.integers(0, mask + 1 - q11)
    ploidy = np.where(rng.random(N) < 0.1, 0x82, 0x02).astype(np.uint8)
    q11[ploidy == 0x82] = 0
    q12[ploidy == 0x82] = 0
    return _pack(q11, q12, ploidy, bits)


def _outputs(k: int):
    return (
        np.zeros((N, k), dtype=np.float64, order="F"),
        np.zeros((N, k), dtype=np.uint16, order="F"),
        np.zeros((N, k), dtype=np.uint16, order="F"),
        np.zeros((N, k), dtype=np.bool_, order="F"),
        np.zeros(k, dtype=np.uint8),
    )


@pytest.mark.parametrize("zlib_input", [False, True])
@pytest.mark.parametrize("n_threads", [1, 3])
def test_decode_matches_oracle_at_every_bit_depth(zlib_input: bool, n_threads: int):
    from jamma.lmm import accel

    rng = np.random.default_rng(7)
    raw = [_variant(rng, bits) for bits in range(1, 17)]
    buffers = [zlib.compress(r) for r in raw] if zlib_input else raw
    lengths = np.array([len(r) for r in raw], dtype=np.int64) if zlib_input else None
    dosages, q11, q12, missing, bit_depth = _outputs(len(raw))

    failure = accel.require().decode_bgen_probabilities_c(
        buffers, lengths, N, dosages, q11, q12, missing, bit_depth, n_threads
    )

    assert failure is None
    for j, block in enumerate(raw):
        e_dosage, e_q11, e_q12, e_missing, e_bits = decode_probabilities(block, N)
        np.testing.assert_array_equal(
            dosages[:, j].view(np.uint64), e_dosage.view(np.uint64)
        )
        np.testing.assert_array_equal(q11[:, j], e_q11, strict=True)
        np.testing.assert_array_equal(q12[:, j], e_q12, strict=True)
        np.testing.assert_array_equal(missing[:, j], e_missing, strict=True)
        assert bit_depth[j] == e_bits


@pytest.mark.parametrize("zlib_input", [False, True])
@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("masked", [False, True])
def test_decode_info_sums_match_numpy(zlib_input: bool, n_threads: int, masked: bool):
    """E, E2, F, N over the non-missing kept rows, exact at every bit depth."""
    from jamma.lmm import accel

    rng = np.random.default_rng(13)
    raw = [_variant(rng, bits) for bits in range(1, 17)]
    buffers = [zlib.compress(r) for r in raw] if zlib_input else raw
    lengths = np.array([len(r) for r in raw], dtype=np.int64) if zlib_input else None
    keep = rng.random(N) < 0.6 if masked else None
    sums = np.zeros((len(raw), 4), dtype=np.int64)

    failure = accel.require().decode_bgen_probabilities_c(
        buffers,
        lengths,
        N,
        *_outputs(len(raw)),
        n_threads,
        info_rows=keep,
        info_sums=sums,
    )

    assert failure is None
    for j, block in enumerate(raw):
        _, q11, q12, missing, _ = decode_probabilities(block, N)
        w = ~missing if keep is None else ~missing & keep
        e = 2 * q11.astype(np.int64) + q12
        expected = [e @ w, (e * e) @ w, (e + 2 * q11.astype(np.int64)) @ w, w.sum()]
        np.testing.assert_array_equal(sums[j], expected, err_msg=f"variant {j}")


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (lambda b: b[:9], "shorter than its 10 + N byte header"),
        (lambda b: struct.pack("<I", N + 1) + b[4:], "sample count differs"),
        (lambda b: b[:4] + struct.pack("<H", 3) + b[6:], "not biallelic"),
        (lambda b: b[:8] + b"\x01" + b[9:], "ploidy is not 2"),
        (lambda b: b[: 8 + N] + b"\x01" + b[9 + N :], "phased data"),
        (lambda b: b[: 8 + N] + b"\x02" + b[9 + N :], "Phased flag"),
        (lambda b: b[: 9 + N] + b"\x00" + b[10 + N :], "outside 1..32"),
        (lambda b: b[: 9 + N] + b"\x11" + b[10 + N :], "above 16"),
        (lambda b: b + b"\x00", "length differs"),
    ],
)
def test_decode_reports_the_first_bad_variant(bad, reason: str):
    from jamma.lmm import accel

    rng = np.random.default_rng(11)
    raw = [_variant(rng, 8) for _ in range(4)]
    raw[2] = bad(raw[2])
    raw[3] = bad(raw[3])
    outputs = _outputs(len(raw))

    failure = accel.require().decode_bgen_probabilities_c(raw, None, N, *outputs, 2)

    assert failure is not None
    position, message = failure
    assert position == 2
    assert reason in message


def test_decode_rejects_probabilities_summing_above_one():
    from jamma.lmm import accel

    block = _pack(np.full(N, 200), np.full(N, 100), np.full(N, 2, np.uint8), 8)
    failure = accel.require().decode_bgen_probabilities_c(
        [block], None, N, *_outputs(1), 1
    )
    assert failure == (0, "stored P(11) + P(12) exceeds 1")


def test_decode_rejects_zlib_data_of_the_wrong_length():
    from jamma.lmm import accel

    raw = _variant(np.random.default_rng(1), 8)
    failure = accel.require().decode_bgen_probabilities_c(
        [zlib.compress(raw)],
        np.array([len(raw) + 1], dtype=np.int64),
        N,
        *_outputs(1),
        1,
    )
    assert failure == (0, "zlib data does not inflate to the declared length")


def test_decode_rejects_c_ordered_outputs():
    from jamma.lmm import accel

    dosages, q11, q12, missing, bit_depth = _outputs(2)
    with pytest.raises(ValueError, match="dosages must be a writeable Fortran"):
        accel.require().decode_bgen_probabilities_c(
            [b"", b""],
            None,
            N,
            np.ascontiguousarray(dosages),
            q11,
            q12,
            missing,
            bit_depth,
            1,
        )


@pytest.mark.parametrize(
    ("info_rows", "info_sums", "match"),
    [
        (np.ones(N - 1, dtype=bool), None, "info_rows must be None"),
        (np.ones(N, dtype=np.uint8), None, "info_rows must be None"),
        (None, np.zeros((2, 3), dtype=np.int64), "info_sums must be None"),
        (None, np.zeros((2, 4), dtype=np.int32), "info_sums must be None"),
        (None, np.zeros((4, 2), dtype=np.int64).T, "info_sums must be None"),
    ],
)
def test_decode_rejects_bad_info_arguments(info_rows, info_sums, match: str):
    from jamma.lmm import accel

    rng = np.random.default_rng(17)
    raw = [_variant(rng, 8) for _ in range(2)]
    with pytest.raises(ValueError, match=match):
        accel.require().decode_bgen_probabilities_c(
            raw, None, N, *_outputs(2), 1, info_rows=info_rows, info_sums=info_sums
        )
