"""``decode_bgen_probabilities_c`` accepts exactly what the NumPy oracle accepts.

Hypothesis builds valid layout-2 blocks, then corrupts them the way a damaged
or hostile ``.bgen`` would: overwritten header fields and payload bytes,
truncation, trailing bytes, lying zlib lengths, and plain garbage. For every
batch the C decoder must reject at the first variant the oracle rejects, and
when the oracle accepts the whole batch the outputs must match it bit for bit.
The sanitizer workflow runs ``tests/lmm_accel`` natively under ASan and UBSan,
so an out-of-bounds read on any of these inputs aborts there even when the
returned values happen to agree.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from tests.reference.bgen import decode_probabilities
from tests.support import requires_c

pytestmark = [pytest.mark.tier0, requires_c]

MAX_SAMPLES = 40
MAX_VARIANTS = 6


@dataclass(frozen=True)
class Batch:
    n: int
    buffers: list[bytes]
    lengths: np.ndarray | None  # declared inflated lengths, zlib input only
    keep: np.ndarray | None
    n_threads: int


def _pack(q11, q12, ploidy, bits: int) -> bytes:
    values = np.column_stack([q11, q12]).ravel().astype(np.uint64)
    bitarr = ((values[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(
        np.uint8
    )
    data = np.packbits(bitarr.ravel(), bitorder="little").tobytes()
    head = struct.pack("<IHBB", len(q11), 2, 2, 2)
    return head + bytes(ploidy) + bytes([0, bits]) + data


@st.composite
def _well_formed_block(draw, n: int) -> bytes:
    """A block whose length matches its header, usually decodable.

    Occasionally it sits one step past a limit with everything else
    consistent, so only that limit's check can reject it: a bit depth of 17
    or 32, or one sample with P(11) + P(12) = 1 + 1/(2^B - 1).
    """
    bits = draw(st.sampled_from([*range(1, 17)] * 2 + [17, 32]))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    mask = (1 << bits) - 1
    q11 = rng.integers(0, mask + 1, n)
    q12 = rng.integers(0, mask + 1 - q11)
    missing = rng.random(n) < draw(st.sampled_from([0.0, 0.2, 1.0]))
    if draw(st.integers(0, 9)) == 0:
        q11[0], q12[0], missing[0] = mask, 1, False
    ploidy = np.where(missing, 0x82, 0x02).astype(np.uint8)
    return _pack(q11, q12, ploidy, bits)


@st.composite
def _one_field_off(draw, block: bytes, n: int) -> bytes:
    """Set one header field of ``block`` just outside what the decoder accepts."""
    offset, fmt, values = draw(
        st.sampled_from(
            [
                (0, "<I", [0, n - 1, n + 1, 2**32 - 1]),  # N
                (4, "<H", [0, 1, 3, 0xFFFF]),  # K
                # A ploidy byte's low six bits must be 2; 0x22 sets bit 5.
                (8 + draw(st.integers(0, n - 1)), "B", [0x01, 0x03, 0x22, 0x81]),
                (8 + n, "B", [1, 2, 0xFF]),  # Phased
                (9 + n, "B", [0, 33, 0xFF]),  # B; 17 and 32 come from blocks
            ]
        )
    )
    value = draw(st.sampled_from(values))
    out = bytearray(block)
    struct.pack_into(fmt, out, offset, value)
    if offset == 9 + n and value == 0:
        # B = 0 packs no data. Without the trim the size check rejects the
        # block, so a dropped B >= 1 check would go unseen.
        del out[10 + n :]
    return bytes(out)


@st.composite
def _corrupt(draw, block: bytes, n: int) -> bytes:
    """Apply one to three arbitrary mutations."""
    out = block
    for _ in range(draw(st.integers(1, 3))):
        kind = draw(
            st.sampled_from(["field", "byte", "overflow", "truncate", "append"])
        )
        if kind == "field" and len(out) >= 10 + n:
            out = draw(_one_field_off(out, n))
        elif kind == "byte" and out:
            pos = draw(st.integers(0, len(out) - 1))
            out = out[:pos] + bytes([draw(st.integers(0, 255))]) + out[pos + 1 :]
        elif kind == "overflow" and len(out) > 10 + n:
            # All-ones values: P(11) + P(12) > 1 for every non-missing sample.
            out = out[: 10 + n] + b"\xff" * (len(out) - 10 - n)
        elif kind == "truncate":
            out = out[: draw(st.integers(0, len(out)))]
        elif kind == "append":
            out += draw(st.binary(min_size=1, max_size=8))
    return out


@st.composite
def _batch(draw) -> Batch:
    n = draw(st.integers(1, MAX_SAMPLES))
    raw = []
    for _ in range(draw(st.integers(1, MAX_VARIANTS))):
        shape = draw(
            st.sampled_from(
                ["clean"] * 5 + ["field"] * 2 + ["corrupt"] * 2 + ["garbage"]
            )
        )
        if shape == "garbage":
            raw.append(draw(st.binary(max_size=2 * n + 24)))
            continue
        block = draw(_well_formed_block(n))
        if shape == "field":
            block = draw(_one_field_off(block, n))
        elif shape == "corrupt":
            block = draw(_corrupt(block, n))
        raw.append(block)

    lengths = None
    buffers = raw
    if draw(st.booleans()):
        buffers = [zlib.compress(r) for r in raw]
        declared = [len(r) for r in raw]
        for j in range(len(raw)):
            if draw(st.integers(0, 4)) == 0:
                declared[j] = draw(st.integers(0, len(raw[j]) + 4))
            if draw(st.integers(0, 4)) == 0:
                buffers[j] = draw(_corrupt(buffers[j], n))
        lengths = np.array(declared, dtype=np.int64)

    keep = None
    if draw(st.booleans()):
        keep = np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)))
    return Batch(n, buffers, lengths, keep, draw(st.integers(1, 4)))


def _oracle(batch: Batch, j: int):
    """The oracle's decode of variant ``j``, or None where it rejects it."""
    raw = batch.buffers[j]
    if batch.lengths is not None:
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            return None
        if len(raw) != batch.lengths[j]:
            return None
    try:
        return decode_probabilities(raw, batch.n)
    except (ValueError, IndexError, struct.error):
        return None


@settings(
    max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
@given(batch=_batch())
def test_decoder_accepts_exactly_what_the_oracle_accepts(batch: Batch):
    from jamma.lmm import accel

    n, k = batch.n, len(batch.buffers)
    dosages = np.zeros((n, k), dtype=np.float64, order="F")
    q11 = np.zeros((n, k), dtype=np.uint16, order="F")
    q12 = np.zeros((n, k), dtype=np.uint16, order="F")
    missing = np.zeros((n, k), dtype=np.bool_, order="F")
    bit_depth = np.zeros(k, dtype=np.uint8)
    sums = np.zeros((k, 4), dtype=np.int64)

    failure = accel.require().decode_bgen_probabilities_c(
        batch.buffers,
        batch.lengths,
        n,
        dosages,
        q11,
        q12,
        missing,
        bit_depth,
        batch.n_threads,
        info_rows=batch.keep,
        info_sums=sums,
    )

    expected = [_oracle(batch, j) for j in range(k)]
    rejected = [j for j, e in enumerate(expected) if e is None]
    event("accepted" if failure is None else failure[1])
    if rejected:
        assert failure is not None, f"oracle rejects variant {rejected[0]}"
        assert failure[0] == rejected[0]
        assert failure[1] != "unknown decode failure"
        return

    assert failure is None
    for j, decoded in enumerate(expected):
        assert decoded is not None
        e_dosage, e_q11, e_q12, e_missing, e_bits = decoded
        np.testing.assert_array_equal(
            dosages[:, j].view(np.uint64), e_dosage.view(np.uint64)
        )
        np.testing.assert_array_equal(q11[:, j], e_q11, strict=True)
        np.testing.assert_array_equal(q12[:, j], e_q12, strict=True)
        np.testing.assert_array_equal(missing[:, j], e_missing, strict=True)
        assert bit_depth[j] == e_bits
        w = ~e_missing if batch.keep is None else ~e_missing & batch.keep
        e = 2 * e_q11.astype(np.int64) + e_q12
        np.testing.assert_array_equal(
            sums[j], [e @ w, (e * e) @ w, (e + 2 * e_q11.astype(np.int64)) @ w, w.sum()]
        )
