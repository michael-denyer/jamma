"""BGEN input through ``GenotypeDataset.open_bgen`` and ``BgenReader``.

The decoding oracle is ``tests/reference/bgen.py``, the NumPy prototype that
walks the ``.bgen`` sequentially and never reads the ``.bgi``. Files are
written at test time by ``bgen.BgenWriter``.
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from bed_reader import open_bed
from loguru import logger

from jamma.genotype.dataset import GenotypeDataset, GenotypeEncoding
from jamma.io.bgen import BgenFormatError, open_bgen_reader
from tests.bgen_files import (
    BgenFiles,
    one_hot_bgen_from_plink,
    random_probabilities,
    write_bgen,
    write_sample,
)
from tests.reference.bgen import decode_file
from tests.support import requires_c

pytestmark = [pytest.mark.tier0, requires_c]

N_SAMPLES = 203  # odd, so packed B-bit rows end mid-byte
N_VARIANTS = 37


def _open(files: BgenFiles, **kwargs) -> GenotypeDataset:
    return GenotypeDataset.open_bgen(files.bgen, files.sample, files.bgi, **kwargs)


def _random_bgen(
    tmp_path: Path, bit_depth: int, compression: str | None, seed: int = 0
) -> BgenFiles:
    rng = np.random.default_rng(seed)
    probs = random_probabilities(rng, N_VARIANTS, N_SAMPLES, missing_rate=0.05)
    return write_bgen(
        tmp_path / f"b{bit_depth}_{compression}.bgen",
        probs,
        bit_depth=bit_depth,
        compression=compression,
    )


def _column_cases(n: int) -> dict[str, np.ndarray | None]:
    rng = np.random.default_rng(3)
    return {
        "unfiltered": None,
        "contiguous": np.arange(4, n - 5, dtype=np.intp),
        "scattered": np.sort(rng.choice(n, size=n // 3, replace=False)).astype(np.intp),
    }


def _sql(bgi: Path, statement: str) -> None:
    con = sqlite3.connect(bgi, isolation_level=None)
    try:
        con.execute(statement)
    finally:
        con.close()


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).view(np.uint64)


@pytest.mark.parametrize("compression", [None, "zlib", "zstd"])
@pytest.mark.parametrize("bit_depth", [1, 3, 8, 10, 16])
def test_decoder_equals_reference_exactly(
    tmp_path: Path, bit_depth: int, compression: str | None
):
    """C dosages are bit-identical, q11/q12/missing identical, to the oracle.

    Swept over block sizes that do and do not divide the variant count,
    contiguous and scattered column selections, and 1 or 4 decode threads.
    """
    files = _random_bgen(tmp_path, bit_depth, compression)
    ref = decode_file(files.bgen)
    assert ref.missing.any()
    assert not ref.missing.all()
    assert np.all(ref.bit_depth == bit_depth)

    for threads in (1, 4):
        reader, *_ = open_bgen_reader(
            files.bgen, files.sample, files.bgi, n_threads=threads
        )
        for name, columns in _column_cases(N_VARIANTS).items():
            cols = np.arange(N_VARIANTS) if columns is None else columns
            for block_size in (1, 7, N_VARIANTS, 64):
                blocks = list(reader.read(cols, block_size, stats_only=False))
                got = {
                    field: np.concatenate([getattr(b, field) for b in blocks], axis=-1)
                    for field in ("dosages", "q11", "q12", "missing", "bit_depth")
                }
                label = f"threads={threads} {name} block_size={block_size}"
                np.testing.assert_array_equal(
                    _bits(got["dosages"]), _bits(ref.dosages[:, cols]), err_msg=label
                )
                for field in ("q11", "q12", "missing"):
                    expected = getattr(ref, field)[:, cols]
                    assert got[field].dtype == expected.dtype, label
                    np.testing.assert_array_equal(
                        got[field], expected, err_msg=label, strict=True
                    )
                np.testing.assert_array_equal(
                    got["bit_depth"], ref.bit_depth[cols], err_msg=label
                )


def test_dataset_blocks_are_the_decoded_dosages(tmp_path: Path):
    """open_bgen's blocks carry the decoder's float64 dosages, NaN for missing."""
    files = _random_bgen(tmp_path, 8, "zlib", seed=4)
    ref = decode_file(files.bgen)
    dataset = _open(files)

    assert dataset.encoding is GenotypeEncoding.PROBABILITIES
    values = np.concatenate([b.dosages() for b in dataset.blocks(10)], axis=1)
    assert values.dtype == np.float64
    np.testing.assert_array_equal(_bits(values), _bits(ref.dosages))


def test_identity_counts_first_allele(tmp_path: Path):
    """Variants come from the .bgi in file order; a1 is the first allele."""
    probs = random_probabilities(np.random.default_rng(1), 3, 5, 0.0)
    files = write_bgen(
        tmp_path / "id.bgen",
        probs,
        alleles=[("C", "T"), ("GA", "G"), ("A", "C")],
        chromosomes=["2", "2", "X"],
        positions=[10, 20, 5],
        rsids=["rsA", "", "rsC"],
        varids=["vA", "chr2:20", "vC"],
        fid=["F1", "F2", "F3", "F4", "F5"],
        iid=["I1", "I2", "I3", "I4", "I5"],
    )
    dataset = _open(files)

    v = dataset.variants
    np.testing.assert_array_equal(v.chr, ["2", "2", "X"])
    np.testing.assert_array_equal(v.rs, ["rsA", "chr2:20", "rsC"])
    np.testing.assert_array_equal(v.pos, [10, 20, 5])
    np.testing.assert_array_equal(v.a1, ["C", "GA", "A"])
    np.testing.assert_array_equal(v.a0, ["T", "G", "C"])
    np.testing.assert_array_equal(dataset.samples.fid, ["F1", "F2", "F3", "F4", "F5"])
    np.testing.assert_array_equal(dataset.samples.iid, ["I1", "I2", "I3", "I4", "I5"])
    assert list(dataset.partitions) == ["2", "X"]


@pytest.mark.parametrize("bit_depth", [1, 8, 16])
def test_one_hot_bgen_equals_plink_dosages(
    tmp_path: Path, asymmetric_plink: Path, bit_depth: int
):
    """Probabilities of exactly 0/1 give the .bed's dosages byte for byte."""
    files = one_hot_bgen_from_plink(
        asymmetric_plink, tmp_path / "onehot.bgen", bit_depth=bit_depth
    )
    plink = GenotypeDataset.open_plink(asymmetric_plink)
    bgen = _open(files)

    (plink_block,) = plink.blocks(plink.n_variants)
    (bgen_block,) = bgen.blocks(bgen.n_variants)
    expected = plink_block.dosages()
    assert np.isnan(expected).any()
    np.testing.assert_array_equal(_bits(bgen_block.dosages()), _bits(expected))
    for field in ("chr", "rs", "pos", "a1", "a0"):
        np.testing.assert_array_equal(
            getattr(bgen.variants, field), getattr(plink.variants, field)
        )
    np.testing.assert_array_equal(bgen.samples.iid, plink.samples.iid)


def test_fingerprint_components(tmp_path: Path):
    files = _random_bgen(tmp_path, 8, "zlib")
    dataset = _open(files)
    st = files.bgen.stat()

    fp = dataset.fingerprint()

    assert set(fp) == {"bgen_fingerprint", "sample_sha256", "variants_sha256"}
    assert fp["bgen_fingerprint"] == f"{files.bgen.name}:{st.st_size}:{st.st_mtime_ns}"
    assert fp["sample_sha256"] == hashlib.sha256(files.sample.read_bytes()).hexdigest()


def test_variants_sha256_tracks_content_not_index_bytes(tmp_path: Path):
    """Rewriting the .bgi with the same rows keeps the hash; a new row set moves it."""
    files = _random_bgen(tmp_path, 8, "zlib")
    before = _open(files).fingerprint()["variants_sha256"]

    _sql(files.bgi, "VACUUM")
    assert _open(files).fingerprint()["variants_sha256"] == before

    _sql(files.bgi, "UPDATE Variant SET rsid = 'renamed' WHERE rsid = 'rs3'")
    assert _open(files).fingerprint()["variants_sha256"] != before


def test_materialize_rejects_bgen(tmp_path: Path):
    dataset = _open(_random_bgen(tmp_path, 8, "zlib"))
    with pytest.raises(ValueError, match="only hard-call datasets"):
        dataset.materialize()


def test_stats_refuse_hwe_and_report_no_unexpected_values(tmp_path: Path):
    dataset = _open(_random_bgen(tmp_path, 8, "zlib"))
    with pytest.raises(ValueError, match="HWE"):
        dataset.stats(None, hwe=True)
    assert dataset.stats(None).n_unexpected == 0


# --------------------------------------------------------------------------
# Rejections at open
# --------------------------------------------------------------------------


def _probs(n_variants: int = 2, n_samples: int = 6) -> np.ndarray:
    return random_probabilities(np.random.default_rng(2), n_variants, n_samples, 0.0)


def test_rejects_layout_1(tmp_path: Path):
    from bgen import BgenWriter

    path = tmp_path / "l1.bgen"
    with BgenWriter(path, 6, compression="zlib", layout=1) as writer:
        writer.add_variant("v", "rs", "1", 1, ["A", "G"], _probs(1)[0])
    write_sample(tmp_path / "l1.sample", list("abcdef"), list("abcdef"))

    with pytest.raises(BgenFormatError, match="layout 1"):
        GenotypeDataset.open_bgen(path, tmp_path / "l1.sample", Path(f"{path}.bgi"))


def test_rejects_multiallelic_variant(tmp_path: Path):
    from bgen import BgenWriter

    path = tmp_path / "multi.bgen"
    with BgenWriter(path, 6, compression="zlib") as writer:
        writer.add_variant("v0", "rs0", "1", 1, ["A", "G"], _probs(1)[0])
        writer.add_variant(
            "v1", "rsTri", "1", 2, ["A", "G", "T"], np.full((6, 6), 1 / 6)
        )
    write_sample(tmp_path / "multi.sample", list("abcdef"), list("abcdef"))

    with pytest.raises(BgenFormatError, match=r"rsTri.*3 alleles"):
        GenotypeDataset.open_bgen(path, tmp_path / "multi.sample", Path(f"{path}.bgi"))


def test_rejects_sample_count_mismatch(tmp_path: Path):
    files = write_bgen(tmp_path / "n.bgen", _probs(), embed_ids=False)
    write_sample(files.sample, list("abcde"), list("abcde"))

    with pytest.raises(BgenFormatError, match=r"5 samples.*header has 6"):
        _open(files)


def test_rejects_variant_count_mismatch(tmp_path: Path):
    files = write_bgen(tmp_path / "m.bgen", _probs(3))
    _sql(files.bgi, "DELETE FROM Variant WHERE rsid = 'rs1'")

    with pytest.raises(BgenFormatError, match=r"indexes 2 variants.*header has 3"):
        _open(files)


def test_rejects_sample_ids_differing_from_embedded(tmp_path: Path):
    files = write_bgen(tmp_path / "ids.bgen", _probs())
    write_sample(files.sample, list("abcdef"), ["s0", "s1", "s2", "sX", "s4", "s5"])

    with pytest.raises(BgenFormatError, match=r"sample 4: .*'s3'.*'sX'"):
        _open(files)


def test_rejects_malformed_sample_header(tmp_path: Path):
    files = write_bgen(tmp_path / "hdr.bgen", _probs())
    files.sample.write_text("ID FID\n0 0\n" + "a b\n" * 6)

    with pytest.raises(BgenFormatError, match="ID_1 ID_2"):
        _open(files)


def test_zstd_without_module_fails_at_open(tmp_path: Path, monkeypatch):
    """A missing zstd module is an environment state, emulated in sys.modules."""
    files = write_bgen(tmp_path / "z.bgen", _probs(), compression="zstd")
    monkeypatch.setitem(sys.modules, "compression.zstd", None)
    monkeypatch.setitem(sys.modules, "backports.zstd", None)

    with pytest.raises(ImportError, match=r"jamma\[zstd\]"):
        _open(files)


def test_open_without_c_extension_raises(tmp_path: Path, no_c_kernels):
    files = write_bgen(tmp_path / "noc.bgen", _probs())
    with pytest.raises(RuntimeError, match="_lmm_accel C extension"):
        _open(files)


def test_missing_file_raises(tmp_path: Path):
    files = write_bgen(tmp_path / "gone.bgen", _probs())
    files.bgi.unlink()
    with pytest.raises(FileNotFoundError, match=r"\.bgi"):
        _open(files)


# --------------------------------------------------------------------------
# Rejections at decode: the per-variant probability layout
# --------------------------------------------------------------------------


def _first_block(dataset: GenotypeDataset) -> None:
    next(dataset.blocks(dataset.n_variants)).dosages()


def test_rejects_phased_data(tmp_path: Path):
    from bgen import BgenWriter

    path = tmp_path / "phased.bgen"
    with BgenWriter(path, 6, compression="zlib") as writer:
        writer.add_variant(
            "v", "rsP", "1", 1, ["A", "G"], np.full((6, 4), 0.5), phased=True
        )
    write_sample(tmp_path / "phased.sample", list("abcdef"), list("abcdef"))
    dataset = GenotypeDataset.open_bgen(
        path, tmp_path / "phased.sample", Path(f"{path}.bgi")
    )

    with pytest.raises(BgenFormatError, match=r"rsP.*phased data is not supported"):
        _first_block(dataset)


def test_rejects_non_diploid_sample(tmp_path: Path):
    from bgen import BgenWriter

    probs = np.full((6, 3), [0.2, 0.3, 0.5])
    probs[5] = [0.4, 0.6, np.nan]
    path = tmp_path / "ploidy.bgen"
    with BgenWriter(path, 6, compression=None) as writer:
        writer.add_variant(
            "v", "rsH", "1", 1, ["A", "G"], probs, ploidy=np.array([2, 2, 2, 2, 2, 1])
        )
    write_sample(tmp_path / "ploidy.sample", list("abcdef"), list("abcdef"))
    dataset = GenotypeDataset.open_bgen(
        path, tmp_path / "ploidy.sample", Path(f"{path}.bgi")
    )

    with pytest.raises(BgenFormatError, match=r"rsH.*ploidy is not 2"):
        _first_block(dataset)


@pytest.mark.parametrize("bit_depth", [17, 24, 32])
def test_rejects_bit_depth_above_16(tmp_path: Path, bit_depth: int):
    files = write_bgen(tmp_path / "wide.bgen", _probs(), bit_depth=bit_depth)
    dataset = _open(files)

    with pytest.raises(BgenFormatError, match=r"rs0.*bit depth B above 16"):
        _first_block(dataset)


def test_rejects_stale_index(tmp_path: Path):
    """A .bgi whose position disagrees with the variant block fails, naming it."""
    files = write_bgen(tmp_path / "stale.bgen", _probs())
    _sql(files.bgi, "UPDATE Variant SET position = 99 WHERE rsid = 'rs1'")
    dataset = _open(files)

    with pytest.raises(BgenFormatError, match=r"rs1.*index is stale"):
        _first_block(dataset)


def test_rejects_corrupt_zlib_data(tmp_path: Path):
    files = write_bgen(tmp_path / "bad.bgen", _probs(1))
    data = bytearray(files.bgen.read_bytes())
    data[-5:] = b"\xff" * 5  # the tail of the only variant's zlib stream
    files.bgen.write_bytes(bytes(data))
    dataset = _open(files)

    with pytest.raises(BgenFormatError, match=r"rs0.*zlib"):
        _first_block(dataset)


def test_header_rejects_bad_magic(tmp_path: Path):
    files = write_bgen(tmp_path / "magic.bgen", _probs())
    data = bytearray(files.bgen.read_bytes())
    data[16:20] = b"nope"
    files.bgen.write_bytes(bytes(data))

    with pytest.raises(BgenFormatError, match="magic"):
        _open(files)


def test_header_layout_flag_is_bits_2_to_5(tmp_path: Path):
    """Setting the layout field to 3 is rejected with the value read."""
    files = write_bgen(tmp_path / "flags.bgen", _probs())
    data = bytearray(files.bgen.read_bytes())
    (header_len,) = struct.unpack_from("<I", data, 4)
    (flags,) = struct.unpack_from("<I", data, header_len)
    struct.pack_into("<I", data, header_len, (flags & ~(0xF << 2)) | (3 << 2))
    files.bgen.write_bytes(bytes(data))

    with pytest.raises(BgenFormatError, match="layout 3"):
        _open(files)


def test_open_logs_projected_decode(tmp_path: Path):
    files = _random_bgen(tmp_path, 8, "zlib")
    messages: list[str] = []
    sink = logger.add(messages.append, level="INFO", format="{message}")
    try:
        _open(files)
    finally:
        logger.remove(sink)
    assert any(f"{N_VARIANTS} variants x {N_SAMPLES} samples" in m for m in messages)


def test_open_bgen_reads_plink_iids_order(tmp_path: Path, asymmetric_plink: Path):
    """Row i of every block is .sample row i."""
    files = one_hot_bgen_from_plink(asymmetric_plink, tmp_path / "order.bgen")
    with open_bed(asymmetric_plink.with_suffix(".bed")) as bed:
        np.testing.assert_array_equal(_open(files).samples.fid, bed.fid)


@pytest.mark.parametrize(("bit_depth", "compression"), [(3, "zstd"), (16, "zlib")])
def test_reference_decoder_agrees_with_bgen_package(
    tmp_path: Path, bit_depth: int, compression: str
):
    """Guard the oracle: its probabilities match ``bgen``'s float32 ones."""
    from bgen import BgenReader as PackageReader

    files = _random_bgen(tmp_path, bit_depth, compression)
    ref = decode_file(files.bgen)
    with PackageReader(files.bgen, delay_parsing=True) as rd:
        theirs = np.stack([v.probabilities for v in rd], axis=1)  # (n, m, 3)
    mask = float(2**bit_depth - 1)
    p11 = np.where(ref.missing, np.nan, ref.q11 / mask)
    p12 = np.where(ref.missing, np.nan, ref.q12 / mask)
    np.testing.assert_allclose(p11, theirs[:, :, 0], rtol=0, atol=1e-7)
    np.testing.assert_allclose(p12, theirs[:, :, 1], rtol=0, atol=1e-7)


def test_rejects_index_size_shorter_than_the_header(tmp_path: Path):
    files = write_bgen(tmp_path / "short.bgen", _probs())
    _sql(files.bgi, "UPDATE Variant SET size_in_bytes = 3 WHERE rsid = 'rs0'")
    dataset = _open(files)

    with pytest.raises(BgenFormatError, match=r"rs0.*header overruns"):
        _first_block(dataset)
