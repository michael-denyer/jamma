"""BGEN v1.2 input: the header, ``.bgi`` and ``.sample`` parsers and the reader.

``BgenReader`` is the ``.bgen`` strategy behind ``GenotypeDataset.open_bgen``.
Only layout 2, biallelic, unphased diploid data with a bit depth of 1 to 16
is read. Variant metadata and block offsets come from the bgenix ``.bgi``
index, so opening never scans the ``.bgen``; the probability data is decoded
by ``decode_bgen_probabilities_c`` in the ``_lmm_accel`` C extension. There
is no NumPy production decoder.

The counted allele is the FIRST allele: dosage = 2*P(11) + P(12), as GCTA
computes it and as GEMMA's BIMBAM column 2 counts.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sqlite3
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np
from loguru import logger

from jamma.genotype.variants import SnpMeta

# Variants decoded per C call. Bounds the transient compressed (and, for
# zstd, inflated) bytes held beside a block to this many variants.
_DECODE_BATCH = 256


def bgen_read_workspace_bytes(n_samples: int, n_variants: int) -> int:
    """Bytes beyond float64 dosages while decoding one block.

    Two uint16 probability arrays and a bool missing mask use five bytes
    per cell. Reserve two layout-2 payloads (10 + 9N bytes each) per batched
    variant for compressed and inflated data. Per-variant bit depths and
    four int64 INFO sums add 33 bytes.
    """
    return (5 * n_samples + 33) * n_variants + 2 * (10 + 9 * n_samples) * min(
        n_variants, _DECODE_BATCH
    )


_COMPRESSION = {0: "none", 1: "zlib", 2: "zstd"}


class BgenFormatError(ValueError):
    """A BGEN, ``.bgi`` or ``.sample`` file this reader cannot or will not read."""


class BgenDependencyError(ImportError):
    """This installation lacks a module a BGEN file needs: zstd or ``_lmm_accel``."""


def _zstd_module() -> ModuleType | None:
    """``compression.zstd`` (3.14+), else ``backports.zstd``, else None."""
    for name in ("compression.zstd", "backports.zstd"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def _require_zstd() -> Callable[[bytes], bytes]:
    module = _zstd_module()
    if module is None:
        raise BgenDependencyError(
            "this BGEN file is zstd-compressed, which needs the zstd module: "
            "install jamma[zstd] (backports.zstd) on Python < 3.14"
        )
    return module.decompress


@dataclass(frozen=True, slots=True)
class BgenHeader:
    """The ``.bgen`` header block.

    Attributes:
        variant_start: File offset of the first variant data block.
        n_variants: M, the header's variant count.
        n_samples: N, the header's sample count.
        compression: 0 none, 1 zlib, 2 zstd.
        layout: Layout flag; only 2 is accepted.
        sample_ids: Embedded sample identifiers, or None when absent.
    """

    variant_start: int
    n_variants: int
    n_samples: int
    compression: int
    layout: int
    sample_ids: np.ndarray | None


def read_bgen_header(path: Path) -> BgenHeader:
    """Parse the header and optional sample identifier block of ``path``.

    Raises:
        BgenFormatError: On a bad magic number, a layout other than 2, a
            reserved compression value, or an inconsistent sample block.
    """
    with open(path, "rb") as fh:
        fixed = fh.read(20)
        if len(fixed) < 20:
            raise BgenFormatError(f"{path}: too short for a BGEN header")
        offset, header_len, n_variants, n_samples = struct.unpack_from("<4I", fixed)
        magic = fixed[16:20]
        if magic not in (b"bgen", b"\x00\x00\x00\x00"):
            raise BgenFormatError(f"{path}: magic number {magic!r} is not 'bgen'")
        if header_len < 20 or header_len > offset:
            raise BgenFormatError(
                f"{path}: header length {header_len} is outside 20..{offset}"
            )
        fh.seek(header_len)
        (flags,) = struct.unpack("<I", fh.read(4))
        compression = flags & 0x3
        layout = (flags >> 2) & 0xF
        if layout != 2:
            raise BgenFormatError(
                f"{path}: layout {layout} is not supported; only BGEN v1.2 "
                "layout 2 files can be read"
            )
        if compression not in _COMPRESSION:
            raise BgenFormatError(f"{path}: reserved compression value {compression}")
        sample_ids = None
        if flags >> 31:
            sample_ids = _read_sample_block(fh, path, header_len, offset, n_samples)
    return BgenHeader(
        variant_start=offset + 4,
        n_variants=n_variants,
        n_samples=n_samples,
        compression=compression,
        layout=layout,
        sample_ids=sample_ids,
    )


def _read_sample_block(
    fh, path: Path, header_len: int, offset: int, n_samples: int
) -> np.ndarray:
    fh.seek(4 + header_len)
    block_len, n_block = struct.unpack("<2I", fh.read(8))
    if block_len + header_len > offset:
        raise BgenFormatError(f"{path}: sample identifier block overruns the data")
    if n_block != n_samples:
        raise BgenFormatError(
            f"{path}: sample identifier block holds {n_block} samples, "
            f"header says {n_samples}"
        )
    data = fh.read(block_len - 8)
    ids, pos = [], 0
    for _ in range(n_samples):
        (length,) = struct.unpack_from("<H", data, pos)
        ids.append(data[pos + 2 : pos + 2 + length].decode())
        pos += 2 + length
    return np.array(ids, dtype=str)


def read_sample_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``ID_1`` and ``ID_2`` of an Oxford ``.sample`` file.

    The first line names the columns and must start ``ID_1 ID_2``; the second
    gives column types and must start ``0 0``. Other columns are ignored.

    Returns:
        ``(id_1, id_2)`` string arrays in file order.

    Raises:
        BgenFormatError: On a missing header or type line, or a short row.
    """
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2 or lines[0][:2] != ["ID_1", "ID_2"]:
        raise BgenFormatError(f"{path}: first line must start with 'ID_1 ID_2'")
    if lines[1][:2] != ["0", "0"]:
        raise BgenFormatError(
            f"{path}: second line must be the column-type line starting '0 0'"
        )
    rows = lines[2:]
    for i, row in enumerate(rows):
        if len(row) < 2:
            raise BgenFormatError(f"{path}: sample row {i + 1} has fewer than 2 fields")
    return (
        np.array([row[0] for row in rows], dtype=str),
        np.array([row[1] for row in rows], dtype=str),
    )


@dataclass(frozen=True, slots=True)
class BgenIndex:
    """The ``.bgi`` ``Variant`` table in file order.

    Attributes:
        variants: chr/rs/pos/a1/a0 per variant, a1 the first allele.
        offset: int64 file offset of each variant data block.
        size: int64 byte length of each variant data block.
    """

    variants: SnpMeta
    offset: np.ndarray
    size: np.ndarray


def read_bgi(path: Path) -> BgenIndex:
    """Read the bgenix ``Variant`` table, ordered by file position.

    ``rs`` is the table's ``rsid``; an empty one is filled from the ``.bgen``
    variant id by the caller.

    Raises:
        BgenFormatError: If any variant has other than two alleles.
    """
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        bad = con.execute(
            "SELECT rsid, chromosome, position, number_of_alleles FROM Variant "
            "WHERE number_of_alleles != 2 ORDER BY file_start_position LIMIT 1"
        ).fetchone()
        if bad is not None:
            rsid, chrom, pos, n_alleles = bad
            raise BgenFormatError(
                f"{path}: variant {rsid or '?'} at {chrom}:{pos} has "
                f"{n_alleles} alleles; only biallelic variants are supported"
            )
        rows = con.execute(
            "SELECT chromosome, position, rsid, allele1, allele2, "
            "file_start_position, size_in_bytes FROM Variant "
            "ORDER BY file_start_position"
        ).fetchall()
    finally:
        con.close()
    columns = list(zip(*rows, strict=True)) if rows else [()] * 7
    chrom, pos, rsid, allele1, allele2, offset, size = columns
    return BgenIndex(
        variants=SnpMeta(
            chr=np.array(chrom, dtype=str),
            rs=np.array(rsid, dtype=str),
            pos=np.array(pos, dtype=np.int64),
            a1=np.array(allele1, dtype=str),
            a0=np.array(allele2, dtype=str),
        ),
        offset=np.array(offset, dtype=np.int64),
        size=np.array(size, dtype=np.int64),
    )


def _variant_id_at(fd: int, offset: int) -> str:
    # The variant id is the first field, so filling an rsid never reads the
    # rest of the variant block.
    (length,) = struct.unpack("<H", os.pread(fd, 2, offset))
    return os.pread(fd, length, offset + 2).decode()


class _VariantHeader(NamedTuple):
    """A layout-2 variant's identifying fields, up to its data length C.

    Attributes:
        rsid: The rsid, possibly empty.
        alleles: The alleles in file order.
        end: Offset of the length C just after the last allele.
    """

    variant_id: str
    rsid: str
    chromosome: str
    position: int
    alleles: tuple[str, ...]
    end: int

    @property
    def display_rsid(self) -> str:
        """The rsid ``open_bgen_reader`` reports: the variant id when empty."""
        return self.rsid or self.variant_id


def _parse_variant_header(blob: memoryview) -> _VariantHeader:
    """Parse the identifying fields at the start of a variant data block.

    Raises:
        struct.error: If a field runs past the end of ``blob``.
        UnicodeDecodeError: If an identifier or allele is not UTF-8.
    """
    p = 0
    identifiers = []
    for _ in range(3):  # variant id, rsid, chromosome
        (length,) = struct.unpack_from("<H", blob, p)
        identifiers.append(bytes(blob[p + 2 : p + 2 + length]).decode())
        p += 2 + length
    position, n_alleles = struct.unpack_from("<IH", blob, p)
    p += 6
    alleles = []
    for _ in range(n_alleles):
        (length,) = struct.unpack_from("<I", blob, p)
        alleles.append(bytes(blob[p + 4 : p + 4 + length]).decode())
        p += 4 + length
    variant_id, rsid, chromosome = identifiers
    return _VariantHeader(variant_id, rsid, chromosome, position, tuple(alleles), p)


@dataclass(frozen=True, slots=True)
class ProbabilityBlock:
    """One decoded block of BGEN variants over every sample.

    All 2-D arrays are ``(n_samples, k)`` in Fortran order.

    Attributes:
        dosages: float64 first-allele dosage (2*q11 + q12) / (2**B - 1); NaN
            for a missing sample.
        q11: uint16 stored P(11) numerator; 0 for a missing sample.
        q12: uint16 stored P(12) numerator; 0 for a missing sample.
        missing: bool, the ploidy byte's missingness bit.
        bit_depth: uint8 (k,) bit depth B of each variant.
        info_sums: int64 ``(k, 4)`` exact INFO sums the decoder accumulated
            over the non-missing samples of ``info_rows``: sum(2*q11 + q12),
            sum((2*q11 + q12)**2), sum(4*q11 + q12) and the sample count.
        info_rows: The sample rows ``info_sums`` covers, or None for every
            row.
    """

    dosages: np.ndarray
    q11: np.ndarray
    q12: np.ndarray
    missing: np.ndarray
    bit_depth: np.ndarray
    info_sums: np.ndarray
    info_rows: np.ndarray | None


class BgenReader:
    """The ``.bgen`` reader behind ``GenotypeDataset``'s reader strategy.

    Args:
        bgen: The ``.bgen`` file.
        sample: The ``.sample`` file, hashed into the fingerprint.
        header: Its parsed header.
        index: Its parsed ``.bgi`` table, with every rsid filled.
        n_threads: OpenMP threads for decoding.
    """

    def __init__(
        self,
        bgen: Path,
        sample: Path,
        header: BgenHeader,
        index: BgenIndex,
        n_threads: int,
    ) -> None:
        self._bgen = bgen
        self._sample = sample
        self._header = header
        self._index = index
        self._n_threads = n_threads
        self._inflate: Callable[[bytes], bytes] | None = (
            _require_zstd() if header.compression == 2 else None
        )

    def read(
        self,
        columns: np.ndarray,
        block_size: int,
        *,
        stats_only: bool,
        info_rows: np.ndarray | None = None,
    ) -> Iterator[ProbabilityBlock]:
        """Yield decoded blocks of ``columns``; ``stats_only`` changes nothing.

        Each block's ``info_sums`` covers ``info_rows`` (every row when None).
        A row listed twice is summed once; the block's ``info_rows`` then
        names the distinct rows, so it never claims to cover the duplicates.

        Raises:
            BgenFormatError: If a variant block disagrees with the ``.bgi``
                or its probability data cannot be decoded; the message names
                the variant.
        """
        from jamma.lmm import accel  # jamma.lmm imports this package

        decode = accel.require().decode_bgen_probabilities_c
        n = self._header.n_samples
        keep = None
        if info_rows is not None:
            keep = np.zeros(n, dtype=np.bool_)
            keep[info_rows] = True
            if np.count_nonzero(keep) != len(info_rows):
                info_rows = np.flatnonzero(keep)
        fd = os.open(self._bgen, os.O_RDONLY)
        try:
            for start in range(0, len(columns), block_size):
                block = columns[start : start + block_size]
                yield self._read_block(fd, decode, block, keep, info_rows)
        finally:
            os.close(fd)

    def _read_block(
        self,
        fd: int,
        decode: Callable[..., tuple[int, str] | None],
        columns: np.ndarray,
        keep: np.ndarray | None,
        info_rows: np.ndarray | None,
    ) -> ProbabilityBlock:
        # Return from a normal call: a suspended reader must not keep this
        # block alive after its consumer takes or releases the dosages.
        n, k = self._header.n_samples, len(columns)
        out = ProbabilityBlock(
            dosages=np.empty((n, k), dtype=np.float64, order="F"),
            q11=np.empty((n, k), dtype=np.uint16, order="F"),
            q12=np.empty((n, k), dtype=np.uint16, order="F"),
            missing=np.empty((n, k), dtype=np.bool_, order="F"),
            bit_depth=np.empty(k, dtype=np.uint8),
            info_sums=np.empty((k, 4), dtype=np.int64),
            info_rows=info_rows,
        )
        for lo in range(0, k, _DECODE_BATCH):
            hi = min(lo + _DECODE_BATCH, k)
            self._decode_batch(fd, decode, columns[lo:hi], out, lo, hi, keep)
        return out

    def _decode_batch(
        self,
        fd: int,
        decode: Callable[..., tuple[int, str] | None],
        columns: np.ndarray,
        out: ProbabilityBlock,
        lo: int,
        hi: int,
        keep: np.ndarray | None,
    ) -> None:
        payloads, lengths = [], []
        for column, blob in zip(columns, self._read_blocks(fd, columns), strict=True):
            payload, inflated = self._payload(int(column), blob)
            payloads.append(payload)
            lengths.append(inflated)
        if self._inflate is not None:
            payloads = [self._inflate(p) for p in payloads]
            for i, (payload, inflated) in enumerate(
                zip(payloads, lengths, strict=True)
            ):
                if len(payload) != inflated:
                    raise BgenFormatError(
                        f"{self._bgen}: variant {self._describe(int(columns[i]))}: "
                        "zstd data does not inflate to the declared length"
                    )
        failure = decode(
            payloads,
            np.array(lengths, dtype=np.int64)
            if self._header.compression == 1
            else None,
            self._header.n_samples,
            out.dosages[:, lo:hi],
            out.q11[:, lo:hi],
            out.q12[:, lo:hi],
            out.missing[:, lo:hi],
            out.bit_depth[lo:hi],
            self._n_threads,
            info_rows=keep,
            info_sums=out.info_sums[lo:hi],
        )
        if failure is not None:
            position, reason = failure
            raise BgenFormatError(
                f"{self._bgen}: variant {self._describe(int(columns[position]))}: "
                f"{reason}"
            )

    def _read_blocks(self, fd: int, columns: np.ndarray) -> Iterator[memoryview]:
        """Yield each column's variant data block, one read per contiguous run."""
        offset, size = self._index.offset[columns], self._index.size[columns]
        breaks = np.flatnonzero(offset[1:] != offset[:-1] + size[:-1]) + 1
        for run in np.split(np.arange(len(columns)), breaks):
            first = int(offset[run[0]])
            total = int(offset[run[-1]] + size[run[-1]]) - first
            data = memoryview(os.pread(fd, total, first))
            if len(data) != total:
                raise BgenFormatError(
                    f"{self._bgen}: truncated; the .bgi places variant data "
                    f"past the end of the file"
                )
            for i in run:
                rel = int(offset[i]) - first
                yield data[rel : rel + int(size[i])]

    def _payload(self, column: int, blob: memoryview) -> tuple[memoryview, int]:
        """Split a variant data block into its probability data and length D.

        Checks chromosome, position, rsid and ordered alleles against the
        ``.bgi``, and length C against the indexed block size, so a stale index
        fails here, naming the first field that differs, instead of decoding
        the wrong bytes.
        """
        try:
            header = _parse_variant_header(blob)
            (c,) = struct.unpack_from("<I", blob, header.end)
        except struct.error:
            raise BgenFormatError(
                f"{self._bgen}: variant {self._describe(column)} header overruns "
                "its .bgi size_in_bytes; the index is stale"
            ) from None
        except UnicodeDecodeError:
            raise BgenFormatError(
                f"{self._bgen}: variant {self._describe(column)} header holds "
                "an identifier or allele that is not UTF-8; the file is corrupt "
                "or the index is stale"
            ) from None
        v = self._index.variants
        for name, in_bgen, in_bgi in (
            ("chromosome", header.chromosome, str(v.chr[column])),
            ("position", header.position, int(v.pos[column])),
            ("rsid", header.display_rsid, str(v.rs[column])),
            ("alleles", header.alleles, (str(v.a1[column]), str(v.a0[column]))),
        ):
            if in_bgen != in_bgi:
                raise BgenFormatError(
                    f"{self._bgen}: variant {self._describe(column)} does not "
                    f"match its .bgi: {name} {in_bgen!r} in the .bgen, "
                    f"{in_bgi!r} in the .bgi; the index is stale"
                )
        p = header.end + 4
        if p + c != len(blob):
            raise BgenFormatError(
                f"{self._bgen}: variant {self._describe(column)} block length "
                "differs from its .bgi size_in_bytes; the index is stale"
            )
        if self._header.compression == 0:
            return blob[p:], c
        (inflated,) = struct.unpack_from("<I", blob, p)
        # BGEN's largest bit depth, 32, bounds the data at 10 + N + 2*N*32/8
        # bytes; a larger declared length is corrupt and would size the C
        # scratch buffers. Depths 17..32 still reach the decoder's own error.
        max_inflated = 10 + 9 * self._header.n_samples
        if inflated > max_inflated:
            raise BgenFormatError(
                f"{self._bgen}: variant {self._describe(column)} declares "
                f"{inflated} uncompressed bytes, more than the {max_inflated} a "
                "biallelic diploid variant can hold at any BGEN bit depth"
            )
        return blob[p + 4 :], inflated

    def _describe(self, column: int) -> str:
        v = self._index.variants
        return f"{v.rs[column]} ({v.chr[column]}:{v.pos[column]})"

    def fingerprint(self) -> dict[str, str]:
        """Return ``bgen_fingerprint``, ``sample_sha256`` and ``variants_sha256``.

        ``variants_sha256`` hashes the parsed variant table and block
        offsets, not the ``.bgi`` bytes, which can differ for the same
        content.
        """
        st = self._bgen.stat()
        with open(self._sample, "rb") as fh:
            sample_sha256 = hashlib.file_digest(fh, "sha256").hexdigest()
        h = hashlib.sha256()
        v = self._index.variants
        for name, column in (
            ("chr", v.chr),
            ("rs", v.rs),
            ("pos", v.pos),
            ("a1", v.a1),
            ("a0", v.a0),
            ("offset", self._index.offset),
            ("size", self._index.size),
        ):
            h.update(f"{name}\x1e".encode())
            h.update("\x1f".join(map(str, column.tolist())).encode())
            h.update(b"\x1d")
        return {
            "bgen_fingerprint": f"{self._bgen.name}:{st.st_size}:{st.st_mtime_ns}",
            "sample_sha256": sample_sha256,
            "variants_sha256": h.hexdigest(),
        }


def open_bgen_reader(
    bgen: Path, sample: Path, bgi: Path, *, n_threads: int
) -> tuple[BgenReader, np.ndarray, np.ndarray, SnpMeta]:
    """Parse and cross-check the header, ``.sample`` and ``.bgi``.

    Returns:
        ``(reader, fid, iid, variants)``: ``fid``/``iid`` are the ``.sample``
        ID_1/ID_2 columns; ``variants`` is in file order with every empty
        rsid replaced by the variant id.

    Raises:
        FileNotFoundError: If any of the three files is missing.
        BgenFormatError: On any unsupported format or count mismatch.
        BgenDependencyError: If the file is zstd-compressed and no zstd
            module imports, or the ``_lmm_accel`` C extension is unavailable.
    """
    from jamma.lmm import accel  # jamma.lmm imports this package

    for path, kind in ((bgen, ".bgen"), (sample, ".sample"), (bgi, ".bgi")):
        if not path.exists():
            raise FileNotFoundError(f"BGEN {kind} file not found: {path}")
    if not accel.available():
        raise BgenDependencyError(
            "reading BGEN needs the _lmm_accel C extension, which is not "
            "available. Recompile: python -m jamma.lmm._compile_accel"
        )
    header = read_bgen_header(bgen)
    fid, iid = read_sample_file(sample)
    index = read_bgi(bgi)
    if header.n_samples < 1:
        raise BgenFormatError(f"{bgen}: the header declares no samples")
    if len(iid) != header.n_samples:
        raise BgenFormatError(
            f"{sample} has {len(iid)} samples, {bgen} header has {header.n_samples}"
        )
    if len(index.offset) != header.n_variants:
        raise BgenFormatError(
            f"{bgi} indexes {len(index.offset)} variants, {bgen} header has "
            f"{header.n_variants}"
        )
    if header.sample_ids is not None and not np.array_equal(header.sample_ids, iid):
        first = int(np.flatnonzero(header.sample_ids != iid)[0])
        raise BgenFormatError(
            f"sample {first + 1}: {bgen} embeds ID {header.sample_ids[first]!r}, "
            f"{sample} ID_2 is {iid[first]!r}"
        )
    index = _fill_empty_rsids(bgen, index)
    reader = BgenReader(bgen, sample, header, index, n_threads)
    logger.info(
        f"Opened BGEN {bgen.name}: {header.n_variants} variants x "
        f"{header.n_samples} samples, {_COMPRESSION[header.compression]}; each "
        f"full pass decodes {int(index.size.sum()) / 1e6:,.1f} MB of variant data "
        f"({header.n_variants * header.n_samples} probability pairs)"
    )
    return reader, fid, iid, index.variants


def _fill_empty_rsids(bgen: Path, index: BgenIndex) -> BgenIndex:
    empty = np.flatnonzero(index.variants.rs == "")
    if len(empty) == 0:
        return index
    rs = index.variants.rs.astype(object)
    fd = os.open(bgen, os.O_RDONLY)
    try:
        for i in empty:
            rs[i] = _variant_id_at(fd, int(index.offset[i]))
    finally:
        os.close(fd)
    v = index.variants
    variants = SnpMeta(chr=v.chr, rs=rs.astype(str), pos=v.pos, a1=v.a1, a0=v.a0)
    return BgenIndex(variants=variants, offset=index.offset, size=index.size)


__all__ = [
    "BgenDependencyError",
    "BgenFormatError",
    "BgenHeader",
    "BgenIndex",
    "BgenReader",
    "ProbabilityBlock",
    "open_bgen_reader",
    "read_bgen_header",
    "read_bgi",
    "read_sample_file",
]
