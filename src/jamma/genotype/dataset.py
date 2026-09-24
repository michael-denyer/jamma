"""Format-neutral genotype input: samples x variants, streamed in blocks.

``GenotypeDataset`` is the one type a genotype consumer needs. File formats
plug in as private ``_GenotypeReader`` strategies that know bytes and nothing
else, so column validation, progress, row selection, statistics and
partitions live here once. ``GenotypeEncoding`` says what one stored value
means; every capability a consumer might branch on derives from it.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, final

import numpy as np
from bed_reader import open_bed

from jamma.core.progress import progress_iterator
from jamma.genotype.snp_stats import SnpStats, collect_snp_stats_from_chunks
from jamma.genotype.variants import SnpMeta
from jamma.io.plink import PlinkReader, validate_plink_dimensions

DEFAULT_BLOCK = 10_000


class GenotypeEncoding(enum.Enum):
    """What one stored genotype value is.

    HARD_CALLS: counts of the counted allele in {0, 1, 2}, or NaN. PLINK
        ``.bed`` files and in-memory matrices. HWE genotype classes exist.
    PROBABILITIES: expected dosages of the counted allele derived from
        genotype probabilities. Fractional values fall in no HWE class.
    """

    HARD_CALLS = "hard_calls"
    PROBABILITIES = "probabilities"

    @property
    def supports_hwe(self) -> bool:
        """True when HWE genotype-class counts are meaningful."""
        return self is GenotypeEncoding.HARD_CALLS

    @property
    def supports_info(self) -> bool:
        """True when INFO can differ from 1, so an INFO filter is meaningful."""
        return self is GenotypeEncoding.PROBABILITIES

    @property
    def validates_hard_calls(self) -> bool:
        """True when statistics count values outside {0, 1, 2, NaN}."""
        return self is GenotypeEncoding.HARD_CALLS


@dataclass(frozen=True, slots=True)
class SampleTable:
    """Sample identities in dataset row order; row i of every block is sample i.

    Attributes:
        fid: Family ID per sample (str).
        iid: Individual ID per sample (str).
    """

    fid: np.ndarray
    iid: np.ndarray

    def __post_init__(self) -> None:
        if len(self.fid) != len(self.iid):
            raise ValueError(
                f"SampleTable fid has {len(self.fid)} rows, iid has {len(self.iid)}"
            )

    def __len__(self) -> int:
        return len(self.iid)


class GenotypeBlock:
    """A run of requested variants over all dataset rows, read once.

    Call ``stats`` any number of times, then ``dosages`` at most once.
    ``dosages`` may hand over the block's own buffer, so the block is spent
    afterwards and every further call raises.

    Attributes:
        columns: Global variant indices in this block, strictly increasing.
        start: Position of the first column in the requested column list
            (``blocks(columns=...)``), or in the dataset when unfiltered.
        end: Exclusive end, in the same coordinate as ``start``.
    """

    __slots__ = ("_encoding", "_values", "columns", "end", "start")

    def __init__(
        self,
        columns: np.ndarray,
        start: int,
        end: int,
        values: np.ndarray,
        encoding: GenotypeEncoding,
    ) -> None:
        self.columns = columns
        self.start = start
        self.end = end
        self._values: np.ndarray | None = values
        self._encoding = encoding

    def _live_values(self) -> np.ndarray:
        if self._values is None:
            raise RuntimeError(
                "GenotypeBlock is spent: dosages() already handed over its values"
            )
        return self._values

    def stats(self, rows: np.ndarray | None = None, *, hwe: bool = False) -> SnpStats:
        """Per-variant statistics over ``rows`` (all rows when None).

        Args:
            rows: Sample row positions to include, or None for every row.
            hwe: Also count HWE genotype classes.

        Returns:
            Statistics with ``global_indices`` equal to ``columns``.
            ``n_unexpected`` counts values outside {0, 1, 2, NaN} when the
            encoding validates hard calls, else 0.

        Raises:
            ValueError: If ``hwe`` is requested for an encoding without HWE
                classes.
            RuntimeError: If the block is spent.
        """
        values = self._live_values()
        return _collect_stats(
            [(values if rows is None else values[rows, :], 0, values.shape[1])],
            n_snps=values.shape[1],
            n_samples=values.shape[0] if rows is None else len(rows),
            columns=self.columns,
            encoding=self._encoding,
            hwe=hwe,
        )

    def dosages(
        self, rows: np.ndarray | None = None, columns: np.ndarray | None = None
    ) -> np.ndarray:
        """Return float64 counted-allele dosages and spend the block.

        NaN marks missing. The caller owns the result and may mutate it in
        place. With neither ``rows`` nor ``columns`` the block's own buffer is
        returned without a copy, in the reader's memory order.

        Args:
            rows: Sample row positions to keep, or None for every row.
            columns: Positions within this block to keep, or None for all.

        Raises:
            RuntimeError: If the block is already spent.
        """
        values = self._live_values()
        self._values = None
        if rows is columns is None:
            return values
        if rows is None:
            return values[:, columns]
        if columns is None:
            return values[rows, :]
        return values[np.ix_(rows, columns)]


def _collect_stats(
    chunks: Iterable[tuple[np.ndarray, int, int]],
    *,
    n_snps: int,
    n_samples: int,
    columns: np.ndarray,
    encoding: GenotypeEncoding,
    hwe: bool,
) -> SnpStats:
    if hwe and not encoding.supports_hwe:
        raise ValueError(f"HWE counts are undefined for {encoding.value} genotypes")
    return collect_snp_stats_from_chunks(
        chunks,
        n_snps=n_snps,
        n_samples=n_samples,
        global_indices=columns,
        include_hwe=hwe,
        validate_genotypes=encoding.validates_hard_calls,
    )


class _GenotypeReader(Protocol):
    """Format strategy. Knows bytes; owns no policy.

    Readers never filter rows, compute statistics or show progress. They read
    the requested global columns in order, in blocks of at most
    ``block_size``, over every dataset row.
    """

    def read(
        self, columns: np.ndarray, block_size: int, *, stats_only: bool
    ) -> Iterator[np.ndarray]:
        """Yield ``(n_samples, k)`` blocks for consecutive runs of ``columns``.

        A block read with ``stats_only=False`` is float64 and owned by the
        caller. ``stats_only=True`` lets a reader return a cheaper dtype or a
        view, since only statistics are computed from it.
        """
        ...

    def fingerprint(self) -> dict[str, str]:
        """Return format-specific cache-key components."""
        ...


class _MatrixReader:
    """An in-memory HARD_CALLS matrix behind the ``_GenotypeReader`` strategy."""

    def __init__(self, genotypes: np.ndarray) -> None:
        self._genotypes = genotypes

    def read(
        self, columns: np.ndarray, block_size: int, *, stats_only: bool
    ) -> Iterator[np.ndarray]:
        for start in range(0, len(columns), block_size):
            block = self._genotypes[:, columns[start : start + block_size]]
            yield block if stats_only else np.asarray(block, dtype=np.float64)

    def fingerprint(self) -> dict[str, str]:
        raise ValueError("an in-memory genotype matrix has no file fingerprint")


@final
class GenotypeDataset:
    """Samples x variants, streamable by column, whatever the file format.

    Immutable after construction. Opening reads metadata only. Each
    ``blocks`` or ``stats`` call opens its own read, so two iterations never
    share cursor state.
    """

    def __init__(
        self,
        reader: _GenotypeReader,
        encoding: GenotypeEncoding,
        samples: SampleTable,
        variants: SnpMeta,
    ) -> None:
        self._reader = reader
        self._encoding = encoding
        self._samples = samples
        self._variants = variants

    @staticmethod
    def open_plink(prefix: Path) -> GenotypeDataset:
        """Open ``prefix.bed``/``.bim``/``.fam``, reading metadata only.

        Args:
            prefix: Path prefix for PLINK files (without extension).

        Returns:
            A HARD_CALLS dataset whose counted allele ``a1`` is ``.bim`` allele 1.

        Raises:
            FileNotFoundError: If any of the .bed, .bim, or .fam files are missing.
            ValueError: If the .bed size does not match the .fam and .bim counts.
        """
        validate_plink_dimensions(prefix)
        with open_bed(Path(f"{prefix}.bed")) as bed:
            samples = SampleTable(fid=bed.fid, iid=bed.iid)
            variants = SnpMeta(
                chr=np.asarray(bed.chromosome).astype(str),
                rs=bed.sid,
                pos=np.asarray(bed.bp_position, dtype=np.int64),
                a1=bed.allele_1,
                a0=bed.allele_2,
            )
        return GenotypeDataset(
            PlinkReader(prefix), GenotypeEncoding.HARD_CALLS, samples, variants
        )

    @staticmethod
    def from_matrix(genotypes: np.ndarray, variants: SnpMeta) -> GenotypeDataset:
        """Wrap an in-memory ``(n_samples, n_variants)`` HARD_CALLS matrix.

        The matrix is not copied; blocks read from it are. Samples get
        positional IDs ``"0"``, ``"1"``, ... for both FID and IID.

        Raises:
            ValueError: If the matrix is not 2-D or its column count differs
                from ``len(variants)``.
        """
        if genotypes.ndim != 2:
            raise ValueError(f"genotypes must be 2-D, got ndim={genotypes.ndim}")
        if genotypes.shape[1] != len(variants):
            raise ValueError(
                f"genotype columns must match variants: "
                f"{genotypes.shape[1]} != {len(variants)}"
            )
        ids = np.arange(genotypes.shape[0]).astype(str)
        return GenotypeDataset(
            _MatrixReader(genotypes),
            GenotypeEncoding.HARD_CALLS,
            SampleTable(fid=ids, iid=ids),
            variants,
        )

    def materialize(self) -> GenotypeDataset:
        """Read every variant into memory once, as a float32 HARD_CALLS dataset.

        float32 holds 0, 1, 2 and NaN exactly and halves the float64
        footprint. Samples, variants and encoding carry over unchanged; the
        result has no file fingerprint.

        Raises:
            ValueError: If the encoding is not HARD_CALLS, whose dosages
                float32 would round.
        """
        if self._encoding is not GenotypeEncoding.HARD_CALLS:
            raise ValueError(
                f"only hard-call datasets can be materialized, "
                f"got {self._encoding.value}"
            )
        cols = np.arange(self.n_variants, dtype=np.intp)
        blocks = list(self._reader.read(cols, max(self.n_variants, 1), stats_only=True))
        genotypes = (
            np.asarray(blocks[0], dtype=np.float32)
            if blocks
            else np.empty((self.n_samples, 0), dtype=np.float32)
        )
        return GenotypeDataset(
            _MatrixReader(genotypes), self._encoding, self._samples, self._variants
        )

    @property
    def samples(self) -> SampleTable:
        return self._samples

    @property
    def variants(self) -> SnpMeta:
        """chr/rs/pos/a1/a0 in dataset column order; a1 is the counted allele."""
        return self._variants

    @property
    def encoding(self) -> GenotypeEncoding:
        return self._encoding

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def n_variants(self) -> int:
        return len(self._variants)

    @cached_property
    def partitions(self) -> Mapping[str, np.ndarray]:
        """Chromosome to ascending global column indices, first-appearance order."""
        chromosomes = self._variants.chr
        _, first = np.unique(chromosomes, return_index=True)
        parts = {}
        for i in np.sort(first):
            indices = np.flatnonzero(chromosomes == chromosomes[i])
            indices.flags.writeable = False
            parts[str(chromosomes[i])] = indices
        return MappingProxyType(parts)

    def fingerprint(self) -> dict[str, str]:
        """Cache-key components that change when the genotype content can.

        PLINK returns ``bed_fingerprint`` (name:size:mtime_ns) and
        ``bim_sha256``, the eigen cache's existing components.

        Raises:
            ValueError: For an in-memory dataset, which has no file identity.
        """
        return self._reader.fingerprint().copy()

    def _resolve_columns(
        self, columns: np.ndarray | None, block_size: int
    ) -> np.ndarray:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if columns is None:
            return np.arange(self.n_variants, dtype=np.intp)
        cols = np.asarray(columns, dtype=np.intp)
        if cols.ndim != 1:
            raise ValueError(f"columns must be 1-D, got ndim={cols.ndim}")
        if len(cols) > 1 and np.any(np.diff(cols) <= 0):
            raise ValueError("columns must be sorted in strictly ascending order")
        if len(cols) > 0 and (cols[0] < 0 or cols[-1] >= self.n_variants):
            raise ValueError(
                f"columns out of bounds: range [{cols[0]}, {cols[-1]}], "
                f"dataset has {self.n_variants} variants"
            )
        return cols

    def _read_spans(
        self,
        cols: np.ndarray,
        block_size: int,
        *,
        stats_only: bool,
        progress: str | None,
        eta_seconds: float | None = None,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield ``(start, end, values)`` per reader block, in requested space."""
        n_cols = len(cols)
        raw = self._reader.read(cols, block_size, stats_only=stats_only)
        if progress is not None:
            raw = progress_iterator(
                raw,
                total=(n_cols + block_size - 1) // block_size,
                desc=progress,
                initial_eta_seconds=eta_seconds,
            )
        starts = range(0, n_cols, block_size)
        for start, values in zip(starts, raw, strict=True):
            yield start, min(start + block_size, n_cols), values

    def blocks(
        self,
        block_size: int,
        *,
        columns: np.ndarray | None = None,
        progress: str | None = None,
        eta_seconds: float | None = None,
    ) -> Iterator[GenotypeBlock]:
        """Stream requested variants in ascending blocks over all rows.

        Args:
            block_size: Max variants per block, in the requested column space.
            columns: Strictly increasing global indices, or None for every
                variant.
            progress: Progress-bar label, or None for no bar.
            eta_seconds: Progress-bar ETA before the first block.

        Raises:
            ValueError: If ``block_size`` < 1 or ``columns`` is not strictly
                increasing and in bounds.
        """
        cols = self._resolve_columns(columns, block_size)
        spans = self._read_spans(
            cols,
            block_size,
            stats_only=False,
            progress=progress,
            eta_seconds=eta_seconds,
        )
        for start, end, values in spans:
            yield GenotypeBlock(cols[start:end], start, end, values, self.encoding)

    def stats(
        self,
        rows: np.ndarray | None,
        *,
        columns: np.ndarray | None = None,
        hwe: bool = False,
        block_size: int = DEFAULT_BLOCK,
        progress: str | None = None,
    ) -> SnpStats:
        """Per-variant statistics over ``rows`` in one pass.

        Args:
            rows: Analysed sample row positions, or None for every row.
            columns: Strictly increasing global indices, or None for every
                variant.
            hwe: Also count HWE genotype classes.
            block_size: Variants per read.
            progress: Progress-bar label, or None for no bar.

        Returns:
            Statistics in requested column order; ``n_unexpected`` is summed
            over blocks for the caller to report.

        Raises:
            ValueError: As ``blocks``, or if ``hwe`` is requested for an
                encoding without HWE classes.
        """
        cols = self._resolve_columns(columns, block_size)
        spans = self._read_spans(cols, block_size, stats_only=True, progress=progress)
        chunks = (
            (values if rows is None else values[rows, :], start, end)
            for start, end, values in spans
        )
        return _collect_stats(
            chunks,
            n_snps=len(cols),
            n_samples=self.n_samples if rows is None else len(rows),
            columns=cols,
            encoding=self.encoding,
            hwe=hwe,
        )
