"""Prepared genotype coordinates shared by NumPy LMM sources."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Protocol

import numpy as np

from jamma.core.snp_stats import (
    SnpFilterSpec,
    SnpSelection,
    SnpStats,
    filter_snp_stats,
)
from jamma.lmm.schema import SnpMeta

if TYPE_CHECKING:
    from jamma.lmm.chunk_runner_numpy import RawLmmChunk


@dataclass(frozen=True, slots=True)
class SampleBasis:
    """Analyzed positions in one source's run-local row coordinates."""

    positions: np.ndarray
    source_row_count: int

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.intp)
        if positions.ndim != 1:
            raise ValueError(f"positions must be 1-D, got ndim={positions.ndim}")
        if self.source_row_count < 0:
            raise ValueError(
                f"source_row_count must be >= 0, got {self.source_row_count}"
            )
        if len(positions) > 0:
            if positions[0] < 0 or positions[-1] >= self.source_row_count:
                raise ValueError("sample positions fall outside the source row range")
            if len(positions) > 1 and np.any(np.diff(positions) <= 0):
                raise ValueError("sample positions must be strictly increasing")
        positions.flags.writeable = False
        object.__setattr__(self, "positions", positions)

    @classmethod
    def from_mask(cls, valid_mask: np.ndarray) -> SampleBasis:
        """Build the analyzed positions from the runner's valid-sample mask."""
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError(f"valid_mask must be 1-D, got ndim={mask.ndim}")
        return cls(np.flatnonzero(mask), mask.shape[0])

    @property
    def analyzed_sample_count(self) -> int:
        return self.positions.shape[0]

    @property
    def is_all_samples(self) -> bool:
        return self.analyzed_sample_count == self.source_row_count


@dataclass(frozen=True, slots=True)
class PreparedGenotypes:
    """One source, sample basis, selection, and aligned chunk stream."""

    snp_meta: SnpMeta
    selection: SnpSelection
    n_unexpected: int
    analyzed_sample_count: int
    sample_basis: SampleBasis
    chunk_factory: Callable[[int], Iterator[RawLmmChunk]] = field(
        compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.n_unexpected < 0:
            raise ValueError(f"n_unexpected must be >= 0, got {self.n_unexpected}")
        if self.analyzed_sample_count < 1:
            raise ValueError(
                f"analyzed_sample_count must be >= 1, got {self.analyzed_sample_count}"
            )
        if self.analyzed_sample_count != self.sample_basis.analyzed_sample_count:
            raise ValueError(
                "analyzed_sample_count must match sample_basis: "
                f"got {self.analyzed_sample_count} and "
                f"{self.sample_basis.analyzed_sample_count}"
            )
        indices = self.selection.indices
        if len(indices) > 0 and (
            np.any(indices < 0) or np.any(indices >= len(self.snp_meta))
        ):
            raise ValueError("selected SNP identities fall outside paired SnpMeta")

    @property
    def n_filtered(self) -> int:
        return self.selection.indices.shape[0]

    @property
    def imputation_means(self) -> np.ndarray:
        return self.selection.filtered_means

    def chunks(self, chunk_size: int) -> Iterator[RawLmmChunk]:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        return self.chunk_factory(chunk_size)


class GenotypeSource(Protocol):
    """A source that binds its storage coordinates in one preparation step."""

    @property
    def n_snps(self) -> int: ...

    def prepare(
        self, samples: SampleBasis, filters: SnpFilterSpec
    ) -> PreparedGenotypes: ...


def bind_prepared_genotypes(
    *,
    snp_meta: SnpMeta,
    stats: SnpStats,
    filters: SnpFilterSpec,
    sample_basis: SampleBasis,
    chunk_source: Callable[[SnpSelection, int], Iterator[RawLmmChunk]],
) -> PreparedGenotypes:
    """Filter one statistics population and bind its matching chunk stream.

    ``chunk_source`` is a generator function taking (selection, chunk_size);
    the filtered selection is bound here so ``PreparedGenotypes.chunks``
    only ever streams the SNPs this preparation selected.
    """
    selection = filter_snp_stats(stats, filters)
    return PreparedGenotypes(
        snp_meta=snp_meta,
        selection=selection,
        n_unexpected=stats.n_unexpected,
        analyzed_sample_count=stats.n_samples,
        sample_basis=sample_basis,
        chunk_factory=partial(chunk_source, selection),
    )
