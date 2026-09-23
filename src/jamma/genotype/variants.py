"""Per-variant metadata shared by every genotype consumer.

``SnpMeta`` lives in ``jamma.genotype`` rather than ``jamma.lmm`` so that
genotype readers can build it without importing the LMM package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

SnpInfoRecord: TypeAlias = Mapping[str, str | int | float]
"""Caller-supplied SNP metadata row accepted by ``SnpMeta.from_dicts``.

The public runners historically accept dictionaries with additional fields
such as MAF. A read-only mapping preserves that compatibility while naming the
value types consumed by ``SnpMeta``.
"""


@dataclass(frozen=True, slots=True)
class SnpMeta:
    """SNP metadata as one array per column in its paired source coordinate.

    Writers and result builders slice these arrays directly; nothing
    materialises a per-SNP dict. ``pos`` is normalised to int64 and ``chr``
    to str at construction, so downstream formatting needs no coercion. A full
    BED source uses file-global positions; a pre-sliced matrix may use local
    positions. The source paired with this value defines the identity.

    Attributes:
        chr: Chromosome identifier per SNP (str).
        rs: SNP identifier / rsID per SNP (str).
        pos: Base-pair position per SNP (int64).
        a1: Minor allele per SNP (str).
        a0: Major allele per SNP (str).
    """

    chr: np.ndarray
    rs: np.ndarray
    pos: np.ndarray
    a1: np.ndarray
    a0: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.rs)
        for name in ("chr", "pos", "a1", "a0"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"SnpMeta columns must share one length; "
                    f"{name} has {len(getattr(self, name))}, rs has {n}"
                )

    def __len__(self) -> int:
        return len(self.rs)

    @classmethod
    def from_dicts(cls, snp_info: Sequence[SnpInfoRecord]) -> SnpMeta:
        """Parse a caller-supplied list of per-SNP dicts.

        The boundary for the public batch API. Requires the canonical keys
        chr/rs/pos/a1/a0 on every dict; raises KeyError on the first miss.
        """
        return cls(
            chr=np.array([str(s["chr"]) for s in snp_info]),
            rs=np.array([s["rs"] for s in snp_info]),
            pos=np.array([int(s["pos"]) for s in snp_info], dtype=np.int64),
            a1=np.array([s["a1"] for s in snp_info]),
            a0=np.array([s["a0"] for s in snp_info]),
        )
