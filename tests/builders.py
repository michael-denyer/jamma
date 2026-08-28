"""Builders for the synthetic inputs the LMM tests share.

The rotated-input recipe (sorted eigenvalues, an intercept or random
covariate block, a phenotype, a genotype block) had been written out
inline in a few dozen tests. The draw order here is the one those tests
used, so a test that moves onto `rotated_lmm_inputs` with the same seed
sees bit-identical arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jamma.lmm.likelihood import compute_Uab

# Matrix sizes the jlinalg BLAS tests sweep. The values were chosen around
# the MR/MC/KC blocking of the own-BLAS kernel deleted at 663a22b; they are
# kept because a spread of powers of two with both neighbours and a few
# primes is what vendor BLAS blocking still trips on. The eigh list stops at
# 200 because eigendecomposition is O(N^3).
BOUNDARY_SIZES = [
    1,
    3,
    5,
    6,
    7,
    8,
    9,
    11,
    13,
    63,
    64,
    65,
    71,
    72,
    73,
    127,
    128,
    129,
    255,
    256,
    257,
    500,
    1000,
]
EIGH_BOUNDARY_SIZES = sorted({*(s for s in BOUNDARY_SIZES if s < 200), 2, 31, 100, 200})


@dataclass(frozen=True)
class LmmInputs:
    """Rotated LMM inputs for one synthetic dataset.

    Attributes:
        eigenvalues: Kinship eigenvalues, ascending, shape (n_samples,).
        UtW: Rotated covariates, shape (n_samples, n_cvt).
        Uty: Rotated phenotype, shape (n_samples,).
        UtG: Rotated genotypes, shape (n_samples, n_snps).
    """

    eigenvalues: np.ndarray
    UtW: np.ndarray
    Uty: np.ndarray
    UtG: np.ndarray

    @property
    def n_samples(self) -> int:
        return self.eigenvalues.shape[0]

    @property
    def n_snps(self) -> int:
        return self.UtG.shape[1]

    @property
    def n_cvt(self) -> int:
        return self.UtW.shape[1]

    def uab_batch(self) -> np.ndarray:
        """Per-SNP Uab, shape (n_snps, n_samples, n_index)."""
        return np.stack(
            [
                compute_Uab(self.UtW, self.Uty, self.UtG[:, i])
                for i in range(self.n_snps)
            ]
        )


def rotated_lmm_inputs(
    n_samples: int,
    n_snps: int,
    n_cvt: int = 1,
    seed: int = 42,
    eig_range: tuple[float, float] = (0.1, 5.0),
) -> LmmInputs:
    """Build synthetic rotated inputs with a seeded generator.

    Eigenvalues are drawn uniformly on ``eig_range`` and sorted ascending.
    ``UtW`` is an intercept column for ``n_cvt == 1`` and standard-normal
    otherwise. ``Uty`` and ``UtG`` are standard-normal.
    """
    rng = np.random.default_rng(seed)
    eigenvalues = np.sort(rng.uniform(*eig_range, n_samples))
    UtW = (
        np.ones((n_samples, 1))
        if n_cvt == 1
        else rng.standard_normal((n_samples, n_cvt))
    )
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))
    return LmmInputs(eigenvalues, UtW, Uty, UtG)


def write_fam(
    path: Path,
    *phenotype_columns: Sequence[float | str],
    missing_at: Iterable[int] = (),
) -> Path:
    """Write a PLINK ``.fam`` with FID/IID/0/0/0 and the given phenotype columns.

    Values are written with ``str``, so a column may mix floats with the
    ``"NA"`` and ``"-9"`` missing markers. ``missing_at`` writes ``NA`` in
    every column for those sample indices.
    """
    n_samples = len(phenotype_columns[0])
    missing = set(missing_at)
    lines = []
    for i in range(n_samples):
        values = ["NA" if i in missing else str(col[i]) for col in phenotype_columns]
        lines.append("\t".join([f"FAM{i:03d}", f"IND{i:03d}", "0", "0", "0", *values]))
    path.write_text("\n".join(lines) + "\n")
    return path
