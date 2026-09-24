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
from bed_reader import open_bed

from jamma.genotype.dataset import GenotypeEncoding
from jamma.lmm.association_plan import (
    ExecutableAssociationPlan,
    ExecutionMode,
    ExecutionPlan,
)
from jamma.lmm.chunk_sizing import LmmChunkPlan
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.pab import compute_Uab
from jamma.lmm.workspace import WorkspaceSpec

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
    intercept: bool = True,
) -> LmmInputs:
    """Build synthetic rotated inputs with a seeded generator.

    Eigenvalues are drawn uniformly on ``eig_range`` and sorted ascending.
    ``UtW`` is an intercept column for ``n_cvt == 1`` unless ``intercept`` is
    False, and standard-normal otherwise. ``Uty`` and ``UtG`` are
    standard-normal.
    """
    rng = np.random.default_rng(seed)
    eigenvalues = np.sort(rng.uniform(*eig_range, n_samples))
    UtW = (
        np.ones((n_samples, 1))
        if n_cvt == 1 and intercept
        else rng.standard_normal((n_samples, n_cvt))
    )
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))
    return LmmInputs(eigenvalues, UtW, Uty, UtG)


def covariate_lmm_inputs(
    n_cvt: int, n_samples: int = 200, n_snps: int = 50, seed: int = 42
) -> LmmInputs:
    """Build the covariate recipe the general (n_cvt >= 2) kernel tests pin.

    Eigenvalues are drawn on (0.1, 2.0) and sorted descending. ``UtW`` is
    ``|normal| + 0.5`` so no covariate column is near zero.
    """
    rng = np.random.default_rng(seed)
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))[::-1]
    UtW = np.abs(rng.standard_normal((n_samples, n_cvt))) + 0.5
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))
    return LmmInputs(eigenvalues, UtW, Uty, UtG)


def gram_uab_batch(
    n_samples: int = 200, n_snps: int = 50, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Build ascending eigenvalues and an n_cvt=1 Uab batch from per-SNP vectors.

    Each SNP draws its own positive ``w``, positive ``x`` and normal ``y``, and
    its six Uab columns are their Gram products, so every Pab recursion is
    well-conditioned.

    Returns:
        ``(eigenvalues, Uab_batch)``, shapes (n_samples,) and
        (n_snps, n_samples, 6).
    """
    rng = np.random.default_rng(seed)
    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))
    Uab_batch = np.zeros((n_snps, n_samples, 6), dtype=np.float64)
    for i in range(n_snps):
        w = np.abs(rng.standard_normal(n_samples)) + 1.0
        x = np.abs(rng.standard_normal(n_samples)) + 0.5
        y = rng.standard_normal(n_samples)
        Uab_batch[i] = np.stack([w * w, w * x, w * y, x * x, x * y, y * y], axis=1)
    return eigenvalues, Uab_batch


def make_runner_synthetic_data(
    n_samples: int = 100, n_snps: int = 50, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Create unrotated genotypes, phenotypes, kinship and SNP info for runner tests."""
    rng = np.random.default_rng(seed)
    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
    phenotypes = rng.standard_normal(n_samples)
    kinship = np.corrcoef(genotypes) + np.eye(n_samples) * 0.1
    kinship = (kinship + kinship.T) / 2
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "T"}
        for i in range(n_snps)
    ]
    return genotypes, phenotypes, kinship, snp_info


def read_plink_genotypes(bfile: Path) -> np.ndarray:
    """Read a whole ``.bed`` as float32 ``(n_samples, n_snps)``, NaN for missing.

    bed-reader's own read, so it is an oracle independent of
    ``GenotypeDataset``; float32 matches the batch pipeline's in-memory
    matrix.
    """
    with open_bed(Path(f"{bfile}.bed")) as bed:
        return bed.read(dtype=np.float32)


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


def empty_workspace(
    dispatch: DispatchPath, n_samples: int, n_input_samples: int, n_cvt: int
) -> WorkspaceSpec:
    """A kernel workspace that holds no bytes, so a quote reads no C sizer."""
    return WorkspaceSpec(
        dispatch, 1, n_samples, n_input_samples, n_cvt, 0, 0, 1, 0, 0, 0
    )


def association_price_plan(
    mode: ExecutionMode,
    *,
    n_samples: int,
    n_snps: int,
    chunk_size: int,
    n_buffers: int = 1,
    n_cvt: int = 1,
    dispatch: DispatchPath = DispatchPath.NUMPY_FALLBACK,
    n_input_samples: int | None = None,
) -> ExecutableAssociationPlan:
    """A real association plan with an empty workspace, priced machine-independently.

    ``price(eigen=None).association_gb`` then carries only the U, genotype,
    rotation-buffer, and Uab/Iab terms, so a test can state them by hand.
    """
    n_input = n_samples if n_input_samples is None else n_input_samples
    return ExecutableAssociationPlan(
        summary=ExecutionPlan(mode, "test"),
        dispatch=dispatch,
        conservative_chunks=LmmChunkPlan(
            chunk_size, -(-n_snps // chunk_size), n_buffers, n_buffers > 1
        ),
        n_samples=n_samples,
        n_input_samples=n_input,
        n_snps_before_filter=n_snps,
        n_cvt=n_cvt,
        mem_budget_gb=None,
        workspace=empty_workspace(dispatch, n_samples, n_input, n_cvt),
        genotype_encoding=GenotypeEncoding.HARD_CALLS,
    )
