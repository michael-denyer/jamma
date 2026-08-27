"""The LOCO-only half of run_lmm_loco's configuration.

Its own module for the same reason ``pipeline_config`` is: this is data, and
``loco`` is behaviour. ``LocoConfig`` also owns the naming of every artifact a
LOCO run reads or writes, so the writer and the cache reader compose filenames
from one place instead of agreeing by convention.

``jamma.lmm.loco`` re-exports both names, so ``from jamma.lmm.loco import
LocoConfig`` keeps working — that is the path ``jamma.pipeline`` and
``jamma.lmm.__init__`` use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LocoConfig:
    """LOCO-specific options for :func:`run_lmm_loco`.

    The nine numerical knobs shared with every other runner live in
    ``LmmConfig``; these are the ones only LOCO has — where kinship and eigen
    artefacts are written, which SNPs take part, and the streaming chunk width.

    Frozen to match ``LmmConfig``. The ndarray fields are frozen by reference
    only, as usual for a dataclass: callers must not mutate an array after
    handing it over.

    Attributes:
        kinship_output_dir: Directory for K_loco files. Set it to write each
            chromosome's K_loco to disk; None writes none.
        prefix: Filename prefix shared by K_loco, eigen and manifest files.
        snps_indices: Global indices of SNPs to test. None tests all.
        ksnps_indices: Global indices of SNPs used to build kinship. None
            uses all.
        col_chunk_size: Columns per streaming chunk when building kinship.
        write_eigen: Write per-chromosome eigenvalues and eigenvectors.
        eigen_dir: Directory for eigen files. Required when write_eigen is set.
        legacy_text: Write kinship and eigen files as GEMMA text rather than
            .npy.
    """

    kinship_output_dir: Path | None = None
    prefix: str = "result"
    snps_indices: np.ndarray | None = None
    ksnps_indices: np.ndarray | None = None
    col_chunk_size: int = 5_000
    write_eigen: bool = False
    eigen_dir: Path | None = None
    legacy_text: bool = False

    def __post_init__(self) -> None:
        # Checked here rather than partway through the run: the caller learns
        # at construction, before any chromosome has been eigendecomposed.
        if self.write_eigen and self.eigen_dir is None:
            raise ValueError(
                "write_eigen=True requires eigen_dir to be set. "
                "Pass eigen_dir=<directory> alongside write_eigen=True."
            )
        if self.col_chunk_size <= 0:
            raise ValueError(
                f"col_chunk_size must be positive, got {self.col_chunk_size}"
            )

    @property
    def artifact_suffix(self) -> str:
        """Extension for kinship and eigen artifacts: .txt for GEMMA, else .npy."""
        return ".txt" if self.legacy_text else ".npy"

    def eigen_stem(self, chr_name: str) -> str:
        """Filename stem for one chromosome's eigenpair, extension excluded.

        ``write_eigen_files`` appends ``.eigenD``/``.eigenU`` and the extension
        itself, so this is what it takes as ``prefix=`` — and what
        :meth:`eigen_paths` composes the read-side names from, which is how the
        writer and the cache reader stay in step.
        """
        return f"{self.prefix}.loco.chr{chr_name}"

    def eigen_paths(self, chr_name: str) -> tuple[Path, Path]:
        """``(eigenD, eigenU)`` paths for one chromosome's cache entry.

        Raises:
            ValueError: If ``eigen_dir`` is None — there is no directory to
                name the files under. Cache readers check ``eigen_dir`` before
                asking; on the write side ``__post_init__`` has it covered.
        """
        if self.eigen_dir is None:
            raise ValueError("eigen_paths() requires eigen_dir, which is None")
        stem = self.eigen_stem(chr_name)
        return (
            self.eigen_dir / f"{stem}.eigenD{self.artifact_suffix}",
            self.eigen_dir / f"{stem}.eigenU{self.artifact_suffix}",
        )

    def kinship_path(self, chr_name: str) -> Path:
        """Path for one chromosome's LOCO kinship matrix.

        Raises:
            ValueError: If ``kinship_output_dir`` is None: nothing asked for
                the kinship to be saved, so there is no directory to name it under.
        """
        if self.kinship_output_dir is None:
            raise ValueError(
                "kinship_path() requires kinship_output_dir, which is None"
            )
        name = f"{self.prefix}.loco.cXX.chr{chr_name}{self.artifact_suffix}"
        return self.kinship_output_dir / name


DEFAULT_LOCO_CONFIG = LocoConfig()
"""The all-defaults LOCO config, shared as run_lmm_loco's default argument.

LocoConfig is frozen, so one instance is safe to share.
"""
