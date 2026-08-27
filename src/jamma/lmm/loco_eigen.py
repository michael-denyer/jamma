"""Where a LOCO run gets its per-chromosome eigenpairs, and what it writes.

Two sources, chosen once by ``run_lmm_loco`` rather than re-tested per
chromosome: read a validated eigen cache, or stream each chromosome's kinship
and eigendecompose it. The artifact writers live here too, since the compute
path is the only thing that calls them.

Filenames come from :class:`~jamma.lmm.loco_config.LocoConfig`, never from a
literal in this module.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.kinship import write_kinship_matrix
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
from jamma.lmm.loco_config import LocoConfig


def _find_loco_eigen_cache(
    loco: LocoConfig,
    chr_names: list[str],
) -> dict[str, tuple[Path, Path]] | None:
    """Check for a complete set of per-chromosome cached eigen files.

    File naming comes from ``loco.eigen_paths()``, the same method the writer
    builds its names with, so the two cannot drift.

    Dimension validation is deferred to the per-chromosome load in
    ``run_lmm_loco``, where ``read_eigen_files(n_samples=...)`` raises
    ``ValueError`` on mismatch. This avoids loading all eigen data
    eagerly just to check dimensions.

    Args:
        loco: LOCO config supplying eigen_dir, eigen_prefix and legacy_text.
        chr_names: List of chromosome names to check.

    Returns:
        Dict mapping chr_name -> (eigenD_path, eigenU_path) if ALL chromosomes
        have both files. None if ANY chromosome is missing either file, or if
        no eigen_dir was configured — all three mean "compute from scratch".
    """
    if loco.eigen_dir is None:
        return None

    if not loco.eigen_dir.is_dir():
        logger.warning(
            f"eigen_dir is not a directory: {loco.eigen_dir}. "
            f"Will compute from scratch."
        )
        return None

    cache: dict[str, tuple[Path, Path]] = {}

    for ch in chr_names:
        d_path, u_path = loco.eigen_paths(ch)

        if not d_path.exists() or not u_path.exists():
            missing = d_path if not d_path.exists() else u_path
            logger.info(
                f"LOCO eigen cache incomplete: missing {missing}. "
                f"Will compute from scratch."
            )
            return None

        cache[ch] = (d_path, u_path)

    return cache


def _save_loco_kinship(
    K_loco: np.ndarray,
    chr_name: str,
    *,
    loco: LocoConfig,
    show_progress: bool,
) -> None:
    """Write one chromosome's LOCO kinship before it is discarded."""
    kinship_path = loco.kinship_path(chr_name)
    try:
        actual_path = write_kinship_matrix(
            K_loco, kinship_path, legacy_text=loco.legacy_text
        )
    except OSError as e:
        raise OSError(
            f"Failed to save LOCO kinship for chromosome {chr_name} "
            f"to {kinship_path}: {e}"
        ) from e
    if show_progress:
        logger.info(f"  Saved LOCO kinship to {actual_path}")


def _write_loco_eigen(
    eigenvalues: np.ndarray,
    U: np.ndarray,
    chr_name: str,
    *,
    loco: LocoConfig,
    eigen_dir: Path,
) -> None:
    """Persist one chromosome's eigenpair to the LOCO eigen cache.

    ``eigen_dir`` is passed separately because the caller has already narrowed
    it out of ``LocoConfig.eigen_dir | None``.
    """
    try:
        write_eigen_files(
            eigenvalues,
            U,
            eigen_dir,
            prefix=loco.eigen_stem(chr_name),
            legacy_text=loco.legacy_text,
        )
    except OSError as e:
        raise OSError(
            f"Failed to write LOCO eigen for chromosome {chr_name} to {eigen_dir}: {e}"
        ) from e
    logger.info(f"  Wrote LOCO eigen for chr {chr_name}")


def _cached_eigen_pairs(
    eigen_cache: dict[str, tuple[Path, Path]],
    chr_names: list[str],
    *,
    n_valid: int,
    partitions: dict[str, np.ndarray],
    show_progress: bool,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield per-chromosome eigenpairs read from a complete eigen cache.

    No kinship is computed on this path: the cache was written by an earlier
    run and validated by the caller before the loop starts.
    """
    for chr_idx, chr_name in enumerate(chr_names):
        d_path, u_path = eigen_cache[chr_name]
        if show_progress:
            logger.info(
                f"LOCO: chromosome {chr_name} ({chr_idx + 1}/{len(chr_names)}), "
                f"{len(partitions[chr_name])} SNPs, loading cached eigen..."
            )
        try:
            eigenvalues, U = read_eigen_files(d_path, u_path, n_samples=n_valid)
        except (ValueError, FileNotFoundError) as e:
            raise type(e)(f"LOCO eigen cache for chromosome {chr_name}: {e}") from e
        yield chr_name, eigenvalues, U


def _computed_eigen_pairs(
    loco_iter: Iterable[tuple[str, np.ndarray]],
    chr_names: list[str],
    *,
    valid_mask: np.ndarray,
    n_valid: int,
    pre_subset: bool,
    all_samples_valid: bool,
    partitions: dict[str, np.ndarray],
    check_memory: bool,
    show_progress: bool,
    loco: LocoConfig,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield per-chromosome eigenpairs by eigendecomposing streamed LOCO kinship.

    Each K_loco is optionally saved, subset to the analysed samples,
    eigendecomposed, optionally written to the eigen cache, then dropped before
    the next chromosome is pulled, so only one lives at a time.

    ``pre_subset`` records that the kinship streamer already accumulated at
    n_valid x n_valid, which lets the subset step skip a post-hoc np.ix_ copy.
    """
    for chr_idx, (chr_name, K_loco) in enumerate(loco_iter):
        if show_progress:
            logger.info(
                f"LOCO: chromosome {chr_name} ({chr_idx + 1}/{len(chr_names)}), "
                f"{len(partitions[chr_name])} SNPs, eigendecomposing..."
            )

        if loco.save_kinship:
            _save_loco_kinship(K_loco, chr_name, loco=loco, show_progress=show_progress)

        if pre_subset:
            if K_loco.shape != (n_valid, n_valid):
                raise RuntimeError(
                    f"Expected K_loco shape ({n_valid}, {n_valid}) from early "
                    f"subsetting, got {K_loco.shape}"
                )
            K_loco_valid = K_loco
            del K_loco
        elif all_samples_valid:
            K_loco_valid = K_loco
            del K_loco
        else:
            K_loco_valid = K_loco[np.ix_(valid_mask, valid_mask)]
            del K_loco
            gc.collect()

        eigenvalues, U = eigendecompose_kinship(K_loco_valid, check_memory=check_memory)
        del K_loco_valid
        gc.collect()

        if loco.write_eigen and loco.eigen_dir is not None:
            _write_loco_eigen(
                eigenvalues, U, chr_name, loco=loco, eigen_dir=loco.eigen_dir
            )

        yield chr_name, eigenvalues, U
