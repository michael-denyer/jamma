"""Where a LOCO run gets its per-chromosome eigenpairs, and what it writes.

Two sources, chosen once by :func:`eigen_pairs_for` rather than re-tested per
chromosome: read a validated eigen cache, or stream each chromosome's kinship
and eigendecompose it. The cache key, the manifest, the directory and the
artifact writers all live here, since the compute path is the only thing that
touches them; ``run_lmm_loco`` only iterates the result.

Filenames come from :class:`~jamma.lmm.loco_config.LocoConfig`, never from a
literal in this module.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.kinship import (
    SnpStatsCache,
    compute_loco_kinship_streaming,
    write_kinship_matrix,
)
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_cache import (
    EigenCacheComponents,
    compute_eigen_cache_key,
    eigen_cache_is_valid,
    invalidate_eigen_cache_manifest,
    write_eigen_cache_manifest,
)
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
from jamma.lmm.loco_config import LocoConfig

EigenPairs = Iterator[tuple[str, np.ndarray, np.ndarray]]
"""``(chr_name, eigenvalues, U)`` per chromosome, in ``chr_names`` order."""


@dataclass(frozen=True)
class EigenPairSource:
    """What ``run_lmm_loco`` iterates, and the SNP statistics that came with it.

    Attributes:
        pairs: One eigenpair per chromosome. Consume in order; each K_loco is
            dropped before the next is pulled.
        snp_stats: Kinship PASS-1 statistics over all samples, for the
            per-chromosome association filter. None when the eigenpairs came
            from the cache, since no kinship pass ran.
    """

    pairs: EigenPairs
    snp_stats: SnpStatsCache | None


@dataclass(frozen=True)
class _EigenCacheWrite:
    """Everything the compute path needs to persist a cache, narrowed once.

    Built only when ``write_eigen`` is set, from a ``LocoConfig`` whose
    ``eigen_dir`` ``__post_init__`` has already guaranteed, so nothing
    downstream re-tests an Optional.
    """

    eigen_dir: Path
    prefix: str
    key: str
    components: EigenCacheComponents


def eigen_pairs_for(
    bed_path: Path,
    chr_names: list[str],
    *,
    loco: LocoConfig,
    maf_threshold: float,
    miss_threshold: float,
    valid_mask: np.ndarray,
    partitions: dict[str, np.ndarray],
    check_memory: bool,
    show_progress: bool,
) -> EigenPairSource:
    """Choose the eigenpair source for one LOCO run.

    A complete, key-validated cache under ``loco.eigen_dir`` is read as-is,
    unless ``loco.write_eigen`` asks for a rewrite. Otherwise each chromosome's
    kinship is streamed and eigendecomposed; with ``write_eigen`` the pairs
    and, once every chromosome has been consumed, the manifest are written.

    Args:
        bed_path: PLINK file prefix (without extension).
        chr_names: Chromosomes in the order the run iterates them.
        loco: Artifact locations and naming, -ksnps restriction, chunk width.
        maf_threshold: Minimum MAF for the kinship SNP filter and cache key.
        miss_threshold: Maximum missing rate, same two uses.
        valid_mask: Boolean (n_samples_total,) analysed-sample mask.
        partitions: chr_name -> global SNP indices, for progress output.
        check_memory: Passed to the kinship streamer and eigendecomposition.
        show_progress: Whether to log per-chromosome progress.
    """
    n_valid = int(np.sum(valid_mask))
    all_samples_valid = n_valid == len(valid_mask)

    cache_write: _EigenCacheWrite | None = None
    if loco.eigen_dir is not None:
        key, components = compute_eigen_cache_key(
            bed_path,
            maf_threshold=maf_threshold,
            miss_threshold=miss_threshold,
            valid_mask=valid_mask,
            ksnps_indices=loco.ksnps_indices,
        )
        if loco.write_eigen:
            cache_write = _EigenCacheWrite(loco.eigen_dir, loco.prefix, key, components)
        else:
            cache = _validated_eigen_cache(
                loco, chr_names, key, eigen_dir=loco.eigen_dir
            )
            if cache is not None:
                if loco.kinship_output_dir is not None:
                    logger.warning(
                        "kinship_output_dir ignored when using cached eigen "
                        "files (kinship is not computed)"
                    )
                logger.warning(
                    "Using cached eigen: SNP filtering will use "
                    "valid-sample-only statistics (not all-sample stats "
                    "from kinship pass). This may produce slightly "
                    "different SNP filter sets compared to the original "
                    "compute run."
                )
                pairs = _cached_eigen_pairs(
                    cache,
                    chr_names,
                    n_valid=n_valid,
                    partitions=partitions,
                    show_progress=show_progress,
                )
                return EigenPairSource(pairs, snp_stats=None)

    # Without a kinship file to save, accumulate at n_valid x n_valid rather
    # than materialising n_samples^2 for a post-hoc subset.
    kinship_valid_indices = (
        None
        if all_samples_valid or loco.kinship_output_dir is not None
        else np.where(valid_mask)[0]
    )
    stream = compute_loco_kinship_streaming(
        bed_path,
        maf_threshold=maf_threshold,
        miss_threshold=miss_threshold,
        check_memory=check_memory,
        show_progress=show_progress,
        ksnps_indices=loco.ksnps_indices,
        valid_indices=kinship_valid_indices,
    )
    pairs = _computed_eigen_pairs(
        stream,
        chr_names,
        valid_mask=valid_mask,
        n_valid=n_valid,
        pre_subset=kinship_valid_indices is not None,
        all_samples_valid=all_samples_valid,
        partitions=partitions,
        check_memory=check_memory,
        show_progress=show_progress,
        loco=loco,
        cache_write=cache_write,
    )
    return EigenPairSource(pairs, snp_stats=stream.snp_stats)


def _validated_eigen_cache(
    loco: LocoConfig, chr_names: list[str], key: str, *, eigen_dir: Path
) -> dict[str, tuple[Path, Path]] | None:
    """A complete per-chromosome cache whose manifest matches ``key``, or None.

    ``eigen_dir`` is ``loco.eigen_dir`` already narrowed by the caller.
    """
    cache = _find_loco_eigen_cache(loco, chr_names)
    if cache is None:
        return None
    ok, reason = eigen_cache_is_valid(eigen_dir, loco.prefix, key)
    if not ok:
        logger.warning(
            f"LOCO eigen cache in {eigen_dir} is stale or unverifiable "
            f"({reason}). Kinship and eigendecomposition will be recomputed."
        )
        return None
    logger.info(
        f"Found complete LOCO eigen cache in {eigen_dir} "
        f"({len(cache)} chromosomes). "
        f"Skipping kinship computation and eigendecomp."
    )
    return cache


def _find_loco_eigen_cache(
    loco: LocoConfig,
    chr_names: list[str],
) -> dict[str, tuple[Path, Path]] | None:
    """Check for a complete set of per-chromosome cached eigen files.

    File naming comes from ``loco.eigen_paths()``, the same method the writer
    builds its names with, so the two cannot drift.

    Dimension validation is deferred to the per-chromosome load, where
    ``read_eigen_files(n_samples=...)`` raises ``ValueError`` on mismatch.
    This avoids loading all eigen data eagerly just to check dimensions.

    Args:
        loco: LOCO config supplying eigen_dir, prefix and legacy_text.
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
    """Persist one chromosome's eigenpair to the LOCO eigen cache."""
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
) -> EigenPairs:
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
    cache_write: _EigenCacheWrite | None,
) -> EigenPairs:
    """Yield per-chromosome eigenpairs by eigendecomposing streamed LOCO kinship.

    Each K_loco is optionally saved, subset to the analysed samples,
    eigendecomposed, optionally written to the eigen cache, then dropped before
    the next chromosome is pulled, so only one lives at a time.

    With ``cache_write``, the stale manifest is removed before the first pair
    and the fresh one written only after the consumer has drained every
    chromosome, so an interrupted rewrite leaves no manifest and the next
    read recomputes rather than trusting a half-rewritten cache.

    ``pre_subset`` records that the kinship streamer already accumulated at
    n_valid x n_valid, which lets the subset step skip a post-hoc np.ix_ copy.
    """
    if cache_write is not None:
        try:
            cache_write.eigen_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"Cannot create eigen cache directory {cache_write.eigen_dir}: {e}"
            ) from e
        invalidate_eigen_cache_manifest(cache_write.eigen_dir, cache_write.prefix)

    for chr_idx, (chr_name, K_loco) in enumerate(loco_iter):
        if show_progress:
            logger.info(
                f"LOCO: chromosome {chr_name} ({chr_idx + 1}/{len(chr_names)}), "
                f"{len(partitions[chr_name])} SNPs, eigendecomposing..."
            )

        if loco.kinship_output_dir is not None:
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

        if cache_write is not None:
            _write_loco_eigen(
                eigenvalues, U, chr_name, loco=loco, eigen_dir=cache_write.eigen_dir
            )

        yield chr_name, eigenvalues, U

    if cache_write is not None:
        write_eigen_cache_manifest(
            cache_write.eigen_dir,
            cache_write.prefix,
            cache_write.key,
            components=cache_write.components,
        )
        logger.info(f"Wrote LOCO eigen cache manifest to {cache_write.eigen_dir}")
