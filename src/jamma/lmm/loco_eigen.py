"""Where a LOCO run gets its per-chromosome eigenpairs, and what it writes.

Two sources, chosen once by :func:`eigen_pairs_for` rather than re-tested per
chromosome: read a validated eigen cache, or stream each chromosome's kinship
and eigendecompose it. The cache key, the manifest, the directory and the
artifact writers all live here, since the compute path is the only thing that
touches them; ``run_lmm_loco`` only iterates the result.

Eigen member names come from :class:`~jamma.lmm.eigen_io.EigenGeneration`, the
same model the whole-genome writer and every reader use.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.core.threading import get_blas_thread_count
from jamma.genotype.snp_stats import SnpStats
from jamma.kinship import compute_loco_kinship_streaming, write_kinship_matrix
from jamma.kinship.loco import LocoRetainedSet, loco_retained_set
from jamma.lmm.association_plan import DEFAULT_STATS_CHUNK, ExecutableAssociationPlan
from jamma.lmm.eigen import (
    center_kinship,
    eigendecompose_kinship_in_scope,
    plan_eigen_driver_for_machine,
)
from jamma.lmm.eigen_cache import (
    EigenCacheComponents,
    compute_eigen_cache_key,
    eigen_cache_manifest_is_valid,
    eigen_cache_manifest_path,
    read_eigen_cache_manifest,
    resolve_eigen_cache,
    write_eigen_cache_manifest,
)
from jamma.lmm.eigen_io import EigenGeneration, read_eigen_files
from jamma.lmm.eigen_plan import EigenDriverPlan
from jamma.lmm.loco_config import LocoConfig, LocoRun
from jamma.lmm.loco_workers import LocoWorkerPlan, solve_eigen_pairs
from jamma.lmm.schema import LmmConfig

EigenPairs = Generator[tuple[str, np.ndarray, np.ndarray], None, None]
"""``(chr_name, eigenvalues, U)`` per chromosome, in the run's chromosome order."""


@dataclass(frozen=True)
class EigenPairSource:
    """What ``run_lmm_loco`` iterates, and the SNP statistics that came with it.

    ``snp_stats`` holds statistics for every SNP over the analysed rows, for
    the per-chromosome association filter. The first kinship pass computes
    them, so they are readable once ``pairs`` has yielded; with cached
    eigenpairs one streamed pass computes them up front.

    Attributes:
        pairs: One eigenpair per chromosome. Consume in order; each K_loco is
            dropped before the next is pulled.
    """

    pairs: EigenPairs
    _snp_stats: Callable[[], SnpStats]

    @property
    def snp_stats(self) -> SnpStats:
        return self._snp_stats()


@dataclass(frozen=True)
class _EigenCacheWrite:
    """Everything the compute path needs to persist a cache, narrowed once.

    Built only when ``write_eigen`` is set, from a ``LocoConfig`` whose
    ``eigen_dir`` ``__post_init__`` has already guaranteed, so nothing
    downstream re-tests an Optional.
    """

    generation: EigenGeneration
    key: str
    components: EigenCacheComponents


def plan_loco_eigen_driver(
    execution: ExecutableAssociationPlan, available_gb: float
) -> EigenDriverPlan:
    """Plan the per-chromosome eigen driver against what the retained set leaves.

    The kinship stream's retained set stays live while each chromosome
    decomposes, so the driver is chosen against the headroom and budget left
    after it. A negative remainder needs no clamp: the planner's own fit
    checks already read it as "does not fit".
    """
    retained_gb = loco_retained_set_for(execution).while_consuming_gb
    budget_gb = execution.mem_budget_gb
    return plan_eigen_driver_for_machine(
        execution.n_samples,
        available_gb - retained_gb,
        budget_gb=None if budget_gb is None else budget_gb - retained_gb,
        inplace_blocker=None,
    )


def loco_retained_set_for(execution: ExecutableAssociationPlan) -> LocoRetainedSet:
    """The kinship stream's retained set for this run's resolved kinship shape."""
    return loco_retained_set(
        execution.resolved_kinship.n_samples,
        execution.n_input_samples,
        DEFAULT_STATS_CHUNK,
        genotype_encoding=execution.genotype_encoding,
    )


def eigen_pairs_for(
    run: LocoRun, chromosomes: dict[str, np.ndarray], workers: LocoWorkerPlan
) -> EigenPairSource:
    """Choose the eigenpair source for one LOCO run.

    A complete, key-validated cache under ``loco.eigen_dir`` is read as-is,
    unless ``loco.write_eigen`` asks for a rewrite. Otherwise each chromosome's
    kinship is streamed and eigendecomposed; with ``write_eigen`` the pairs
    and, once every chromosome has been consumed, the manifest are written.

    Args:
        run: The resolved run.
        chromosomes: chr_name -> global SNP indices, in the order the run
            iterates them. The indices only feed progress output.
        workers: How many chromosomes decompose at once, from
            ``plan_loco_workers``; its ``consumer_gb`` is what the kinship
            streamer reserves for the eigen consumer.
    """
    loco, config = run.loco, run.config
    rows = run.analysed_rows
    all_samples_valid = len(rows) == run.dataset.n_samples

    cache_write: _EigenCacheWrite | None = None
    if loco.eigen_dir is not None:
        key, components = compute_eigen_cache_key(
            run.dataset,
            maf_threshold=config.maf_threshold,
            miss_threshold=config.miss_threshold,
            valid_mask=run.samples.valid_mask,
            ksnps_indices=loco.ksnps_indices,
            info_threshold=loco.info_threshold,
        )
        if loco.write_eigen:
            cache_write = _EigenCacheWrite(
                EigenGeneration(loco.eigen_dir, loco.prefix), key, components
            )
        else:
            cache = _validated_eigen_cache(
                loco, list(chromosomes), key, eigen_dir=loco.eigen_dir
            )
            if cache is not None:
                logger.info("LOCO workers: 0 (cached eigenpairs)")
                if loco.kinship_output_dir is not None:
                    logger.warning(
                        "kinship_output_dir ignored when using cached eigen "
                        "files (kinship is not computed)"
                    )
                pairs = _cached_eigen_pairs(
                    cache,
                    chromosomes,
                    n_valid=len(rows),
                    show_progress=config.show_progress,
                )
                stats = run.dataset.stats(
                    None if all_samples_valid else rows,
                    block_size=DEFAULT_STATS_CHUNK,
                    progress="LOCO: SNP statistics" if config.show_progress else None,
                )
                if stats.n_unexpected > 0:
                    logger.warning(
                        f"Genotype validation: {stats.n_unexpected} values outside "
                        "expected range {0, 1, 2, NaN}"
                    )
                return EigenPairSource(pairs, lambda: stats)

    logger.info(workers.describe())
    kinship_is_analysed = run.execution.resolved_kinship.n_samples == len(rows)
    stream = compute_loco_kinship_streaming(
        run.dataset,
        chunk_size=DEFAULT_STATS_CHUNK,
        maf_threshold=config.maf_threshold,
        miss_threshold=config.miss_threshold,
        check_memory=config.check_memory,
        show_progress=config.show_progress,
        ksnps_indices=loco.ksnps_indices,
        valid_indices=None if all_samples_valid or not kinship_is_analysed else rows,
        filter_sample_indices=None if all_samples_valid else rows,
        mem_budget=config.mem_budget,
        consumer_gb=workers.consumer_gb,
        info_threshold=loco.info_threshold,
    )
    pairs = _computed_eigen_pairs(
        stream,
        chromosomes,
        subset_rows=None if kinship_is_analysed else rows,
        config=config,
        loco=loco,
        cache_write=cache_write,
        eigen_plan=run.eigen_plan,
        workers=workers.workers,
    )
    return EigenPairSource(pairs, lambda: stream.snp_stats)


def _validated_eigen_cache(
    loco: LocoConfig, chr_names: list[str], key: str, *, eigen_dir: Path
) -> dict[str, tuple[Path, Path]] | None:
    """A complete per-chromosome cache whose manifest matches ``key``, or None.

    ``eigen_dir`` is ``loco.eigen_dir`` already narrowed by the caller.
    """
    manifest = read_eigen_cache_manifest(eigen_dir, loco.prefix)
    if manifest is None:
        return None
    ok, reason = eigen_cache_manifest_is_valid(
        manifest, eigen_cache_manifest_path(eigen_dir, loco.prefix), key
    )
    if not ok:
        logger.warning(
            f"LOCO eigen cache in {eigen_dir} is stale or unverifiable "
            f"({reason}). Kinship and eigendecomposition will be recomputed."
        )
        return None
    cache = resolve_eigen_cache(manifest, eigen_dir, loco.prefix, chr_names)
    if cache is None:
        logger.warning(
            f"LOCO eigen cache manifest in {eigen_dir} is incomplete or unsafe"
        )
        return None
    logger.info(
        f"Found complete LOCO eigen cache in {eigen_dir} "
        f"({len(cache)} chromosomes). "
        f"Skipping kinship computation and eigendecomp."
    )
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
    generation: EigenGeneration,
    legacy_text: bool,
) -> tuple[Path, Path]:
    """Persist one chromosome's eigenpair to the LOCO eigen cache."""
    try:
        paths = generation.write_member(
            chr_name, eigenvalues, U, legacy_text=legacy_text
        )
    except OSError as e:
        raise OSError(
            f"Failed to write LOCO eigen for chromosome {chr_name} to "
            f"{generation.directory}: {e}"
        ) from e
    logger.info(f"  Wrote LOCO eigen for chr {chr_name}")
    return paths


def _cached_eigen_pairs(
    eigen_cache: dict[str, tuple[Path, Path]],
    chromosomes: dict[str, np.ndarray],
    *,
    n_valid: int,
    show_progress: bool,
) -> EigenPairs:
    """Yield per-chromosome eigenpairs read from a complete eigen cache.

    No kinship is computed on this path: the cache was written by an earlier
    run and validated by the caller before the loop starts.
    """
    for chr_idx, (chr_name, snp_indices) in enumerate(chromosomes.items()):
        d_path, u_path = eigen_cache[chr_name]
        if show_progress:
            logger.info(
                f"LOCO: chromosome {chr_name} ({chr_idx + 1}/{len(chromosomes)}), "
                f"{len(snp_indices)} SNPs, loading cached eigen..."
            )
        try:
            eigenvalues, U = read_eigen_files(d_path, u_path, n_samples=n_valid)
        except (ValueError, FileNotFoundError) as e:
            raise type(e)(f"LOCO eigen cache for chromosome {chr_name}: {e}") from e
        yield chr_name, eigenvalues, U
        del eigenvalues, U


def _analysed_subset(
    K_loco: np.ndarray, subset_rows: np.ndarray | None, *, copy: bool
) -> np.ndarray:
    """K_loco over the analysed samples, owned by the caller when ``copy`` is set.

    ``subset_rows`` is None when the stream already accumulated over the
    analysed samples. Its buffer is then returned as-is when the caller
    consumes it before the next pull; ``copy`` makes an owned array of it
    instead, for a worker that outlives that pull. The np.ix_ subset is a
    fresh array either way.
    """
    if subset_rows is not None:
        return K_loco[np.ix_(subset_rows, subset_rows)]
    return K_loco.copy() if copy else K_loco


def _computed_eigen_pairs(
    loco_iter: Iterable[tuple[str, np.ndarray]],
    chromosomes: dict[str, np.ndarray],
    *,
    subset_rows: np.ndarray | None,
    config: LmmConfig,
    loco: LocoConfig,
    cache_write: _EigenCacheWrite | None,
    eigen_plan: EigenDriverPlan,
    workers: int = 1,
    solve: Callable[
        ..., tuple[np.ndarray, np.ndarray]
    ] = eigendecompose_kinship_in_scope,
) -> EigenPairs:
    """Yield per-chromosome eigenpairs by eigendecomposing streamed LOCO kinship.

    Each K_loco is optionally saved, subset to ``subset_rows`` unless the
    streamer already accumulated over the analysed samples, and
    eigendecomposed. Cache artifacts are written as the consumer advances.

    Concurrent workers receive owned copies of the stream's reusable buffer
    and pairs come out in chromosome order with at most ``workers`` solves in
    flight. ``solve_eigen_pairs`` owns scheduling, the one BLAS scope every
    solve runs under, and worker cleanup.

    With ``cache_write``, every pair is written under one fresh generation.
    The manifest is replaced only after the consumer drains every chromosome,
    so interruption leaves the prior generation committed and readable.
    """
    show_progress = config.show_progress
    members: dict[str, tuple[Path, Path]] = {}
    if cache_write is not None:
        eigen_dir = cache_write.generation.directory
        try:
            eigen_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"Cannot create eigen cache directory {eigen_dir}: {e}"
            ) from e

    n_threads = get_blas_thread_count()

    def decompose(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center_kinship(K)
        return solve(
            K,
            check_memory=config.check_memory,
            mem_budget=config.mem_budget,
            eigen_plan=eigen_plan,
            show_progress=show_progress and workers == 1,
            n_threads=n_threads,
        )

    def inputs() -> Generator[tuple[str, np.ndarray], None, None]:
        chr_idx = 0
        for chr_name, K_loco in loco_iter:
            chr_idx += 1  # noqa: SIM113 -- enumerate retains the preceding matrix
            if show_progress:
                logger.info(
                    f"LOCO: chromosome {chr_name} ({chr_idx}/{len(chromosomes)}), "
                    f"{len(chromosomes[chr_name])} SNPs, eigendecomposing..."
                )
            if loco.kinship_output_dir is not None:
                _save_loco_kinship(
                    K_loco, chr_name, loco=loco, show_progress=show_progress
                )
            # Yield the owned copy directly so this frame does not retain it
            # after a non-inplace solver returns a separate eigenvector array.
            yield (
                chr_name,
                _analysed_subset(K_loco, subset_rows, copy=workers > 1),
            )
            del K_loco

    with closing(
        solve_eigen_pairs(inputs(), decompose, workers=workers, n_threads=n_threads)
    ) as pairs:
        # Do not use enumerate: it retains the preceding eigenvector matrix.
        for chr_name, eigenvalues, U in pairs:
            if cache_write is not None:
                members[chr_name] = _write_loco_eigen(
                    eigenvalues,
                    U,
                    chr_name,
                    generation=cache_write.generation,
                    legacy_text=loco.legacy_text,
                )
            yield chr_name, eigenvalues, U
            del eigenvalues, U

    if cache_write is not None:
        write_eigen_cache_manifest(
            cache_write.generation,
            cache_write.key,
            components=cache_write.components,
            members=members,
        )
        logger.info(
            f"Wrote LOCO eigen cache manifest to {cache_write.generation.directory}"
        )
