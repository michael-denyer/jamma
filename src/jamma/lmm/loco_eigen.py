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
import uuid
from collections import deque
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from loguru import logger

from jamma.core import memory
from jamma.core.eigen_plan import EigenDriverPlan
from jamma.core.threading import get_physical_core_count
from jamma.kinship import (
    SnpStatsCache,
    compute_loco_kinship_streaming,
    write_kinship_matrix,
)
from jamma.kinship.loco import LocoRetainedSet, loco_retained_set
from jamma.lmm.association_plan import DEFAULT_STATS_CHUNK, ExecutableAssociationPlan
from jamma.lmm.eigen import (
    center_kinship,
    eigendecompose_kinship,
    plan_eigen_driver_for_machine,
)
from jamma.lmm.eigen_cache import (
    EigenCacheComponents,
    compute_eigen_cache_key,
    eigen_cache_manifest_is_valid,
    eigen_cache_manifest_path,
    loco_eigen_paths_from_manifest,
    read_eigen_cache_manifest,
    write_eigen_cache_manifest,
)
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_generation_members
from jamma.lmm.loco_config import LocoConfig

EigenPairs = Generator[tuple[str, np.ndarray, np.ndarray], None, None]
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
    generation: str


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
        inplace_eligible=True,
    )


def loco_retained_set_for(execution: ExecutableAssociationPlan) -> LocoRetainedSet:
    """The kinship stream's retained set for this run's resolved kinship shape."""
    return loco_retained_set(
        execution.resolved_kinship.n_samples,
        execution.n_input_samples,
        DEFAULT_STATS_CHUNK,
    )


class LocoWorkerPlan(NamedTuple):
    """How many chromosomes eigendecompose at once, and what that reserves.

    Attributes:
        workers: Concurrent eigendecompositions. One is the sequential path,
            unchanged from before workers existed.
        memory_allows: The memory ceiling on its own, before the request, the
            chromosome count and the core count cap it. Logged so a clamp
            names its cause.
        cores: Physical cores, the third cap.
        consumer_gb: What the kinship stream reserves for the eigen consumer:
            the driver's peak alone sequentially, else the larger of one
            owned K_loco copy plus one driver peak per worker, and the
            association pass plus the ``workers - 1`` still in flight.
    """

    workers: int
    memory_allows: int
    cores: int
    consumer_gb: float


def plan_loco_workers(
    requested: int,
    *,
    n_chr: int,
    retained: LocoRetainedSet,
    eigen_plan: EigenDriverPlan,
    available_gb: float,
    budget_gb: float | None,
    association_gb: float,
    cores: int | None = None,
) -> LocoWorkerPlan:
    """Pick how many chromosomes eigendecompose at once.

    Pure sizing math against the two ceilings ``plan_loco_passes`` applies:
    ``memory.fits`` against ``available_gb`` and, when set, ``budget_gb``.
    The stream's retained set stays live underneath; each worker adds one
    owned copy of the streamed K_loco (the stream's buffer is overwritten on
    the next pull) and the eigen driver's peak on top of it. The consumer's
    peak is the larger of two moments: every worker solving at once, and the
    association pass running while the other ``workers - 1`` still solve.
    The count is the largest that fits, capped by the request, the
    chromosome count and the physical cores, and never below one. At one
    worker nothing is copied, so the reservation is the driver's peak alone.

    Args:
        requested: ``JAMMA_LOCO_WORKERS``, already parsed and at least one.
        n_chr: Chromosomes in the run; no point in more workers than that.
        retained: The stream's live matrices and disk buffer.
        eigen_plan: The driver every chromosome runs; its peak is per worker.
        available_gb: Free RAM in GB (the caller reads psutil once).
        budget_gb: User ceiling in GB without the physical-RAM margin, or None.
        association_gb: Peak of one chromosome's association pass, which
            overlaps the solves still in flight.
        cores: Physical core cap, or None to read the machine.

    Returns:
        The worker count, the memory-only bound, the core cap and the
        reservation the stream must hold for the consumer.
    """
    per_worker_gb = retained.matrix_gb + eigen_plan.required_gb

    def consumer(workers: int) -> float:
        if workers == 1:
            return eigen_plan.required_gb
        return max(
            workers * per_worker_gb, association_gb + (workers - 1) * per_worker_gb
        )

    def fits(workers: int) -> bool:
        peak_gb = retained.while_consuming_gb + consumer(workers)
        within_budget = budget_gb is None or peak_gb <= budget_gb
        return within_budget and memory.fits(peak_gb, available_gb)

    memory_allows = next((w for w in range(requested, 1, -1) if fits(w)), 1)
    if cores is None:
        cores = get_physical_core_count()
    workers = max(1, min(requested, n_chr, cores, memory_allows))
    return LocoWorkerPlan(workers, memory_allows, cores, consumer(workers))


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
    eigen_plan: EigenDriverPlan,
    workers: LocoWorkerPlan,
    mem_budget: float | None = None,
    association_peak_gb: float = 0.0,
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
        eigen_plan: The driver every chromosome's decomposition runs, from
            ``plan_loco_eigen_driver``.
        workers: How many chromosomes decompose at once, from
            ``plan_loco_workers``; its ``consumer_gb`` is what the kinship
            streamer reserves for the eigen consumer.
        mem_budget: User-set ceiling in GB, or None for no ceiling. Reaches the
            kinship streamer's veto and each decomposition's gate.
        association_peak_gb: Peak the association phase will hold, so the kinship
            streamer sizes its chromosome batch around the larger of that and the
            eigen consumer.
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
            cache_write = _EigenCacheWrite(
                loco.eigen_dir, loco.prefix, key, components, uuid.uuid4().hex
            )
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
        chunk_size=DEFAULT_STATS_CHUNK,
        maf_threshold=maf_threshold,
        miss_threshold=miss_threshold,
        check_memory=check_memory,
        show_progress=show_progress,
        ksnps_indices=loco.ksnps_indices,
        valid_indices=kinship_valid_indices,
        filter_sample_indices=None if all_samples_valid else np.where(valid_mask)[0],
        mem_budget=mem_budget,
        consumer_gb=max(workers.consumer_gb, association_peak_gb),
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
        eigen_plan=eigen_plan,
        mem_budget=mem_budget,
        workers=workers.workers,
    )
    return EigenPairSource(pairs, snp_stats=stream.snp_stats)


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
    cache = _find_loco_eigen_cache(loco, chr_names, manifest=manifest)
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


def _find_loco_eigen_cache(
    loco: LocoConfig,
    chr_names: list[str],
    *,
    manifest: dict[str, object] | None = None,
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

    if manifest is None:
        manifest = read_eigen_cache_manifest(loco.eigen_dir, loco.prefix)
    if manifest is None:
        return None
    return loco_eigen_paths_from_manifest(
        loco.eigen_dir, loco.prefix, chr_names, manifest
    )


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
    generation: str,
) -> tuple[Path, Path]:
    """Persist one chromosome's eigenpair to the LOCO eigen cache."""
    try:
        paths = write_eigen_generation_members(
            eigenvalues,
            U,
            eigen_dir,
            prefix=loco.prefix,
            generation=generation,
            legacy_text=loco.legacy_text,
            label=f"loco.chr{chr_name}",
        )
    except OSError as e:
        raise OSError(
            f"Failed to write LOCO eigen for chromosome {chr_name} to {eigen_dir}: {e}"
        ) from e
    logger.info(f"  Wrote LOCO eigen for chr {chr_name}")
    return paths


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
        del eigenvalues, U


def _analysed_subset(
    K_loco: np.ndarray,
    *,
    valid_mask: np.ndarray,
    n_valid: int,
    pre_subset: bool,
    all_samples_valid: bool,
    copy: bool,
) -> np.ndarray:
    """K_loco over the analysed samples, owned by the caller when ``copy`` is set.

    The stream's buffer is returned as-is when it already has the analysed
    shape and the caller consumes it before the next pull; ``copy`` makes an
    owned array of it instead, for a worker that outlives that pull. The
    np.ix_ subset is a fresh array either way.
    """
    if pre_subset:
        if K_loco.shape != (n_valid, n_valid):
            raise RuntimeError(
                f"Expected K_loco shape ({n_valid}, {n_valid}) from early "
                f"subsetting, got {K_loco.shape}"
            )
    elif not all_samples_valid:
        return K_loco[np.ix_(valid_mask, valid_mask)]
    return K_loco.copy() if copy else K_loco


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
    eigen_plan: EigenDriverPlan,
    mem_budget: float | None,
    workers: int = 1,
    solve: Callable[..., tuple[np.ndarray, np.ndarray]] = eigendecompose_kinship,
) -> EigenPairs:
    """Yield per-chromosome eigenpairs by eigendecomposing streamed LOCO kinship.

    Each K_loco is optionally saved, subset to the analysed samples,
    eigendecomposed, optionally written to the eigen cache, then dropped before
    the next chromosome is pulled, so only one lives at a time.

    With ``workers`` above one, that many chromosomes decompose at once on a
    thread pool. The stream yields one shared buffer, overwritten on the next
    pull, so each submitted chromosome is copied first; that copy is the
    whole per-worker footprint, since the driver overwrites it with U. Pairs
    still come out in chromosome order, the oldest future resolved first,
    with at most ``workers`` in flight. ``jlinalg`` releases the GIL around
    the solver, so the solves overlap. Each solve keeps its own BLAS thread
    count. On Accelerate every solve is single-threaded and the results are
    identical to the sequential path's; MKL and OpenBLAS split each solve
    across one shared pool, and concurrent callers move where that split
    lands, so results there agree to rounding (last-bit differences in U),
    and pairing ``JAMMA_LOCO_WORKERS=W`` with ``JAMMA_BLAS_THREADS=cores//W``
    avoids oversubscription. Accelerate runs DSYEVD on one core whatever the
    setting, which is what makes the overlap worth having there. The first
    failed solve propagates in chromosome order; the rest are cancelled and
    the pool is shut down, also when the consumer closes the generator early.

    ``solve`` is ``eigendecompose_kinship``; tests inject a stand-in with the
    same keyword signature to observe the overlap without patching.

    With ``cache_write``, every pair is written under one fresh generation.
    The manifest is replaced only after the consumer drains every chromosome,
    so interruption leaves the prior generation committed and readable.

    ``pre_subset`` records that the kinship streamer already accumulated at
    n_valid x n_valid, which lets the subset step skip a post-hoc np.ix_ copy.
    """
    artifacts: dict[str, dict[str, str]] = {}
    if cache_write is not None:
        try:
            cache_write.eigen_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"Cannot create eigen cache directory {cache_write.eigen_dir}: {e}"
            ) from e

    def decompose(K_loco_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center_kinship(K_loco_valid)
        return solve(
            K_loco_valid,
            check_memory=check_memory,
            mem_budget=mem_budget,
            eigen_plan=eigen_plan,
            show_progress=show_progress and workers == 1,
        )

    def publish(chr_name: str, eigenvalues: np.ndarray, U: np.ndarray) -> None:
        if cache_write is None:
            return
        d_path, u_path = _write_loco_eigen(
            eigenvalues,
            U,
            chr_name,
            loco=loco,
            eigen_dir=cache_write.eigen_dir,
            generation=cache_write.generation,
        )
        artifacts[chr_name] = {"eigenD": d_path.name, "eigenU": u_path.name}

    pending: deque[tuple[str, Future[tuple[np.ndarray, np.ndarray]]]] = deque()

    def oldest() -> tuple[str, np.ndarray, np.ndarray]:
        # Built here rather than in the loop so the generator frame holds no
        # name for the yielded eigenvectors once the consumer has them.
        chr_name, future = pending.popleft()
        eigenvalues, U = future.result()
        del future
        publish(chr_name, eigenvalues, U)
        return chr_name, eigenvalues, U

    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        # No enumerate() here. CPython's enumerate holds its previous result
        # tuple, and through it the previous U, until this generator yields the
        # next item, so chromosome c's eigenvectors would stay live through
        # c+1's eigendecomposition. The counter feeds the progress line below.
        chr_idx = -1
        for chr_name, K_loco in loco_iter:
            chr_idx += 1
            if show_progress:
                logger.info(
                    f"LOCO: chromosome {chr_name} ({chr_idx + 1}/{len(chr_names)}), "
                    f"{len(partitions[chr_name])} SNPs, eigendecomposing..."
                )

            if loco.kinship_output_dir is not None:
                _save_loco_kinship(
                    K_loco, chr_name, loco=loco, show_progress=show_progress
                )

            K_loco_valid = _analysed_subset(
                K_loco,
                valid_mask=valid_mask,
                n_valid=n_valid,
                pre_subset=pre_subset,
                all_samples_valid=all_samples_valid,
                copy=pool is not None,
            )
            del K_loco

            if pool is None:
                eigenvalues, U = decompose(K_loco_valid)
                del K_loco_valid
                gc.collect()
                publish(chr_name, eigenvalues, U)
                yield chr_name, eigenvalues, U
                del eigenvalues, U
                continue

            pending.append((chr_name, pool.submit(decompose, K_loco_valid)))
            del K_loco_valid
            if len(pending) == workers:
                yield oldest()

        while pending:
            yield oldest()
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    if cache_write is not None:
        write_eigen_cache_manifest(
            cache_write.eigen_dir,
            cache_write.prefix,
            cache_write.key,
            components=cache_write.components,
            generation=cache_write.generation,
            artifacts=artifacts,
        )
        logger.info(f"Wrote LOCO eigen cache manifest to {cache_write.eigen_dir}")
