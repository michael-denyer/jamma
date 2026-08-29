"""The ``-gk`` path: compute a kinship matrix and write it to disk.

Split out of ``pipeline.py`` because ``-gk`` is a different program from
``-lmm``. It shares nothing with the association pipeline but the config and
the startup banner, it is reached only from ``jamma.cli``, and it returns a
``KinshipResult`` rather than a ``PipelineResult``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from loguru import logger

from jamma.io.plink import get_plink_metadata
from jamma.io.snp_list import resolve_snp_list_file
from jamma.kinship import (
    compute_kinship_streaming,
    compute_loco_kinship_streaming,
    write_kinship_matrix,
    write_loco_kinship_matrices,
)
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import write_eigen_files
from jamma.pipeline_banner import log_dataset_banner
from jamma.pipeline_config import KinshipResult, PipelineConfig

__all__ = ["compute_kinship"]


def compute_kinship(config: PipelineConfig, mode: Literal[1, 2]) -> KinshipResult:
    """Compute and write the kinship matrix (the ``-gk`` path).

    Orchestrates kinship computation end-to-end so the CLI is a thin shell,
    as ``PipelineRunner.run`` is for the ``-lmm`` path. Honours ``config.loco``
    (writes per-chromosome LOCO matrices), ``config.write_eigen``
    (eigendecomposes and writes the eigen files), and ``config.ksnps_file``
    (restricts the SNPs used). The flag-combination guards are checked here,
    before anything is read from disk, so a direct caller gets the same
    errors the CLI does.

    Args:
        config: Pipeline configuration. Only the kinship knobs are read.
        mode: Kinship mode — 1 (centered) or 2 (standardized). Both stream.

    Returns:
        A KinshipResult with the written paths, dimensions, and timing.

    Raises:
        ValueError: If ``mode`` is not 1 or 2, or if ``config.loco`` is
            combined with ``mode == 2`` or with ``config.write_eigen``.
        FileNotFoundError: If the PLINK ``.bed`` file does not exist.
    """
    if mode not in (1, 2):
        raise ValueError(f"invalid kinship mode {mode}. Use -gk 1 or -gk 2.")
    if config.loco and config.write_eigen:
        raise ValueError(
            "-eigen not supported with -gk -loco mode. "
            "Use -lmm -loco -eigen to write per-chromosome eigen files."
        )
    if config.loco and mode == 2:
        raise ValueError(
            "-gk 2 (standardized) is not supported with -loco. "
            "LOCO kinship uses centered mode (-gk 1). "
            "Use 'jamma -gk 2 -bfile X' without -loco for standardized kinship."
        )

    meta = get_plink_metadata(config.bfile)
    n_samples = meta.n_samples
    n_snps = meta.n_snps

    # GEMMA-style banner — kinship uses all samples (n_analyzed == n_total).
    log_dataset_banner(n_total=n_samples, n_analyzed=n_samples, n_snps=n_snps)

    ksnps_indices = resolve_snp_list_file(config.ksnps_file, meta.sid, "-ksnps")

    t_kinship = time.perf_counter()

    if config.loco:
        logger.info(f"Computing LOCO kinship matrices from {config.bfile}")
        loco_stream = compute_loco_kinship_streaming(
            config.bfile,
            maf_threshold=config.maf,
            miss_threshold=config.miss,
            check_memory=config.check_memory,
            show_progress=config.show_progress,
            ksnps_indices=ksnps_indices,
            mem_budget=config.mem_budget,
        )
        written_paths = write_loco_kinship_matrices(
            loco_stream,
            output_dir=config.output_dir,
            prefix=config.output_prefix,
            legacy_text=config.legacy_text,
        )
        kinship_s = time.perf_counter() - t_kinship
        logger.info(
            f"Wrote {len(written_paths)} LOCO kinship matrices in {kinship_s:.2f}s"
        )
        return KinshipResult(
            kinship_paths=written_paths,
            eigen_paths=None,
            n_samples=n_samples,
            n_snps=n_snps,
            mode=mode,
            is_loco=True,
            kinship_s=kinship_s,
        )

    if config.maf > 0.0 or config.miss < 1.0:
        logger.info(f"Filtering: MAF >= {config.maf}, missing rate <= {config.miss}")

    if mode == 1:
        logger.info("Computing centered kinship matrix (streaming)")
    else:
        logger.info("Computing standardized kinship matrix (streaming)")
    K = compute_kinship_streaming(
        config.bfile,
        maf_threshold=config.maf,
        miss_threshold=config.miss,
        check_memory=config.check_memory,
        show_progress=config.show_progress,
        ksnps_indices=ksnps_indices,
        mode="centered" if mode == 1 else "standardized",
    )

    kinship_s = time.perf_counter() - t_kinship

    kinship_base = config.output_dir / f"{config.output_prefix}.cXX.txt"
    kinship_path = write_kinship_matrix(K, kinship_base, legacy_text=config.legacy_text)
    logger.info(f"Kinship matrix written to {kinship_path}")
    n_out = K.shape[0]

    eigen_paths: tuple[Path, Path] | None = None
    if config.write_eigen:
        eigenvalues, eigenvectors = eigendecompose_kinship(
            K, check_memory=config.check_memory
        )
        del K  # K may be overwritten by eigendecomp; prevent accidental reuse
        d_path, u_path = write_eigen_files(
            eigenvalues,
            eigenvectors,
            config.output_dir,
            config.output_prefix,
            legacy_text=config.legacy_text,
        )
        eigen_paths = (d_path, u_path)
        logger.info(f"Eigenvalues written to {d_path}")
        logger.info(f"Eigenvectors written to {u_path}")

    return KinshipResult(
        kinship_paths=[kinship_path],
        eigen_paths=eigen_paths,
        n_samples=n_out,
        n_snps=n_snps,
        mode=mode,
        is_loco=False,
        kinship_s=kinship_s,
    )
