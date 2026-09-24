"""Guard/precondition tests for the shared NumPy LMM chunk runner.

These cover the cheap, isolated failure paths that the end-to-end parity
suites never exercise: the ``run_lmm_chunk_source_numpy_group`` argument
preconditions.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import psutil
import pytest
from loguru import logger

from jamma.genotype.snp_stats import SnpSelection
from jamma.genotype.variants import SnpMeta
from jamma.lmm import accel
from jamma.lmm.chunk_runner_numpy import (
    PhenotypeChunkJob,
    RawLmmChunk,
    run_lmm_chunk_source_numpy_group,
)
from jamma.lmm.chunk_sizing import LmmChunkPlan
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.genotype_source import PreparedGenotypes, SampleBasis
from jamma.lmm.prepare_common import NullFit, RotatedBasis
from jamma.lmm.schema import LmmConfig
from jamma.lmm.workspace import WorkspaceSpec
from tests.support import requires_c

pytestmark = pytest.mark.tier0


@requires_c
def test_association_logs_the_workspace_thread_cap_used_by_real_kernels(monkeypatch):
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 18)
    n_samples, n_snps = 20, 6
    rng = np.random.default_rng(21)
    G = rng.integers(0, 3, size=(n_samples, n_snps)).astype(float)
    config = LmmConfig(lmm_mode=1, show_progress=False)
    fit = replace(
        _null_fit(n_samples, Uty=rng.normal(size=n_samples)),
        Hi_eval_null=np.full(n_samples, 0.5),
    )
    genotypes = replace(
        _prepared_genotypes(n_samples, n_snps),
        chunk_factory=lambda width: (
            RawLmmChunk(G[:, i : i + width].copy(), i, i + width)
            for i in range(0, n_snps, width)
        ),
    )
    workspace = WorkspaceSpec.build(
        DispatchPath.FUSED,
        config.lmm_mode,
        n_samples,
        n_samples,
        1,
        config.n_grid,
        config.n_refine,
        2,
    )
    written = []
    messages = []
    sink = logger.add(messages.append, level="INFO", format="{message}")
    try:
        run_lmm_chunk_source_numpy_group(
            genotypes=genotypes,
            basis=_basis(n_samples),
            jobs=(
                PhenotypeChunkJob(
                    fit,
                    lambda arrays, _start, _end: written.append(arrays["betas"]),
                ),
            ),
            config=config,
            dispatch=DispatchPath.FUSED,
            chunks=LmmChunkPlan(2, 3, 2, True),
            workspace=workspace,
        )
    finally:
        logger.remove(sink)
    expected = 2 if accel.HAS_OPENMP else 1
    line = next(msg for msg in messages if msg.startswith("Association threads:"))
    assert line.endswith(f" | C-ext={expected}\n")
    assert sum(len(betas) for betas in written) == n_snps
    assert all(np.isfinite(betas).all() for betas in written)


# ---------------------------------------------------------------------------
# run_lmm_chunk_source_numpy_group preconditions
# ---------------------------------------------------------------------------


def _prepared_genotypes(n_samples: int, n_filtered: int) -> PreparedGenotypes:
    indices = np.arange(n_filtered, dtype=np.intp)
    return PreparedGenotypes(
        snp_meta=SnpMeta(
            chr=np.full(n_filtered, "1"),
            rs=np.array([f"rs{i}" for i in indices]),
            pos=indices,
            a1=np.full(n_filtered, "A"),
            a0=np.full(n_filtered, "G"),
        ),
        selection=SnpSelection(
            indices=indices,
            local_indices=indices,
            mask=np.ones(n_filtered, dtype=bool),
            filtered_afs=np.zeros(n_filtered),
            filtered_miss=np.zeros(n_filtered, dtype=int),
            filtered_means=np.zeros(n_filtered),
        ),
        n_unexpected=0,
        analyzed_sample_count=n_samples,
        sample_basis=SampleBasis(np.arange(n_samples), n_samples),
        chunk_factory=lambda _chunk_size: iter(()),
    )


def _workspace(n_samples: int, config: LmmConfig | None = None) -> WorkspaceSpec:
    config = LmmConfig(show_progress=False) if config is None else config
    return WorkspaceSpec.build(
        DispatchPath.NUMPY_FALLBACK,
        config.lmm_mode,
        n_samples,
        n_samples,
        1,
        config.n_grid,
        config.n_refine,
        1,
    )


def test_prepared_genotype_sample_count_must_match_prepared_run() -> None:
    with pytest.raises(ValueError, match="sample count does not match"):
        run_lmm_chunk_source_numpy_group(
            **_run_kwargs(genotypes=_prepared_genotypes(3, 5))
        )


def _basis(n_samples: int) -> RotatedBasis:
    return RotatedBasis(
        eigenvalues=np.ones(n_samples),
        U=np.eye(n_samples),
        W=np.ones((n_samples, 1)),
        UtW=np.ones((n_samples, 1)),
    )


def _null_fit(n_samples: int, Uty: np.ndarray) -> NullFit:
    return NullFit(
        Uty=Uty,
        logl_H0=-1.0,
        Hi_eval_null=np.ones(n_samples),
        pve=None,
        pve_se=None,
    )


def _run_kwargs(**overrides):
    """Minimal valid arguments for the shared chunk interface."""
    n_samples = 4
    config = LmmConfig(lmm_mode=1, show_progress=False)
    base = {
        "genotypes": _prepared_genotypes(n_samples, 5),
        "basis": _basis(n_samples),
        "jobs": (
            PhenotypeChunkJob(
                _null_fit(n_samples, np.linspace(-1.0, 1.0, n_samples)),
                lambda _arrays, _start, _end: None,
            ),
        ),
        "config": config,
        "dispatch": DispatchPath.NUMPY_FALLBACK,
        "chunks": LmmChunkPlan(5, 1, 1, False),
        "workspace": _workspace(n_samples, config),
    }
    base.update(overrides)
    return base


def test_shared_chunk_entry_resets_the_p_yy_warning():
    """Every runner reaches this entry, so the per-run reset belongs here.

    The flag deduplicates the negative-P_yy warning within one run. Resetting
    it in the batch runner only meant a streaming or LOCO run that followed
    one in the same process never warned again.
    """
    from jamma.lmm import pab

    n_samples, n_snps = 4, 5
    G = np.array(
        [[0, 1, 2, 0, 1], [1, 2, 0, 2, 0], [2, 0, 1, 1, 2], [0, 1, 1, 2, 0]],
        dtype=float,
    )
    genotypes = replace(
        _prepared_genotypes(n_samples, n_snps),
        chunk_factory=lambda width: (
            RawLmmChunk(G[:, i : i + width].copy(), i, i + width)
            for i in range(0, n_snps, width)
        ),
    )
    pab._p_yy_state.warned = True
    run_lmm_chunk_source_numpy_group(**_run_kwargs(genotypes=genotypes))
    assert getattr(pab._p_yy_state, "warned", False) is False
