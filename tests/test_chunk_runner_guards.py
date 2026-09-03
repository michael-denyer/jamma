"""Guard/precondition tests for the shared NumPy LMM chunk runner.

These cover the cheap, isolated failure paths that the end-to-end parity
suites never exercise: the ``run_lmm_chunk_source_numpy`` argument
preconditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from jamma.core.snp_stats import SnpSelection
from jamma.lmm.chunk_runner_numpy import run_lmm_chunk_source_numpy
from jamma.lmm.chunk_sizing import LmmChunkPlan
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.genotype_source import PreparedGenotypes
from jamma.lmm.prepare_common import PreparedLmmRun
from jamma.lmm.schema import ChunkRunStats, LmmConfig, SnpMeta

pytestmark = pytest.mark.tier0

# ---------------------------------------------------------------------------
# run_lmm_chunk_source_numpy preconditions
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
        chunk_factory=lambda _chunk_size: iter(()),
    )


def test_prepared_run_and_config_define_an_empty_chunk_run() -> None:
    """The chunk interface derives counts instead of accepting parallel integers."""
    n_samples = 4
    prepared = PreparedLmmRun(
        eigenvalues=np.ones(n_samples),
        U=np.eye(n_samples),
        UtW=np.ones((n_samples, 1)),
        Uty=np.ones(n_samples),
        logl_H0=-1.0,
        Hi_eval_null=np.ones(n_samples),
        pve=None,
        pve_se=None,
    )

    stats = run_lmm_chunk_source_numpy(
        genotypes=_prepared_genotypes(n_samples, 0),
        chunk_sink=lambda _arrays, _start, _end: None,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        chunks=LmmChunkPlan(1, 0, 1, False),
        prepared=prepared,
        config=LmmConfig(show_progress=False),
    )

    assert stats == ChunkRunStats()


def test_prepared_genotype_sample_count_must_match_prepared_run() -> None:
    with pytest.raises(ValueError, match="sample count does not match"):
        run_lmm_chunk_source_numpy(**_run_kwargs(genotypes=_prepared_genotypes(3, 5)))


def _run_kwargs(**overrides):
    """Minimal valid arguments for the shared chunk interface.

    The prepared source and sink are never reached by the empty-run paths.
    """
    n_samples = 4
    prepared = PreparedLmmRun(
        eigenvalues=np.ones(n_samples),
        U=np.eye(n_samples),
        UtW=np.ones((n_samples, 1)),
        Uty=np.ones(n_samples),
        logl_H0=-1.0,
        Hi_eval_null=np.ones(n_samples),
        pve=None,
        pve_se=None,
    )
    base = {
        "genotypes": _prepared_genotypes(n_samples, 5),
        "chunk_sink": lambda _arrays, _start, _end: None,
        "dispatch": DispatchPath.NUMPY_FALLBACK,
        "chunks": LmmChunkPlan(5, 1, 1, False),
        "prepared": prepared,
        "config": LmmConfig(lmm_mode=1, show_progress=False),
    }
    base.update(overrides)
    return base


def test_empty_filtered_returns_zeroed_stats():
    """No SNPs means no work and no time spent, on every field the caller reads."""
    stats = run_lmm_chunk_source_numpy(
        **_run_kwargs(genotypes=_prepared_genotypes(4, 0))
    )
    assert stats == ChunkRunStats()


def test_shared_chunk_entry_resets_the_p_yy_warning():
    """Every runner reaches this entry, so the per-run reset belongs here.

    The flag deduplicates the negative-P_yy warning within one run. Resetting
    it in the batch runner only meant a streaming or LOCO run that followed
    one in the same process never warned again.
    """
    from jamma.lmm import likelihood

    likelihood._p_yy_state.warned = True
    run_lmm_chunk_source_numpy(**_run_kwargs(genotypes=_prepared_genotypes(4, 0)))
    assert getattr(likelihood._p_yy_state, "warned", False) is False
