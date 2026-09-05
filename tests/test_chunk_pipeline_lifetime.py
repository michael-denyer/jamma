"""Process-wide BLAS ownership for the overlapped LMM chunk pipeline."""

from __future__ import annotations

import threading
from contextlib import contextmanager

import numpy as np
import pytest

from jamma.lmm import chunk_pipeline, chunk_runner_numpy
from jamma.lmm.chunk_kernel import RunInvariants, make_kernel
from jamma.lmm.chunk_runner_numpy import (
    RawLmmChunk,
    _ChunkEngine,
    _PhenotypeConsumer,
)
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.prepare_common import PreparedLmmRun
from jamma.lmm.schema import LmmConfig
from jamma.lmm.workspace import WorkspaceSpec
from tests.conftest import requires_c

pytestmark = pytest.mark.tier0


class _ObservedEngine:
    """Wrap the production engine with a deterministic overlap window."""

    def __init__(
        self,
        engine: _ChunkEngine,
        active_limit: list[int],
        *,
        fail_compute: int | None = None,
    ):
        self.engine = engine
        self.active_limit = active_limit
        self.fail_compute = fail_compute
        self.prepare_calls = 0
        self.compute_calls = 0
        self.background_started = threading.Event()
        self.compute_started = threading.Event()
        self.observed: list[tuple[str, int]] = []

    @property
    def omp_threads(self) -> int:
        return self.engine.omp_threads

    def prepare(self):
        self.prepare_calls += 1
        prepared = self.engine.prepare()
        if self.prepare_calls == 3:
            self.background_started.set()
            assert self.compute_started.wait(timeout=5)
            self.observed.append(("rotation", self.active_limit[0]))
        return prepared

    def compute_and_write(self, prepared) -> None:
        self.compute_calls += 1
        if self.compute_calls == 2:
            self.compute_started.set()
            assert self.background_started.wait(timeout=5)
            self.observed.append(("compute", self.active_limit[0]))
        self.engine.compute_and_write(prepared)
        if self.compute_calls == self.fail_compute:
            raise RuntimeError("compute failed")


def _real_engine(
    active_limit: list[int],
    *,
    dispatch: DispatchPath,
    fail_compute: int | None = None,
):
    rng = np.random.default_rng(27)
    n_samples = 12
    n_snps = 3
    eigenvalues = np.linspace(0.2, 1.4, n_samples)
    U, _ = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))
    UtW = U.T @ np.ones((n_samples, 1))
    Uty = U.T @ rng.standard_normal(n_samples)
    config = LmmConfig(
        lmm_mode=1,
        maf_threshold=0.0,
        miss_threshold=1.0,
        show_progress=False,
    )
    prepared = PreparedLmmRun(
        eigenvalues=eigenvalues,
        U=U,
        UtW=UtW,
        Uty=Uty,
        logl_H0=-1.0,
        Hi_eval_null=1.0 / (eigenvalues + 1.0),
        pve=None,
        pve_se=None,
    )
    invariants = RunInvariants.build(dispatch, prepared, config, n_snps)
    workspace = WorkspaceSpec.build(
        dispatch, 1, n_samples, n_samples, 1, config.n_grid, config.n_refine, 1
    )
    kernel = make_kernel(invariants, workspace)
    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
    raw_chunks = iter(
        RawLmmChunk(genotypes[:, i : i + 1].copy(), i, i + 1) for i in range(n_snps)
    )
    written: list[np.ndarray] = []
    engine = _ChunkEngine(
        consumers=(
            _PhenotypeConsumer(
                invariants,
                kernel,
                lambda arrays, _start, _end: written.append(arrays["betas"]),
            ),
        ),
        U=U,
        filtered_means=genotypes.mean(axis=0),
        raw_chunks=raw_chunks,
        chunk_size=1,
        n_buffers=2,
        omp_threads=1,
    )
    return _ObservedEngine(engine, active_limit, fail_compute=fail_compute), written


def _fake_blas_controller(active_limit: list[int], transitions: list[tuple[str, int]]):
    @contextmanager
    def control(limit: int):
        previous = active_limit[0]
        transitions.append(("enter", limit))
        active_limit[0] = limit
        try:
            yield
        finally:
            active_limit[0] = previous
            transitions.append(("restore", previous))

    return control


def _drive(engine: _ObservedEngine, rotation_threads: int = 2) -> float:
    return chunk_pipeline._drive_pipeline(
        engine,  # type: ignore[arg-type]
        n_chunks=3,
        rotation_threads=rotation_threads,
        n_samples=4,
        n_filtered=3,
        show_progress=False,
        progress_label="test",
    )


@requires_c
@pytest.mark.parametrize("dispatch", [DispatchPath.FUSED, DispatchPath.NUMPY_FALLBACK])
def test_pipeline_holds_one_blas_limit_while_rotation_and_compute_overlap(
    monkeypatch: pytest.MonkeyPatch,
    dispatch: DispatchPath,
) -> None:
    active_limit = [11]
    transitions: list[tuple[str, int]] = []
    monkeypatch.setattr(
        chunk_pipeline,
        "blas_threads",
        _fake_blas_controller(active_limit, transitions),
    )
    monkeypatch.setattr(
        chunk_runner_numpy,
        "blas_threads",
        _fake_blas_controller(active_limit, transitions),
    )
    engine, written = _real_engine(active_limit, dispatch=dispatch)

    _drive(engine, rotation_threads=2)

    assert sorted(engine.observed) == [("compute", 2), ("rotation", 2)]
    assert transitions == [("enter", 2), ("restore", 11)]
    assert active_limit == [11]
    assert len(written) == 3
    assert all(np.isfinite(chunk).all() for chunk in written)


@requires_c
def test_pipeline_restores_blas_limit_when_foreground_compute_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_limit = [13]
    transitions: list[tuple[str, int]] = []
    monkeypatch.setattr(
        chunk_pipeline,
        "blas_threads",
        _fake_blas_controller(active_limit, transitions),
    )
    monkeypatch.setattr(
        chunk_runner_numpy,
        "blas_threads",
        _fake_blas_controller(active_limit, transitions),
    )
    engine, written = _real_engine(
        active_limit, dispatch=DispatchPath.FUSED, fail_compute=2
    )

    with pytest.raises(RuntimeError, match="compute failed"):
        _drive(engine, rotation_threads=3)

    assert sorted(engine.observed) == [("compute", 3), ("rotation", 3)]
    assert transitions == [("enter", 3), ("restore", 13)]
    assert active_limit == [13]
    assert len(written) == 2


@requires_c
def test_per_chunk_process_limit_would_override_the_overlap_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control for the removed ``blas_threads(1)`` kernel wrapper."""
    active_limit = [17]
    transitions: list[tuple[str, int]] = []
    controller = _fake_blas_controller(active_limit, transitions)
    monkeypatch.setattr(chunk_pipeline, "blas_threads", controller)
    engine, _written = _real_engine(active_limit, dispatch=DispatchPath.FUSED)
    real_compute = engine.compute_and_write

    def compute_with_legacy_limit(prepared) -> None:
        if engine.engine.kernel.uses_c:
            with controller(1):
                real_compute(prepared)
        else:
            real_compute(prepared)

    monkeypatch.setattr(engine, "compute_and_write", compute_with_legacy_limit)

    _drive(engine, rotation_threads=4)

    assert ("compute", 1) in engine.observed
    assert ("enter", 1) in transitions


def test_thread_plan_never_exceeds_preallocated_compute_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chunk_pipeline, "get_physical_core_count", lambda: 16)

    plan = chunk_pipeline.plan_thread_budget(
        n_samples=5_000,
        omp_threads=16,
        max_omp_threads=4,
        use_pipeline=True,
    )

    assert plan.omp == 4
    assert plan.rotation == 12
    assert plan.rotation + plan.omp == plan.total_cores
