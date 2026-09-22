"""Prepared batch genotypes own only the rows needed by the phenotype loop."""

from dataclasses import replace
from weakref import ref

import numpy as np
import pytest
from bed_reader import to_bed

import jamma.pipeline_phenotype_loop as phenotype_loop
from jamma.lmm import runner_numpy
from jamma.pipeline import PipelineConfig, PipelineRunner
from tests.builders import write_fam

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize("subset", [False, True])
def test_pipeline_releases_replaced_batch_matrix(tmp_path, monkeypatch, subset):
    rng = np.random.default_rng(190)
    bfile = tmp_path / "study"
    values = rng.integers(0, 3, (100, 300)).astype(np.float64)
    values[::13, ::7] = np.nan
    to_bed(bfile.with_suffix(".bed"), values)
    phenotypes = rng.normal(size=(2, 100))
    if subset:
        phenotypes[:, [1, 4]] = np.nan
    write_fam(bfile.with_suffix(".fam"), *phenotypes.tolist())
    observations = []

    class ObservedMatrixSource(runner_numpy.MatrixSource):
        def prepare(self, samples, filters):
            original = ref(self._genotypes)
            prepared = super().prepare(samples, filters)
            chunks = prepared.chunk_factory

            def observe_chunks(chunk_size):
                observations.append(original() is not None)
                yield from chunks(chunk_size)

            return replace(prepared, chunk_factory=observe_chunks)

    # allow-patch: observe lifetime while delegating all reads and computation.
    monkeypatch.setattr(phenotype_loop, "MatrixSource", ObservedMatrixSource)
    result = PipelineRunner(
        PipelineConfig(
            bfile=bfile,
            backend="numpy",
            phenotype_columns=[1, 2],
            output_dir=tmp_path / "output",
            show_progress=False,
            no_telemetry=True,
            check_memory=False,
        )
    ).run()

    assert observations
    assert all(alive == (not subset) for alive in observations)
    assert result.n_snps_tested > 0
    assert len(result.phenotype_results) == 2
    assert all(item.assoc_path.stat().st_size > 0 for item in result.phenotype_results)
