"""The batch pipeline reads genotypes from disk once for every phenotype."""

import numpy as np
import pytest
from bed_reader import to_bed

import jamma.io.plink as plink_io
from jamma.genotype.dataset import GenotypeDataset
from jamma.pipeline import PipelineConfig, PipelineRunner
from tests.builders import write_fam

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize("subset", [False, True])
def test_pipeline_batch_reads_bed_once_before_association(
    tmp_path, monkeypatch, subset
):
    rng = np.random.default_rng(190)
    bfile = tmp_path / "study"
    values = rng.integers(0, 3, (100, 300)).astype(np.float64)
    values[::13, ::7] = np.nan
    to_bed(bfile.with_suffix(".bed"), values)
    phenotypes = rng.normal(size=(2, 100))
    if subset:
        phenotypes[:, [1, 4]] = np.nan
    write_fam(bfile.with_suffix(".fam"), *phenotypes.tolist())
    bed_opens = []
    reads_at_materialize = []
    real_open_bed = plink_io.open_bed
    real_materialize = GenotypeDataset.materialize

    def counting_open_bed(*args, **kwargs):
        bed_opens.append(args[0])
        return real_open_bed(*args, **kwargs)

    def observed_materialize(self):
        materialized = real_materialize(self)
        reads_at_materialize.append(len(bed_opens))
        return materialized

    # allow-patch: count genotype reads while delegating to bed-reader.
    monkeypatch.setattr(plink_io, "open_bed", counting_open_bed)
    # allow-patch: observe the batch load while delegating to the real method.
    monkeypatch.setattr(GenotypeDataset, "materialize", observed_materialize)
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

    # One in-memory load shared by both phenotypes; statistics and every
    # association chunk then read memory, never the .bed again.
    assert len(reads_at_materialize) == 1
    assert len(bed_opens) == reads_at_materialize[0]
    assert result.n_snps_tested > 0
    assert len(result.phenotype_results) == 2
    assert all(item.assoc_path.stat().st_size > 0 for item in result.phenotype_results)
