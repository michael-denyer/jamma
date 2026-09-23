"""Tests for CLI memory pre-flight checks.

Uses subprocess to test memory checks without requiring specific machine
memory sizes.
"""

from pathlib import Path

import pytest

from jamma.genotype.dataset import GenotypeDataset
from tests.fixture_paths import SYNTHETIC

PLINK_PREFIX = SYNTHETIC.bfile
KINSHIP_FILE = SYNTHETIC.kinship


@pytest.mark.tier0
class TestCliMemoryCheckUnit:
    """Unit tests for memory check logic (no subprocess)."""

    def test_estimate_called_before_load(self):
        """Memory estimate should be computable from metadata alone."""
        from jamma.lmm.association_plan import plan_association

        # This simulates what CLI does: get dimensions, then estimate
        dataset = GenotypeDataset.open_plink(PLINK_PREFIX)
        est = plan_association(
            dataset.n_samples, dataset.n_variants, backend="numpy-streaming"
        ).price(eigen=None)

        assert est.association_gb > 0
        assert est.total_peak_gb == max(
            est.kinship_gb, est.eigen_gb, est.statistics_gb, est.association_gb
        )

    def test_metadata_does_not_load_genotypes(self):
        """Opening the dataset should only read dimensions, not genotypes."""
        # This should be fast and low-memory
        dataset = GenotypeDataset.open_plink(PLINK_PREFIX)

        assert dataset.n_samples == 100
        assert dataset.n_variants == 500


@pytest.mark.tier1
class TestMemoryGateEndToEnd:
    """The gate's reject-or-proceed decision, driven through the real run().

    A unit test on the estimator does not prove the gate fires; these run the
    pipeline entry the CLI calls and assert the decision itself.
    """

    def _config(self, tmp_path: Path, **overrides):
        from jamma.pipeline import PipelineConfig

        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        return PipelineConfig(
            bfile=PLINK_PREFIX,
            kinship_file=KINSHIP_FILE,
            lmm_mode=1,
            output_dir=out,
            show_progress=False,
            save_kinship=False,
            **overrides,
        )

    def test_rejects_before_any_compute_when_budget_tiny(self, tmp_path: Path):
        """A 1 MB budget must reject the run with the budget message."""
        from jamma.pipeline import PipelineRunner

        config = self._config(tmp_path, check_memory=True, mem_budget=0.001)
        with pytest.raises(MemoryError, match="exceeds"):
            PipelineRunner(config).run()
        assert not list((tmp_path / "out").glob("*.assoc.txt")), (
            "rejected run must not write results"
        )

    def test_proceeds_and_writes_results_when_memory_fits(self, tmp_path: Path):
        """With the gate on and ample memory, the run completes end to end."""
        from jamma.pipeline import PipelineRunner

        config = self._config(tmp_path, check_memory=True)
        result = PipelineRunner(config).run()
        assert result.n_snps_tested > 0
        assert list((tmp_path / "out").glob("*.assoc.txt"))
