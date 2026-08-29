"""compute_kinship and PipelineRunner.load_kinship behaviour.

The first section validates compute_kinship's flag combinations before it
touches disk. The rest, moved from test_pipeline.py, exercises
load_kinship's early sample filtering, weight application, and MAF/missing
filter parity with -gk — the ~300 lines of "compute_kinship or load_kinship"
tests F6 named as their own seam.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from jamma.pipeline import PipelineConfig, PipelineRunner
from jamma.pipeline_kinship import compute_kinship
from tests.builders import write_fam
from tests.conftest import require_fixture
from tests.fixture_paths import MOUSE, SYNTHETIC

BFILE = SYNTHETIC.bfile
_MOUSE_BFILE = MOUSE.bfile

# A bfile that does not exist: every case below must fail on the guard, not on
# the missing .bed, which is what proves the guard runs first.
MISSING = Path("/nonexistent/p15/study")


@pytest.mark.tier0
def test_mode_outside_1_2_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid kinship mode 3"):
        compute_kinship(PipelineConfig(bfile=MISSING), 3)  # type: ignore[arg-type]


@pytest.mark.tier0
def test_loco_with_write_eigen_is_rejected() -> None:
    config = PipelineConfig(bfile=MISSING, loco=True, write_eigen=True)
    with pytest.raises(ValueError, match="-eigen not supported with -gk -loco"):
        compute_kinship(config, 1)


@pytest.mark.tier0
def test_loco_with_standardized_mode_is_rejected() -> None:
    config = PipelineConfig(bfile=MISSING, loco=True)
    with pytest.raises(ValueError, match=r"-gk 2 .* not supported with -loco"):
        compute_kinship(config, 2)


@pytest.mark.tier0
def test_missing_bed_is_reported_once_the_guards_pass() -> None:
    with pytest.raises(FileNotFoundError, match=r"\.bed file not found"):
        compute_kinship(PipelineConfig(bfile=MISSING), 1)


def _copy_plink_genotypes(dest: Path) -> Path:
    """Copy .bed and .bim from gemma_synthetic fixture to dest directory.

    Returns:
        bfile prefix (dest / "test")
    """
    for ext in (".bed", ".bim"):
        shutil.copy(SYNTHETIC.dir / f"test{ext}", dest / f"test{ext}")
    return dest / "test"


@pytest.mark.tier1
def test_weight_file_applied_to_kinship(tmp_path: Path) -> None:
    """Pipeline applies weights to kinship matrix when weight_file is set."""
    # Create a weight file with non-trivial weights
    weight_file = tmp_path / "weights.txt"
    with open(weight_file, "w") as f:
        for _ in range(100):
            f.write("4.0\n")  # All weights = 4.0

    # Run pipeline with weights
    config_weighted = PipelineConfig(
        bfile=BFILE,
        lmm_mode=1,
        maf=0.01,
        miss=0.05,
        output_dir=tmp_path / "weighted",
        check_memory=False,
        show_progress=False,
        weight_file=weight_file,
    )

    # Run pipeline without weights for comparison
    config_unweighted = PipelineConfig(
        bfile=BFILE,
        lmm_mode=1,
        maf=0.01,
        miss=0.05,
        output_dir=tmp_path / "unweighted",
        check_memory=False,
        show_progress=False,
    )

    runner_w = PipelineRunner(config_weighted)
    runner_u = PipelineRunner(config_unweighted)

    # Load kinship with and without weights
    K_weighted = runner_w.load_kinship(100)
    K_unweighted = runner_u.load_kinship(100)

    # With uniform weights=4.0, K_weighted[i,j] = K[i,j] / sqrt(4*4) = K[i,j] / 4
    np.testing.assert_allclose(K_weighted, K_unweighted / 4.0, rtol=1e-10)


@pytest.mark.tier1
def test_lmm_kinship_applies_config_maf_miss() -> None:
    """-lmm internally-computed kinship applies config MAF/missing filters.

    Regression for Bug 6: load_kinship built the kinship with no MAF or
    missingness filter (thresholds defaulted to 0.0 / 1.0), while -gk applied
    them. On mouse_hs1940 the default maf=0.01 removes SNPs, so the filtered
    kinship differs from the unfiltered one, and load_kinship must match the
    filtered computation, not the unfiltered one.
    """
    from jamma.kinship.stream import compute_kinship_streaming

    require_fixture(_MOUSE_BFILE.with_suffix(".bed"), _MOUSE_BFILE.with_suffix(".fam"))

    config = PipelineConfig(
        bfile=_MOUSE_BFILE,
        maf=0.01,
        miss=0.05,
        check_memory=False,
        show_progress=False,
    )
    K_load = PipelineRunner(config).load_kinship(1940)

    K_filtered = compute_kinship_streaming(
        _MOUSE_BFILE,
        maf_threshold=0.01,
        miss_threshold=0.05,
        check_memory=False,
        show_progress=False,
    )
    K_unfiltered = compute_kinship_streaming(
        _MOUSE_BFILE, check_memory=False, show_progress=False
    )

    # The filter must actually change the kinship, or the test proves nothing.
    assert not np.allclose(K_filtered, K_unfiltered, rtol=1e-8)
    np.testing.assert_allclose(
        K_load,
        K_filtered,
        rtol=1e-12,
        atol=1e-14,
        err_msg="load_kinship must apply config maf/miss like -gk does",
    )


_N_SAMPLES = 100
_NAN_INDICES = {5, 10, 15}


def _valid_indices_excluding(
    n_samples: int = _N_SAMPLES, exclude: set[int] | None = None
) -> np.ndarray:
    """Return sorted array of sample indices excluding the given set."""
    exclude = exclude or _NAN_INDICES
    return np.array([i for i in range(n_samples) if i not in exclude])


@pytest.mark.tier1
class TestEarlySampleFiltering:
    """Tests for early sample filtering before kinship computation."""

    def test_early_sample_filter_pipeline(self, tmp_path: Path) -> None:
        """Early filtering: NaN phenotypes + save_kinship=False.

        Verifies the pipeline computes valid_mask before kinship and
        passes valid_indices, producing identical eigenvalues to a
        direct valid-subset kinship computation.
        """
        from jamma.kinship.stream import compute_kinship_streaming

        bfile = _copy_plink_genotypes(tmp_path)
        write_fam(
            tmp_path / "test.fam",
            [1.0 + i * 0.1 for i in range(_N_SAMPLES)],
            missing_at=_NAN_INDICES,
        )
        valid_indices = _valid_indices_excluding()

        out = tmp_path / "output_early"
        out.mkdir()
        config = PipelineConfig(
            bfile=bfile,
            lmm_mode=1,
            output_dir=out,
            check_memory=False,
            show_progress=False,
            save_kinship=False,
            backend="numpy",
        )

        runner = PipelineRunner(config)
        K_with_vi = runner.load_kinship(_N_SAMPLES, valid_indices=valid_indices)
        n_valid = len(valid_indices)
        assert K_with_vi.shape == (n_valid, n_valid), (
            f"Expected ({n_valid}, {n_valid}) kinship, got {K_with_vi.shape}"
        )

        K_ref = compute_kinship_streaming(
            bfile,
            maf_threshold=config.maf,
            miss_threshold=config.miss,
            check_memory=False,
            show_progress=False,
            valid_indices=valid_indices,
        )
        np.testing.assert_allclose(
            K_with_vi,
            K_ref,
            rtol=1e-12,
            err_msg="load_kinship with valid_indices must match direct streaming",
        )

    def test_save_kinship_full_size(self, tmp_path: Path) -> None:
        """save_kinship=True: the file is full-size, the return is the subset."""
        bfile = _copy_plink_genotypes(tmp_path)
        write_fam(tmp_path / "test.fam", [1.0 + i * 0.1 for i in range(_N_SAMPLES)])

        out = tmp_path / "output_save"
        out.mkdir()
        config = PipelineConfig(
            bfile=bfile,
            lmm_mode=1,
            output_dir=out,
            check_memory=False,
            show_progress=False,
            save_kinship=True,
            backend="numpy",
        )

        valid_indices = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9])
        K = PipelineRunner(config).load_kinship(_N_SAMPLES, valid_indices=valid_indices)
        assert K.shape == (len(valid_indices), len(valid_indices))

        K_saved = np.load(out / "result.cXX.npy")
        assert K_saved.shape == (_N_SAMPLES, _N_SAMPLES)
        np.testing.assert_array_equal(K, K_saved[np.ix_(valid_indices, valid_indices)])

    def test_weight_file_valid_indices(self, tmp_path: Path) -> None:
        """Weights filtered to match valid_indices under early filtering."""
        bfile = _copy_plink_genotypes(tmp_path)
        write_fam(tmp_path / "test.fam", [1.0 + i * 0.1 for i in range(_N_SAMPLES)])

        weight_file = tmp_path / "weights.txt"
        np.savetxt(weight_file, np.arange(1.0, _N_SAMPLES + 1.0))

        out = tmp_path / "output_wt"
        out.mkdir()
        config = PipelineConfig(
            bfile=bfile,
            lmm_mode=1,
            output_dir=out,
            check_memory=False,
            show_progress=False,
            weight_file=weight_file,
        )

        valid_indices = _valid_indices_excluding()
        K = PipelineRunner(config).load_kinship(_N_SAMPLES, valid_indices=valid_indices)
        n_valid = len(valid_indices)
        assert K.shape == (n_valid, n_valid), (
            f"Expected ({n_valid}, {n_valid}) with valid_indices, got {K.shape}"
        )

    def test_precomputed_kinship_still_works(self, tmp_path: Path) -> None:
        """Pre-computed kinship from file is subsetted post-load with valid_indices."""
        from jamma.kinship.stream import compute_kinship_streaming

        bfile = _copy_plink_genotypes(tmp_path)
        write_fam(tmp_path / "test.fam", [1.0 + i * 0.1 for i in range(_N_SAMPLES)])

        K_full = compute_kinship_streaming(
            bfile, check_memory=False, show_progress=False
        )
        kinship_file = tmp_path / "kinship.cXX.txt"
        np.savetxt(kinship_file, K_full)

        out = tmp_path / "output_precomp"
        out.mkdir()
        config = PipelineConfig(
            bfile=bfile,
            lmm_mode=1,
            output_dir=out,
            check_memory=False,
            show_progress=False,
            kinship_file=kinship_file,
        )

        valid_indices = _valid_indices_excluding()
        K = PipelineRunner(config).load_kinship(_N_SAMPLES, valid_indices=valid_indices)

        n_valid = len(valid_indices)
        assert K.shape == (n_valid, n_valid)
        np.testing.assert_allclose(
            K,
            K_full[np.ix_(valid_indices, valid_indices)],
            rtol=1e-12,
            err_msg="Pre-computed kinship with valid_indices must match np.ix_",
        )

    def test_run_end_to_end_with_nan_phenotypes(self, tmp_path: Path) -> None:
        """Full run() with NaN phenotypes triggers early filtering and completes."""
        bfile = _copy_plink_genotypes(tmp_path)
        write_fam(
            tmp_path / "test.fam",
            [1.0 + i * 0.1 for i in range(_N_SAMPLES)],
            missing_at=_NAN_INDICES,
        )
        n_valid = _N_SAMPLES - len(_NAN_INDICES)

        out = tmp_path / "output_e2e"
        out.mkdir()
        config = PipelineConfig(
            bfile=bfile,
            lmm_mode=1,
            output_dir=out,
            check_memory=False,
            show_progress=False,
            save_kinship=False,
            backend="numpy",
        )

        result = PipelineRunner(config).run()

        assert result.n_samples == n_valid, (
            f"Expected {n_valid} samples after NaN filtering, got {result.n_samples}"
        )
        assert result.n_snps_tested > 0, "Should test at least some SNPs"

    def test_run_end_to_end_save_kinship_with_nan(self, tmp_path: Path) -> None:
        """Full run() with save_kinship=True and NaN phenotypes.

        Verifies save_kinship does not change statistical results: the
        filtered kinship is saved and eigenpairs match the non-save path.
        """
        bfile = _copy_plink_genotypes(tmp_path)
        write_fam(
            tmp_path / "test.fam",
            [1.0 + i * 0.1 for i in range(_N_SAMPLES)],
            missing_at=_NAN_INDICES,
        )
        n_valid = _N_SAMPLES - len(_NAN_INDICES)

        # backend stays out of the dict: splatting it would widen the literal
        # to str and no longer satisfy PipelineConfig's Literal[...] field.
        common_kwargs = {
            "bfile": bfile,
            "lmm_mode": 1,
            "check_memory": False,
            "show_progress": False,
        }

        out_no_save = tmp_path / "output_nosave"
        out_no_save.mkdir()
        result_no_save = PipelineRunner(
            PipelineConfig(
                **common_kwargs,
                backend="numpy",
                output_dir=out_no_save,
                save_kinship=False,
            )
        ).run()

        out_save = tmp_path / "output_save"
        out_save.mkdir()
        result_save = PipelineRunner(
            PipelineConfig(
                **common_kwargs,
                backend="numpy",
                output_dir=out_save,
                save_kinship=True,
            )
        ).run()

        assert result_save.n_samples == result_no_save.n_samples == n_valid
        assert result_save.n_snps_tested == result_no_save.n_snps_tested

        # Saved kinship should be full (n_samples, n_samples) for reuse
        K_saved = np.load(out_save / "result.cXX.npy")
        assert K_saved.shape == (_N_SAMPLES, _N_SAMPLES), (
            f"save_kinship must write full ({_N_SAMPLES}, {_N_SAMPLES}) "
            f"kinship for reuse, got {K_saved.shape}"
        )
