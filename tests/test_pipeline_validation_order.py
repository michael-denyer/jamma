"""Which error ``validate_inputs`` reports when a config breaks several rules.

The per-rule tests in ``test_pipeline.py`` each violate one thing, so they pass
under any ordering of the checks. Precedence is still observable behaviour: a
user who passes two bad options sees one specific message, and reshaping the
method is exactly the change that would silently reorder them.

These cases were recorded from the implementation as it stood before the
file-existence checks were table-driven. They pin the contract, not a
preference. If a deliberate reordering is ever wanted, change these
expectations in the same commit and say why.

The phenotype column range check used to be in this ordering, between the PLINK
files and the file-existence table. It moved to
``PipelineConfig.__post_init__``, so a bad index now fails at construction and
never reaches ``validate_inputs`` to be ordered against anything. Its
replacement lives in ``test_pipeline.py::TestMultiPhenotypeConfig``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jamma.pipeline import PipelineConfig, PipelineRunner

BFILE = Path(__file__).resolve().parent / "fixtures/mouse_hs1940/mouse_hs1940"

pytestmark = pytest.mark.tier1


def _raises(config: PipelineConfig) -> tuple[type, str]:
    runner = PipelineRunner(config)
    with pytest.raises((ValueError, FileNotFoundError)) as excinfo:
        runner.validate_inputs()
    return type(excinfo.value), str(excinfo.value)


def test_missing_plink_beats_every_other_violation(tmp_path):
    """Nothing is checked until the dataset itself is known to exist."""
    kind, message = _raises(
        PipelineConfig(
            bfile=tmp_path / "nonexistent",
            kinship_file=tmp_path / "missing.cXX.txt",
            hwe_threshold=-1.0,
            check_memory=False,
        )
    )
    assert kind is FileNotFoundError
    assert "PLINK .bed file" in message


def test_loco_kinship_conflict_beats_the_file_not_existing(tmp_path):
    """The conflict is reported even though the same file is also missing."""
    kind, message = _raises(
        PipelineConfig(
            bfile=BFILE,
            loco=True,
            kinship_file=tmp_path / "missing.cXX.txt",
            check_memory=False,
        )
    )
    assert kind is ValueError
    assert "mutually exclusive" in message


def test_eigen_pairing_beats_eigen_file_existence(tmp_path):
    """An unpaired -d is reported before we ask whether the file is there."""
    kind, message = _raises(
        PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=tmp_path / "missing.d.txt",
            check_memory=False,
        )
    )
    assert kind is ValueError
    assert "must be provided together" in message


def test_loco_eigen_conflict_beats_eigen_file_existence(tmp_path):
    kind, message = _raises(
        PipelineConfig(
            bfile=BFILE,
            loco=True,
            eigenvalue_file=tmp_path / "missing.d.txt",
            eigenvector_file=tmp_path / "missing.u.txt",
            check_memory=False,
        )
    )
    assert kind is ValueError
    assert "not supported with -loco" in message


def test_eigenvalue_file_checked_before_kinship_file(tmp_path):
    """File-existence checks run in a fixed order; eigen comes first."""
    kind, message = _raises(
        PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=tmp_path / "missing.d.txt",
            eigenvector_file=tmp_path / "missing.u.txt",
            kinship_file=tmp_path / "missing.cXX.txt",
            check_memory=False,
        )
    )
    assert kind is FileNotFoundError
    assert "Eigenvalue file not found" in message


@pytest.mark.parametrize(
    ("earlier", "later", "expected"),
    [
        ("kinship_file", "covariate_file", "Kinship matrix file not found"),
        ("covariate_file", "snps_file", "Covariate file not found"),
        ("snps_file", "ksnps_file", "SNP list file not found"),
        ("ksnps_file", "weight_file", "Kinship SNP list file not found"),
    ],
)
def test_file_existence_checks_keep_their_relative_order(
    tmp_path, earlier, later, expected
):
    """Two missing files: the one earlier in the sequence is the one reported."""
    kind, message = _raises(
        PipelineConfig(
            bfile=BFILE,
            check_memory=False,
            **{
                earlier: tmp_path / f"missing_{earlier}",
                later: tmp_path / f"missing_{later}",
            },
        )
    )
    assert kind is FileNotFoundError
    assert expected in message


def test_weight_file_existence_beats_its_loco_conflict(tmp_path):
    """The file is checked before the -widv/-loco incompatibility."""
    kind, message = _raises(
        PipelineConfig(
            bfile=BFILE,
            loco=True,
            weight_file=tmp_path / "missing.weights.txt",
            check_memory=False,
        )
    )
    assert kind is FileNotFoundError
    assert "Weight file not found" in message


def test_cat_requires_covariate_before_checking_column_indices():
    kind, message = _raises(
        PipelineConfig(
            bfile=BFILE,
            cat_columns=[0],
            check_memory=False,
        )
    )
    assert kind is ValueError
    assert "-cat requires -c" in message


def test_hwe_range_is_checked_last(tmp_path):
    """Every file check precedes the hwe range check."""
    kind, message = _raises(
        PipelineConfig(
            bfile=BFILE,
            hwe_threshold=-1.0,
            covariate_file=tmp_path / "missing.covar.txt",
            check_memory=False,
        )
    )
    assert kind is FileNotFoundError
    assert "Covariate file not found" in message
