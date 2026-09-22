"""Which error ``validate_inputs`` reports when a config breaks several rules.

The per-rule tests in ``test_pipeline.py`` each violate one thing, so they pass
under any ordering of the checks. Precedence is still observable behaviour: a
user who passes two bad options sees one specific message, and reshaping the
method is exactly the change that would silently reorder them.

These cases were recorded from the implementation as it stood before the
file-existence checks were table-driven. They pin the contract, not a
preference. If a deliberate reordering is ever wanted, change these
expectations in the same commit and say why.

The phenotype column range check, the hwe range and hwe-with-loco checks, and
the two ``cat_columns`` checks used to be in this ordering. They moved to
``PipelineConfig.__post_init__``, so a bad value now fails at construction and
never reaches ``validate_inputs`` to be ordered against anything. Their
replacements live in ``test_pipeline.py`` (``TestMultiPhenotypeConfig`` and
``TestValidateInputsSnpsFields``).

The five kinship and eigen source rules (-k with -loco, -d/-u pairing, -d/-u
with -loco, -widv with -loco, -widv with -d/-u) moved to
``PipelineConfig.source()`` for the same reason. They keep their relative
order, but they now beat every filesystem check, so two expectations flipped
deliberately: a source conflict is reported before a missing PLINK file, and
the -widv/-loco conflict before a missing weight file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jamma.pipeline import PipelineConfig, PipelineRunner

BFILE = Path(__file__).resolve().parent / "fixtures/mouse_hs1940/mouse_hs1940"

pytestmark = pytest.mark.tier0


def _rejected_at_construction(**fields: object) -> str:
    with pytest.raises(ValueError) as excinfo:
        PipelineConfig(check_memory=False, **fields)  # type: ignore[arg-type]
    return str(excinfo.value)


def _raises(config: PipelineConfig) -> tuple[type, str]:
    runner = PipelineRunner(config)
    with pytest.raises((ValueError, FileNotFoundError)) as excinfo:
        runner.validate_inputs()
    return type(excinfo.value), str(excinfo.value)


def test_missing_plink_beats_every_missing_input_file(tmp_path):
    """No input file is checked until the dataset itself is known to exist."""
    kind, message = _raises(
        PipelineConfig(
            bfile=tmp_path / "nonexistent",
            kinship_file=tmp_path / "missing.cXX.txt",
            check_memory=False,
        )
    )
    assert kind is FileNotFoundError
    assert "PLINK .bed file" in message


def test_source_conflict_beats_missing_plink(tmp_path):
    """A source conflict needs no filesystem, so it fails at construction."""
    message = _rejected_at_construction(
        bfile=tmp_path / "nonexistent",
        kinship_file=tmp_path / "missing.cXX.txt",
        loco=True,
    )
    assert "mutually exclusive" in message


def test_loco_kinship_conflict_beats_eigen_pairing(tmp_path):
    message = _rejected_at_construction(
        bfile=BFILE,
        loco=True,
        kinship_file=tmp_path / "missing.cXX.txt",
        eigenvalue_file=tmp_path / "missing.d.txt",
    )
    assert "mutually exclusive" in message


def test_eigen_pairing_beats_loco_eigen_conflict(tmp_path):
    message = _rejected_at_construction(
        bfile=BFILE,
        loco=True,
        eigenvalue_file=tmp_path / "missing.d.txt",
    )
    assert "must be provided together" in message


def test_loco_eigen_conflict_beats_loco_weight_conflict(tmp_path):
    message = _rejected_at_construction(
        bfile=BFILE,
        loco=True,
        eigenvalue_file=tmp_path / "missing.d.txt",
        eigenvector_file=tmp_path / "missing.u.txt",
        weight_file=tmp_path / "missing.weights.txt",
    )
    assert "-d/-u (pre-computed eigen) not supported with -loco" in message


def test_loco_weight_conflict_beats_weight_file_existence(tmp_path):
    message = _rejected_at_construction(
        bfile=BFILE,
        loco=True,
        weight_file=tmp_path / "missing.weights.txt",
    )
    assert "-widv (individual weights) is not yet supported with -loco" in message


def test_eigen_weight_conflict_beats_eigen_dir_without_loco(tmp_path):
    message = _rejected_at_construction(
        bfile=BFILE,
        eigenvalue_file=tmp_path / "missing.d.txt",
        eigenvector_file=tmp_path / "missing.u.txt",
        weight_file=tmp_path / "missing.weights.txt",
        eigen_dir=tmp_path,
    )
    assert "cannot be used with -d/-u" in message


def test_eigen_dir_without_loco_is_rejected(tmp_path):
    """gwas(eigen_dir=..., loco=False) used to be accepted and ignored."""
    message = _rejected_at_construction(bfile=BFILE, eigen_dir=tmp_path)
    assert "--eigen-dir is only supported with -loco" in message


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


def test_cat_requires_covariate_before_checking_column_indices():
    """Both -cat rules are config-time now; the missing -c is still reported first."""
    with pytest.raises(ValueError, match="-cat requires -c"):
        PipelineConfig(bfile=BFILE, cat_columns=[0], check_memory=False)
