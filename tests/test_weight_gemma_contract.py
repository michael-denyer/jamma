"""Regression coverage for GEMMA 0.98.5 ``-widv`` semantics."""

import json
import shutil

import pytest

from tests.math_validation.compare import compare_files
from tests.math_validation.evidence import run_pipeline
from tests.math_validation.weight_contract import (
    compare_weights,
    fixed_lambda_differences,
    load_nonpositive_weight_cases,
    load_weight_cases,
    require_reference,
    write_oracle,
)


@pytest.mark.tier1
def test_weight_reference_is_hash_verified():
    source, provenance = require_reference()
    assert provenance["gemma"]["version"] == "0.98.5"
    assert provenance["gemma"]["source_revision"] == (
        "c37b0445f820b682836a1d20009ce1817546493a"
    )
    assert (source / "gemma.assoc.txt").is_file()


@pytest.mark.tier1
def test_weight_formula_matches_gemma_at_reported_lambda():
    differences = fixed_lambda_differences()
    assert differences["beta"] < 1e-6
    assert differences["se"] < 1e-6
    assert differences["p_wald"] < 1e-6


@pytest.mark.tier1
@pytest.mark.parametrize("case", load_nonpositive_weight_cases(), ids=lambda c: c["id"])
def test_nonpositive_weight_rows_match_external_gemma(case, math_evidence_dir):
    result = compare_weights(math_evidence_dir, case_ids=(case["id"],))
    assert result["status"] == "VERIFIED", result


@pytest.mark.tier1
def test_weight_dense_oracle_matches_external_gemma(tmp_path):
    source, _ = require_reference()
    oracle = tmp_path / "oracle.assoc.txt"
    write_oracle(oracle)
    result = compare_files(
        oracle,
        source / "gemma.assoc.txt",
        af_contract="counted-allele",
        mode=4,
        reference_optional_logl=True,
    )
    assert result["status"] == "VERIFIED", result


@pytest.mark.tier1
def test_weight_oracle_rejects_corrupt_model_identity_and_af(tmp_path):
    source, _ = require_reference()
    model = json.loads((source / "model.json").read_text())
    corrupt_identity = json.loads(json.dumps(model))
    corrupt_identity["selected_snp_ids"][0] = "not-in-raw-bim"
    with pytest.raises(ValueError, match="absent from raw BIM"):
        write_oracle(
            tmp_path / "bad-identity.assoc.txt", model_override=corrupt_identity
        )

    corrupt_af = json.loads(json.dumps(model))
    corrupt_af["selected_af"][0] += 0.2
    oracle = tmp_path / "bad-af.assoc.txt"
    write_oracle(oracle, model_override=corrupt_af)
    result = compare_files(
        oracle,
        source / "gemma.assoc.txt",
        af_contract="counted-allele",
        mode=4,
        reference_optional_logl=True,
    )
    assert result["status"] == "NOT VERIFIED"
    assert set(result["failure_ids"]) == {
        "boundary1:af_orientation",
        "boundary1:af",
    }


@pytest.mark.tier1
@pytest.mark.parametrize("case", load_weight_cases(), ids=lambda c: c["id"])
def test_pipeline_weight_semantics_match_external_gemma(case, math_evidence_dir):
    bundle = compare_weights(math_evidence_dir, case_ids=(case["id"],))
    assert bundle["status"] == "VERIFIED", bundle
    default, refined = bundle["cases"][0]["runs"]
    if default["status"] != "VERIFIED":
        assert default["comparison"]["failure_ids"] == ["boundary5:l_mle"], default
    assert refined["status"] == "VERIFIED", refined


@pytest.mark.tier1
def test_weighted_eigen_files_round_trip_through_reload(tmp_path):
    """Eigen files written under -widv reproduce the weighted run when reloaded.

    GEMMA scales the eigenvector rows by sqrt(w) before writing, so its
    weighted eigenU round-trips through -d/-u without the weight file. JAMMA
    wrote the unscaled U, and -d/-u refuses -widv, so a reload silently gave
    a model that was neither the weighted nor the unweighted one.
    """
    source, _ = require_reference()
    common = {
        "covariate_file": source / "covariates.txt",
        "lmm_mode": 4,
        "maf": 0.1,
        "miss": 0.1,
        "n_refine": 30,
        "backend": "numpy",
    }
    weighted, _, _ = run_pipeline(
        source,
        tmp_path / "weighted",
        weight_file=source / "weights.txt",
        output_prefix="weighted",
        write_eigen=True,
        **common,
    )
    reloaded, _, _ = run_pipeline(
        source,
        tmp_path / "reload",
        eigenvalue_file=tmp_path / "weighted" / "weighted.eigenD.npy",
        eigenvector_file=tmp_path / "weighted" / "weighted.eigenU.npy",
        output_prefix="reload",
        **common,
    )
    comparison = compare_files(
        reloaded.assoc_paths[0],
        weighted.assoc_paths[0],
        af_contract="counted-allele",
        mode=4,
    )
    assert comparison["status"] == "VERIFIED", comparison


@pytest.mark.tier1
def test_weights_are_applied_once_across_multiple_phenotypes(tmp_path):
    source, _ = require_reference()
    for suffix in ("bed", "bim"):
        shutil.copyfile(source / f"tiny.{suffix}", tmp_path / f"tiny.{suffix}")
    fam_rows = [line.split() for line in (source / "tiny.fam").read_text().splitlines()]
    (tmp_path / "tiny.fam").write_text(
        "".join("\t".join([*row, row[5]]) + "\n" for row in fam_rows)
    )
    result, _, _ = run_pipeline(
        tmp_path,
        tmp_path / "out",
        covariate_file=source / "covariates.txt",
        weight_file=source / "weights.txt",
        phenotype_columns=(1, 2),
        lmm_mode=4,
        maf=0.1,
        miss=0.1,
        n_refine=30,
        backend="numpy-streaming",
        output_prefix="jamma",
    )
    for path in result.assoc_paths:
        comparison = compare_files(
            path,
            source / "gemma.assoc.txt",
            af_contract="counted-allele",
            mode=4,
            reference_optional_logl=True,
        )
        assert comparison["status"] == "VERIFIED", comparison
