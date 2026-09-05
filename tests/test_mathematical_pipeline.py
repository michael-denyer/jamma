"""Phase 2 raw-input-to-output external validation."""

import json

import numpy as np
import pytest

from tests.math_validation.pipeline_cases import (
    compare_pipeline,
    load_pipeline_manifest,
    require_pipeline_reference,
)


@pytest.mark.tier0
def test_pipeline_manifest_covers_all_modes_and_boundaries():
    manifest = load_pipeline_manifest()
    assert {case["mode"] for case in manifest["cases"]} == {1, 2, 3, 4}
    assert manifest["factors"]["sample_selection"].startswith("phenotype and covariate")
    assert "just below and just above" in manifest["factors"]["genotype_missingness"]


@pytest.mark.tier0
def test_filter_boundaries_cross_when_analysis_population_changes():
    from tests.math_validation.pipeline_cases import _arrays

    x = np.asarray(_arrays()["genotypes"])
    analysed = x[2:]
    full_maf = np.nanmean(x, axis=0) / 2
    analysed_maf = np.nanmean(analysed, axis=0) / 2
    full_miss = np.isnan(x).mean(axis=0)
    analysed_miss = np.isnan(analysed).mean(axis=0)
    assert full_maf[0] >= 0.1 > analysed_maf[0]
    assert full_miss[3] <= 0.1 < analysed_miss[3]


@pytest.mark.tier1
@pytest.mark.parametrize(
    "case", load_pipeline_manifest()["cases"], ids=lambda c: c["id"]
)
def test_pipeline_reference_is_hash_verified(case):
    directory, provenance = require_pipeline_reference(case)
    assert provenance["case"] == case
    assert (directory / "gemma.assoc.txt").is_file()
    model = json.loads((directory / "model.json").read_text())
    assert len(model["selected_sample_ids"]) == 38
    assert "boundary0" not in model["selected_snp_ids"]
    assert "boundary1" in model["selected_snp_ids"]
    assert "boundary2" in model["selected_snp_ids"]
    assert "boundary3" not in model["selected_snp_ids"]
    assert "boundary4" not in model["selected_snp_ids"]


@pytest.mark.tier0
def test_pipeline_comparison_rejects_no_backends(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        compare_pipeline(tmp_path / "empty", backends=())


@pytest.mark.tier1
@pytest.mark.parametrize(
    "case", load_pipeline_manifest()["cases"], ids=lambda c: c["id"]
)
def test_raw_pipeline_matches_external_and_oracle(case, math_evidence_dir):
    result = compare_pipeline(
        math_evidence_dir,
        case_ids=(case["id"],),
    )
    assert result["status"] == "VERIFIED", result
    assert len(result["cases"][0]["runs"]) == 4
