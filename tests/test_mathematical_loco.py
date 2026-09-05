"""Phase 2 LOCO cold/warm external anchors."""

import json

import pytest

from tests.math_validation.loco_cases import (
    compare_loco,
    load_loco_manifest,
    require_loco_reference,
)


@pytest.mark.tier0
def test_loco_manifest_has_all_modes_and_singleton_chromosome():
    manifest = load_loco_manifest()
    assert {case["mode"] for case in manifest["cases"]} == {1, 2, 3, 4}
    assert manifest["chromosomes"]["3"] == 1


@pytest.mark.tier1
@pytest.mark.parametrize("case", load_loco_manifest()["cases"], ids=lambda c: c["id"])
def test_loco_reference_is_hash_verified(case):
    directory, provenance = require_loco_reference(case)
    assert provenance["case"] == case
    model = json.loads((directory / "model.json").read_text())
    assert model["snps_by_chromosome"]["3"] == ["loco8"]


@pytest.mark.tier1
@pytest.mark.parametrize("case", load_loco_manifest()["cases"], ids=lambda c: c["id"])
def test_loco_cold_and_warm_each_match_gemma(case, tmp_path):
    result = compare_loco(tmp_path / case["id"], case_ids=(case["id"],))
    assert result["status"] == "VERIFIED", result
    assert [run["route"] for run in result["cases"][0]["runs"]] == ["cold", "warm"]
    assert result["cases"][0]["runs"][1]["cache_reused"] is True
