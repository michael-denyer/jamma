"""Contract tests for the bounded mathematical mutation runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import mathematical_mutations as mutations

pytestmark = pytest.mark.tier0


def test_manifest_covers_phase3_categories() -> None:
    manifest = mutations.load_manifest(mutations.DEFAULT_MANIFEST)
    categories = {item["category"] for item in manifest["mutations"]}
    assert len(categories) == 12
    assert len(manifest["mutations"]) == 12


@pytest.mark.tier1
def test_tiny_wald_results_preserve_external_allele_orientation() -> None:
    from jamma.io import load_plink_binary, read_fam_phenotypes
    from jamma.kinship.io import read_kinship_matrix
    from jamma.lmm.runner_numpy import run_lmm_association_numpy
    from jamma.lmm.schema import LmmConfig, SnpMeta
    from jamma.validation import compare_assoc_results, load_gemma_assoc
    from tests.conftest import require_fixture

    fixture = (
        mutations.ROOT
        / "tests/fixtures/mathematical_validation/tiny-wald-supplied-kinship"
    )
    prefix = fixture / "tiny"
    reference_path = fixture / "gemma.assoc.txt"
    require_fixture(
        prefix.with_suffix(".bed"),
        prefix.with_suffix(".bim"),
        prefix.with_suffix(".fam"),
        fixture / "kinship.txt",
        reference_path,
    )
    plink = load_plink_binary(prefix)
    result = run_lmm_association_numpy(
        genotypes=plink.genotypes,
        phenotypes=read_fam_phenotypes(prefix.with_suffix(".fam")),
        kinship=read_kinship_matrix(fixture / "kinship.txt"),
        snp_info=SnpMeta.from_plink_meta(plink.meta),
        config=LmmConfig(
            lmm_mode=1,
            maf_threshold=0.0,
            miss_threshold=1.0,
            check_memory=False,
            show_progress=False,
        ),
    ).associations
    reference = load_gemma_assoc(reference_path)
    assert [(row.allele1, row.allele0) for row in result] == [
        (row.allele1, row.allele0) for row in reference
    ]
    assert compare_assoc_results(result, reference).passed


def test_manifest_rejects_unsafe_source_path(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mutations": [
                    {
                        "id": "escape",
                        "category": "bad",
                        "path": "../outside.py",
                        "find": "x",
                        "replace": "y",
                        "detector": "tests/test_x.py::test_x",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="unsafe"):
        mutations.load_manifest(path)


def test_single_mutation_is_isolated_and_detected() -> None:
    manifest = mutations.load_manifest(mutations.DEFAULT_MANIFEST)
    mutation = next(m for m in manifest["mutations"] if m["id"] == "stale-loco-cache")
    original = mutations.ROOT / mutation["path"]
    before = mutations._sha256(original)
    report = mutations.run_mutation(mutation, timeout=60)
    assert report["status"] == "VERIFIED", report
    assert report["baseline_rc"] == 0
    assert report["actual_rc"] != 0
    assert report["patch_match_count"] == 1
    assert report["actual_detectors"]
    assert all(case["outcome"] != "error" for case in report["actual_testcases"])
    assert report["source_sha256_before"] == report["source_sha256_after"] == before
