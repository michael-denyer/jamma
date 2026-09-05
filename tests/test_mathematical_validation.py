"""Independent-oracle formulas and Phase 0 evidence acceptance checks."""

import ast
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.math_validation import dense_oracle
from tests.math_validation.compare import (
    check_boundary_coverage,
    compare_files,
    read_rows,
)
from tests.math_validation.fixtures import (
    REFERENCE,
    WALD_HEADER,
    load_manifest,
    materialize,
    verify_reference,
)
from tests.math_validation.oracle_io import write_oracle_assoc
from tests.math_validation.supplied_cases import compare


@pytest.mark.tier0
def test_dense_ols_hand_calculation_and_reml_normalization():
    w = np.ones((5, 1))
    x = np.array([-2, -1, 0, 1, 2.0])
    residual = np.array([1, -2, 2, -2, 1.0])
    y = 3 + 2 * x + residual
    fit = dense_oracle.evaluate(np.zeros((5, 5)), w, x, y, 1)
    assert fit["beta"] == pytest.approx(2)
    assert fit["rss"] == pytest.approx(14)
    assert fit["se"] == pytest.approx(np.sqrt(14 / 30))
    assert fit["mle"] == pytest.approx(-2.5 * (np.log(2 * np.pi * 14 / 5) + 1))
    assert fit["reml"] == pytest.approx(-1.5 * (np.log(2 * np.pi * 14 / 3) + 1))
    scaled = dense_oracle.evaluate(np.zeros((5, 5)), 7 * w, x, y, 1)
    assert scaled["reml"] == pytest.approx(fit["reml"])


@pytest.mark.tier0
def test_dense_oracle_rejects_rank_deficiency():
    with pytest.raises(ValueError, match="full-rank"):
        dense_oracle.evaluate(np.eye(4), np.ones((4, 1)), np.ones(4), np.arange(4.0), 1)


@pytest.mark.tier0
def test_oracle_import_boundary():
    # Explicit handoff requirement. A value-only test cannot establish source
    # independence: importing the same formula would reproduce the same error.
    source = Path(dense_oracle.__file__).read_text()
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "relative import")
    assert set(imports) <= {"numpy", "scipy.optimize", "scipy.stats"}


@pytest.mark.tier0
@pytest.mark.parametrize(
    "injection", ["", "import jamma", "__import__('tests.reference')"]
)
def test_oracle_runs_with_production_imports_forbidden(tmp_path, injection):
    path = tmp_path / "oracle.py"
    path.write_text(Path(dense_oracle.__file__).read_text() + "\n" + injection + "\n")
    code = """
import importlib.abc, runpy, sys
class BlockProduction(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"jamma", "tests"}:
            raise ImportError("oracle production import forbidden")
sys.meta_path.insert(0, BlockProduction())
module = runpy.run_path(sys.argv[1])
np = module["np"]
module["evaluate"](np.eye(4), np.ones((4,1)), np.arange(4.),
                   np.array([1.,0.,2.,4.]), 1.)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (1 if injection else 0), result.stderr
    if injection:
        assert "oracle production import forbidden" in result.stderr


@pytest.mark.tier0
def test_manifest_rejects_duplicate_or_unsafe_case_ids(tmp_path):
    manifest = load_manifest()
    for changed in ("../outside", manifest["cases"][0]["id"]):
        altered = copy.deepcopy(manifest)
        altered["cases"].append({**altered["cases"][0], "id": changed})
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="case ID"):
            load_manifest(path)


@pytest.mark.tier1
@pytest.mark.parametrize("case", load_manifest()["cases"], ids=lambda case: case["id"])
def test_manifest_case_matches_gemma_and_dense_oracle(case, math_evidence_dir):
    manifest = {**load_manifest(), "cases": [case]}
    result = compare(manifest, REFERENCE, math_evidence_dir)
    assert result["status"] == "VERIFIED", result
    assert [c["id"] for c in result["cases"]] == [case["id"]]
    assert result["cases"][0]["selected_snp_ids"] == [
        f"snp{i}" for i in range(case["n_snps"])
    ]
    assert len(result["cases"][0]["selected_sample_ids"]) == case["n_samples"]


@pytest.mark.tier1
def test_reference_hash_and_missing_file_fail(tmp_path):
    case = load_manifest()["cases"][0]
    source, _ = verify_reference(case)
    target = tmp_path / case["id"]
    shutil.copytree(source, target)
    (target / "tiny.bed").write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_reference(case, tmp_path)
    (target / "tiny.bed").unlink()
    with pytest.raises(FileNotFoundError):
        verify_reference(case, tmp_path)


@pytest.mark.tier1
def test_fixture_recipe_reproduces_inputs(tmp_path):
    case = load_manifest()["cases"][0]
    source, _ = verify_reference(case)
    model = materialize(case, tmp_path)
    # Raw PLINK must encode the same IDs, phenotypes and dosages as the model.
    fam = np.loadtxt(tmp_path / "tiny.fam", dtype=str)
    np.testing.assert_array_equal(fam[:, 5].astype(float), model["phenotype"])
    assert fam.shape == (case["n_samples"], 6)
    # Genotypes and IDs are RNG-only. Phenotype generation uses NumPy
    # factorization and is not a bit-level cross-LAPACK fixture gate.
    # Numerical comparisons always read the immutable committed FAM instead.
    for suffix in ("bed", "bim"):
        assert (tmp_path / f"tiny.{suffix}").read_bytes() == (
            source / f"tiny.{suffix}"
        ).read_bytes()


@pytest.mark.tier1
@pytest.mark.parametrize("column", ["beta", "se", "logl_H1", "l_remle", "p_wald", "af"])
def test_each_numeric_column_mutation_is_rejected(tmp_path, column):
    case = load_manifest()["cases"][0]
    source, _ = verify_reference(case)
    path = tmp_path / "mutant.assoc.txt"
    lines = (source / "gemma.assoc.txt").read_text().splitlines()
    row = lines[1].split("\t")
    index = WALD_HEADER.index(column)
    row[index] = str(float(row[index]) * 1.5)
    lines[1] = "\t".join(row)
    path.write_text("\n".join(lines) + "\n")
    result = compare_files(path, source / "gemma.assoc.txt")
    assert result["status"] == "NOT VERIFIED", result
    assert f"snp0:{column}" in result["failure_ids"]


@pytest.mark.tier1
def test_header_and_record_order_are_observed(tmp_path):
    case = load_manifest()["cases"][0]
    source, _ = verify_reference(case)
    model = json.loads((source / "model.json").read_text())
    path = tmp_path / "oracle.assoc.txt"
    write_oracle_assoc(model, path)
    assert tuple(read_rows(path)[0]) == WALD_HEADER
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
    assert compare_files(path, source / "gemma.assoc.txt")["status"] == "NOT VERIFIED"
    path.write_text("\n".join(["\t".join(reversed(WALD_HEADER)), *lines[1:]]) + "\n")
    with pytest.raises(ValueError, match="header"):
        read_rows(path)


@pytest.mark.tier0
def test_wrong_af_orientation_preserving_maf_is_rejected(tmp_path):
    case = load_manifest()["cases"][0]
    source, _ = verify_reference(case)
    path = tmp_path / "mutant.assoc.txt"
    lines = (source / "gemma.assoc.txt").read_text().splitlines()
    row = lines[1].split("\t")
    index = WALD_HEADER.index("af")
    row[index] = str(1 - float(row[index]))
    lines[1] = "\t".join(row)
    path.write_text("\n".join(lines) + "\n")
    result = compare_files(path, source / "gemma.assoc.txt")
    assert result["status"] == "NOT VERIFIED"
    assert "snp0:af_orientation" in result["failure_ids"]


@pytest.mark.tier0
def test_boundary_coverage_cannot_pass_vacuously():
    case = {"id": "declared-boundary"}
    expectations = {"declared-boundary": {"snp0": "upper"}}
    assert check_boundary_coverage(case, [], expectations)["status"] == "NOT VERIFIED"
    assert (
        check_boundary_coverage(
            case, [{"rs": "snp0", "classes": ["upper", "upper"]}], expectations
        )["status"]
        == "VERIFIED"
    )


@pytest.mark.tier0
@pytest.mark.parametrize(
    ("actual_af", "reference_af", "status"),
    [
        ("0.538", "0.537", "VERIFIED"),
        ("0.387", "0.388", "VERIFIED"),
        ("0.539", "0.537", "NOT VERIFIED"),
    ],
)
def test_printed_af_rounding_limit_is_decimal_exact(
    tmp_path, actual_af, reference_af, status
):
    source, _ = verify_reference(load_manifest()["cases"][0])
    paths = [tmp_path / "actual.txt", tmp_path / "reference.txt"]
    for path, af in zip(paths, (actual_af, reference_af), strict=True):
        lines = (source / "gemma.assoc.txt").read_text().splitlines()
        row = lines[1].split("\t")
        row[WALD_HEADER.index("af")] = af
        lines[1] = "\t".join(row)
        path.write_text("\n".join(lines) + "\n")
    result = compare_files(*paths)
    assert result["status"] == status, result
