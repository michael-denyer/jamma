#!/usr/bin/env python3
"""Compare the declared fixture cases with pytest's actual tier1 collection."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.math_validation.fixtures import ROOT, load_manifest
from tests.math_validation.loco_cases import load_loco_manifest
from tests.math_validation.pipeline_cases import load_pipeline_manifest
from tests.math_validation.weight_contract import (
    load_nonpositive_weight_cases,
    load_weight_cases,
)


def check_inventory(
    collected,
    cases,
    *,
    test_file="tests/test_mathematical_validation.py",
    test_function="test_manifest_case_matches_gemma_and_dense_oracle",
):
    expected = {f"{test_file}::{test_function}[{case['id']}]" for case in cases}
    actual_list = [item for item in collected if f"::{test_function}[" in item]
    actual = set(actual_list)
    return {
        "status": "VERIFIED"
        if actual == expected and len(actual_list) == len(actual)
        else "NOT VERIFIED",
        "expected": sorted(expected),
        "collected": sorted(actual),
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
        "duplicate_count": len(actual_list) - len(actual),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_mathematical_validation.py",
        "tests/test_mathematical_pipeline.py",
        "tests/test_mathematical_loco.py",
        "tests/test_weight_gemma_contract.py",
        "--collect-only",
        "-q",
        "-m",
        "tier1",
        "-o",
        "addopts=",
        "-p",
        "no:rerunfailures",
    ]
    collected = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    groups = {
        "loco": check_inventory(
            collected.stdout.splitlines(),
            load_loco_manifest()["cases"],
            test_file="tests/test_mathematical_loco.py",
            test_function="test_loco_cold_and_warm_each_match_gemma",
        ),
        "supplied_kinship": check_inventory(
            collected.stdout.splitlines(), load_manifest()["cases"]
        ),
        "internal_kinship": check_inventory(
            collected.stdout.splitlines(),
            load_pipeline_manifest()["cases"],
            test_file="tests/test_mathematical_pipeline.py",
            test_function="test_raw_pipeline_matches_external_and_oracle",
        ),
        "weights": check_inventory(
            collected.stdout.splitlines(),
            load_weight_cases(),
            test_file="tests/test_weight_gemma_contract.py",
            test_function="test_pipeline_weight_semantics_match_external_gemma",
        ),
        "nonpositive_weights": check_inventory(
            collected.stdout.splitlines(),
            load_nonpositive_weight_cases(),
            test_file="tests/test_weight_gemma_contract.py",
            test_function="test_nonpositive_weight_rows_match_external_gemma",
        ),
    }
    result = {
        "status": "VERIFIED"
        if all(g["status"] == "VERIFIED" for g in groups.values())
        else "NOT VERIFIED",
        "groups": groups,
        "command": command,
        "exit_code": collected.returncode,
        "stdout": collected.stdout,
        "stderr": collected.stderr,
    }
    if collected.returncode:
        result["status"] = "NOT VERIFIED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])
    return int(result["status"] != "VERIFIED")


if __name__ == "__main__":
    raise SystemExit(main())
