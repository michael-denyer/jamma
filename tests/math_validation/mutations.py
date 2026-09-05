#!/usr/bin/env python3
"""Run bounded mathematical mutations in an import-isolated repository copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "tests/math_validation/mutations.json"
COMPARATOR_PATH = "tests/math_validation/compare.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_relative(value: str, *, production: bool = False) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe repository-relative path: {value!r}")
    normalized = path.as_posix()
    if production and not (
        normalized.startswith("src/jamma/") or normalized == COMPARATOR_PATH
    ):
        raise ValueError(
            f"mutation target is not production/validation code: {value!r}"
        )
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1 or not isinstance(raw.get("mutations"), list):
        raise ValueError("mutation manifest must have schema_version 1 and mutations")
    seen: set[str] = set()
    for mutation in raw["mutations"]:
        mutation_id = mutation.get("id")
        if not isinstance(mutation_id, str) or not mutation_id or mutation_id in seen:
            raise ValueError(f"invalid or duplicate mutation id: {mutation_id!r}")
        seen.add(mutation_id)
        _safe_relative(mutation["path"], production=True)
        if not mutation.get("find") or mutation["find"] == mutation.get("replace"):
            raise ValueError(f"{mutation_id}: mutation must change non-empty text")
        if not mutation.get("detector", "").startswith("tests/"):
            raise ValueError(
                f"{mutation_id}: detector must be an explicit tests/ nodeid"
            )
    return raw


def _copy_tree(destination: Path) -> None:
    for name in ("src", "tests", "scripts"):
        shutil.copytree(
            ROOT / name,
            destination / name,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def _pytest(
    copy_root: Path, nodeid: str, xml_path: Path, timeout: int
) -> dict[str, Any]:
    env = os.environ.copy()
    env.pop("JAMMA_MATH_EVIDENCE_DIR", None)
    env["PYTHONPATH"] = str(copy_root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        nodeid,
        "-q",
        "-n",
        "1",
        "-o",
        "addopts=",
        "--tb=short",
        f"--junitxml={xml_path}",
        "-p",
        "no:rerunfailures",
    ]
    try:
        run = subprocess.run(
            command,
            cwd=copy_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return {"rc": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr
        )
        return {
            "rc": 124,
            "stdout": stdout or "",
            "stderr": (stderr or "") + f"\npytest timed out after {timeout}s",
        }


def _xml_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    results = []
    for case in ET.parse(path).iter("testcase"):
        outcome = "passed"
        detail = ""
        for tag in ("failure", "error", "skipped"):
            child = case.find(tag)
            if child is not None:
                outcome = tag
                detail = (child.get("message", "") + "\n" + (child.text or "")).strip()
                break
        results.append(
            {
                "testcase": f"{case.get('classname', '')}::{case.get('name', '')}",
                "outcome": outcome,
                "detail": detail,
            }
        )
    return results


def _excerpt(
    run: dict[str, Any], cases: list[dict[str, str]], limit: int = 1200
) -> str:
    failed = "\n".join(c["detail"] for c in cases if c["outcome"] == "failure")
    text = failed or (run["stdout"] + "\n" + run["stderr"])
    return text.strip()[-limit:]


def _is_assertion_failure(detail: str) -> bool:
    """Distinguish an assertion from an exception reported as JUnit failure."""
    return (
        "AssertionError" in detail
        or detail.startswith("assert ")
        or "\nE   assert " in detail
        or "Not equal to tolerance" in detail
    )


def run_mutation(mutation: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    relative = _safe_relative(mutation["path"], production=True)
    original = ROOT / relative
    before = _sha256(original)
    result: dict[str, Any] = {
        "id": mutation["id"],
        "category": mutation["category"],
        "kind": mutation.get("kind", "production"),
        "expected_detector": mutation["detector"],
        "source": relative.as_posix(),
        "source_sha256_before": before,
    }
    if relative.suffix in {".c", ".h"} and not mutation.get("rebuild"):
        result.update(
            status="NOT VERIFIED", reason="C mutation has no isolated rebuild step"
        )
        result["source_sha256_after"] = _sha256(original)
        return result

    with tempfile.TemporaryDirectory(prefix="jamma-mutation-") as tmp:
        copy_root = Path(tmp) / "repo"
        _copy_tree(copy_root)
        baseline_xml = copy_root / "baseline.xml"
        baseline = _pytest(copy_root, mutation["detector"], baseline_xml, timeout)
        baseline_cases = _xml_results(baseline_xml)
        result["baseline_rc"] = baseline["rc"]
        result["baseline_testcases"] = baseline_cases
        if (
            baseline["rc"] != 0
            or not baseline_cases
            or any(c["outcome"] != "passed" for c in baseline_cases)
        ):
            result.update(
                status="INCONCLUSIVE",
                reason="named detector did not pass on the untouched isolated baseline",
                failure_excerpt=_excerpt(baseline, baseline_cases),
            )
        else:
            target = copy_root / relative
            source = target.read_text()
            count = source.count(mutation["find"])
            result["patch_match_count"] = count
            if count != 1:
                result.update(
                    status="INCONCLUSIVE",
                    reason=f"patch matched {count} times; exact-once required",
                )
            else:
                target.write_text(
                    source.replace(mutation["find"], mutation["replace"], 1)
                )
                mutant_xml = copy_root / "mutant.xml"
                mutant = _pytest(copy_root, mutation["detector"], mutant_xml, timeout)
                cases = _xml_results(mutant_xml)
                expected = str(
                    mutation.get("testcase", mutation["detector"].split("::")[-1])
                )
                intended = [
                    c
                    for c in cases
                    if expected in c["testcase"]
                    and c["outcome"] == "failure"
                    and _is_assertion_failure(c["detail"])
                ]
                errors = [c for c in cases if c["outcome"] == "error"]
                result.update(
                    actual_rc=mutant["rc"],
                    actual_testcases=cases,
                    actual_detectors=[c["testcase"] for c in intended],
                    failure_excerpt=_excerpt(mutant, cases),
                )
                if mutant["rc"] != 0 and intended and not errors:
                    result.update(status="VERIFIED", reason="intended assertion failed")
                else:
                    result.update(
                        status="INCONCLUSIVE",
                        reason="mutation did not produce an intended assertion failure",
                    )
    result["source_sha256_after"] = _sha256(original)
    if result["source_sha256_after"] != before:
        raise RuntimeError(f"original source changed while running {mutation['id']}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mutation", action="append", default=[])
    parser.add_argument(
        "--all", action="store_true", help="run slow and smoke mutations"
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest.resolve())
    chosen = [
        m
        for m in manifest["mutations"]
        if (not args.mutation and (args.all or m.get("tier", "smoke") == "smoke"))
        or m["id"] in args.mutation
    ]
    unknown = set(args.mutation) - {m["id"] for m in manifest["mutations"]}
    if unknown:
        parser.error(f"unknown mutation(s): {', '.join(sorted(unknown))}")
    report = {
        "schema_version": 1,
        "status": "VERIFIED",
        "mutations": [run_mutation(m, timeout=args.timeout) for m in chosen],
    }
    if any(m["status"] != "VERIFIED" for m in report["mutations"]):
        report["status"] = "NOT VERIFIED"
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    verified = sum(item["status"] == "VERIFIED" for item in report["mutations"])
    total = len(report["mutations"])
    print(f"{report['status']}: {verified}/{total} mutations verified")
    for item in report["mutations"]:
        print(f"{item['status']}\t{item['id']}")
    if args.output:
        print(f"report: {args.output}")
    return 0 if report["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
