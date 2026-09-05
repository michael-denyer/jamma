"""Compare association files using the canonical JAMMA error policy."""

import csv
from dataclasses import asdict
from decimal import Decimal

from jamma.lmm.schema import HEADERS, TEST_TYPE_MAP
from jamma.validation.compare import compare_assoc_results, load_gemma_assoc


def _check_header(header, mode, optional_logl):
    expected = tuple(HEADERS[TEST_TYPE_MAP[mode]].split("\t"))
    allowed = [expected]
    if optional_logl and mode == 4:
        allowed.append(tuple(field for field in expected if field != "logl_H1"))
    if header not in allowed:
        raise ValueError(f"unexpected mode {mode} header: {header}")
    return [field for field in expected if field not in header]


def read_rows(path, mode=1, *, optional_logl=False):
    with path.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        header = tuple(reader.fieldnames or [])
        _check_header(header, mode, optional_logl)
        rows = list(reader)
    if len({row["rs"] for row in rows}) != len(rows):
        raise ValueError("duplicate SNP IDs")
    return rows


def compare_files(
    actual,
    reference,
    *,
    af_contract="counted-allele",
    mode=1,
    reference_optional_logl=False,
):
    if af_contract != "counted-allele":
        raise ValueError(f"unknown allele-frequency contract: {af_contract}")

    for path, optional in ((actual, False), (reference, reference_optional_logl)):
        with path.open() as stream:
            absent_fields = _check_header(
                tuple(stream.readline().strip().split("\t")), mode, optional
            )
    a_rows, b_rows = load_gemma_assoc(actual), load_gemma_assoc(reference)
    for rows in (a_rows, b_rows):
        if len({row.rs for row in rows}) != len(rows):
            raise ValueError("duplicate SNP IDs")
    result = compare_assoc_results(a_rows, b_rows)
    errors = []
    if [r.rs for r in a_rows] != [r.rs for r in b_rows]:
        errors.append("ordered SNP IDs")
    for a, b in zip(a_rows, b_rows, strict=False):
        for field in ("chr", "rs", "ps", "n_miss", "allele1", "allele0"):
            if getattr(a, field) != getattr(b, field):
                errors.append(f"{b.rs}:{field}")
        # Both current writers emit BIM A1 dosage frequency. Keep its direction:
        # folding to MAF would hide flips. Two .3f values have at most 1e-3
        # combined rounding uncertainty; this supplements the existing gate.
        # Compare the printed decimals exactly at the formatting limit.
        # Binary subtraction makes 0.538 - 0.537 slightly greater than 0.001.
        actual_af = Decimal(str(a.af))
        reference_af = Decimal(str(b.af))
        if not (
            actual_af.is_finite()
            and reference_af.is_finite()
            and 0 <= actual_af <= 1
            and abs(actual_af - reference_af) <= Decimal("0.001")
        ):
            errors.append(f"{b.rs}:af_orientation")
    fields = asdict(result)
    failures = [
        f"{b_rows[i].rs if i < len(b_rows) else a_rows[i].rs}:{field}"
        for field, value in fields.items()
        if isinstance(value, dict) and not value["passed"]
        for i in value["failed_indices"]
    ]
    return {
        "status": "VERIFIED" if result.passed and not errors else "NOT VERIFIED",
        "fields": fields,
        "failure_ids": errors + failures,
        "af_contract": af_contract,
        "reference_absent_fields": absent_fields,
    }


def check_boundary_coverage(case, records, expectations):
    """Require the boundary SNPs and classes declared before comparison."""
    expected = expectations.get(case["id"], {})
    observed = {record["rs"]: record["classes"][0] for record in records}
    passed = bool(expected) and observed == expected
    return {
        "status": "VERIFIED" if passed else "NOT VERIFIED",
        "expected": expected,
        "observed": observed,
        "failure_ids": [] if passed else [f"{case['id']}:boundary-coverage"],
    }
