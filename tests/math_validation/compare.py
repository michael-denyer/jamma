"""Compare association files using the canonical JAMMA error policy."""

from dataclasses import asdict

from jamma.lmm.schema import MODE_SPECS, parse_lmm_mode
from jamma.validation.compare import compare_assoc_results, load_gemma_assoc


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
    mode = parse_lmm_mode(mode)
    a_rows = load_gemma_assoc(actual, mode=mode, require_logl=True)
    b_rows = load_gemma_assoc(
        reference, mode=mode, require_logl=not reference_optional_logl
    )
    result = compare_assoc_results(a_rows, b_rows)
    errors = []
    if [r.rs for r in a_rows] != [r.rs for r in b_rows]:
        errors.append("ordered SNP IDs")
    for a, b in zip(a_rows, b_rows, strict=False):
        for field in ("chr", "rs", "ps", "n_miss", "allele1", "allele0"):
            if getattr(a, field) != getattr(b, field):
                errors.append(f"{b.rs}:{field}")
    failures = [
        f"{b_rows[i].rs if i < len(b_rows) else a_rows[i].rs}:{field}"
        for field, column in result.columns.items()
        for i in column.failed_indices
    ]
    carried = {c.field_name for c in MODE_SPECS[mode].stat_columns}
    return {
        "status": "VERIFIED" if result.passed and not errors else "NOT VERIFIED",
        "fields": {field: asdict(column) for field, column in result.columns.items()},
        "failure_ids": errors + failures,
        "af_contract": af_contract,
        "reference_absent_fields": (
            ["logl_H1"]
            if "logl_H1" in carried and any(r.logl_H1 is None for r in b_rows)
            else []
        ),
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
