"""Regression tests for the isolated mathematical efficiency probes."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_math_efficiencies import (
    report_scale,
    verify_block_projection,
    verify_low_rank_inverse,
)

pytestmark = pytest.mark.tier0


def test_block_projection_matches_packed_pab() -> None:
    rng = np.random.default_rng(20260721)

    worst_scaled_error = verify_block_projection(rng, trials=2)

    assert worst_scaled_error < 2e-11


def test_low_rank_inverse_matches_dense_solve() -> None:
    rng = np.random.default_rng(20260721)

    worst_backward_error = verify_low_rank_inverse(rng, trials=2)

    assert worst_backward_error < 32 * np.finfo(np.float64).eps


def test_scale_report_includes_each_candidate(capsys) -> None:
    report_scale(
        n_samples=125_632,
        kinship_snps=91_586,
        n_grid=50,
        n_refine=20,
        threads=48,
    )

    output = capsys.readouterr().out
    assert "packed Pab recursion" in output
    assert "current=100 shared=50" in output
    assert "rank_ratio=72.900%" in output
