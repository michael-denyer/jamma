"""NumPy LOCO tests that run without JAX.

Kept in a separate file from test_loco.py because test_loco.py has
``pytest.importorskip("jax")`` at module level, which skips the entire
module when JAX is not installed. Tests here exercise the NumPy backend
only and must not import JAX.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jamma.lmm.loco import run_lmm_loco
from tests.conftest import load_phenotypes_from_fam

# Fixture with 3 chromosomes — required for LOCO (needs >1 chromosome to leave one out)
_LOCO_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "gemma_loco"
_LOCO_BFILE = _LOCO_FIXTURE_ROOT / "test"


@pytest.mark.tier1
def test_loco_numpy_show_progress_true():
    """NumPy LOCO with show_progress=True completes without error.

    Exercises the tqdm progress bars and logger.info calls in
    _compute_loco_kinship_streaming_numpy and run_lmm_loco.
    Not marked @requires_jax — runs in NumPy-only CI.
    """
    if not _LOCO_BFILE.with_suffix(".bed").exists():
        pytest.skip("gemma_loco fixture not available")

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    results, n_tested = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        lmm_mode=1,
        show_progress=True,
        check_memory=False,
        backend="numpy",
    )

    assert n_tested > 0, "Expected at least one SNP to be tested"
    assert len(results) > 0, "Expected at least one association result"
