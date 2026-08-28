"""Guard that the GEMMA equivalence report actually passes.

docs/GEMMA_EQUIVALENCE.md presents ``scripts/demonstrate_equivalence.py`` as the
empirical backing for the tolerance table in CLAUDE.md, but nothing ran it. It
regressed in v4.1.0 and stayed red for four months: the script reused one
kinship array across every section while ``eigendecompose_kinship`` consumes
its input, so every section after the first ran on eigenvectors instead of a
kinship matrix.

The script exits non-zero when any field exceeds tolerance, so running it is
the whole test. It runs in the default suite rather than the slow tier: the
whole report takes about 8s, and a guard that only fires in test-slow.yml is
most of the way back to a guard nobody runs.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.tier1

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/demonstrate_equivalence.py"


# The report runs in about 8s standalone, but it shells out to a subprocess that
# spins its own BLAS threads. Under the suite's `-n 3` xdist workers an unlucky
# ordering co-schedules it with other heavy tests, oversubscribes the cores, and
# pushes the run past the 120s default. Seed 11 hit that on three consecutive
# weekly flaky-detect runs (#159), each a timeout rather than a tolerance failure.
# justified: subprocess BLAS oversubscription under xdist, not slow test code (#159)
@pytest.mark.timeout(300)
def test_equivalence_report_passes_all_tolerances():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    if result.returncode != 0:
        # The script logs heavily to stderr. Lead with the tolerance table rows
        # so the actual failure is not buried under eigendecomp chatter; fall
        # back to stderr only when the script died before reporting anything.
        failures = [ln.strip() for ln in result.stdout.splitlines() if "FAIL" in ln]
        detail = (
            "\n".join(failures)
            if failures
            else f"no FAIL rows; script exited {result.returncode}. "
            f"stderr tail:\n{result.stderr[-1500:]}"
        )
        pytest.fail(
            f"demonstrate_equivalence.py reported fields outside tolerance:\n{detail}"
        )

    assert "VERDICT: ALL FIELDS PASS TOLERANCES" in result.stdout


def _load_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import demonstrate_equivalence
    finally:
        if sys.path and sys.path[0] == str(ROOT / "scripts"):
            sys.path.pop(0)
    return demonstrate_equivalence


@pytest.mark.tier0
def test_rank_correlation_matches_scipy_spearman_without_importing_scipy():
    """The report's rho is scipy's Spearman on untied data, computed without scipy.

    The script must not import scipy (installing it replaces the ILP64 numpy),
    so the check compares against scipy here, where the dev environment has it.
    """
    from scipy.stats import spearmanr

    script = _load_script()
    rng = np.random.default_rng(7)
    x = rng.standard_normal(500)
    y = x + rng.standard_normal(500) * 0.5

    expected, _ = spearmanr(x, y)
    assert script._rank_correlation(x, y) == pytest.approx(expected, abs=1e-12)
    assert script._rank_correlation(x, -x) == pytest.approx(-1.0)
    source = (ROOT / "scripts/demonstrate_equivalence.py").read_text()
    assert "from scipy" not in source
    assert "import scipy" not in source
