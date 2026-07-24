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

import pytest

pytestmark = pytest.mark.tier1

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/demonstrate_equivalence.py"


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
