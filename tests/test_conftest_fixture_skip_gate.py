"""Self-tests for the fixture-unavailability skip gate in conftest.py.

Everything under tests/fixtures/ is committed, so a test that skips because
it "could not find" a fixture is really a test with a wrong path. That bug
presents as a green run, which is how two GEMMA-parity tests stayed dormant
for their whole lifetime (#147). The gate turns such a skip into a session
failure.

The gate is a meta-rule, so it needs its own regression tests: if a future
refactor drops the hook, inverts the predicate, or loses the exit-status
mutation, dormant tests would silently re-enter the suite.

Two properties are easy to get wrong and are covered by subprocess tests
rather than reasoning. The gate must fire under ``-n`` (the tier gate above
it in conftest.py was once empirically a no-op under xdist), and mutating
``session.exitstatus`` in ``pytest_sessionfinish`` must actually change the
process exit code.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from tests.conftest import _FIXTURE_UNAVAILABLE_RE

pytestmark = pytest.mark.tier0

pytest_plugins = ["pytester"]

# Stub conftest importing the real hooks by file path. Pytester runs in a
# tmpdir without our pyproject, so `tests.conftest` will not import as a
# package. Delegating to the real module means a regression in it fails
# these tests instead of them passing against a copy.
_GATE_CONFTEST = textwrap.dedent(
    f'''
    """Stub conftest delegating to the real fixture-skip gate."""
    from __future__ import annotations

    import importlib.util
    from pathlib import Path

    _REAL_CONFTEST = Path({str(Path(__file__).parent / "conftest.py")!r})

    _spec = importlib.util.spec_from_file_location(
        "_jamma_real_conftest", _REAL_CONFTEST
    )
    _real = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_real)

    pytest_runtest_logreport = _real.pytest_runtest_logreport
    pytest_sessionfinish = _real.pytest_sessionfinish
    '''
)


class TestPattern:
    """The phrase must discriminate fixture skips from environment skips."""

    @pytest.mark.parametrize(
        "reason",
        [
            "Skipped: gemma_loco fixture not available",
            "Skipped: GEMMA synthetic fixture not available",
            "Skipped: Mouse HS1940 fixture not available",
            "Skipped: mouse_hs1940 fixture not available",
        ],
    )
    def test_matches_fixture_skips(self, reason: str) -> None:
        assert _FIXTURE_UNAVAILABLE_RE.search(reason)

    @pytest.mark.parametrize(
        "reason",
        [
            "Skipped: C extension not available",
            "Skipped: Fused mode-4 C extension not available",
            "Skipped: compute_score_batch_general_c not available",
            "Skipped: uv not available on PATH",
            "Skipped: source not available",
            "Skipped: Skipping benchmark (--benchmark-skip active).",
        ],
    )
    def test_ignores_environment_skips(self, reason: str) -> None:
        assert not _FIXTURE_UNAVAILABLE_RE.search(reason)

    def test_covers_every_live_site(self) -> None:
        """Every in-tree fixture-availability skip uses the gated phrase.

        Guards against a new site written as "fixture missing" or similar,
        which the gate would not see. A skip that mentions fixtures without
        claiming one is absent is left alone, so test_fixture_manifest.py's
        "no tracked fixtures to drift-test against" does not trip this.
        """
        absence = re.compile(
            r"not available|unavailable|not found|missing|does ?n[o']t exist|no such",
            re.IGNORECASE,
        )
        suspicious: list[str] = []
        for path in sorted(Path(__file__).parent.glob("test_*.py")):
            if path.name == Path(__file__).name:
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if "pytest.skip(" not in line or "fixture" not in line.lower():
                    continue
                if absence.search(line) and not _FIXTURE_UNAVAILABLE_RE.search(line):
                    suspicious.append(f"{path.name}:{lineno}: {line.strip()}")
        assert not suspicious, (
            "These fixture skips use wording the gate cannot see. Use the "
            "phrase 'fixture not available':\n  " + "\n  ".join(suspicious)
        )


class TestGateFires:
    """Subprocess tests: the gate must change the real process exit code."""

    def _write(self, pytester: pytest.Pytester, skip_reason: str) -> None:
        pytester.makeconftest(_GATE_CONFTEST)
        pytester.makepyfile(
            test_skipper=textwrap.dedent(
                f"""
                import pytest

                def test_needs_data():
                    pytest.skip({skip_reason!r})
                """
            )
        )

    def test_fixture_skip_fails_session(self, pytester: pytest.Pytester) -> None:
        """A fixture-unavailability skip must fail an otherwise green run."""
        self._write(pytester, "gemma_loco fixture not available")
        result = pytester.runpytest_subprocess("-p", "no:randomly")
        assert result.ret != 0, "gate did not fail the session"
        combined = "\n".join([*result.errlines, *result.outlines])
        assert "test_skipper.py::test_needs_data" in combined
        assert "looking at the wrong path" in combined

    def test_environment_skip_passes(self, pytester: pytest.Pytester) -> None:
        """Counter-test: an unrelated skip must leave the session green."""
        self._write(pytester, "C extension not available")
        result = pytester.runpytest_subprocess("-p", "no:randomly")
        assert result.ret == 0, (
            f"gate fired on an environment skip. ret={result.ret}\n"
            f"stdout={result.outlines!r}\nstderr={result.errlines!r}"
        )

    def test_clean_run_passes(self, pytester: pytest.Pytester) -> None:
        """Counter-test: no skips at all must leave the session green."""
        pytester.makeconftest(_GATE_CONFTEST)
        pytester.makepyfile(test_ok="def test_a(): pass\n")
        result = pytester.runpytest_subprocess("-p", "no:randomly")
        assert result.ret == 0, (
            f"gate fired on a clean run. ret={result.ret}\nstderr={result.errlines!r}"
        )

    def test_gate_fires_under_xdist(self, pytester: pytest.Pytester) -> None:
        """The skip happens on a worker; the controller must still fail.

        This is the regression the tier gate taught us to write. A hook that
        only sees the controller's own reports passes here while missing
        every real skip, because the default addopts run with -n 3.
        """
        self._write(pytester, "gemma_loco fixture not available")
        result = pytester.runpytest_subprocess("-p", "no:randomly", "-n", "2")
        assert result.ret != 0, (
            "gate should fail under -n 2; a controller-only hook would "
            "silently pass here"
        )
        combined = "\n".join([*result.errlines, *result.outlines])
        assert "looking at the wrong path" in combined

    def test_gate_passes_under_xdist_when_clean(
        self, pytester: pytest.Pytester
    ) -> None:
        """Counter-test: an environment skip under ``-n 2`` stays green."""
        self._write(pytester, "C extension not available")
        result = pytester.runpytest_subprocess("-p", "no:randomly", "-n", "2")
        assert result.ret == 0, (
            f"gate fired on an environment skip under -n 2. ret={result.ret}\n"
            f"stdout={result.outlines!r}\nstderr={result.errlines!r}"
        )
