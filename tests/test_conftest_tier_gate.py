"""Self-tests for the tier-marker gate in ``scripts/check_test_markers.py``.

The gate is a meta-rule: every test *item* in this suite must declare a tier
marker. If the gate silently fails-open (e.g. a future refactor inverts the
predicate, swaps the marker check for ``True``, or wraps the report in
``contextlib.suppress``), unmarked tests would silently re-enter the default
CI run.

The gate is per-item, not per-file: it unions the module ``pytestmark``, the
enclosing class's decorators, and the function's own decorators for every
``test_*`` item, and reports the function when that union carries no tier
marker. A file-granular predecessor of this gate passed a file the moment
any one test in it carried a marker, so a sibling test with none went
unnoticed; ``test_one_marked_one_unmarked_function_reports_only_the_unmarked_one``
and ``test_untiered_function_in_tiered_module_fails_naming_the_function``
below both pin that a mixed file reports exactly the gap, not a false pass.

The gate is source-parsing, run as a pre-commit and CI lint rather than from
a pytest hook, so no collection filter or xdist distribution can hide it.
These tests exercise the helpers directly with synthetic ASTs and run the
lint end to end against a synthetic tree.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from check_test_markers import (
    _file_untiered_functions,
    _module_level_marker_names,
    _tier_marker_message,
    _untiered_test_functions,
)

from tests.support import install_lint_script

pytestmark = pytest.mark.tier0


def _parse(src: str) -> ast.Module:
    return ast.parse(textwrap.dedent(src))


class TestModuleLevelMarkerNames:
    def test_single_marker(self) -> None:
        tree = _parse(
            """
            import pytest
            pytestmark = pytest.mark.tier0
            """
        )
        assert _module_level_marker_names(tree) == {"tier0"}

    def test_list_of_markers(self) -> None:
        tree = _parse(
            """
            import pytest
            pytestmark = [pytest.mark.tier1, pytest.mark.slow]
            """
        )
        assert _module_level_marker_names(tree) == {"tier1", "slow"}

    def test_marker_with_args(self) -> None:
        """``pytest.mark.foo(...)`` parametrised marker still recognised."""
        tree = _parse(
            """
            import pytest
            pytestmark = pytest.mark.skipif(True, reason="x")
            """
        )
        assert _module_level_marker_names(tree) == {"skipif"}

    def test_no_pytestmark(self) -> None:
        tree = _parse("def test_x(): pass\n")
        assert _module_level_marker_names(tree) == set()

    def test_unrelated_assignment(self) -> None:
        tree = _parse("pytestmark_unused = 1\n")
        assert _module_level_marker_names(tree) == set()


class TestUntieredTestFunctions:
    """Per-item marker union: module ∪ class ∪ function, reported by name."""

    def test_module_pytestmark_covers_every_function(self) -> None:
        tree = _parse(
            """
            import pytest
            pytestmark = pytest.mark.tier0

            def test_a(): pass
            def test_b(): pass
            """
        )
        assert _untiered_test_functions(tree) == []

    def test_per_function_marker_passes(self) -> None:
        tree = _parse(
            """
            import pytest

            @pytest.mark.tier1
            def test_a(): pass
            """
        )
        assert _untiered_test_functions(tree) == []

    def test_no_markers_reports_the_function(self) -> None:
        tree = _parse("def test_a(): pass\n")
        assert _untiered_test_functions(tree) == ["test_a"]

    def test_one_marked_one_unmarked_function_reports_only_the_unmarked_one(
        self,
    ) -> None:
        """A module with mixed per-function coverage: only the gap is named.

        This is exactly the case the old file-granular gate could not see:
        the file *has* a tier marker (on ``test_a``), so it passed, while
        ``test_b`` ran untiered.
        """
        tree = _parse(
            """
            import pytest

            @pytest.mark.tier0
            def test_a(): pass

            def test_b(): pass
            """
        )
        assert _untiered_test_functions(tree) == ["test_b"]

    def test_class_decorator_covers_its_methods(self) -> None:
        tree = _parse(
            """
            import pytest

            @pytest.mark.tier0
            class TestY:
                def test_a(self): pass
                def test_b(self): pass
            """
        )
        assert _untiered_test_functions(tree) == []

    def test_one_tiered_class_one_untiered_class_reports_only_the_gap(self) -> None:
        tree = _parse(
            """
            import pytest

            @pytest.mark.tier0
            class TestMarked:
                def test_a(self): pass

            class TestUnmarked:
                def test_b(self): pass
            """
        )
        assert _untiered_test_functions(tree) == ["TestUnmarked.test_b"]

    def test_unrelated_marker_does_not_satisfy_the_gate(self) -> None:
        """``custom`` is not in the required-tier set."""
        tree = _parse(
            """
            import pytest

            @pytest.mark.custom
            def test_a(): pass
            """
        )
        assert _untiered_test_functions(tree) == ["test_a"]

    def test_non_test_functions_are_ignored(self) -> None:
        tree = _parse(
            """
            def helper(): pass
            def test_a(): pass
            """
        )
        assert _untiered_test_functions(tree) == ["test_a"]


class TestFileUntieredFunctions:
    def _write(self, tmp_path, src: str):
        path = tmp_path / "test_target.py"
        path.write_text(textwrap.dedent(src))
        return path

    def test_module_pytestmark_passes(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            """
            import pytest
            pytestmark = pytest.mark.tier0
            def test_a(): pass
            """,
        )
        assert _file_untiered_functions(path) == []

    def test_per_test_marker_passes(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            """
            import pytest

            @pytest.mark.tier1
            def test_a(): pass
            """,
        )
        assert _file_untiered_functions(path) == []

    def test_no_markers_fails(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            """
            def test_a(): pass
            """,
        )
        assert _file_untiered_functions(path) == ["test_a"]

    def test_one_marked_one_unmarked_function_fails_naming_the_function(
        self, tmp_path
    ) -> None:
        path = self._write(
            tmp_path,
            """
            import pytest

            @pytest.mark.tier0
            def test_a(): pass

            def test_b(): pass
            """,
        )
        assert _file_untiered_functions(path) == ["test_b"]

    def test_unrelated_marker_fails(self, tmp_path) -> None:
        """``custom`` is not in the required-tier set."""
        path = self._write(
            tmp_path,
            """
            import pytest

            @pytest.mark.custom
            def test_a(): pass
            """,
        )
        assert _file_untiered_functions(path) == ["test_a"]

    def test_syntax_error_treated_as_missing(self, tmp_path) -> None:
        """Unparsable source surfaces via the gate, not a swallowed exception."""
        path = tmp_path / "test_broken.py"
        path.write_text("def test_a(:\n    pass\n")
        assert _file_untiered_functions(path) == ["<unparsable file>"]

    def test_vanished_file_is_not_part_of_the_suite(self, tmp_path) -> None:
        """A path the walk saw but the read cannot find reports nothing.

        Transient files planted by other tests come and go under xdist; a
        gate that flags them as unparsable fails the suite at random.
        """
        gone = tmp_path / "test_gone.py"
        assert not gone.exists()
        assert _file_untiered_functions(gone) == []


class TestEnforceTierMarkersInProcess:
    def test_real_suite_passes(self) -> None:
        """The actual jamma test suite must satisfy the gate.

        If this fails, a real test file is missing a tier marker — fix the
        file rather than the gate. This is also a smoke test that
        ``_tier_marker_message`` walks ``tests/`` correctly.
        """
        assert _tier_marker_message(Path(__file__).parent) is None


class TestGateAsLint:
    """End to end: the lint exits 1 and names the gap on a synthetic tree."""

    _SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_test_markers.py"

    def _run(
        self, tmp_path: Path, name: str, source: str
    ) -> subprocess.CompletedProcess[str]:
        script = install_lint_script(self._SCRIPT, tmp_path / "scripts")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / name).write_text(textwrap.dedent(source))
        return subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, check=False
        )

    def test_gate_fires_on_an_unmarked_file(self, tmp_path: Path) -> None:
        result = self._run(
            tmp_path,
            "test_unmarked.py",
            """
            def test_does_a_thing():
                assert True
            """,
        )
        assert result.returncode == 1, result.stderr
        assert "tests/test_unmarked.py" in result.stderr
        assert "no tier marker" in result.stderr

    def test_gate_passes_when_marked(self, tmp_path: Path) -> None:
        """Counter-test: a marked file passes."""
        result = self._run(
            tmp_path,
            "test_marked.py",
            """
            import pytest
            pytestmark = pytest.mark.tier0

            def test_a(): pass
            def test_b(): pass
            """,
        )
        assert result.returncode == 0, result.stderr

    def test_untiered_function_in_tiered_module_fails_naming_the_function(
        self, tmp_path: Path
    ) -> None:
        """A marker on one function does not paper over a sibling with none.

        Regression for the file-granular predecessor: it passed the moment
        the file had *a* marker anywhere, so a lone ``@pytest.mark.tier0``
        decorator on one function made the whole file (including untiered
        siblings) look fully covered. The per-item gate must instead name
        the specific function that carries no marker.
        """
        result = self._run(
            tmp_path,
            "test_mixed.py",
            """
            import pytest

            @pytest.mark.tier0
            def test_covered(): pass

            def test_gap(): pass
            """,
        )
        assert result.returncode == 1, result.stderr
        assert "tests/test_mixed.py: test_gap" in result.stderr
        assert "test_covered" not in result.stderr
