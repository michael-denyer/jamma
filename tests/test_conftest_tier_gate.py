"""Self-tests for the tier-marker enforcement gate in conftest.py.

The gate is a meta-rule: every test file in this suite must declare a tier
marker. If the gate silently fails-open (e.g. a future refactor inverts the
predicate, swaps the marker check for ``True``, or wraps the raise in
``contextlib.suppress``), unmarked tests would silently re-enter the default
CI run.

The gate is implemented as **source-parsing** in ``pytest_configure``
(not ``pytest_collection_modifyitems``) because xdist forks workers AFTER
``pytest_configure`` and the controller's items list is empty under
``-n``. The collection-based design that preceded this one was empirically
a no-op when ``-n 3`` (the default in pyproject) was active. See
``test_gate_fires_under_xdist`` for the regression check.

These tests exercise the helper functions directly with synthetic ASTs and
add one ``pytester`` subprocess test that runs the gate under ``-n 2`` to
prove the xdist path works.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from tests.conftest import (
    _enforce_tier_markers,
    _file_declares_tier_marker,
    _module_level_marker_names,
    _per_test_marker_names,
)

pytestmark = pytest.mark.tier0

pytest_plugins = ["pytester"]


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


class TestPerTestMarkerNames:
    def test_function_decorator(self) -> None:
        tree = _parse(
            """
            import pytest

            @pytest.mark.tier1
            def test_x():
                pass
            """
        )
        assert "tier1" in _per_test_marker_names(tree)

    def test_class_decorator(self) -> None:
        tree = _parse(
            """
            import pytest

            @pytest.mark.benchmark
            class TestY:
                def test_a(self): pass
            """
        )
        assert "benchmark" in _per_test_marker_names(tree)

    def test_call_form_decorator(self) -> None:
        """``@pytest.mark.skipif(...)`` decorator with args."""
        tree = _parse(
            """
            import pytest

            @pytest.mark.skipif(True, reason="r")
            def test_x():
                pass
            """
        )
        assert "skipif" in _per_test_marker_names(tree)

    def test_no_decorators(self) -> None:
        tree = _parse("def test_x(): pass\n")
        assert _per_test_marker_names(tree) == set()


class TestFileDeclaresTierMarker:
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
        assert _file_declares_tier_marker(path)

    def test_per_test_marker_passes(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            """
            import pytest

            @pytest.mark.tier1
            def test_a(): pass
            """,
        )
        assert _file_declares_tier_marker(path)

    def test_no_markers_fails(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            """
            def test_a(): pass
            """,
        )
        assert not _file_declares_tier_marker(path)

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
        assert not _file_declares_tier_marker(path)

    def test_syntax_error_treated_as_missing(self, tmp_path) -> None:
        """Unparsable source surfaces via the gate, not a swallowed exception."""
        path = tmp_path / "test_broken.py"
        path.write_text("def test_a(:\n    pass\n")
        assert not _file_declares_tier_marker(path)


class TestEnforceTierMarkersInProcess:
    def test_real_suite_passes(self) -> None:
        """The actual jamma test suite must satisfy the gate.

        If this fails, a real test file is missing a tier marker — fix the
        file rather than the gate. This is also a smoke test that
        ``_enforce_tier_markers`` walks ``tests/`` correctly.
        """
        _enforce_tier_markers()


class TestGateUnderXdist:
    """End-to-end regression: the gate must fire under ``-n N`` (xdist).

    The previous collection-based gate silently no-op'd because xdist's
    controller hook receives an empty ``items`` list. Running this under
    ``-n 2`` proves the source-parse approach is xdist-safe.
    """

    _GATE_CONFTEST = textwrap.dedent(
        '''
        """Stub conftest mirroring the real source-parse gate.

        Kept in-line so the self-test does not depend on importing from
        the parent conftest (which would pull in fixtures unrelated to
        the gate).
        """
        from __future__ import annotations

        import ast
        from pathlib import Path

        import pytest

        _REQUIRED = frozenset({"tier0", "tier1", "tier2", "slow", "benchmark"})


        def _module_marks(tree):
            names = set()
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                if not (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "pytestmark"
                ):
                    continue
                cs = (
                    node.value.elts
                    if isinstance(node.value, (ast.List, ast.Tuple))
                    else [node.value]
                )
                for c in cs:
                    t = c.func if isinstance(c, ast.Call) else c
                    if (
                        isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Attribute)
                        and isinstance(t.value.value, ast.Name)
                        and t.value.value.id == "pytest"
                        and t.value.attr == "mark"
                    ):
                        names.add(t.attr)
            return names


        def _per_test_marks(tree):
            names = set()
            for node in ast.walk(tree):
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    continue
                for d in node.decorator_list:
                    t = d.func if isinstance(d, ast.Call) else d
                    if (
                        isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Attribute)
                        and isinstance(t.value.value, ast.Name)
                        and t.value.value.id == "pytest"
                        and t.value.attr == "mark"
                    ):
                        names.add(t.attr)
            return names


        def pytest_configure(config):
            if hasattr(config, "workerinput"):
                return
            tests_dir = Path(str(config.rootpath))
            missing = []
            for path in sorted(tests_dir.rglob("test_*.py")):
                try:
                    tree = ast.parse(path.read_text())
                except (SyntaxError, OSError):
                    missing.append(path.name)
                    continue
                if (_module_marks(tree) | _per_test_marks(tree)) & _REQUIRED:
                    continue
                missing.append(path.name)
            if missing:
                raise pytest.UsageError(
                    "no tier marker: " + ", ".join(sorted(set(missing)))
                )
        '''
    )

    _INI = textwrap.dedent(
        """
        [pytest]
        markers =
            tier0: fast
            tier1: parity
            tier2: scale
            slow: independent
            benchmark: pytest-benchmark
        """
    )

    def test_gate_fires_under_xdist(self, pytester: pytest.Pytester) -> None:
        """Unmarked file must fail the gate even with ``-n 2``."""
        pytester.makeini(self._INI)
        pytester.makeconftest(self._GATE_CONFTEST)
        pytester.makepyfile(
            test_unmarked="""
            def test_does_a_thing():
                assert True
            """,
        )
        result = pytester.runpytest_subprocess("-n", "2")
        assert result.ret != 0, (
            "Gate should fail under -n 2; the previous collection-based "
            "gate silently passed."
        )
        # Either stderr or stdout depending on xdist's plumbing — search both.
        combined = "\n".join([*result.errlines, *result.outlines])
        assert "test_unmarked.py" in combined
        assert "no tier marker" in combined

    def test_gate_passes_when_marked_under_xdist(
        self, pytester: pytest.Pytester
    ) -> None:
        """Counter-test: a marked file passes under ``-n 2``."""
        pytester.makeini(self._INI)
        pytester.makeconftest(self._GATE_CONFTEST)
        pytester.makepyfile(
            test_marked="""
            import pytest
            pytestmark = pytest.mark.tier0

            def test_a(): pass
            def test_b(): pass
            """,
        )
        result = pytester.runpytest_subprocess("-n", "2")
        assert result.ret == 0, (
            f"Gate should pass on marked file. ret={result.ret}\n"
            f"stdout={result.outlines!r}\nstderr={result.errlines!r}"
        )
