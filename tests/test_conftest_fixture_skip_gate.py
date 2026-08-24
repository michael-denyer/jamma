"""Self-tests for the mechanisms that keep a dormant test loud.

A test that skips for a reason that is not true reports green forever. Three
ways in have bitten this repo: a wrong fixture path (#147), a guard reading a
source file deleted years earlier (#156), and a capability probe left pointing
at a flag a refactor removed (#182).

``require_fixture`` is the mechanism the suite uses for fixture paths: it raises
``FileNotFoundError`` naming every missing path, so a wrong path fails loudly and
a wrong *directory* reports all of its files at once.

``_enforce_no_dormant_skips`` is the gate that stops such a guard coming back. It
parses every test file at ``pytest_configure`` and rejects three shapes: a skip
whose reason names a fixture, a skip control-dependent on a filesystem check, and
a skip gated on whether a name still exists.

These are meta-rules, so they need regression tests of their own: if a refactor
softens ``require_fixture`` back into a skip, or inverts one of the gate's
predicates, dormant tests would silently re-enter the suite.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.conftest import (
    _attribute_probed_skip_lines,
    _enforce_no_dormant_skips,
    _fixture_skip_lines,
    _path_guarded_skip_lines,
    require_fixture,
)

pytestmark = pytest.mark.tier0

_FIXTURES = Path(__file__).parent / "fixtures"
# A real committed fixture, so deleting it fails these tests too.
_PRESENT = _FIXTURES / "gemma_loco" / "test.bed"


class TestRequireFixture:
    """The mechanism must raise, and must name every miss in one failure."""

    def test_returns_none_when_every_path_exists(self) -> None:
        assert require_fixture(_PRESENT, _PRESENT.with_suffix(".bim")) is None

    def test_raises_file_not_found_on_a_missing_path(self) -> None:
        with pytest.raises(FileNotFoundError):
            require_fixture(_FIXTURES / "nope" / "missing.bed")

    def test_names_every_missing_path(self) -> None:
        """A wrong directory must report all of its files, not stop at the first.

        This is the half of #147 a single ``.exists()`` check could not show: the
        directory and the filename were both wrong, and one check collapsed both
        misses into one reason.
        """
        with pytest.raises(FileNotFoundError) as excinfo:
            require_fixture(
                _FIXTURES / "nope" / "a.bed",
                _FIXTURES / "nope" / "b.bim",
                _FIXTURES / "nope" / "c.fam",
            )
        message = str(excinfo.value)
        assert "a.bed" in message
        assert "b.bim" in message
        assert "c.fam" in message
        assert "3 of 3" in message

    def test_omits_paths_that_exist(self) -> None:
        with pytest.raises(FileNotFoundError) as excinfo:
            require_fixture(_PRESENT, _FIXTURES / "nope" / "missing.bed")
        message = str(excinfo.value)
        assert "1 of 2" in message
        assert _PRESENT.name not in message

    def test_renders_missing_paths_relative_to_the_repo_root(self) -> None:
        with pytest.raises(FileNotFoundError) as excinfo:
            require_fixture(_FIXTURES / "nope" / "missing.bed")
        assert "tests/fixtures/nope/missing.bed" in str(excinfo.value)

    def test_renders_a_path_outside_the_repo_absolutely(self, tmp_path: Path) -> None:
        outside = tmp_path / "elsewhere.bed"
        with pytest.raises(FileNotFoundError) as excinfo:
            require_fixture(outside)
        assert str(outside) in str(excinfo.value)


def _lines(source: str) -> list[int]:
    return _fixture_skip_lines(ast.parse(source))


class TestFixtureSkipDetection:
    """The gate must see every way of writing the guard, and only those."""

    @pytest.mark.parametrize(
        "source",
        [
            'pytest.skip("gemma_loco fixture not available")',
            'pytest.skip("fixture missing")',
            'pytest.skip("no FIXTURE data here")',
            'pytest.skip(reason="mouse_hs1940 fixture absent")',
            '@pytest.mark.skipif(True, reason="fixture not built")\ndef test_x(): pass',
        ],
    )
    def test_flags_a_skip_naming_a_fixture(self, source: str) -> None:
        """Wording must not matter.

        The runtime backstop this replaced matched the exact phrase "fixture not
        available", so every other wording here sailed straight through it.
        """
        assert _lines(source), f"gate missed: {source}"

    @pytest.mark.parametrize(
        "source",
        [
            'pytest.skip("C extension not available")',
            'pytest.skip("uv not available on PATH")',
            'pytest.skip("no vendor DSYRK on this build")',
            '@pytest.mark.skipif(True, reason="C extension not compiled")\ndef t(): 0',
        ],
    )
    def test_leaves_environment_skips_alone(self, source: str) -> None:
        """Optional C kernels and missing tooling are legitimate skips."""
        assert not _lines(source), f"gate over-reached: {source}"

    def test_ignores_a_computed_reason(self) -> None:
        """A non-literal reason cannot be judged from source, so it is not guessed."""
        assert not _lines('pytest.skip(f"{name} fixture not available")')

    def test_ignores_an_unrelated_skip_attribute(self) -> None:
        assert not _lines('shutil.skip("fixture not available")')

    def test_reports_the_line_of_the_call(self) -> None:
        source = 'x = 1\ny = 2\npytest.skip("fixture not available")\n'
        assert _lines(source) == [3]


def _path_lines(source: str) -> list[int]:
    return _path_guarded_skip_lines(ast.parse(source))


class TestPathGuardedSkipDetection:
    """The second detector: a skip reached because a path was not found.

    The word-based detector above only fires when the reason says "fixture".
    That let a real one through for months: ``TestDstedcNoAbort`` read a
    ``dstedc.c`` deleted at 663a22b and skipped with the reason "source not
    available", so the word never appeared and the test reported green on every
    run until #156 removed it. Wording is the wrong thing to key on, because
    the author of the next guard picks it freely. What cannot be avoided is the
    shape: the skip is control-dependent on a filesystem check.
    """

    @pytest.mark.parametrize(
        "source",
        [
            # The exact shape that hid TestDstedcNoAbort.
            'if not p.exists():\n    pytest.skip("source not available")',
            # Positive test with the skip in the else branch.
            'if p.exists():\n    run()\nelse:\n    pytest.skip("nope")',
            'if not p.is_file():\n    pytest.skip("nope")',
            'if not p.is_dir():\n    pytest.skip("nope")',
            'if not os.path.exists(p):\n    pytest.skip("nope")',
            'if not os.path.isfile(p):\n    pytest.skip("nope")',
            # Nested inside a function body, which is how it is really written.
            (
                "def test_x():\n"
                "    src = ROOT / 'a.c'\n"
                "    if not src.exists():\n"
                '        pytest.skip("not built")\n'
            ),
            # Decorator form.
            '@pytest.mark.skipif(not P.exists(), reason="anything")\ndef t(): 0',
            '@pytest.mark.skipif(condition=not P.exists(), reason="x")\ndef t(): 0',
        ],
    )
    def test_flags_a_skip_guarded_by_a_path_check(self, source: str) -> None:
        """Wording is irrelevant; not one of these reasons says "fixture"."""
        assert _path_lines(source), f"gate missed: {source}"

    @pytest.mark.parametrize(
        "source",
        [
            # Optional tooling and optional imports are legitimate skips.
            'if shutil.which("uv") is None:\n    pytest.skip("uv not on PATH")',
            'if find_spec("numba") is None:\n    pytest.skip("numba absent")',
            'if not HAS_C_EXTENSION:\n    pytest.skip("C extension not built")',
            'if os.environ.get("CI") != "1":\n    pytest.skip("CI only")',
            # A path check that fails rather than skips is the correct pattern.
            'if not p.exists():\n    pytest.fail("recorder wrote no file")',
            # An assertion on a path is not a skip either.
            "assert p.exists()",
        ],
    )
    def test_leaves_legitimate_guards_alone(self, source: str) -> None:
        assert not _path_lines(source), f"gate over-reached: {source}"

    def test_reports_the_line_of_the_skip_not_the_if(self) -> None:
        source = 'x = 1\nif not p.exists():\n    pytest.skip("nope")\n'
        assert _path_lines(source) == [3]


def _probe_lines(source: str) -> list[int]:
    return _attribute_probed_skip_lines(ast.parse(source))


class TestAttributeProbedSkipDetection:
    """The third detector: a skip gated on whether a name still exists.

    Neither detector above sees this one. The reason is honest and says nothing
    about a fixture, and there is no filesystem check anywhere. The guard reads
    as a capability check, and behaves as one until the attribute it probes is
    renamed or deleted, at which point it answers False forever and every test
    behind it goes quiet. #182 deleted the ``_C_FUSED_AVAILABLE`` flag while
    ``test_lmm_accel_fused.py`` still probed for it, and nine tests covering the
    live fused Wald kernel skipped from that merge onward on a machine with the
    extension fully built.
    """

    @pytest.mark.parametrize(
        "source",
        [
            # The exact shape that hid TestFusedParity: the probe is in a
            # module-level flag, and the decorator only references the flag.
            (
                'AVAILABLE = mod._accel is not None and hasattr(mod, "_C_FUSED")\n'
                '@pytest.mark.skipif(not AVAILABLE, reason="fused C absent")\n'
                "def test_x(): pass\n"
            ),
            # Probe written straight into the decorator.
            '@pytest.mark.skipif(not hasattr(m, "k"), reason="x")\ndef t(): 0',
            '@pytest.mark.skipif(condition=not hasattr(m, "k"))\ndef t(): 0',
            # getattr with a default is the same silent answer.
            'FLAG = getattr(m, "k", None)\n@pytest.mark.skipif(not FLAG)\ndef t(): 0',
            # Control-flow form.
            'if not hasattr(m, "k"):\n    pytest.skip("no kernel")',
            # One hop through a second binding.
            (
                'RAW = hasattr(m, "k")\n'
                "FLAG = RAW and other\n"
                "@pytest.mark.skipif(not FLAG)\n"
                "def t(): 0\n"
            ),
        ],
    )
    def test_flags_a_skip_gated_on_an_attribute_probe(self, source: str) -> None:
        assert _probe_lines(source), f"gate missed: {source}"

    @pytest.mark.parametrize(
        "source",
        [
            # Gating on the capability itself is the correct pattern.
            '@pytest.mark.skipif(m._accel is None, reason="C absent")\ndef t(): 0',
            'if not HAS_C_EXTENSION:\n    pytest.skip("C extension not built")',
            # A probe that asserts rather than skips is what the fix looks like.
            'assert hasattr(m, "k")',
            'if not hasattr(m, "k"):\n    pytest.fail("kernel went missing")',
            # Probes outside any skip are ordinary reflection.
            'def test_x():\n    assert hasattr(result, "beta")',
            # An unrelated skip attribute is not pytest's.
            'if not hasattr(m, "k"):\n    shutil.skip("nope")',
        ],
    )
    def test_leaves_legitimate_guards_alone(self, source: str) -> None:
        assert not _probe_lines(source), f"gate over-reached: {source}"

    def test_a_self_referential_binding_terminates(self) -> None:
        """Following bindings must not loop on a name that refers to itself."""
        assert not _probe_lines("FLAG = FLAG and other\n")

    def test_reports_the_line_of_the_skip_not_the_if(self) -> None:
        source = 'x = 1\nif not hasattr(m, "k"):\n    pytest.skip("nope")\n'
        assert _probe_lines(source) == [3]


class TestGateOverTheRealSuite:
    """The gate must pass on the tree as committed, and be able to fail."""

    def test_the_real_suite_is_clean(self) -> None:
        """No in-tree test guards a fixture or a path with a skip.

        Runs the same sweep ``pytest_configure`` runs. #149 replaced the guards it
        knew about; ``test_fixture_manifest.py`` kept one whose reason avoided the
        old backstop's phrase, and this sweep is what found it. #156 removed the
        last path-guarded one, which no wording-based check could ever have seen.
        """
        _enforce_no_dormant_skips()

    def test_a_planted_fixture_skip_is_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prove the gate can fail, so a clean run means something."""
        planted = tmp_path / "test_planted.py"
        planted.write_text(
            "import pytest\n\n\n"
            "def test_needs_data():\n"
            '    pytest.skip("mouse_hs1940 fixture not available")\n'
        )
        monkeypatch.setattr("tests.conftest._TESTS_DIR", tmp_path)
        with pytest.raises(pytest.UsageError, match=r"name a fixture in their reason"):
            _enforce_no_dormant_skips()

    def test_a_planted_path_guarded_skip_is_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reason says nothing incriminating; only the shape gives it away."""
        planted = tmp_path / "test_planted.py"
        planted.write_text(
            "import pytest\n"
            "from pathlib import Path\n\n\n"
            "def test_reads_a_source_file():\n"
            "    src = Path('src/jamma/jlinalg/src/dstedc.c')\n"
            "    if not src.exists():\n"
            '        pytest.skip("source not available")\n'
        )
        monkeypatch.setattr("tests.conftest._TESTS_DIR", tmp_path)
        with pytest.raises(pytest.UsageError, match=r"guarded by a filesystem check"):
            _enforce_no_dormant_skips()

    def test_a_planted_attribute_probed_skip_is_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The path is fine and the reason is honest; only the probe gives it away."""
        planted = tmp_path / "test_planted.py"
        planted.write_text(
            "import pytest\n\n"
            "import jamma.lmm.compute_numpy as compute_numpy\n\n"
            "_available = compute_numpy._accel is not None and hasattr(\n"
            '    compute_numpy, "_C_FUSED_AVAILABLE"\n'
            ")\n\n\n"
            '@pytest.mark.skipif(not _available, reason="fused C not available")\n'
            "def test_fused_kernel():\n"
            "    pass\n"
        )
        monkeypatch.setattr("tests.conftest._TESTS_DIR", tmp_path)
        with pytest.raises(pytest.UsageError, match=r"gated on whether a name exists"):
            _enforce_no_dormant_skips()

    def test_every_category_reports_together(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One run must name every offender, not stop at the first category.

        Same reasoning as ``require_fixture`` naming every missing path at once:
        fixing one and re-running to discover the next is the slow path.
        """
        (tmp_path / "test_word.py").write_text(
            'import pytest\n\n\ndef test_a():\n    pytest.skip("fixture absent")\n'
        )
        (tmp_path / "test_shape.py").write_text(
            "import pytest\n\n\n"
            "def test_b():\n"
            "    if not p.exists():\n"
            '        pytest.skip("nope")\n'
        )
        (tmp_path / "test_probe.py").write_text(
            "import pytest\n\n\n"
            "def test_c():\n"
            '    if not hasattr(m, "k"):\n'
            '        pytest.skip("nope")\n'
        )
        monkeypatch.setattr("tests.conftest._TESTS_DIR", tmp_path)
        with pytest.raises(pytest.UsageError) as excinfo:
            _enforce_no_dormant_skips()
        message = str(excinfo.value)
        assert "test_word.py" in message
        assert "test_shape.py" in message
        assert "test_probe.py" in message
