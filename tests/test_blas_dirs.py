"""Tests for jamma.jlinalg._blas_dirs.probe_plan().

probe_plan() is the Python replacement for the pathlib-through-CPython-API
directory discovery that used to live in blas_dispatch.c. It only returns
candidate directories; it never dlopens or dlsyms anything, so these tests
check the returned (kind, path) pairs against a faked site-packages tree
rather than the actual BLAS backend on this machine.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from jamma.jlinalg import _blas_dirs

pytestmark = pytest.mark.tier0


def _fake_spec(origin: Path) -> importlib.machinery.ModuleSpec:
    spec = importlib.util.spec_from_file_location("fake", str(origin))
    assert spec is not None
    return spec


def _find_spec_for(numpy_origin: Path | None, mkl_origin: Path | None):
    """Build a find_spec stub that answers for "numpy" and "mkl" only."""

    def _find_spec(name: str, *args, **kwargs):
        if name == "numpy" and numpy_origin is not None:
            return _fake_spec(numpy_origin)
        if name == "mkl" and mkl_origin is not None:
            return _fake_spec(mkl_origin)
        return None

    return _find_spec


class TestProbePlanNumpyLayouts:
    """Candidate dirs for numpy's bundled-library layouts (OpenBLAS, Accelerate)."""

    def test_openblas_layout(self, tmp_path):
        """A numpy wheel with .libs/, _core/.libs/, and the sibling dir all appear."""
        site_packages = tmp_path / "site-packages"
        numpy_init = site_packages / "numpy" / "__init__.py"
        numpy_init.parent.mkdir(parents=True)
        numpy_init.write_text("")

        with patch(
            "importlib.util.find_spec",
            side_effect=_find_spec_for(numpy_init, None),
        ):
            plan = _blas_dirs.probe_plan()

        kinds = {kind for kind, _ in plan}
        assert kinds == {"openblas_or_mkl"}
        paths = {Path(p) for _, p in plan}
        numpy_dir = numpy_init.resolve().parent
        assert numpy_dir / ".libs" in paths
        assert numpy_dir / "_core" / ".libs" in paths
        assert numpy_dir.parent / "numpy.libs" in paths

    def test_accelerate_layout_no_bundled_libs(self, tmp_path):
        """No .libs directories on disk still yields the three candidate paths.

        C does the existence check via opendir; probe_plan() only names
        candidates, matching the old code's behaviour of trying every path
        whether or not it exists (e.g. macOS Accelerate ships no numpy.libs).
        """
        site_packages = tmp_path / "site-packages"
        numpy_init = site_packages / "numpy" / "__init__.py"
        numpy_init.parent.mkdir(parents=True)
        numpy_init.write_text("")

        with patch(
            "importlib.util.find_spec",
            side_effect=_find_spec_for(numpy_init, None),
        ):
            plan = _blas_dirs.probe_plan()

        assert len(plan) == 3
        assert all(kind == "openblas_or_mkl" for kind, _ in plan)

    def test_numpy_not_importable(self, tmp_path):
        """No numpy spec (find_spec returns None) yields no numpy candidates."""
        with patch("importlib.util.find_spec", side_effect=_find_spec_for(None, None)):
            plan = _blas_dirs.probe_plan()
        assert plan == []


class TestProbePlanMklLayout:
    """Candidate dirs for a pip-installed ILP64 MKL package."""

    def test_mkl_layout(self, tmp_path):
        site_packages = tmp_path / "site-packages"
        mkl_init = site_packages / "mkl" / "__init__.py"
        mkl_init.parent.mkdir(parents=True)
        mkl_init.write_text("")

        with patch(
            "importlib.util.find_spec",
            side_effect=_find_spec_for(None, mkl_init),
        ):
            plan = _blas_dirs.probe_plan()

        kinds = {kind for kind, _ in plan}
        assert kinds == {"mkl"}
        paths = {Path(p) for _, p in plan}
        mkl_dir = mkl_init.resolve().parent
        assert mkl_dir / "mkl.libs" in paths
        assert mkl_dir.parent / "mkl.libs" in paths

    def test_mkl_not_importable(self, tmp_path):
        with patch("importlib.util.find_spec", side_effect=_find_spec_for(None, None)):
            plan = _blas_dirs.probe_plan()
        assert not any(kind == "mkl" for kind, _ in plan)


class TestProbePlanOrderAndCombination:
    """Both numpy and mkl candidates appear, numpy candidates first."""

    def test_numpy_then_mkl_order(self, tmp_path):
        site_packages = tmp_path / "site-packages"
        numpy_init = site_packages / "numpy" / "__init__.py"
        numpy_init.parent.mkdir(parents=True)
        numpy_init.write_text("")
        mkl_init = site_packages / "mkl" / "__init__.py"
        mkl_init.parent.mkdir(parents=True)
        mkl_init.write_text("")

        with patch(
            "importlib.util.find_spec",
            side_effect=_find_spec_for(numpy_init, mkl_init),
        ):
            plan = _blas_dirs.probe_plan()

        kinds = [kind for kind, _ in plan]
        assert kinds == ["openblas_or_mkl"] * 3 + ["mkl"] * 2

    def test_plan_entries_are_str_str_tuples(self, tmp_path):
        numpy_init = tmp_path / "numpy" / "__init__.py"
        numpy_init.parent.mkdir(parents=True)
        numpy_init.write_text("")

        with patch(
            "importlib.util.find_spec",
            side_effect=_find_spec_for(numpy_init, None),
        ):
            plan = _blas_dirs.probe_plan()

        for entry in plan:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            kind, path = entry
            assert isinstance(kind, str)
            assert isinstance(path, str)
