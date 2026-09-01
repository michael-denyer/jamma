"""Tests for scripts/check_forbidden_patches.py.

The gate resolves each patch target through the test module's import
table and then through the source tree's import tables, so the cases here
build a tiny ``src/jamma`` tree beside the test file. A regex gate would
pass every "wrapped" and "aliased" case below; that is what the AST
rewrite exists to close.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.conftest import install_lint_script

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check_forbidden_patches.py"

_SRC = {
    "jamma/__init__.py": "",
    "jamma/jlinalg/__init__.py": (
        "def eigh(K, inplace=False): ...\nblas_has_dsyevd = 1\n"
    ),
    "jamma/lmm/__init__.py": "",
    "jamma/lmm/accel.py": "_accel = None\n",
    "jamma/lmm/eigen.py": (
        "from jamma import jlinalg\n"
        "from jamma.core.memory import check_memory_available\n"
        "\ndef eigendecompose_kinship(K): ...\n"
    ),
    "jamma/lmm/pab.py": "def calc_pab(): ...\n",
    "jamma/lmm/likelihood.py": "def reml_log_likelihood(): ...\n",
    "jamma/pipeline.py": "from jamma.lmm.eigen import eigendecompose_kinship\n",
    "jamma/core/__init__.py": "",
    "jamma/core/memory.py": "def check_memory_available(): ...\n",
}

_FORBIDDEN = {
    "string": """
        from unittest.mock import patch
        patch("jamma.lmm.pab.calc_pab")
        """,
    "wrapped": """
        from unittest.mock import patch
        patch(
            "jamma.lmm.pab.calc_pab",
            side_effect=None,
        )
        """,
    "patch.object-alias": """
        from unittest.mock import patch
        import jamma.lmm.pab as lik
        patch.object(lik, "calc_pab")
        """,
    "setattr-from-alias": """
        from jamma.lmm import pab as lk
        def test(monkeypatch):
            monkeypatch.setattr(lk, "calc_pab", None)
        """,
    "setattr-string": """
        def test(monkeypatch):
            monkeypatch.setattr("jamma.lmm.pab.calc_pab", None)
        """,
    "likelihood-string": """
        from unittest.mock import patch
        patch("jamma.lmm.likelihood.reml_log_likelihood")
        """,
    "mocker": """
        def test(mocker):
            mocker.patch("numpy.linalg.eigh")
        """,
    "patch.object-np": """
        import numpy as np
        from unittest.mock import patch
        patch.object(np.linalg, "eigh")
        """,
    "whole-module-at-import-site": """
        from unittest.mock import patch
        patch("jamma.lmm.eigen.jlinalg")
        """,
    "attr-at-import-site": """
        from unittest.mock import patch
        patch("jamma.lmm.eigen.jlinalg.eigh")
        """,
    "reexport-at-import-site": """
        from unittest.mock import patch
        patch("jamma.pipeline.eigendecompose_kinship")
        """,
    "mock.patch": """
        from unittest import mock
        mock.patch("scipy.stats.f")
        """,
}

_ALLOWED = {
    "os-boundary": """
        from unittest.mock import patch
        patch("psutil.virtual_memory")
        """,
    "memory-probe-via-reexport": """
        from unittest.mock import patch
        patch("jamma.lmm.eigen.check_memory_available")
        """,
    "accel-seam": """
        from jamma.lmm import accel
        def test(monkeypatch):
            monkeypatch.setattr(accel, "_accel", None)
        """,
    "detection-flag": """
        from unittest.mock import patch
        patch("jamma.lmm.eigen.jlinalg.blas_has_dsyevd", 1)
        """,
    "all-caps-knob": """
        from jamma.lmm import special
        def test(monkeypatch):
            monkeypatch.setattr(special, "_CF_MAX_ITER", 0)
        """,
    "local-object": """
        def test(runner, monkeypatch):
            monkeypatch.setattr(runner, "_emit", None)
        """,
    "patch.dict": """
        from unittest.mock import patch
        patch.dict("os.environ", {})
        """,
}

_MARKED = {
    "same-line": """
        from unittest.mock import patch
        patch("jamma.lmm.pab.calc_pab")  # allow-patch: spy
        """,
    "line-above": """
        from unittest.mock import patch
        # allow-patch: spy
        patch(
            "jamma.lmm.pab.calc_pab",
        )
        """,
    "closing-line": """
        from unittest.mock import patch
        patch(
            "jamma.lmm.pab.calc_pab",
        )  # allow-patch: spy
        """,
    "comment-block-above": """
        from unittest.mock import patch
        # allow-patch: spy, forwarding to the real function
        # so the numbers are still real
        patch("jamma.lmm.pab.calc_pab")
        """,
    "above-with-header": """
        from unittest.mock import patch
        def test():
            # allow-patch: spy
            with (
                patch("os.getcwd"),
                patch("jamma.lmm.pab.calc_pab"),
            ):
                pass
        """,
}


def _run(tmp_path: Path, test_source: str, *args: str) -> subprocess.CompletedProcess:
    script_copy = install_lint_script(_SCRIPT, tmp_path / "scripts")
    for rel, body in _SRC.items():
        target = tmp_path / "src" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    test_file = tmp_path / "tests" / "test_x.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(textwrap.dedent(test_source))
    return subprocess.run(
        [sys.executable, str(script_copy), *args, str(test_file)],
        capture_output=True,
        text=True,
        check=False,
    )


pytestmark = pytest.mark.tier0


@pytest.mark.parametrize("source", _FORBIDDEN.values(), ids=_FORBIDDEN.keys())
def test_forbidden_target_fails(tmp_path, source):
    result = _run(tmp_path, source)
    assert result.returncode == 1, result.stderr
    assert "tests/test_x.py:" in result.stderr


@pytest.mark.parametrize("source", _ALLOWED.values(), ids=_ALLOWED.keys())
def test_allowed_target_passes(tmp_path, source):
    result = _run(tmp_path, source)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source", _MARKED.values(), ids=_MARKED.keys())
def test_allow_marker_covers_the_call(tmp_path, source):
    result = _run(tmp_path, source)
    assert result.returncode == 0, result.stderr


def test_marker_without_reason_does_not_count(tmp_path):
    source = """
        from unittest.mock import patch
        patch("jamma.lmm.pab.calc_pab")  # allow-patch:
        """
    assert _run(tmp_path, source).returncode == 1


def test_list_prints_every_resolved_site(tmp_path):
    source = """
        from unittest.mock import patch
        patch("psutil.virtual_memory")
        patch("jamma.lmm.eigen.jlinalg.eigh")
        """
    result = _run(tmp_path, source, "--list")
    assert result.returncode == 0
    assert "tests/test_x.py:3: patch psutil.virtual_memory" in result.stdout
    assert "tests/test_x.py:4: patch jamma.jlinalg.eigh" in result.stdout


def test_unreadable_file_fails(tmp_path):
    _run(tmp_path, "")
    bad = tmp_path / "tests" / "test_x.py"
    bad.write_bytes(b"\xff\xfe")
    result = subprocess.run(
        [sys.executable, str(tmp_path / "scripts" / _SCRIPT.name), str(bad)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "could not be read" in result.stderr
