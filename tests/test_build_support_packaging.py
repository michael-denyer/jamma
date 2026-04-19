"""Tests that build_support/ ships in sdist but NOT in wheel.

Contract: build_support/compile_and_link.py is a PEP 517 build helper only.
Shipping it in the wheel would silently turn jamma.core.recompile (the
runtime ABI-mismatch recompile path end users depend on) into dead code,
because installed wheels would then find build_support/ on-disk and skip
the runtime fallback.

Runs via ``uv build``. Marked tier2 because it shells out to the build
backend and takes several seconds.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _build(tmp_out: Path) -> tuple[Path, Path]:
    """Run ``uv build -o <tmp_out>`` and return (sdist_path, wheel_path)."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not available on PATH")

    result = subprocess.run(
        [uv, "build", "-o", str(tmp_out)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            f"uv build failed (likely missing C toolchain in test env): "
            f"{result.stderr[-400:]}"
        )

    sdists = list(tmp_out.glob("*.tar.gz"))
    wheels = list(tmp_out.glob("*.whl"))
    if not sdists or not wheels:
        pytest.skip(
            f"uv build did not produce both sdist and wheel: "
            f"sdists={sdists}, wheels={wheels}"
        )
    return sdists[0], wheels[0]


@pytest.mark.tier2
@pytest.mark.slow
def test_build_support_ships_in_sdist_not_wheel(tmp_path):
    """Source builds need build_support/ (hatch hook imports from it).
    Installed wheels must NOT have build_support/ — runtime recompile
    uses jamma.core.recompile, which ships inside src/jamma.
    """
    sdist_path, wheel_path = _build(tmp_path)

    with tarfile.open(sdist_path) as tar:
        sdist_names = tar.getnames()
    with zipfile.ZipFile(wheel_path) as zf:
        wheel_names = zf.namelist()

    # sdist MUST contain build_support/compile_and_link.py.
    sdist_has_compile_and_link = any(
        name.endswith("build_support/compile_and_link.py") for name in sdist_names
    )
    assert sdist_has_compile_and_link, (
        f"build_support/compile_and_link.py missing from sdist {sdist_path.name}. "
        "Source builds will fail: the hatch hook imports from build_support."
    )

    # Wheel MUST NOT contain any build_support/ entry.
    wheel_has_build_support = any(
        "build_support/" in name or name.startswith("build_support")
        for name in wheel_names
    )
    assert not wheel_has_build_support, (
        f"build_support/ was shipped in wheel {wheel_path.name}. "
        "This breaks the runtime-recompile contract: installed wheels must "
        "use jamma.core.recompile, not build_support/. Files found: "
        f"{[n for n in wheel_names if 'build_support' in n]}"
    )
