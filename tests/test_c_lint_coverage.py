"""Every C source in the tree is covered by static analysis.

The cppcheck hook was scoped to ``^src/jamma/jlinalg/src/`` from the day it was
added, so it never ran on ``src/jamma/lmm/*.c``. That went unnoticed for the
whole life of the LMM accelerator, including two translation-unit splits. A
lint hook whose ``files:`` pattern silently misses a tree looks identical to a
lint hook that passes, which is why the coverage is asserted rather than
assumed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _REPO_ROOT / ".pre-commit-config.yaml"
_CI_WORKFLOW = _REPO_ROOT / ".github/workflows/ci.yml"
_C_TREES = ("src/jamma/lmm", "src/jamma/jlinalg/src")

pytestmark = pytest.mark.tier0


def _hook(hook_id: str) -> dict:
    config = yaml.safe_load(_CONFIG.read_text())
    for repo in config["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == hook_id:
                return hook
    pytest.fail(f"no {hook_id!r} hook in {_CONFIG.name}")


def _c_sources() -> list[str]:
    found = [
        str(path.relative_to(_REPO_ROOT))
        for tree in _C_TREES
        for path in sorted((_REPO_ROOT / tree).glob("*.c"))
    ]
    assert found, "no C sources found; the tree layout moved"
    return found


def test_cppcheck_covers_every_c_source():
    pattern = re.compile(_hook("cppcheck")["files"])
    uncovered = [src for src in _c_sources() if not pattern.search(src)]
    assert not uncovered, (
        f"cppcheck does not run on {uncovered}. Widen the hook's files: "
        "pattern; an unlinted source is indistinguishable from a clean one."
    )


def test_cppcheck_defines_the_numpy_format_macro():
    """Undefined, NPY_INTP_FMT makes cppcheck give up on _lmm_accel.c.

    The define carries quotes, and a pre-commit ``entry:`` is split into
    arguments without a shell, so putting it there leaves the macro expanding
    to a bare ``ld``. It has to live in the wrapper script.
    """
    runner = _REPO_ROOT / _hook("cppcheck")["entry"]
    assert runner.exists(), f"{runner.name} is missing"
    assert "-DNPY_INTP_FMT='\"ld\"'" in runner.read_text(), (
        "cppcheck reports unknownMacro on NPY_INTP_FMT and stops analysing "
        "_lmm_accel.c, which then looks clean because nothing was checked. "
        "Define it rather than suppressing the id, which would also hide real "
        "parse failures."
    )


def test_ci_pins_an_exact_cppcheck_version():
    """CI and the hook have to run the same linter, deterministically.

    The bare ``apt-get install -y cppcheck`` this replaced resolved to
    whatever the runner image shipped — 2.13.0, against the 2.21.0 the hook
    runs locally. That is two failure modes at once: the lint job can start
    failing on an image bump with no code change, and a CI linter eight
    releases behind the gating hook can never catch what the hook missed.
    """
    lint_env = yaml.safe_load(_CI_WORKFLOW.read_text())["jobs"]["lint"].get("env", {})
    version = str(lint_env.get("CPPCHECK_VERSION", ""))
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), (
        f"the lint job pins CPPCHECK_VERSION to {version!r}. It must be an "
        "exact X.Y.Z so the linter's check set does not move with the runner "
        "image."
    )
    assert re.fullmatch(r"[0-9a-f]{40}", str(lint_env.get("CPPCHECK_COMMIT", ""))), (
        "CPPCHECK_COMMIT must be a full 40-character SHA. Tags are mutable, "
        "so the version alone does not pin what gets built."
    )


def test_ci_does_not_install_cppcheck_from_apt():
    """A package-manager install reintroduces the drift the pin removes.

    Reads the parsed ``run:`` blocks with shell comments stripped, not raw
    file lines: prose mentioning apt is not an apt install, and a scan that
    cannot tell them apart is the defect PR #92 fixed in the section parser.
    """
    jobs = yaml.safe_load(_CI_WORKFLOW.read_text())["jobs"].values()
    commands = [
        line.strip()
        for job in jobs
        for step in job.get("steps", [])
        for line in str(step.get("run", "")).splitlines()
        if not line.lstrip().startswith("#")
    ]
    offenders = [
        line
        for line in commands
        if "cppcheck" in line and re.search(r"\b(apt-get|apt|snap)\b", line)
    ]
    assert not offenders, (
        f"cppcheck is installed by package manager in ci.yml: {offenders}. "
        "Build it from the pinned source instead; the resolved version "
        "otherwise tracks the runner image rather than the hook."
    )


def test_lmm_opts_out_of_clang_format_explicitly():
    """Without this file the repo-root config silently rewrites half of lmm/."""
    opt_out = _REPO_ROOT / "src/jamma/lmm/.clang-format"
    assert opt_out.exists(), (
        "src/jamma/lmm/.clang-format is missing. clang-format falls back to the "
        "repo-root config, which reformats ~5,100 lines of _lmm_accel.c."
    )
    assert "DisableFormat: true" in opt_out.read_text()
