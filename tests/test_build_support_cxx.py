"""C++ formatting has independent compiler and strict floating-point policy."""

from pathlib import Path

import pytest

from jamma._build_support.build_execution import Toolchain, detect_toolchain
from jamma._build_support.build_models import (
    LMM_ACCEL_SPEC,
    MATRIX_TEXT_SPEC,
    resolve_build_spec,
    resolve_cflags_for,
)
from jamma._build_support.find_compiler import find_cxx_compiler

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize("compiler", ["", "does-not-exist-jamma-cxx"])
def test_explicit_invalid_cxx_does_not_fall_back(monkeypatch, compiler: str) -> None:
    monkeypatch.setenv("CXX", compiler)
    assert find_cxx_compiler() is None


def test_cpp_does_not_probe_c_or_openmp(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("C++ text formatting must not depend on C/OpenMP detection")

    monkeypatch.setattr(
        "jamma._build_support.find_compiler.find_c_compiler", unexpected
    )
    monkeypatch.setattr(
        "jamma._build_support.find_compiler.find_cxx_compiler",
        lambda: ("c++", ["-pthread"]),
    )
    monkeypatch.setattr(
        "jamma._build_support.openmp_detect.detect_openmp_flags", unexpected
    )
    toolchain = detect_toolchain(language="c++", uses_openmp=False)
    assert isinstance(toolchain, Toolchain)
    assert toolchain.cc_cmd == "c++"
    assert toolchain.cc_extra == ("-pthread",)
    assert toolchain.omp_compile == toolchain.omp_link == ()


def test_cpp_flags_preserve_special_values_and_c_policy() -> None:
    env = {"CFLAGS": "-march=x86-64-v3", "CXXFLAGS": "-Ofast"}
    cpp_extra = resolve_build_spec(MATRIX_TEXT_SPEC, dev_mode=False, env=env)
    assert cpp_extra == ("-Ofast",)
    flags = resolve_cflags_for(Path("_matrix_text.cpp"), set(), [], list(cpp_extra))
    assert "-std=c++17" in flags
    assert "-std=c11" not in flags
    assert flags.index("-fno-fast-math") > flags.index("-Ofast")
    assert resolve_build_spec(LMM_ACCEL_SPEC, dev_mode=False, env=env) == (
        "-march=x86-64-v3",
    )
