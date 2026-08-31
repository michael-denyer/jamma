"""Tests for the JAMMA_SENTINEL_UB env-var injection resolved by
``resolve_build_spec`` for ``LMM_ACCEL_SPEC``.

When ``JAMMA_SENTINEL_UB`` is truthy, ``resolve_build_spec`` appends
``-DJAMMA_SENTINEL_UB`` to the base compile flags so the sanitizer workflow's
sentinel-meta-test job can rebuild ``_lmm_accel.so`` with the gated heap-OOB
function exposed.

``resolve_build_spec`` is a pure function of the spec and the environment, so
these tests pass ``env`` as a plain dict and assert on the returned tuple. No
compiler runs and nothing is mocked — the earlier version drove a full
``compile_extension`` and swallowed the post-link import failure to capture
kwargs off a monkeypatched ``execute_build``.
"""

from __future__ import annotations

import pytest

from jamma._build_support.build_models import LMM_ACCEL_SPEC, resolve_build_spec

pytestmark = pytest.mark.tier0

_SENTINEL = "-DJAMMA_SENTINEL_UB"


def _resolve(env: dict[str, str]) -> tuple[str, ...]:
    return resolve_build_spec(LMM_ACCEL_SPEC, dev_mode=True, env=env)


def test_no_sentinel_no_injection():
    """Unset env: only the dev base flag, no sentinel macro."""
    assert _resolve({}) == ("-march=native",)


def test_sentinel_set_to_1_injects_macro():
    """JAMMA_SENTINEL_UB=1: sentinel present, appended AFTER -march=native."""
    resolved = _resolve({"JAMMA_SENTINEL_UB": "1"})
    assert resolved == ("-march=native", _SENTINEL)
    assert resolved.index(_SENTINEL) > resolved.index("-march=native")


@pytest.mark.parametrize("value", ["0", "", "  "])
def test_sentinel_off_values(value):
    """'', '0', '  ' (after .strip()): no injection."""
    assert _SENTINEL not in _resolve({"JAMMA_SENTINEL_UB": value})


@pytest.mark.parametrize("value", ["1", "true", "yes", " 1 "])
def test_sentinel_truthy_values_engage(value):
    """Various truthy values all engage the gate."""
    assert _SENTINEL in _resolve({"JAMMA_SENTINEL_UB": value})


def test_sentinel_orthogonal_to_sanitize():
    """JAMMA_SANITIZE does not affect the resolved base flags.

    The two env vars are orthogonal: the sanitizer flags are layered on later
    by ``apply_sanitizer_overrides`` inside ``run_build`` (covered by
    test_build_support_sanitizer_override.py), so ``resolve_build_spec`` returns
    the sentinel macro and nothing sanitizer-related regardless of
    JAMMA_SANITIZE.
    """
    resolved = _resolve(
        {"JAMMA_SENTINEL_UB": "1", "JAMMA_SANITIZE": "address,undefined"}
    )
    assert resolved == ("-march=native", _SENTINEL)


def test_wheel_mode_never_march_or_sentinel():
    """The wheel path honours CFLAGS only — never -march=native or the sentinel.

    This is the portability guarantee: dev-only flags cannot leak into the
    portable wheel, by construction, even with the sentinel env var set.
    """
    resolved = resolve_build_spec(
        LMM_ACCEL_SPEC,
        dev_mode=False,
        env={"JAMMA_SENTINEL_UB": "1", "CFLAGS": "-march=x86-64-v3 -O2"},
    )
    assert resolved == ("-march=x86-64-v3", "-O2")
    assert "-march=native" not in resolved
    assert _SENTINEL not in resolved
