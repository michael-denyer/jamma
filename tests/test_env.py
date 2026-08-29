"""Tests for jamma.core.constants.Env, the single JAMMA_* env-var parser (F3).

Env.current() must read os.environ fresh on every call rather than caching
a module-level singleton: monkeypatch.setenv/delenv only takes effect for
reads that happen after the patch, and the rest of the suite (test_threading,
test_loco_numpy, test_force_numpy_fallback, ...) depends on that per-test
mutability continuing to work.
"""

from __future__ import annotations

import pytest

from jamma.core.constants import Env, env_flag

pytestmark = pytest.mark.tier0

_BOOL_FIELDS = (
    "force_numpy_fallback",
    "no_telemetry",
    "no_openmp",
    "sentinel_ub",
)

_BOOL_FIELD_TO_VAR = {
    "force_numpy_fallback": "JAMMA_FORCE_NUMPY_FALLBACK",
    "no_telemetry": "JAMMA_NO_TELEMETRY",
    "no_openmp": "JAMMA_NO_OPENMP",
    "sentinel_ub": "JAMMA_SENTINEL_UB",
}

_TRUTHY_SPELLINGS = ("1", "true", "yes", "anything")


@pytest.fixture(autouse=True)
def _clear_jamma_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a clean slate for all JAMMA_* vars Env reads."""
    for var in (
        "JAMMA_BLAS_THREADS",
        "JAMMA_LOCO_WORKERS",
        "JAMMA_BACKEND",
        "JAMMA_FORCE_NUMPY_FALLBACK",
        "JAMMA_NO_TELEMETRY",
        "JAMMA_NO_OPENMP",
        "JAMMA_SANITIZE",
        "JAMMA_SENTINEL_UB",
    ):
        monkeypatch.delenv(var, raising=False)


class TestBooleanFieldsTruthiness:
    """Every boolean field shares env_flag's presence-based truthiness rule."""

    @pytest.mark.parametrize("field", _BOOL_FIELDS)
    @pytest.mark.parametrize("value", _TRUTHY_SPELLINGS)
    def test_truthy_spellings_read_true(
        self, monkeypatch: pytest.MonkeyPatch, field: str, value: str
    ) -> None:
        monkeypatch.setenv(_BOOL_FIELD_TO_VAR[field], value)
        assert getattr(Env.current(), field) is True

    @pytest.mark.parametrize("field", _BOOL_FIELDS)
    def test_zero_reads_false(
        self, monkeypatch: pytest.MonkeyPatch, field: str
    ) -> None:
        monkeypatch.setenv(_BOOL_FIELD_TO_VAR[field], "0")
        assert getattr(Env.current(), field) is False

    @pytest.mark.parametrize("field", _BOOL_FIELDS)
    def test_empty_reads_false(
        self, monkeypatch: pytest.MonkeyPatch, field: str
    ) -> None:
        monkeypatch.setenv(_BOOL_FIELD_TO_VAR[field], "")
        assert getattr(Env.current(), field) is False

    @pytest.mark.parametrize("field", _BOOL_FIELDS)
    def test_unset_reads_false(self, field: str) -> None:
        assert getattr(Env.current(), field) is False

    @pytest.mark.parametrize("field", _BOOL_FIELDS)
    def test_matches_env_flag(
        self, monkeypatch: pytest.MonkeyPatch, field: str
    ) -> None:
        """Env's boolean fields must never drift from env_flag's own rule."""
        var = _BOOL_FIELD_TO_VAR[field]
        for value in (*_TRUTHY_SPELLINGS, "0", "", "false", "no", "off"):
            monkeypatch.setenv(var, value)
            assert getattr(Env.current(), field) == env_flag(var), value


class TestRawStringFields:
    """blas_threads_raw / loco_workers_raw / backend_raw pass the raw string
    through unparsed — their int()/choice parsing stays at the call site so
    the existing malformed-value warnings keep firing."""

    def test_blas_threads_raw_unset_is_none(self) -> None:
        assert Env.current().blas_threads_raw is None

    def test_blas_threads_raw_passthrough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JAMMA_BLAS_THREADS", "4")
        assert Env.current().blas_threads_raw == "4"

    def test_loco_workers_raw_unset_is_none(self) -> None:
        assert Env.current().loco_workers_raw is None

    def test_loco_workers_raw_passthrough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JAMMA_LOCO_WORKERS", "4")
        assert Env.current().loco_workers_raw == "4"

    def test_backend_raw_unset_is_none(self) -> None:
        assert Env.current().backend_raw is None

    def test_backend_raw_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JAMMA_BACKEND", "numpy")
        assert Env.current().backend_raw == "numpy"


class TestSanitizeField:
    """sanitize is a stripped, comma-separated string, not a bool."""

    def test_unset_is_empty_string(self) -> None:
        assert Env.current().sanitize == ""

    def test_passthrough_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JAMMA_SANITIZE", "  address,undefined  ")
        assert Env.current().sanitize == "address,undefined"


class TestFreshPerCall:
    """Env.current() must never be memoized: a later monkeypatch has to be
    visible to the very next call, since tests and the sanitizer/telemetry
    CLI flows both rely on process-lifetime env mutation."""

    def test_reads_change_between_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert Env.current().no_telemetry is False
        monkeypatch.setenv("JAMMA_NO_TELEMETRY", "1")
        assert Env.current().no_telemetry is True
        monkeypatch.delenv("JAMMA_NO_TELEMETRY")
        assert Env.current().no_telemetry is False


class TestFrozen:
    def test_env_is_frozen(self) -> None:
        env = Env.current()
        with pytest.raises(AttributeError):
            env.no_telemetry = True  # type: ignore[misc]
