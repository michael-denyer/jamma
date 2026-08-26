"""Tests for jamma.core.recompile.auto_recompile_c_extension.

Covers the four observable outcomes of the runtime recompile shim:
  1. Compiler module import fails -> False + debug log
  2. Compiler module raises during compile -> False + warning log + fallback msg
  3. Compiler module returns False -> False + warning log + fallback msg
  4. Compiler module returns True -> True + sys.modules key evicted

The shim is a thin import-retry wrapper around the compile_extension()
entry points in jamma.lmm._compile_accel and jamma.jlinalg._compile_jlinalg.
Tests here run against the shim's public contract only; no real
subprocess / compilation ever fires. End-to-end wheel-install ->
auto_recompile_c_extension -> real compile coverage is tracked as a
follow-up (tier2, scratch venv) and is not yet implemented.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from jamma.core.recompile import _import_and_validate, auto_recompile_c_extension


def _fake_spec(*, module_name, compiler_module, sys_module_key, label):
    """Build a BuildSpec carrying only the load identity these tests exercise.

    The build fields are dummies — auto_recompile_c_extension reads only
    module_name / compiler_module / sys_module_key / fallback_label.
    """
    from jamma._build_support.compile_and_link import BuildSpec

    return BuildSpec(
        package_parts=(),
        source_parts=(),
        include_parts=(),
        sources=(),
        lapack_sources=(),
        output_stem=module_name,
        module_name=module_name,
        compiler_module=compiler_module,
        sys_module_key=sys_module_key,
        fallback_label=label,
    )


def _recompile(*, module_name, compiler_module, sys_module_key, label):
    return auto_recompile_c_extension(
        _fake_spec(
            module_name=module_name,
            compiler_module=compiler_module,
            sys_module_key=sys_module_key,
            label=label,
        )
    )


@pytest.fixture(autouse=True)
def _isolate_lock_files(monkeypatch, tmp_path):
    """Redirect every test's lock file into tmp_path.

    Without this, _lock_path_for derives the lock path from the fake
    sys_module_key (e.g. "jamma._fake_ext_success") and writes a
    .lock file into the real src/jamma/ source tree on every run.
    """
    from jamma.core import recompile as recompile_mod

    monkeypatch.setattr(
        recompile_mod,
        "_lock_path_for",
        lambda key: tmp_path / f"{key.replace('.', '_')}.lock",
    )


def _make_fake_compiler(module_name: str, *, outcome):
    """Build a fake compiler module exposing ``compile_extension(verbose=...)``.

    ``outcome`` controls behavior:
      - True  -> returns True
      - False -> returns False
      - Exception subclass -> raised when compile_extension is called
    """
    mod = types.ModuleType(module_name)

    def compile_extension(verbose: bool = False, on_retry=None) -> bool:
        del verbose, on_retry
        if isinstance(outcome, type) and issubclass(outcome, BaseException):
            raise outcome("fake failure")
        return bool(outcome)

    mod.compile_extension = compile_extension  # type: ignore[attr-defined]
    return mod


@pytest.mark.tier0
def test_compiler_module_missing_returns_false(monkeypatch):
    """Corrupted install case: the compiler module itself is missing."""
    module_name = "jamma._compiler_that_does_not_exist"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    result = _recompile(
        module_name="_fake_ext",
        compiler_module=module_name,
        sys_module_key="jamma._fake_ext",
        label="fake",
    )

    assert result is False


@pytest.mark.tier0
def test_compiler_raises_returns_false_and_does_not_evict(monkeypatch):
    """Exception during compile -> False; stale module entry must remain.

    If the shim popped sys.modules on failure, a subsequent import would
    try to rebuild from scratch and likely re-hit the same crash loop.
    """
    compiler_name = "jamma._fake_compiler_raises"
    sys_key = "jamma._fake_ext_raises"

    monkeypatch.setitem(
        sys.modules,
        compiler_name,
        _make_fake_compiler(compiler_name, outcome=RuntimeError),
    )
    sentinel = types.ModuleType(sys_key)
    monkeypatch.setitem(sys.modules, sys_key, sentinel)

    result = _recompile(
        module_name="_fake_ext_raises",
        compiler_module=compiler_name,
        sys_module_key=sys_key,
        label="fake",
    )

    assert result is False
    assert sys.modules.get(sys_key) is sentinel, (
        "stale module must NOT be evicted on failure — eviction would "
        "force re-import into the same broken path"
    )


@pytest.mark.tier0
def test_compiler_returns_false_does_not_evict(monkeypatch):
    """compile_extension returned False -> shim returns False, no eviction."""
    compiler_name = "jamma._fake_compiler_false"
    sys_key = "jamma._fake_ext_false"

    monkeypatch.setitem(
        sys.modules,
        compiler_name,
        _make_fake_compiler(compiler_name, outcome=False),
    )
    sentinel = types.ModuleType(sys_key)
    monkeypatch.setitem(sys.modules, sys_key, sentinel)

    result = _recompile(
        module_name="_fake_ext_false",
        compiler_module=compiler_name,
        sys_module_key=sys_key,
        label="fake",
    )

    assert result is False
    assert sys.modules.get(sys_key) is sentinel


@pytest.mark.tier0
def test_successful_recompile_evicts_stale_module(monkeypatch):
    """Success path -> returns True AND pops the stale sys.modules entry so
    subsequent import picks up the freshly compiled .so."""
    compiler_name = "jamma._fake_compiler_success"
    sys_key = "jamma._fake_ext_success"

    monkeypatch.setitem(
        sys.modules,
        compiler_name,
        _make_fake_compiler(compiler_name, outcome=True),
    )
    stale = types.ModuleType(sys_key)
    monkeypatch.setitem(sys.modules, sys_key, stale)

    result = _recompile(
        module_name="_fake_ext_success",
        compiler_module=compiler_name,
        sys_module_key=sys_key,
        label="fake",
    )

    assert result is True
    assert sys_key not in sys.modules, (
        "success must evict the stale module so re-import picks up the "
        "freshly compiled .so — a regression here would silently serve "
        "stale bytecode forever"
    )


@pytest.mark.tier0
def test_successful_recompile_with_no_prior_sys_modules_entry(monkeypatch):
    """pop(key, None) must not raise when the key was never present."""
    compiler_name = "jamma._fake_compiler_no_prior"
    sys_key = "jamma._fake_ext_no_prior"

    monkeypatch.setitem(
        sys.modules,
        compiler_name,
        _make_fake_compiler(compiler_name, outcome=True),
    )
    monkeypatch.delitem(sys.modules, sys_key, raising=False)

    result = _recompile(
        module_name="_fake_ext_no_prior",
        compiler_module=compiler_name,
        sys_module_key=sys_key,
        label="fake",
    )

    assert result is True


@pytest.mark.tier0
def test_on_retry_callback_is_wired_and_emits_warning(monkeypatch, capsys):
    """The runtime recompile shim must pass a non-None on_retry callback
    to compile_extension AND invoking it must emit a warning the user
    can see. Without this, runtime recompile silently falls back to
    single-threaded with no user-visible signal — the exact gap this
    test guards.
    """
    from loguru import logger as _logger

    compiler_name = "jamma._fake_compiler_retry"
    sys_key = "jamma._fake_ext_retry"

    mod = types.ModuleType(compiler_name)
    captured_retry: list[object] = []

    def compile_extension(verbose: bool = False, on_retry=None) -> bool:
        del verbose
        captured_retry.append(on_retry)
        if on_retry is not None:
            on_retry("OpenMP compilation failed, retrying without OpenMP")
        return True

    mod.compile_extension = compile_extension  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, compiler_name, mod)
    monkeypatch.delitem(sys.modules, sys_key, raising=False)

    # Route loguru to stderr so capsys can observe it.
    sink_id = _logger.add(sys.stderr, level="WARNING")
    try:
        result = _recompile(
            module_name="_fake_ext_retry",
            compiler_module=compiler_name,
            sys_module_key=sys_key,
            label="fake",
        )
    finally:
        _logger.remove(sink_id)

    assert result is True
    assert captured_retry, "compile_extension must be called"
    assert captured_retry[0] is not None, (
        "auto_recompile_c_extension must pass a non-None on_retry to "
        "compile_extension so OMP downgrade signals surface"
    )
    captured = capsys.readouterr()
    assert "OpenMP compilation failed" in captured.err, (
        "on_retry invocation must produce a user-visible warning — "
        "loguru must emit, not silently discard, retry notices"
    )


@pytest.mark.tier0
def test_concurrent_recompiles_serialize(monkeypatch, tmp_path):
    """Regression for jamma-oy1c: N threads triggering auto_recompile must
    not interleave inside compile_extension. The file lock around the
    compiler call must serialize them.

    Uses 8 threads with a 100ms critical section and a shared lock path.
    The timestamp-based overlap check is strict: any two threads whose
    [enter, exit] intervals intersect are a lock failure. Exit counter
    plus overlap check together prove mutual exclusion.
    """
    import threading
    import time

    from jamma.core import recompile as recompile_mod

    # All threads share one lock file — the invariant under test.
    shared_lock = tmp_path / "shared.lock"
    monkeypatch.setattr(recompile_mod, "_lock_path_for", lambda key: shared_lock)

    compiler_name = "jamma._fake_compiler_concurrent"
    n_workers = 8
    critical_sleep_s = 0.1

    # Record (enter_ns, exit_ns) per call.
    intervals: list[tuple[int, int]] = []
    intervals_lock = threading.Lock()

    def slow_compile(verbose: bool = False, on_retry=None) -> bool:
        del verbose, on_retry
        enter_ns = time.monotonic_ns()
        time.sleep(critical_sleep_s)
        exit_ns = time.monotonic_ns()
        with intervals_lock:
            intervals.append((enter_ns, exit_ns))
        return True

    fake_mod = types.ModuleType(compiler_name)
    fake_mod.compile_extension = slow_compile  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, compiler_name, fake_mod)

    # Each worker uses a distinct sys_module_key so the post-lock re-import
    # check cannot short-circuit them — isolates lock serialization from
    # the already-built fast-path.
    sys_keys = [f"jamma._fake_ext_concurrent_{i}" for i in range(n_workers)]
    for k in sys_keys:
        monkeypatch.setitem(sys.modules, k, types.ModuleType(k))

    results: list[bool] = []
    results_lock = threading.Lock()

    def worker(sys_key: str) -> None:
        r = _recompile(
            module_name="_fake_ext_concurrent",
            compiler_module=compiler_name,
            sys_module_key=sys_key,
            label="fake",
        )
        with results_lock:
            results.append(r)

    threads = [threading.Thread(target=worker, args=(k,)) for k in sys_keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [True] * n_workers, (
        f"all {n_workers} recompiles must succeed, got {results!r}"
    )
    assert len(intervals) == n_workers, (
        f"compile_extension must run {n_workers} times, got {len(intervals)}"
    )

    # Observable invariant: no two intervals overlap. Sort by enter time
    # and check each subsequent enter is >= previous exit.
    intervals.sort()
    for i in range(1, len(intervals)):
        prev_exit = intervals[i - 1][1]
        this_enter = intervals[i][0]
        assert this_enter >= prev_exit, (
            f"intervals {intervals[i - 1]} and {intervals[i]} overlap — "
            f"file lock failed to serialize. Concurrent linkers will race "
            f"on the .so output path."
        )


@pytest.mark.tier0
def test_concurrent_recompiles_fail_without_lock(monkeypatch, tmp_path):
    """Negative control: if the lock is stubbed to a no-op, overlap IS
    observed. Proves the preceding test would catch a broken lock — it
    isn't just passing because the scheduler happens to serialize.
    """
    import contextlib
    import threading
    import time

    from jamma.core import recompile as recompile_mod

    shared_lock = tmp_path / "shared.lock"
    monkeypatch.setattr(recompile_mod, "_lock_path_for", lambda key: shared_lock)

    # Replace _file_lock with a no-op context manager — simulates a broken
    # lock implementation.
    @contextlib.contextmanager
    def _no_op_lock(_path):
        yield

    monkeypatch.setattr(recompile_mod, "_file_lock", _no_op_lock)

    compiler_name = "jamma._fake_compiler_no_lock"
    n_workers = 8
    critical_sleep_s = 0.1

    intervals: list[tuple[int, int]] = []
    intervals_lock = threading.Lock()

    def slow_compile(verbose: bool = False, on_retry=None) -> bool:
        del verbose, on_retry
        enter_ns = time.monotonic_ns()
        time.sleep(critical_sleep_s)
        exit_ns = time.monotonic_ns()
        with intervals_lock:
            intervals.append((enter_ns, exit_ns))
        return True

    fake_mod = types.ModuleType(compiler_name)
    fake_mod.compile_extension = slow_compile  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, compiler_name, fake_mod)

    sys_keys = [f"jamma._fake_ext_nolock_{i}" for i in range(n_workers)]
    for k in sys_keys:
        monkeypatch.setitem(sys.modules, k, types.ModuleType(k))

    def worker(sys_key: str) -> None:
        _recompile(
            module_name="_fake_ext_nolock",
            compiler_module=compiler_name,
            sys_module_key=sys_key,
            label="fake",
        )

    threads = [threading.Thread(target=worker, args=(k,)) for k in sys_keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Without the lock, threads MUST overlap given 8 workers * 100ms sleep
    # on any modern scheduler. If this assertion ever spuriously fails
    # (e.g. CI on a single-core VM serializes everything), raise n_workers
    # or critical_sleep_s — do not relax the assertion.
    intervals.sort()
    overlap_found = any(
        intervals[i][0] < intervals[i - 1][1] for i in range(1, len(intervals))
    )
    assert overlap_found, (
        "Negative control did not detect overlap — the positive test above "
        "may pass whether or not the lock works. Review test timing."
    )


# --- _lock_path_for coverage ---
#
# The autouse _isolate_lock_files fixture monkeypatches
# recompile_mod._lock_path_for for every test. These tests need the REAL
# function, so they undo the monkeypatch first.


@pytest.mark.tier0
def test_lock_path_for_installed_package_lives_in_package_dir(monkeypatch):
    """For an installed package, the lock file is placed next to the .so so
    concurrent interpreters sharing site-packages serialize on it.

    Uses the ``jamma.core`` package — always present in-tree and packaged.
    """
    monkeypatch.undo()  # drop the autouse _lock_path_for patch

    from jamma.core.recompile import _lock_path_for

    path = _lock_path_for("jamma.core._fake_ext")

    import jamma.core as jamma_core

    # Resolve both sides for symlink parity (e.g. macOS /var vs /private/var).
    pkg_dir = Path(next(iter(jamma_core.__path__))).resolve()
    assert path.resolve().is_relative_to(pkg_dir), (
        f"lock path {path} must live inside package dir {pkg_dir} "
        f"so site-packages-shared interpreters serialize on it"
    )
    import sysconfig as _sysconfig

    ext_suffix = _sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    assert path.name.endswith(f"{ext_suffix}.lock"), (
        f"lock path {path} must end with {ext_suffix}.lock, got {path.name!r}"
    )


@pytest.mark.tier0
def test_lock_path_for_unknown_package_falls_back_to_tempdir(monkeypatch):
    """When find_spec returns None (package truly missing), the helper must
    fall back to tempdir rather than raising.
    """
    monkeypatch.undo()

    import tempfile as _tempfile

    from jamma.core.recompile import _lock_path_for

    path = _lock_path_for("jamma_nonexistent_pkg_xyz._x")

    tempdir = Path(_tempfile.gettempdir()).resolve()
    assert path.resolve().is_relative_to(tempdir), (
        f"fallback path {path} must live in tempdir {tempdir}"
    )
    assert path.name.endswith(".lock")


@pytest.mark.tier0
def test_lock_path_for_toplevel_module_falls_back_to_tempdir(monkeypatch):
    """A sys_module_key with no dot (no package) must hit the tempdir
    fallback branch — the ``if package_name:`` check gates the package-dir
    path and must not raise on top-level names.
    """
    monkeypatch.undo()

    import tempfile as _tempfile

    from jamma.core.recompile import _lock_path_for

    path = _lock_path_for("_toplevel_ext")

    tempdir = Path(_tempfile.gettempdir()).resolve()
    assert path.resolve().is_relative_to(tempdir)
    assert path.name.endswith(".lock")


@pytest.mark.tier0
def test_lock_skipped_when_sibling_recompiled(monkeypatch, tmp_path):
    """After acquiring the lock, if a sibling already rebuilt the .so the
    shim must skip its own compile and return True. Avoids redundant
    rebuilds in pytest-xdist worker pools where N workers all hit the
    same ABI mismatch on first import.
    """
    import importlib as importlib_mod

    from jamma.core import recompile as recompile_mod

    monkeypatch.setattr(
        recompile_mod, "_lock_path_for", lambda key: tmp_path / "shared.lock"
    )

    compiler_name = "jamma._fake_compiler_skip"
    sys_key = "jamma._fake_ext_skip"

    compile_calls = [0]

    def should_not_compile(verbose: bool = False, on_retry=None) -> bool:
        del verbose, on_retry
        compile_calls[0] += 1
        return True

    fake_mod = types.ModuleType(compiler_name)
    fake_mod.compile_extension = should_not_compile  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, compiler_name, fake_mod)

    # The shim's post-lock recheck is:
    #   if sys_module_key not in sys.modules:
    #       importlib.import_module(sys_module_key)  # success -> skip rebuild
    # So we must (a) delete the key from sys.modules and (b) make
    # import_module succeed (simulating "sibling process already rebuilt
    # the .so and registered it"). Patching import_module to populate
    # sys.modules and return a fake module is the cleanest way.
    monkeypatch.delitem(sys.modules, sys_key, raising=False)
    sibling_built = types.ModuleType(sys_key)

    real_import = importlib_mod.import_module

    def fake_import(name):
        if name == sys_key:
            sys.modules[name] = sibling_built
            return sibling_built
        return real_import(name)

    monkeypatch.setattr(recompile_mod.importlib, "import_module", fake_import)

    result = _recompile(
        module_name="_fake_ext_skip",
        compiler_module=compiler_name,
        sys_module_key=sys_key,
        label="fake",
    )

    assert result is True
    assert compile_calls[0] == 0, (
        "compile_extension must be skipped when sibling process already "
        f"rebuilt the .so, but it was called {compile_calls[0]} time(s)"
    )


@pytest.mark.tier0
@pytest.mark.timeout(30)
def test_recompile_refuses_to_recurse_into_its_own_import_probe(monkeypatch):
    """Re-entry must return False, not block on the lock this call already holds.

    ``compile_extension`` verifies its build by deleting ``jamma.<pkg>*`` from
    ``sys.modules`` and re-importing. That re-executes the package ``__init__``
    that called in here, and if the extension still will not load, that
    ``__init__`` calls straight back in. ``flock`` is per open-file-description,
    so the second call opens a second fd and the process blocks against its own
    lock: 0% CPU, two fds, and the .so already written.

    The timeout is what makes a regression a failure rather than a hung suite.
    """
    compiler_name = "jamma._fake_compiler_reentrant"
    sys_key = "jamma._fake_ext_reentrant"
    calls: list[str] = []
    nested_result: list[bool] = []

    mod = types.ModuleType(compiler_name)

    def compile_extension(verbose: bool = False, on_retry=None) -> bool:
        del verbose, on_retry
        calls.append("compile")
        nested_result.append(
            _recompile(
                module_name="_fake_ext_reentrant",
                compiler_module=compiler_name,
                sys_module_key=sys_key,
                label="fake",
            )
        )
        return True

    mod.compile_extension = compile_extension  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, compiler_name, mod)

    result = _recompile(
        module_name="_fake_ext_reentrant",
        compiler_module=compiler_name,
        sys_module_key=sys_key,
        label="fake",
    )

    assert result is True
    assert nested_result == [False], (
        "the re-entrant call must decline rather than recurse; returning True "
        "would let the build run twice inside one lock"
    )
    assert calls == ["compile"], (
        f"compile_extension must run once, not once per re-entry: {calls}"
    )


# --- WARNING-level logging on load failure (surface reason, not silence it) ---


def _fake_build_spec(*, module_name, sys_module_key, fallback_label, required_attrs=()):
    from jamma._build_support.compile_and_link import BuildSpec

    return BuildSpec(
        package_parts=(),
        source_parts=(),
        include_parts=(),
        sources=(),
        lapack_sources=(),
        output_stem=module_name,
        module_name=module_name,
        compiler_module=f"jamma._nonexistent_compiler_for_{module_name}",
        sys_module_key=sys_module_key,
        fallback_label=fallback_label,
        required_attrs=required_attrs,
    )


@pytest.mark.tier0
def test_compiler_module_missing_logs_warning_with_reason(monkeypatch, capsys):
    """Compiler module ImportError must be a WARNING carrying the exception
    text, not a DEBUG line the user never sees. Without this, the
    user-facing 'not available' message never explains why.
    """
    from loguru import logger as _logger

    module_name = "jamma._compiler_that_does_not_exist"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    sink_id = _logger.add(sys.stderr, level="WARNING")
    try:
        result = _recompile(
            module_name="_fake_ext",
            compiler_module=module_name,
            sys_module_key="jamma._fake_ext",
            label="fake",
        )
    finally:
        _logger.remove(sink_id)

    assert result is False
    captured = capsys.readouterr()
    assert "not available" in captured.err
    assert module_name in captured.err
    assert "No module named" in captured.err, (
        "the ImportError text must be included so the user knows why the "
        "compiler module could not be imported"
    )


@pytest.mark.tier0
def test_import_and_validate_import_error_logs_warning_with_reason(capsys):
    """A genuine import failure (dlopen error, missing .so) must be a
    WARNING carrying the exception text, not a DEBUG line the user never
    sees behind the generic 'not available' message.
    """
    from loguru import logger as _logger

    spec = _fake_build_spec(
        module_name="_fake_ext_importerr",
        sys_module_key="jamma._fake_ext_that_does_not_exist_importerr",
        fallback_label="fake-fallback",
    )

    sink_id = _logger.add(sys.stderr, level="WARNING")
    try:
        result = _import_and_validate(spec, expected_abi=1)
    finally:
        _logger.remove(sink_id)

    assert result is None
    captured = capsys.readouterr()
    assert "_fake_ext_importerr" in captured.err
    assert "fake-fallback" in captured.err
    assert "No module named" in captured.err, (
        "the ImportError text must be included so the user knows why the "
        "compiled extension failed to import"
    )


@pytest.mark.tier0
def test_import_and_validate_missing_abi_version_logs_warning(monkeypatch, capsys):
    """A module with no ABI_VERSION attribute must log a WARNING naming the
    missing attribute, not a silent DEBUG line.
    """
    from loguru import logger as _logger

    sys_key = "jamma._fake_ext_no_abi"
    fake_mod = types.ModuleType(sys_key)
    monkeypatch.setitem(sys.modules, sys_key, fake_mod)

    spec = _fake_build_spec(
        module_name="_fake_ext_no_abi",
        sys_module_key=sys_key,
        fallback_label="fake-fallback",
    )

    sink_id = _logger.add(sys.stderr, level="WARNING")
    try:
        result = _import_and_validate(spec, expected_abi=1)
    finally:
        _logger.remove(sink_id)

    assert result is None
    captured = capsys.readouterr()
    assert "ABI_VERSION missing" in captured.err
    assert "fake-fallback" in captured.err


@pytest.mark.tier0
def test_import_and_validate_missing_required_attrs_logs_warning_with_names(
    monkeypatch, capsys
):
    """A module missing a declared required_attrs symbol must log a WARNING
    naming the missing attribute(s), not a silent DEBUG line. A
    required_attrs typo would otherwise trigger a wasted rebuild and then a
    silent fallback with no way to tell why.
    """
    from loguru import logger as _logger

    sys_key = "jamma._fake_ext_missing_attr"
    fake_mod = types.ModuleType(sys_key)
    fake_mod.ABI_VERSION = 1  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, sys_key, fake_mod)

    spec = _fake_build_spec(
        module_name="_fake_ext_missing_attr",
        sys_module_key=sys_key,
        fallback_label="fake-fallback",
        required_attrs=("dgemm", "eigh"),
    )

    sink_id = _logger.add(sys.stderr, level="WARNING")
    try:
        result = _import_and_validate(spec, expected_abi=1)
    finally:
        _logger.remove(sink_id)

    assert result is None
    captured = capsys.readouterr()
    assert "dgemm" in captured.err
    assert "eigh" in captured.err
    assert "fake-fallback" in captured.err


@pytest.mark.tier0
def test_reentrancy_decline_logs_info_that_rebuild_succeeded(monkeypatch, capsys):
    """When the reentrancy guard declines a nested recompile call, the log
    must say the .so was rebuilt successfully and takes effect next
    process — otherwise the caller's own "failed to load" message and this
    line contradict each other (the .so was, in fact, just rebuilt).
    """
    from loguru import logger as _logger

    compiler_name = "jamma._fake_compiler_reentrant_log"
    sys_key = "jamma._fake_ext_reentrant_log"

    mod = types.ModuleType(compiler_name)

    def compile_extension(verbose: bool = False, on_retry=None) -> bool:
        del verbose, on_retry
        _recompile(
            module_name="_fake_ext_reentrant_log",
            compiler_module=compiler_name,
            sys_module_key=sys_key,
            label="fake",
        )
        return True

    mod.compile_extension = compile_extension  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, compiler_name, mod)

    sink_id = _logger.add(sys.stderr, level="INFO")
    try:
        result = _recompile(
            module_name="_fake_ext_reentrant_log",
            compiler_module=compiler_name,
            sys_module_key=sys_key,
            label="fake",
        )
    finally:
        _logger.remove(sink_id)

    assert result is True
    captured = capsys.readouterr()
    assert "rebuilt successfully" in captured.err
    assert "next" in captured.err
