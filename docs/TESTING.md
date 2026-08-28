# Testing

JAMMA's pytest suite balances CI speed against tight numerical parity with
GEMMA. This document is split into three parts:

1. **[Running tests](#1-running-tests)** — setup, commands, CI workflows.
2. **[Test design rules](#2-test-design-rules)** — philosophy, mocking
   policy, anti-patterns, when to skip.
3. **[Suite map and current state](#3-suite-map-and-current-state)** —
   what each subsystem's tests cover, plus a list of tests to improve or
   fold.

> **Source of truth.** The marker list, default `addopts`, and timeout live
> in [`pyproject.toml`](../pyproject.toml) under
> `[tool.pytest.ini_options]`. If anything here disagrees with that file,
> `pyproject.toml` wins.

---

## 1. Running Tests

### 1.1 Framework

| Tool | Min version | Purpose |
|------|-------------|---------|
| `pytest` | `8.0.0` | Test runner |
| `pytest-xdist` | `3.5.0` | Parallel execution (`-n 3`) |
| `pytest-randomly` | `3.15.0` | Order randomization (`--randomly-seed=last`) |
| `pytest-timeout` | `2.3.0` | Per-test cap (`--timeout=120`) |
| `pytest-cov` | `4.0.0` | Local coverage (CI uses slipcover) |
| `pytest-benchmark` | `5.0.0` | Microbenchmarks (run separately, `-n0`) |
| `hypothesis` | `6.100.0` | Property-based tests |
| `scipy` | `1.10.0` | **Test-only.** Used by `test_special.py` and `test_lmm_io_validation.py` for `scipy.stats` reference values |

> **Never make `scipy` a runtime dependency.** It overwrites the ILP64
> numpy build and breaks 100k+ sample eigendecomposition. Production code
> uses the stdlib-only `jamma.special` module instead. New stat functions
> go there first; reach for `scipy.stats` only to produce a reference value.

### 1.2 Setup

```bash
uv sync
uv run python -m jamma.lmm._compile_accel
uv run python -m jamma.jlinalg._compile_jlinalg
```

`tests/conftest.py` warns at session start if any C extension is stale
relative to its source. The pre-push hook
(`scripts/check_c_extension_freshness.py`) is the blocking gate.

### 1.3 Default test run

```bash
uv run pytest tests/ -x
```

Picks up `addopts` from `pyproject.toml` automatically:

```text
-n 3 --randomly-seed=last --benchmark-skip --timeout=120
-m 'not slow and not tier2' --no-cov
```

**Never `-n auto`** — it spawns too many workers and contaminates
BLAS-threaded tests. Use `-n 3` (or `-n0` for benchmarks).

### 1.4 The 120-second timeout

Every test is killed at 120 seconds. Tier0 should finish in <5s, tier1 in
<60s — the cap exists to catch accidental BLAS-on-100k-samples calls,
deadlocks, and infinite loops.

If a test legitimately needs longer:

```python
@pytest.mark.tier2
@pytest.mark.timeout(600)
def test_streaming_100k_samples(): ...
```

A test that needs `@pytest.mark.timeout(N)` for `N > 120` almost always
needs `tier2`/`slow` too — it should not run under the default filter.

### 1.5 Tier system

| Marker | Speed | Description | Runs in CI |
|--------|-------|-------------|-----------|
| `tier0` | <5s each | Pure computation, no I/O, no GEMMA reference | Every push/PR |
| `tier1` | <60s each | Numerical parity against GEMMA reference fixtures | Every push/PR |
| `tier2` | Minutes | Large samples (10k+), memory-constrained scenarios | Slow Tests workflow |
| `slow` | — | Independent slow marker — long but not memory-bound | Slow Tests workflow |
| `benchmark` | — | `pytest-benchmark` microbenchmarks | Manually only |

> **`slow` is independent of `tier2`.** Earlier docs called it an alias; it
> is not. The Slow Tests workflow runs `tier2 or slow`. Use `slow` when a
> test is too long for the default suite but does not need the large-memory
> CI runner that `tier2` implies.

### 1.6 Mandatory tier marker

Every test file must declare at least one tier marker — per-test or
module-level:

```python
import pytest
pytestmark = pytest.mark.tier0
```

A `pytest_configure` hook in `tests/conftest.py` aborts the run with
`tests/<name>.py: file is missing a tier marker (tier0/tier1/tier2/slow/benchmark)`
when this is missing. The gate runs once on the controller before xdist
forks workers (a previous collection-based gate failed open under `-n N`
because xdist controllers skip collection of the worker test files).
Recognises parametrised markers (`@pytest.mark.skipif(...)`) and list-form
`pytestmark`. Promote the file to its correct tier rather than silencing
the check.

### 1.7 Common commands

```bash
# By tier
uv run pytest -m tier0 -x
uv run pytest -m "tier0 or tier1" -x
uv run pytest -m tier2 -v -o 'addopts='        # local, large memory

# Single file / test
uv run pytest tests/test_likelihood_numpy.py -x
uv run pytest tests/test_kinship_validation.py -k "matches_gemma" -x

# Reproduce a random order
uv run pytest tests/ -x                          # reuse last seed
uv run pytest tests/ --randomly-seed=12345
uv run pytest tests/ -p no:randomly              # disable

# Coverage (matches CI)
uv run slipcover --source src/jamma --fail-under 80 -m pytest \
  -m "not tier2 and not slow and not benchmark" -v -n0 -o 'addopts='
```

### 1.8 Benchmarks

Always `-n0` to avoid cross-test timing interference.

```bash
# Microbenchmarks
uv run pytest tests/test_jlinalg_dgemm.py tests/test_jlinalg_dsyrk.py tests/lmm_accel/ \
  -v -n0 --benchmark-only -m benchmark

# End-to-end backend comparison (vs GEMMA on mouse_hs1940)
uv run python scripts/bench_all_backends.py
```

Update the Performance table in `README.md` after any change to runner
logic, chunk sizing, BLAS threading, C extensions, or likelihood
computation.

### 1.9 CI workflows

| Workflow | Trigger | Test command |
|----------|---------|-------------|
| `ci.yml` → `lint` | push/PR | `prek run --all-files` |
| `ci.yml` → `test` (Linux 3.11/3.12, ARM Mac 3.12, Linux MKL ILP64) | push/PR | `pytest -m "not tier2 and not slow and not benchmark" -v -n 3` |
| `ci.yml` → `coverage` | push/PR | `slipcover --fail-under 80 -m pytest ... -n0` plus per-subsystem floors via `scripts/check_subsystem_coverage.py` (lmm 80%, jlinalg 18%, kinship 50%, io 80%) |
| `ci.yml` → `package-smoke` | push/PR | `uv build`, then assert sdist and wheel both ship `_build_support/` and the wheel imports in a clean venv |
| `ci.yml` → `link-check` | push/PR | lychee `--offline` over every `.md`, sharing `lychee.toml` with the pre-commit hook |
| `test-slow.yml` | push to master | `pytest -m "tier2 or slow" -v -o 'addopts=' --no-cov` |
| `fingerprint.yml` | PR touching `_lmm_*.c`, `_lmm_*.h`, `_build_support/`, or the fingerprint scripts | Builds both sides of the merge base and diffs per-entry-point result digests. Tolerance-based tests do not catch last-bit drift |
| `sanitizers.yml` | Wednesday cron + dispatch | `pytest -m "not benchmark and not slow" -n 0 -p no:randomly` (under ASAN/UBSAN) |
| `flaky-detect.yml` | Sunday 06:00 UTC + dispatch | `pytest` under five distinct `--randomly-seed` values, opens an issue on disagreement |

CI overrides `addopts` via `-o 'addopts='` so markers and parallelism are
controlled per-job, independent of the local default.

### 1.10 Running under sanitizers (local repro of CI)

The weekly `Sanitizers (ASAN + UBSAN)` workflow rebuilds the C extensions
with `-fsanitize=address,undefined` and runs the test suite under ASAN.
Linux is the supported repro target — macOS works for UBSAN-only, but
ASAN on macOS requires Xcode's clang and a runtime libasan that is not
typically on the default `LD_PRELOAD` path.

```bash
# Force the NumPy fallback so MKL / dlopen don't confuse ASAN's
# interceptors (RESEARCH §"Pitfall 4": ASAN + dlopen interaction can
# produce false-positive heap-buffer-overflow reports inside
# dispatched BLAS calls).
export JAMMA_FORCE_NUMPY_FALLBACK=1

# Tell the build helpers to inject sanitizer flags via
# apply_sanitizer_overrides.
export JAMMA_SANITIZE=address,undefined

# Use gcc to match the LD_PRELOAD libasan path (mixing gcc-built .so
# and clang's libasan crashes ASAN at startup).
export CC=gcc

# Rebuild both extensions with sanitizers.
uv run python -m jamma.jlinalg._compile_jlinalg
uv run python -m jamma.lmm._compile_accel

# Preload libasan from the SAME compiler used to build the .so files.
export LD_PRELOAD="$(gcc -print-file-name=libasan.so)"
# scripts/asan-suppressions.txt is in LSAN-format (`leak:<symbol>` lines);
# it must be referenced via LSAN_OPTIONS, NOT ASAN_OPTIONS — ASan aborts
# at startup with "failed to parse suppressions" if it tries to read leak:
# patterns as ASan-format ones.
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:strict_string_checks=1:allocator_may_return_null=1:symbolize=1:print_stacktrace=1"
export LSAN_OPTIONS="suppressions=$PWD/scripts/asan-suppressions.txt:print_suppressions=0"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1:symbolize=1"

uv run pytest -m "not benchmark and not slow" -n 0 -p no:randomly
```

To restore your local environment to the default (non-sanitizer) build
after testing, unset the env vars and recompile:

```bash
unset JAMMA_SANITIZE JAMMA_FORCE_NUMPY_FALLBACK LD_PRELOAD ASAN_OPTIONS LSAN_OPTIONS UBSAN_OPTIONS
uv run python -m jamma.jlinalg._compile_jlinalg
uv run python -m jamma.lmm._compile_accel
```

#### Interpreting failures

| Output | Meaning | Action |
|--------|---------|--------|
| `AddressSanitizer: heap-buffer-overflow` in `_lmm_accel.c:NNN` or `_jlinalg/<file>.c:NNN` | Real bug in JAMMA C code | Fix the bug — file/line and stack trace point at the offending site |
| `LeakSanitizer: detected memory leaks` with frames in `_PyImport_LoadDynamic`, `PyType_Ready`, `PyArray_API`, `OPENSSL_init_crypto`, etc. | Expected interpreter / NumPy / OpenSSL init noise | Verify the symbol is covered in `scripts/asan-suppressions.txt`; if not, add it with an upstream-issue citation per the file's header. NEVER add a `leak:jamma_*` suppression — that defeats the workflow. |
| `runtime error: signed integer overflow` (UBSAN) | Real bug — usually arithmetic on int sizes/strides | Fix or add an explicit cast with a comment explaining the safety argument |
| Workflow exits 0 with no `AddressSanitizer:` lines anywhere in the asan-ubsan log | Either (a) clean run (good) or (b) ASAN not actually wired (BAD) | The `asan-sentinel-meta-test` job exists exactly to distinguish these cases — if it's also green, ASAN is wired and the asan-ubsan green is real |

> **Note:** The sanitizer workflow uses `JAMMA_FORCE_NUMPY_FALLBACK=1`,
> so the C BLAS dispatch in `_jlinalg.so` and the C compute path in
> `_lmm_accel.so` are SKIPPED at import time. The sanitizer therefore
> exercises the NumPy fallback paths under instrumentation — not the
> vendor-BLAS dispatch. This is intentional; BLAS-allocator interactions
> with ASAN are too noisy to catch genuine bugs in our own code.

See also: [`.github/workflows/sanitizers.yml`](../.github/workflows/sanitizers.yml),
[`scripts/asan-suppressions.txt`](../scripts/asan-suppressions.txt),
[`src/jamma/_build_support/compile_and_link.py`](../src/jamma/_build_support/compile_and_link.py)
(the `apply_sanitizer_overrides` helper).

### 1.11 A missing fixture is a bug, not a skip

Everything under `tests/fixtures/` is committed and hash-verified by the
`GEMMA fixture sha256 manifest` pre-commit hook. So a test that cannot find
a fixture is not reporting a missing input, it is reporting its own wrong
path. Guarded with `pytest.skip`, that bug is invisible: the run stays green
and the test simply never executes.

Two GEMMA-parity tests in `tests/test_likelihood_derivatives.py` sat dormant
for their whole lifetime this way. The fixture directory was wrong *and* the
kinship filename was wrong, and a single `.exists()` guard turned both into
one skip (#147).

**Guard with `require_fixture`, not `pytest.skip`.** `require_fixture` in
`tests/conftest.py` raises `FileNotFoundError` when any argument does not
exist, naming every missing path relative to the repository root, and
returns `None` otherwise:

```python
from tests.conftest import require_fixture

def test_something():
    require_fixture(MOUSE_BFILE.with_suffix(".bed"), MOUSE_KINSHIP)
```

Pass every path the test is about to read, in one call. A wrong directory
then reports all of its files at once rather than stopping at the first,
which is the half of #147 a single check could not show. For a file whose
tests all need the same fixture, call it once at module level, as
`tests/test_loco_eigen_cache.py` does; a wrong path there fails collection.

**The skip gate is the backstop.** `_enforce_no_dormant_skips` in
`tests/conftest.py` parses every `tests/**/test_*.py` at `pytest_configure` and
fails the session, listing each file and line. Same mechanism as the §1.6 tier
gate: source-parsed and run once, so it holds under xdist, `-k` and `-m`, and it
flags the guard even in a file whose tests never ran.

It applies **three independent detectors**, and reports every category in one
failure so a sweep clears in a single pass:

**1. The reason names a fixture** (`_fixture_skip_lines`) — reads the reason
string of a `pytest.skip(...)` or `@pytest.mark.skipif(..., reason=...)`.

- **Any wording is caught.** The detector looks for the word `fixture`, so
  `fixture missing`, `no fixture data` and `fixture absent` all fail. It replaced
  a runtime check that matched only the exact phrase `fixture not available` in
  skip *reports*, which could fire only when the guarded test actually ran and
  only for that one wording. `test_fixture_manifest.py` carried a fixture skip
  worded around it, and the source-parsed gate is what found it.
- **A computed reason is not judged.** Only string literals are inspected; an
  f-string reason cannot be read from source and is left alone.

**2. The skip is guarded by a filesystem check** (`_path_guarded_skip_lines`) —
ignores the reason entirely and reads the control flow, flagging a
`pytest.skip` reached because `.exists()`, `.is_file()`, `.is_dir()`,
`os.path.isfile()` or `os.path.isdir()` was False, in either the `if` or the
`else` branch, or in a `@pytest.mark.skipif(not P.exists(), ...)` decorator.

Detector 2 exists because detector 1 has a blind spot that a real test sat in
for months. `TestDstedcNoAbort` in `tests/test_jlinalg_eigh.py` read a
`src/jamma/jlinalg/src/dstedc.c` that commit `663a22b` had deleted, and skipped
with the reason `source not available`. The word `fixture` never appeared, so
the gate passed it and the test reported green on every run until #156 deleted
it. Wording is the wrong thing to key on, because whoever writes the next guard
chooses it freely; the shape is not optional.

A guard now has to evade both detectors, and the evasions pull against each
other: avoid the word and the shape still shows, keep the check implicit and the
wording has nothing left to describe it with.

**3. The skip is gated on whether a name exists**
(`_attribute_probed_skip_lines`) — flags a `pytest.skip` or
`@pytest.mark.skipif` whose condition reaches a `hasattr` or `getattr` call,
following module-level bindings so the usual
`AVAILABLE = ... hasattr(mod, "FLAG")` plus `skipif(not AVAILABLE, ...)` pair is
caught rather than just the inline form.

Detector 3 covers what neither of the others can see: a guard whose path is fine
and whose reason is honest, but which asks whether a name is still there.
`hasattr` answers False for a deleted name exactly as readily as for one that was
never built, so the guard turns itself off during an unrelated refactor.
`test_lmm_accel_fused.py` probed `compute_numpy._C_FUSED_AVAILABLE`; #182
collapsed the thirteen `_C_*_AVAILABLE` flags to a single capability bit and
removed it, and the nine `TestFusedParity` tests covering the live fused Wald
kernel skipped from that merge onward, on machines with the extension fully
built.

**Gate on the capability, not on the name.** For the C extension that is
`compute_numpy._accel is not None`, which is the one bit the ABI-equality gate
actually admits. If a test needs a specific attribute to be present, `assert` it:
the assert fails when the name goes, which is the whole point.

- **Skips about the environment are untouched.** `C extension not available`,
  `uv not available on PATH`, an absent optional import and an env-var gate are
  all genuine conditional skips and stay skips. Neither detector fires on them.
- **Fail on a path, do not skip on it.** If a file should be there and is not,
  use `pytest.fail`, an `assert`, or `require_fixture`, as
  `tests/test_fingerprint_harness.py` does with `pytest.fail` when the recorder
  writes nothing. If the file genuinely may be absent because it is a build
  output, gate on the flag that predicts it (`HAS_C_EXTENSION` and the like)
  rather than on the path.

All three mechanisms have regression tests in
`tests/test_conftest_fixture_skip_gate.py`, including one that plants the exact
shape `TestDstedcNoAbort` had.

---

## 2. Test Design Rules

### 2.1 Test philosophy

Tests validate **observable behavior** — return values, exceptions,
warnings, files written, DataFrame columns. Default to no comments;
default to no source inspection. If you find yourself reaching for
`inspect.getsource()`, you are testing the wrong layer.

### 2.2 Boundary catalogue (where mocking is allowed)

`@patch` and `MagicMock` are allowed **only** at OS/hardware/process
boundaries. Everywhere else, write a fake.

| Allowed mock target | Why it qualifies |
|---|---|
| `psutil.virtual_memory`, `psutil.Process(...).memory_info` | OS state; cannot be set without large allocations |
| `gc.collect` | Process-level side effect |
| `jamma.core.memory._check_available` | Thin wrapper around OS memory probe |
| `jlinalg.blas_is_ilp64` | Library detection that cannot be flipped at runtime |
| `os.environ` setters | Process state |
| `subprocess.run` / `subprocess.Popen` | Spawns external processes |
| `progressbar.ProgressBar` | External UI library — see §2.4 |

If your patch target is not on this list, you are mocking the wrong
layer. Either justify the addition in PR review or write a fake.

### 2.3 Fakes over mocks

For non-boundary collaborators, write a fake class implementing the real
interface. Fakes catch interface drift; `MagicMock()` silently accepts
any attribute access and hides renames.

Shared fakes live in the
[`tests/fakes/`](../tests/fakes/) package. The canonical example is
[`FakeAssocWriter`](../tests/fakes/assoc_writer.py): it replaces
`MagicMock()` for the `IncrementalAssocWriter` interface and captures
`write_arrays_batch` calls in a list. The package also ships
`FakeProgressBar`/`FakeProgressbarModule`
([`progress.py`](../tests/fakes/progress.py)) and
`FakePipelineRunnerFactory`
([`pipeline.py`](../tests/fakes/pipeline.py)). Each fake's self-tests are
in [`tests/fakes/test_fakes.py`](../tests/fakes/test_fakes.py); accessing
an undeclared attribute raises `AttributeError` (the contract that
distinguishes a fake from `MagicMock`).

### 2.4 Structural source tests (narrow exception)

Most "read source code and grep" tests are anti-patterns, but a few are
legitimate **build/assembly guardrails** that no behavioral test can
catch. The carve-out is:

| Allowed structural test | Why behavior tests can't replace it |
|---|---|
| LOCO iterator-None guard uses `raise RuntimeError`, not bare `assert` ([`tests/test_safety_gates.py:278`](../tests/test_safety_gates.py#L278)) | `python -O` strips bare `assert`; behavior-only test passes in dev and silently breaks in prod |
| Compile-flag literals not in three forbidden entry points ([`scripts/check_compile_flag_literals.py`](../scripts/check_compile_flag_literals.py)) | Drift between `hatch_build.py` and runtime recompile produces ABI mismatch at runtime |
| `_lmm_accel.c` reaches `Python.h` before any header that pulls in `<math.h>` ([`tests/test_c_include_order.py`](../tests/test_c_include_order.py)) | `M_PI` is not C11. glibc defines it only under `_XOPEN_SOURCE`, which `Python.h` sets; macOS defines it unconditionally. Get the order wrong and the local build and ARM Mac CI pass while every Linux job fails to compile |

**Rules for adding a new structural source test:**

1. The thing being checked must be uncatchable by behavioral tests (a
   process abort, a compiler-stripped check, a build-time flag, an
   ABI-relevant directive).
2. Include a comment explaining *why* a behavior test cannot replace it.
3. Mark `tier0` so it runs everywhere — these are guardrails, not parity.
4. Assert on something that exists. A structural test whose target file is
   deleted degrades to a permanent skip and reads as coverage it is not
   providing. `test_avx2_vzeroupper_source` sat that way from `663a22b`
   until it was removed.

If the rule is "X function should call Y", that is a behavior test, not
a structural test. Test the behavior.

### 2.5 Anti-patterns

| Anti-pattern | Preferred approach |
|---|---|
| `inspect.getsource()` assertions | Assert on warnings, exceptions, return values |
| `MagicMock()` for data classes (`MemoryBreakdown`, `ExecutionPlan`, `Path`) | Construct real instances with test values |
| `MagicMock(spec=Path)` to stand in for a path | Use `tmp_path` or `Path("/tmp/x")` |
| `@patch` on a non-boundary collaborator (`PipelineRunner`, `numpy.linalg.eigh`) | Inject a fake; or test against real values |
| Mocking numerical functions (eigh, BLAS, likelihood) | Use small synthetic data with known results |
| `@patch` on the function under test | Patch only its external dependencies |
| `assert mock.call_count == N` for internal functions | Assert on observable output |
| Performance assertions in pass/fail tests (`assert c_time < py_time / 2`) | Use `pytest-benchmark`, track over time |

`assert_called_once_with` is acceptable at boundaries — external APIs,
subprocess calls, dispatch routers where delegation IS the observable
behavior. Not for internal functions.

The "mocking numerical functions" anti-pattern is enforced by
[`scripts/check_forbidden_patches.py`](../scripts/check_forbidden_patches.py),
an AST-based pre-commit hook. It bans patching `numpy.linalg.*`,
`scipy.*`, and JAMMA's own numerical modules
(`compute_numpy`/`cn`, `likelihood`, `jlinalg`/`jl`,
`kinship_compute`/`kc`). The capability seam `compute_numpy._accel` and
`_*_ENABLED` constants are excluded. To exercise the pure-NumPy path, set
`_accel` to None; that is one bit, because the ABI-equality gate admits all
of the C extension's `methods[]` table or none of it, so there is no build
that exports some kernels and not others. If you have a legitimate reason to
patch something else here, add an inline `# allow-patch:` comment explaining
why. Read failures (`OSError`, `UnicodeDecodeError`) exit
non-zero rather than passing vacuously.

### 2.6 When `pytest.skip` is acceptable

The suite has ~220 skip/xfail calls. Three categories — only the first
two are acceptable:

1. **Hardware/library availability** — vendor LAPACK absent, ILP64 not
   active, BLAS backend mismatch. Use module-level
   `pytestmark = pytest.mark.skipif(...)` so the file skips at collection
   time. Example: [`tests/test_jlinalg_dispatch.py:13`](../tests/test_jlinalg_dispatch.py#L13).
2. **Optional fixture absent** — a dataset too large to commit. There is no
   fixture in this category, and §1.11's gate now rejects any skip reason that
   names a fixture, whatever the wording. `gemma_loco` and `mouse_hs1940` are
   both committed in full and their guards use `require_fixture`, which raises.
   If a genuinely un-committable dataset ever arrives, gate it on an environment
   variable or a marker rather than on the word "fixture" in a skip reason.
3. **Test is broken / commented-out** — *not acceptable*. Either fix or
   delete.

`@pytest.mark.xfail` is only for known bugs with an open beads/GitHub
issue. Include the issue ID in the reason string.

### 2.7 Bug fix workflow

1. **Reproduce**: write a failing test that demonstrates the bug.
2. **Fix**: change production code to make the test pass.
3. **Keep**: the regression test stays permanently. Fold it into the
   canonical module test file (e.g. `test_loco_numpy.py`) or, if the
   bug class warrants its own file, name it after the *behavior*
   (`test_loco_orchestration.py`) — never after the trigger
   (`test_loco_bugs.py`, `test_review_fixes.py`).
4. **Scope**: only test the broken behavior, not surrounding code.

### 2.8 Agent-generated test rules

- Only write tests for code modified in the current task.
- Never modify production code to make a test pass — if a test fails,
  the test is wrong or there's a real bug.
- Each test must be traceable to a specific behavior change or bug fix.
- Do not write speculative tests for code you did not touch.

### 2.9 Tier selection for new tests

| If the test... | Use tier |
|---|---|
| Pure computation, no I/O, no reference data | `tier0` |
| Validates output against GEMMA reference fixtures | `tier1` |
| Needs >1 GB memory or runtime >60s | `tier2` |
| Long but not memory-bound | `slow` |

### 2.10 Numerical comparison tolerances

Use `ToleranceConfig` from
[`src/jamma/validation/tolerances.py`](../src/jamma/validation/tolerances.py).
The defaults are calibrated from formal error propagation analysis (see
[`docs/GEMMA_EQUIVALENCE.md`](GEMMA_EQUIVALENCE.md)):

```python
from jamma.validation import ToleranceConfig

config = ToleranceConfig()
np.testing.assert_allclose(result, reference, rtol=config.pvalue_rtol, atol=config.atol)
```

Do not relax tolerances to make tests pass. If a tolerance is too tight,
either fix the algorithm or update [`docs/GEMMA_EQUIVALENCE.md`](GEMMA_EQUIVALENCE.md)
*and* `ToleranceConfig` in one PR.

### 2.11 Shared fixtures

Defined in [`tests/conftest.py`](../tests/conftest.py):

| Fixture | Description |
|---------|-------------|
| `sample_plink_data` | Path prefix for synthetic PLINK files (`tests/fixtures/gemma_synthetic/test`) |
| `output_dir` | Temporary output directory wrapping `tmp_path` |
| `tolerance_config` | `ToleranceConfig()` for numerical comparisons |
| `synthetic_covariate_data_ncvt2` | Rotated data with 2 covariates (200 samples, 50 SNPs) |
| `synthetic_covariate_data_ncvt4` | Rotated data with 4 covariates (200 samples, 50 SNPs) |

If you add a fixture, also add a row here.

### 2.12 Property-based tests

Live in [`tests/test_hypothesis.py`](../tests/test_hypothesis.py):

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=10, max_value=100))
def test_kinship_symmetry(n_samples): ...
```

Run with `uv run pytest tests/test_hypothesis.py -x`.

---

## 3. Suite Map and Current State

### 3.1 Suite map by subsystem

| Subsystem | Test files | What's covered |
|---|---|---|
| **LMM core** | `lmm_accel/` (10 per-kernel-family modules), `test_lmm_unit.py`, `test_lmm_score.py`, `test_lmm_dispatch.py`, `test_lmm_compute_dispatch.py`, `test_lmm_audit.py`, `test_lmm_io_validation.py`, `test_likelihood_numpy.py`, `test_likelihood_derivatives.py` | C accelerator parity vs NumPy reference; Pab/Uab math; Wald/score/LRT statistics; dispatch routing; numerical guards; assoc-line/dispatch-table validation; REML 2nd/3rd derivatives |
| **LMM runners** | `test_runner_numpy.py`, `test_runner_dispatch.py`, `test_numpy_streaming.py`, `test_compute_numpy.py`, `test_chunk_runner_guards.py`, `test_pipeline.py`, `test_pipeline_helpers.py`, `test_pipeline_banner.py`, `test_pipeline_validation_order.py` | Batch + streaming runners; shared chunk runner and its guards; backend selection; pipeline orchestration and validation ordering; CLI banner |
| **Kinship** | `test_kinship_numpy.py`, `test_kinship_io.py`, `test_kinship_validation.py` | DSYRK-based kinship computation; .cXX.txt I/O; GEMMA parity |
| **jlinalg (BLAS dispatch)** | `test_jlinalg_dgemm.py`, `test_jlinalg_dsyrk.py`, `test_jlinalg_eigh.py`, `test_jlinalg_dispatch.py`, `test_jlinalg_unity.py`, `test_jlinalg_build.py`, `test_eigh_inplace.py` | DGEMM/DSYRK/eigh wrappers; LP64 vs ILP64 dispatch; vendor BLAS/LAPACK dispatch correctness; build artefact sanity |
| **LOCO** | `test_loco_numpy.py`, `test_loco_eigen_cache.py`, `test_loco_orchestration.py` | Leave-one-chromosome-out orchestration; per-chromosome eigen cache |
| **I/O** | `test_io.py`, `test_io_error_paths.py`, `test_error_paths.py`, `test_eigen_io.py`, `test_matrix_reader.py`, `test_matrix_writer.py`, `test_incremental_writer.py`, `test_kinship_io.py`, `test_snp_list.py`, `test_plink_validation.py` | PLINK reader; eigenvector cache I/O; incremental .assoc.txt writer; SNP filters; error and rollback paths |
| **Memory & gates** | `test_memory.py`, `test_memory_gates.py`, `test_memory_chunk_coupling.py`, `test_eigendecomp_memory.py`, `test_safety_gates.py`, `test_auto_tune_chunk.py` | Memory estimation; OOM gates; `compute_chunk_size_numpy` chunk sizing and pipeline-buffer pricing |
| **CLI / API** | `test_cli.py`, `test_cli_memory.py`, `test_gwas_api.py` | Click entry point; `-lmm` flag handling; programmatic GWAS API |
| **Backend / hardware** | `test_backend_detection.py`, `test_hardware_context.py`, `test_threading.py`, `test_jlinalg_dispatch.py`, `test_force_numpy_fallback.py` | Backend autodetection; physical core count; threading limits; the `JAMMA_FORCE_NUMPY_FALLBACK` escape hatch |
| **Build support** | `test_build_support_compile_and_link.py`, `test_build_support_openmp_detect.py`, `test_build_support_packaging.py`, `test_build_support_sanitizer_override.py`, `test_check_c_extension_freshness.py`, `test_check_compile_flag_literals.py`, `test_check_quiet_flags.py`, `test_check_test_timeouts.py`, `test_check_doc_anchors.py`, `test_verify_compile_invocations_match.py`, `test_c_extensions_ci.py`, `test_c_include_order.py`, `test_c_lint_coverage.py`, `test_core_recompile.py` | Compile-flag invariants; OpenMP detection; wheel packaging; sanitizer flag injection; include order; cppcheck coverage; doc line anchors; runtime recompile |
| **Fingerprint / sanitizer harness** | `test_fingerprint_harness.py`, `test_compare_fingerprints.py`, `test_lmm_accel_sections.py`, `test_sanitizer_sentinel.py`, `test_compile_accel_sentinel_injection.py`, `test_sanitizer_workflow_yaml.py`, `test_asan_suppressions.py` | The machinery behind `fingerprint.yml` and `sanitizers.yml`. These test the gates themselves, so a broken harness cannot go green by doing nothing |
| **Validation / parity** | `test_validation.py`, `test_validation_assoc.py`, `test_validate_runner_inputs.py`, `test_kinship_validation.py`, `test_demonstrate_equivalence.py` | GEMMA parity machinery; tolerance config; assoc file diff; the equivalence demonstration script |
| **Suite meta** | `test_conftest_tier_gate.py`, `test_fixture_manifest.py`, `tests/fakes/test_fakes.py` | The mandatory-tier-marker gate (§1.6), the fixture manifest (§3.5), and the fakes' own contract tests |
| **Reference oracles** | `tests/reference/likelihood.py`, `tests/reference/stats.py` | GEMMA-literal scalar ports (CalcPPab, CalcPPPab, LogRL_dev2, CalcRLWald, CalcRLScore, CalcLRT, `f_sf`, `safe_sqrt`) with no production caller; the batch and C paths are held to them |
| **Numerics / utilities** | `test_special.py`, `test_schema.py`, `test_snp_filter.py`, `test_snp_filter_perf.py`, `test_snp_stats.py`, `test_core_snp_stats.py`, `test_categorical.py`, `test_missingness.py`, `test_weights.py`, `test_prepare_common.py`, `test_telemetry.py`, `test_progress.py`, `test_hypothesis.py` | Cephes betainc / chi2_sf; data-class schemas; SNP filtering and statistics; phenotype prep; progress bars |

### 3.2 Tests to improve

| Test | Issue | Action |
|---|---|---|
| `test_jlinalg_dgemm.py`, `test_jlinalg_dsyrk.py`, `test_jlinalg_eigh.py` | Heavy parametrised boundary coverage runs by default (currently `tier0` at the module level with `@slow`/`@benchmark` carving out the expensive cases) | Consider hypothesis property tests for shape/transpose invariants |

### 3.3 Tests / files to fold

| Files | Action |
|---|---|
| ~~`test_audit_fixes.py`~~ → `test_lmm_audit.py` | Renamed via `git mv` (history preserved). Content unchanged — still an LMM-numerical-guard suite, but the name no longer reads as a one-shot scratch bin |
| ~~`test_review_fixes.py`~~ → `test_lmm_io_validation.py` | Renamed. Heterogeneous (assoc format, build_results, erfc parity, degenerate-SNP NaN); bound together by being I/O- and dispatch-validation-shaped |
| ~~`test_loco_bugs.py`~~ → `test_loco_orchestration.py` | Renamed |
| ~~`test_lmm_likelihood_dev2.py`~~ → `test_likelihood_derivatives.py` | Renamed. `dev2` is GEMMA jargon for "second derivative" (`LogRL_dev2`); the file's symbols inherit it but the file itself is named for the *behavior* (REML 2nd/3rd derivatives wrt lambda, used for `se(pve)`) |

### 3.4 Suite-wide stats (snapshot)

Counted at v7.2.0.

- 107 test files, ~44k lines.
- Largest: `test_likelihood_numpy.py` (~2,100 lines). `test_lmm_accel.py` was
  split into `tests/lmm_accel/`, eleven modules by kernel family, in 6.0.0.
- ~220 `skip`/`skipif`/`xfail` calls — most legitimate (vendor LAPACK, optional fixtures).
- 11 files use `@patch`/`MagicMock` (~25 occurrences). Four of them are the `tests/fakes/` package itself. The rest sit at the boundaries catalogued in §2.2.
- `inspect.getsource()`: zero uses. The ban holds.

### 3.5 Fixture manifest

[`tests/fixtures/MANIFEST.toml`](../tests/fixtures/MANIFEST.toml) tracks the
SHA-256 of every git-tracked fixture (55 entries). The manifest is enforced
by two gates:

1. **Pre-commit hook** (fast):
   [`scripts/check_fixture_manifest.py`](../scripts/check_fixture_manifest.py)
   verifies on-disk hashes match, flags untracked additions, and flags
   manifest-without-disk entries.
2. **Tier0 self-test** (slow):
   [`tests/test_fixture_manifest.py`](../tests/test_fixture_manifest.py)
   runs the same check inside the test suite.

After intentionally regenerating fixtures (e.g. updating the GEMMA
reference assoc files, regenerating the kinship reference matrix), run
`uv run python scripts/check_fixture_manifest.py --write` to rebuild the
manifest. `--write` auto-extracts `GEMMA Version` and `Command Line Input`
from `.log.txt` headers, so the manifest also serves as a provenance record.

Editing or adding a fixture without updating the manifest will fail the
pre-commit gate with a `sha256 drift` message that points at the stale
hash and tells you exactly which command to run.
