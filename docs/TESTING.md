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
| `test-slow.yml` | push to master | `pytest -m "tier2 or slow" -v -o 'addopts=' --no-cov` |
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
| `_mm256_zeroupper()` / `vzeroupper` present in AVX2 kernel source ([`tests/test_jlinalg_dgemm.py:568`](../tests/test_jlinalg_dgemm.py#L568)) | Missing this corrupts SSE registers in *callers'* code, not in our test |
| LOCO iterator-None guard uses `raise RuntimeError`, not bare `assert` ([`tests/test_safety_gates.py:311`](../tests/test_safety_gates.py#L311)) | `python -O` strips bare `assert`; behavior-only test passes in dev and silently breaks in prod |
| Compile-flag literals not in three forbidden entry points ([`scripts/check-compile-flag-literals.py`](../scripts/check-compile-flag-literals.py)) | Drift between `hatch_build.py` and runtime recompile produces ABI mismatch at runtime |

**Rules for adding a new structural source test:**

1. The thing being checked must be uncatchable by behavioral tests (a
   process abort, a compiler-stripped check, a build-time flag, an
   ABI-relevant directive).
2. Include a comment explaining *why* a behavior test cannot replace it.
3. Mark `tier0` so it runs everywhere — these are guardrails, not parity.

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
[`scripts/check-forbidden-patches.py`](../scripts/check-forbidden-patches.py),
an AST-based pre-commit hook. It bans patching `numpy.linalg.*`,
`scipy.*`, and JAMMA's own numerical modules
(`compute_numpy`/`cn`, `likelihood`, `jlinalg`/`jl`,
`kinship_compute`/`kc`). Feature-flag constants (`_C_*_AVAILABLE`,
`_*_ENABLED`) are excluded. If you have a legitimate reason to patch one
of these (typically toggling a dispatch boundary in a test that exists
specifically to verify dispatch), add an inline `# allow-patch:` comment
explaining why. Read failures (`OSError`, `UnicodeDecodeError`) exit
non-zero rather than passing vacuously.

### 2.6 When `pytest.skip` is acceptable

The suite has ~180 skip/xfail calls. Three categories — only the first
two are acceptable:

1. **Hardware/library availability** — vendor LAPACK absent, ILP64 not
   active, BLAS backend mismatch. Use module-level
   `pytestmark = pytest.mark.skipif(...)` so the file skips at collection
   time. Example: [`tests/test_jlinalg_dispatch.py:12`](../tests/test_jlinalg_dispatch.py#L12).
2. **Optional fixture absent** — large datasets shipped out-of-band
   (e.g. `gemma_loco`). Skip with a message naming the missing fixture path.
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
`docs/EQUIVALENCE.md`):

```python
from jamma.validation import ToleranceConfig

config = ToleranceConfig()
np.testing.assert_allclose(result, reference, rtol=config.pvalue_rtol, atol=config.atol)
```

Do not relax tolerances to make tests pass. If a tolerance is too tight,
either fix the algorithm or update `docs/EQUIVALENCE.md` *and*
`ToleranceConfig` in one PR.

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
| **LMM core** | `lmm_accel/` (11 per-kernel-family modules), `test_lmm_unit.py`, `test_lmm_score.py`, `test_lmm_dispatch.py`, `test_lmm_audit.py`, `test_lmm_io_validation.py`, `test_likelihood_numpy.py`, `test_likelihood_derivatives.py` | C accelerator parity vs NumPy reference; Pab/Uab math; Wald/score/LRT statistics; dispatch routing; numerical guards; assoc-line/dispatch-table validation; REML 2nd/3rd derivatives |
| **LMM runners** | `test_runner_numpy.py`, `test_runner_dispatch.py`, `test_numpy_streaming.py`, `test_compute_numpy.py`, `test_pipeline.py`, `test_pipeline_helpers.py`, `test_pipeline_banner.py` | Batch + streaming runners; shared chunk runner; backend selection; pipeline orchestration; CLI banner |
| **Kinship** | `test_kinship_numpy.py`, `test_kinship_io.py`, `test_kinship_validation.py` | DSYRK-based kinship computation; .cXX.txt I/O; GEMMA parity |
| **jlinalg (BLAS dispatch)** | `test_jlinalg_dgemm.py`, `test_jlinalg_dsyrk.py`, `test_jlinalg_eigh.py`, `test_jlinalg_lapack.py`, `test_jlinalg_level1.py`, `test_jlinalg_dispatch.py`, `test_jlinalg_unity.py`, `test_jlinalg_build.py`, `test_eigh_inplace.py` | DGEMM/DSYRK/eigh wrappers; LP64 vs ILP64 dispatch; AVX2 microkernel guards; build artefact sanity |
| **LOCO** | `test_loco_numpy.py`, `test_loco_eigen_cache.py`, `test_loco_orchestration.py` | Leave-one-chromosome-out orchestration; per-chromosome eigen cache |
| **I/O** | `test_io.py`, `test_io_error_paths.py`, `test_eigen_io.py`, `test_matrix_reader.py`, `test_matrix_writer.py`, `test_incremental_writer.py`, `test_kinship_io.py`, `test_snp_list.py`, `test_plink_validation.py` | PLINK reader; eigenvector cache I/O; incremental .assoc.txt writer; SNP filters |
| **Memory & gates** | `test_memory.py`, `test_memory_gates.py`, `test_memory_chunk_coupling.py`, `test_eigendecomp_memory.py`, `test_safety_gates.py`, `test_auto_tune_chunk.py`, `test_rss_logging.py` | Memory estimation; OOM gates; chunk-size auto-tuning; RSS telemetry |
| **CLI / API** | `test_cli.py`, `test_cli_memory.py`, `test_gwas_api.py` | Click entry point; `-lmm` flag handling; programmatic GWAS API |
| **Backend / hardware** | `test_backend_detection.py`, `test_hardware_context.py`, `test_threading.py`, `test_jlinalg_dispatch.py` | Backend autodetection; physical core count; threading limits |
| **Build support** | `test_build_support_compile_and_link.py`, `test_build_support_openmp_detect.py`, `test_build_support_packaging.py`, `test_check_c_extension_freshness.py`, `test_check_compile_flag_literals.py`, `test_check_quiet_flags.py`, `test_check_test_timeouts.py`, `test_verify_compile_invocations_match.py`, `test_c_extensions_ci.py`, `test_core_recompile.py` | Compile-flag invariants; OpenMP detection; wheel packaging; runtime recompile |
| **Validation / parity** | `test_validation.py`, `test_validation_assoc.py`, `test_validate_runner_inputs.py`, `test_kinship_validation.py` | GEMMA parity machinery; tolerance config; assoc file diff |
| **Numerics / utilities** | `test_special.py`, `test_schema.py`, `test_snp_filter.py`, `test_snp_filter_perf.py`, `test_snp_stats.py`, `test_categorical.py`, `test_missingness.py`, `test_weights.py`, `test_prepare_common.py`, `test_telemetry.py`, `test_progress.py`, `test_hypothesis.py` | Cephes betainc / chi2_sf; data-class schemas; SNP filtering; phenotype prep; progress bars |

### 3.2 Tests to improve

| Test | Issue | Action |
|---|---|---|
| `test_jlinalg_dgemm.py`, `test_jlinalg_dsyrk.py`, `test_jlinalg_eigh.py` | Heavy parametrised boundary coverage runs by default (currently `tier0` at the module level with `@slow`/`@benchmark` carving out the expensive cases) | Consider hypothesis property tests for shape/transpose invariants |

### 3.3 Tests / files to fold

| Files | Action |
|---|---|
| `test_jlinalg_lapack.py` lines [47](../tests/test_jlinalg_lapack.py#L47), [66](../tests/test_jlinalg_lapack.py#L66), [146](../tests/test_jlinalg_lapack.py#L146) | Duplicate large QR/SVD reconstruction-and-orthogonality checks (5000×200 each, all `slow`). Fold into one parametrised tier2 case per decomposition |
| ~~`test_audit_fixes.py`~~ → `test_lmm_audit.py` | Renamed via `git mv` (history preserved). Content unchanged — still an LMM-numerical-guard suite, but the name no longer reads as a one-shot scratch bin |
| ~~`test_review_fixes.py`~~ → `test_lmm_io_validation.py` | Renamed. Heterogeneous (assoc format, build_results, erfc parity, degenerate-SNP NaN); bound together by being I/O- and dispatch-validation-shaped |
| ~~`test_loco_bugs.py`~~ → `test_loco_orchestration.py` | Renamed |
| ~~`test_lmm_likelihood_dev2.py`~~ → `test_likelihood_derivatives.py` | Renamed. `dev2` is GEMMA jargon for "second derivative" (`LogRL_dev2`); the file's symbols inherit it but the file itself is named for the *behavior* (REML 2nd/3rd derivatives wrt lambda, used for `se(pve)`) |

### 3.4 Suite-wide stats (snapshot)

- 85 test files, ~39k lines.
- Largest: `test_likelihood_numpy.py` (~2,100 lines). `test_lmm_accel.py` was
  split into `tests/lmm_accel/`, eleven modules by kernel family, in 6.0.0.
- ~178 `skip`/`skipif`/`xfail` calls — most legitimate (vendor LAPACK, optional fixtures).
- 8 files use `@patch`/`MagicMock` (~31 occurrences). Most are at allowed boundaries; the violations called out in §3.2 are the exceptions.
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
[`scripts/regenerate_fixture_manifest.py`](../scripts/regenerate_fixture_manifest.py)
to rebuild the manifest. The regen script auto-extracts `GEMMA Version`
and `Command Line Input` from `.log.txt` headers, so the manifest also
serves as a provenance record.

Editing or adding a fixture without updating the manifest will fail the
pre-commit gate with a `sha256 drift` message that points at the stale
hash and tells you exactly which command to run.
