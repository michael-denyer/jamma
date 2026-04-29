# Changelog

All notable changes to JAMMA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Tier-marker enforcement gate**: `tests/conftest.py` now AST-parses every
  collected test file in `pytest_configure` and fails the run when any file
  lacks a tier (`tier0`/`tier1`/`tier2`), `slow`, or `benchmark` marker. The
  gate runs once on the controller before xdist forks workers, fixing the
  silent fail-open under `-n N` that the previous collection-based gate had.
  Recognises parametrised markers (`@pytest.mark.skipif(...)`) and list-form
  `pytestmark`. Regression test `test_gate_fires_under_xdist` asserts the
  gate fires under `-n 2`.
- **Forbidden-patches gate**: new `scripts/check-forbidden-patches.py` +
  pre-commit hook bans patching `numpy.linalg.*`, `scipy.*`, and JAMMA's own
  numerical functions in tests. Feature-flag constants (`_C_*_AVAILABLE`)
  are excluded; `# allow-patch:` escape hatch documented. Now uses AST
  scanning rather than regex, covers `patch.object(<module>, ...)`,
  `mocker.patch(...)`, and `monkeypatch.setattr("dotted.path"...)`. Module-
  arg `monkeypatch.setattr(<module>, "<func>")` is also caught (closes a
  hole where two test files set callables on numerical modules and slipped
  past the previous gate). Read failures raise `_ScanError` and exit
  non-zero rather than passing vacuously on docs-only batches.
- **AST + runtime safety gates**: replaced regex source-greps in
  `TestLOCOIteratorRuntimeError` and `TestJlinalgABIValidation` with
  `ast.parse` structural checks plus runtime tests that exercise the
  guards (`python -O` subprocess for `loco_iter`; in-subprocess monkey-
  patched `_EXPECTED_JLINALG_ABI` for ABI drift, asserting on exit code
  and stderr).
- **Fakes package**: `tests/fakes/` provides `FakePipelineRunner`,
  `FakePipelineRunnerFactory`, `FakeAssocWriter`, `FakeProgressbarModule`,
  and `FakeProgressBar`. Type-narrowed to real `PipelineConfig` /
  `PipelineResult` so adding a required field actually breaks tests.
  `TestFakeProductionDrift` compares `inspect.signature` of each fake
  method to the real production method and fails with a specific drift
  message instead of silently masking new args. Adopted by `test_progress.py`
  (10 nested `patch(...) + MagicMock` blocks → one `fake_progressbar`
  fixture) and `test_cli.py` (4 `MagicMock` chains → one factory).
- **GEMMA fixture manifest**: `tests/fixtures/MANIFEST.toml` (55 entries)
  with SHA-256 of every git-tracked fixture. `scripts/check_fixture_manifest.py`
  verifies on-disk hashes match, flags untracked additions, and flags
  manifest-without-disk entries. `scripts/regenerate_fixture_manifest.py`
  rebuilds the manifest after intentional updates and auto-extracts
  `GEMMA Version` and `Command Line Input` from `.log.txt` headers.
  Pre-commit hook (fast) + tier0 self-test `tests/test_fixture_manifest.py`
  (slow) gate it.
- **Scheduled flaky-test detection**: `.github/workflows/flaky-detect.yml`
  runs the default suite under five distinct `pytest-randomly` seeds every
  Sunday 06:00 UTC. Non-blocking; opens an issue on disagreement.
- **Subsystem coverage gates**: per-subsystem coverage floors enforced in
  CI (`src/jamma/jlinalg/` floor at 18% to accommodate the Linux-vs-macOS
  vendor-LAPACK fallback delta — Linux measured 21.8% without MKL-ILP64,
  macOS-Accelerate measured 33.6%; both reference numbers documented in
  the threshold comment).

### Changed

- **Tier marker hygiene**: 8 previously-unmarked test files now have
  module-level `pytestmark`. `test_jlinalg_dispatch.py` converts
  `pytestmark = skipif(...)` to a list combining `tier0` + the existing
  `skipif`. `test_runner_numpy.py`: `:443`/`:518` GEMMA-parity tests
  promoted to tier1; `:396` internal dispatch test reclassified tier1 →
  tier0.
- **Tier3 marker removed** from `pyproject.toml`, both CI workflows,
  `conftest.py`, and both docs — defined and excluded everywhere but
  never used.
- **Scratch-bin renames** (git mv preserves history):
  `test_audit_fixes.py` → `test_lmm_audit.py`,
  `test_review_fixes.py` → `test_lmm_io_validation.py`,
  `test_loco_bugs.py` → `test_loco_orchestration.py`,
  `test_lmm_likelihood_dev2.py` → `test_likelihood_derivatives.py`.
- **Fakes drop call-count integers**: `FakeAssocWriter.call_count`,
  `FakePipelineRunner.run_calls`, `FakePipelineRunnerFactory.call_count`,
  `FakeProgressBar.start_calls`/`finish_calls` replaced with state
  booleans and lifecycle-violation `AssertionError`s. `update_calls:
  list[int]` retained because it records observable values, not counts.
- **`FakeProgressbarModule.widgets`** simplified from nested class to
  `SimpleNamespace(WidgetBase=_FakeWidget)`.
- **`test_jlinalg_lapack.py`**: folded `test_reconstruction_accuracy_large`
  and `test_orthogonality_large` into one
  `test_large_5000x200_reconstruction_and_orthogonality` (both checked
  the same 5000×200 QR — running it twice wasted CI minutes). Loosened
  orthogonality bound for the large case from 1e-14 to 1e-13 (theoretical
  floor for sqrt(5000) accumulation is ~1.6e-14).
- **`blas_backend` known-backends set** extended with `system-BLAS-ILP64`
  and `system-BLAS-LP64` (returned by `blas_dispatch.c:132` when a vendor
  library is loaded but path-string detection cannot identify it — typical
  on Linux distros linking against alias-only `libblas.so`).
- **`test_blas_backend_string_has_known_value`** asserts membership in a
  documented set (incl. `Accelerate-ILP64`) instead of printing.

### Fixed

- **Tier-marker gate failed open under xdist**: collection-based gate
  silently no-op'd whenever `-n N` was active (default `-n 3`). Empirically
  reproduced — an unmarked file ran cleanly under `-n 2`. Switched the
  gate to source-parsing in `pytest_configure` (runs once on the controller
  before xdist forks workers).
- **`monkeypatch.setattr(<module>, "<func>")`** previously bypassed the
  forbidden-patches policy. `test_lmm_accel.py:207` set
  `_compute_lmm_batch_c` to a sentinel and `test_prepare_common.py:282`
  set `_compute_score_batch_c` to `None` — both exited 0 under the old
  gate. Added a module-form rule keyed off the documented forbidden-module
  aliases (`compute_numpy`, `cn`, `likelihood`, `jlinalg`, `jl`,
  `kinship_compute`, `kc`), still allowing `_AVAILABLE`/`_ENABLED` flags.
  Audited the existing call sites and added `# allow-patch:` comments to
  the 5 legitimate dispatch toggles.
- **`scripts/check-forbidden-patches.py`** no longer swallows `OSError` /
  `UnicodeDecodeError`. Read failures now exit non-zero rather than silently
  producing zero findings (the silent-failure mode the gate is meant to
  prevent). Detects "argv passed but no `.py` among them" and falls back
  to a repo-wide scan with a stderr note instead of passing vacuously when
  pre-commit hands the hook a docs-only batch.
- **`tests/conftest.py`**: replaced silent `except ImportError: return` in
  `pytest_configure` with a stderr warning so a broken freshness script
  is visible.
- **`TestEigendecompLP64Threshold`**: replaced
  `contextlib.suppress(...)` with `pytest.raises(RuntimeError, match="test
  stub")`. The previous form could not distinguish "RuntimeError propagated
  to caller" from "caller silently caught and returned a default" — both
  passed the warning-routing assertion.
- **`.github/workflows/ci.yml`**: dropped `not tier3` from the default
  pytest filter (the marker was removed from `pyproject` / `conftest` /
  docs in `6d9ab15` but this one workflow line was missed).
- **`git mv` rename deletes**: the renames in `6d9ab15` staged the new
  files but the matching `D` entries for the old files were never added
  to the index, so the new files shipped alongside the old ones. Staged
  the deletes for `test_audit_fixes.py`, `test_review_fixes.py`,
  `test_loco_bugs.py`, and `test_lmm_likelihood_dev2.py`.
- **`tests/test_conftest_tier_gate.py`**: previously embedded a parallel
  stub of the old collection-based gate; after the xdist fail-open fix it
  was no longer testing the implementation it claimed to. Rewired the
  stub conftest to `importlib`-load the real `_enforce_tier_markers` from
  `tests/conftest.py`.
- **Removed dead `scripts/pre-push`**: standalone bash hook duplicated
  the `.pre-commit-config.yaml`'s `ruff-format-all` pre-push entry and
  was never wired into any git hook (`.git/hooks/pre-push` is prek-managed).

### Removed

- `tier3` pytest marker (defined but never used).
- `scripts/pre-push` (dead code; functionality lives in pre-commit).
- `docs/TESTING.md` §3.3 "Tests / markers to remove" (all rows were
  already done); subsequent sections renumbered.
- Stale 35-line "Test Tier System" block from `conftest.py` (claimed
  three tiers, listed nonexistent example tests, duplicated TESTING.md
  §1.5); replaced with a pointer to the source-of-truth doc.
- Three near-identical "@pytest.mark.slow on individual tests still
  applies" comments (restated standard pytest semantics).
- Transitional `FakeAssocWriter` re-export comment in
  `test_runner_numpy.py`.

## [5.2.1] - 2026-04-21

### Fixed

- Restore `#define _GNU_SOURCE` at the top of
  `src/jamma/jlinalg/src/blas_dispatch.c`. The BLIS strip in 5.2.0
  removed the define along with the `dladdr` scaffolding that
  originally motivated it, but two surviving `RTLD_DEFAULT` call sites
  silently relied on it too. `RTLD_DEFAULT` is exposed by glibc's
  `<dlfcn.h>` only under `_GNU_SOURCE`; the standard manylinux image
  happens to enable it via default CFLAGS, but the AVX2 manylinux
  image (gcc-toolset-14) does not — so 5.2.0 wheel builds failed on
  both Linux jobs and no wheels reached PyPI. 5.2.0 should be
  considered unreleased; install 5.2.1 directly.

## [5.2.0] - 2026-04-21

### Added

- **Build-support consolidation**: new internal `jamma._build_support`
  package (`compile_and_link.py`, `openmp_detect.py`, `find_compiler.py`)
  is the single source of truth for compile flags, source lists, and
  link flags used by `hatch_build.py` (PEP 517 wheel path),
  `_compile_jlinalg.py` and `_compile_accel.py` (dev-mode and runtime
  recompile entry points), and the `jamma.core.recompile` ABI-mismatch
  shim. Every bare compile flag (`-O3`, `-fno-fast-math`, `-fopenmp`,
  etc.) now lives in one file; two pre-commit hooks
  (`check-compile-flag-literals.py`, `verify_compile_invocations_match.py`)
  enforce this.
- **Runtime recompile hardening**: new `jamma.core.recompile` shim uses
  a file-lock + atomic `os.replace` to serialize concurrent recompiles
  (pytest-xdist workers, parallel Databricks jobs, multiple notebook
  kernels) so they no longer race on the same `.so` path and produce a
  corrupted file. The `_compile_accel` path now verifies the freshly
  compiled `.so` actually imports before returning success — a missing
  export or bad RPATH previously let the recompile report success with
  an unusable extension.
- **Stale C extension drift detection**: new `check_c_extension_freshness.py`
  pre-push hook detects when a committed `.so` is older than its source,
  preventing pushes that would ship stale binaries.
- **CI/lint discipline**: new pre-commit hooks `check-quiet-flags.py`
  (bans `-q` / `--silent` / `--quiet` and pre-commit skip flags in
  committed code), `check-test-timeouts.py` (flags unjustified long
  pytest timeouts), and `ruff BLE001` (bans blind `except Exception`).
  New `package-smoke` CI job inspects sdist + wheel contents to prevent
  missing `_build_support` files from shipping.

### Changed

- **Pipeline refactor**: `PipelineRunner._run_inner` split into
  `_memory_preflight`, `_load_phenotypes_and_intersect_masks`, and
  `_run_loco` helpers. Shared LMM compute helpers promoted to public
  names (`build_uab_tab`, `_build_results`, etc.) to support the
  extracted dispatch-path selector.
- **LMM dispatch extracted** from `run_lmm_association_numpy` into
  `src/jamma/lmm/dispatch.py` — the ~60-line logic for selecting
  between fused/split/general kernels by `n_cvt × lmm_mode ×
  kernel-availability` is now independently unit-testable.
- **OpenMP downgrade visibility**: runtime recompile retries that fall
  back to single-threaded execution now surface a `warnings.warn()`
  rather than disappearing silently (closes the gap between build-time
  and runtime diagnostics).
- **Documentation**: WHY_JAMMA tolerance table disambiguates golden-section
  vs Brent optimizer attribution. Mermaid diagrams across
  `README.md`/`docs/` migrated from literal `\n` (which renders as
  backslash-n) to `<br/>` for proper line breaks. Build-plumbing
  references refreshed to match the `_build_support` consolidation.
- **CI**: Node runtime bumped from 22 to 24. `michael-denyer/numpy-mkl`
  references bumped to 2.4.4. Dependabot SHA-pinning now covers
  `github-actions` ecosystem.

### Fixed

- **AccelImport retry-path drift** (latent bug): the post-auto-recompile
  unpack in `compute_numpy.py` had 33 targets vs the 35-field
  `AccelImport` NamedTuple, missing `compute_score_split_general_c` and
  `compute_lrt_split_general_c`. A successful runtime recompile would
  have raised `ValueError: too many values to unpack` instead of
  recovering. Both unpack sites replaced with field-by-field binds so
  they cannot drift.
- **`_compile_accel` reported false success** (latent bug): returned
  True on compile+link without verifying the produced `.so` imports,
  so bad RPATH / missing runtime lib / ABI mismatch let
  `python -m jamma.lmm._compile_accel` exit 0 and `auto_recompile`
  report success while the real `import` still raised. Import
  verification re-added (mirrors `_compile_jlinalg`).
- **`jlinalg` recompile diagnostics invisible on Databricks**: replaced
  two `print(..., file=sys.stderr)` blocks with `warnings.warn()` so
  recompile-skipped / recompile-but-import-failed messages route
  through the same channel as the surrounding `warnings` and aren't
  swallowed by notebook stderr capture.
- **`_build_support/__init__.py` docstring** described a non-existent
  `sys.path.insert` loader; rewritten to match the actual
  `importlib.util.spec_from_file_location` + `jamma_build_support.*`
  namespace mechanism used by `hatch_build.py`.
- Runtime recompile lock-file paths (`*.so.lock`) now gitignored to
  prevent accidental commits.
- Batch LMM memory preflight threads `n_cvt` through
  `check_memory_before_run` so multi-covariate runs don't silently pass
  a single-covariate preflight and OOM at the real allocation.

### Removed

- **BLIS dispatch path** from `src/jamma/jlinalg/src/blas_dispatch.c`.
  The `discover_bundled_blis()` discovery routine, the `is_blis`
  parameter threaded through six resolver functions, and the
  co-located `libblis-firestorm.dylib` binary (never tracked in git,
  never shipped in any wheel) are gone. jlinalg now dispatches to
  vendor ILP64 BLAS/LAPACK (Accelerate on macOS 13.3+, MKL-ILP64 on
  Linux/Windows via the `michael-denyer/numpy-mkl` index) with NumPy
  fallback otherwise — no middle tier. BLIS was BLAS-only; eigh fell
  through to NumPy anyway, so the dispatch path offered no net speedup
  on any active install. Net: `-184 / +49` lines in `blas_dispatch.c`,
  plus related cleanup across `jlinalg.h`, two tests, and two core
  docstrings.
- Dead LP64 branch in `jlinalg.select_best_backend` and stale legacy
  fields from `jlinalg_eigh_status_t` — jlinalg was never wiring LP64
  backends anyway; the dead code inflated the API surface.
- Orphaned `_compile_utils.py` and legacy `openmp_detect.py` in
  `jamma.core` (moved to `jamma._build_support`).
- Redundant `auto_recompile` re-export shim in `jamma.lmm`.

## [5.1.6] - 2026-04-15

### Fixed

- Batch LMM memory preflight now propagates `n_cvt` to `estimate_lmm_memory`
  at both call sites (`PipelineRunner._run_inner` batch branch and
  `run_lmm_association_numpy`). Previously these passed only
  `(n_samples, n_snps)`, silently defaulting `n_cvt=1`, so multi-covariate
  runs could pass the preflight and then OOM at the real `Uab_batch` /
  `Iab_batch` allocations (which scale with `n_cvt`). The streaming branch
  was already correct.

## [5.1.5] - 2026-04-11

### Added

- Warn when fewer than 50 samples enter the LMM (after phenotype/covariate
  filtering). LMM-based GWAS has insufficient statistical power below this
  scale, and JAMMA's batch golden-section lambda optimizer assumes unimodal
  log-likelihoods — an assumption most likely to fail at very small n. The
  warning fires once per run from both the pipeline (CLI) and `run_lmm()`
  (programmatic API). See `docs/GEMMA_DIVERGENCES.md` §6.

### Changed

- Rename `EQUIVALENCE.md` → `GEMMA_EQUIVALENCE.md` and
  `NUMERICAL_EQUIVALENCE_BOUND.md` → `GEMMA_NUMERICAL_EQUIVALENCE_BOUND.md`;
  update all cross-references across docs, tests, README, and CHANGELOG
- Link previously orphaned `GEMMA_NUMERICAL_EQUIVALENCE_BOUND.md` from README

## [5.1.4] - 2026-04-08

### Changed

- Remove 7 `inspect.getsource()` anti-pattern tests and rewrite
  `test_lapack_no_ffast_math` to parse build config files as text
- Replace `MagicMock` with real types and fakes across test suite
- Add pre-commit hook banning `inspect.getsource()` in tests
- Add test type routing and bug fix workflow sections to TESTING.md

## [5.1.3] - 2026-04-07

### Fixed

- Compiler detection now uses cc/clang/gcc fallback chain instead of failing
  when `CC` is unset or points to a missing compiler
- `hatch_build.py` uses the same fallback chain for wheel builds
- Narrow exception catches in `_compile_jlinalg.py` — no longer swallows
  unexpected errors during C extension compilation
- Assert C extension is loaded in CI to catch silent compilation failures

### Added

- Sigstore build provenance attestations on PyPI publish
- OSV vulnerability scanning on pull requests
- YAML-form issue templates (bug report, feature request)
- Streaming covariate integration tests

### Changed

- Replace pre-commit with prek (Rust-based, no Python dependency)
- Pin all GitHub Actions to commit SHAs (Dependabot keeps them updated)
- Pin `hatchling==1.29.0` and `numpy==2.4.3` in build-system.requires
- Use `--index-url` instead of `--extra-index-url` for custom package indexes

### Security

- Harden supply chain: pinned actions, Sigstore attestations, osv-scanner
- Dependabot configured for GitHub Actions ecosystem (weekly)

## [5.1.2] - 2026-04-02

### Fixed

- Ctrl+C during eigendecomposition now exits immediately instead of blocking
  until the LAPACK call finishes
- Progress bar no longer shows 100% before propagating worker exceptions
  (MemoryError, LinAlgError)
- Broken pipe on stdout no longer masks eigendecomposition results
- Remove meaningless AdaptiveETA widget from time-based progress bar

### Added

- Tests for `timed_progress()`: exception propagation, 99% cap, error display,
  `estimated_seconds=0` edge case

## [5.1.1] - 2026-04-01

### Fixed

- Time estimates now show BLAS backend caveat when not running on MKL
  (estimates are calibrated to MKL ILP64 on 48-core Xeon)
- Memory pre-flight check logs active BLAS backend and ILP64 status; warns
  when >40k samples without ILP64 or when time estimates are uncalibrated
- Fix pip install order in docs: deps first, numpy-mkl second, jamma --no-deps
  last to prevent ILP64 overwrite

### Changed

- High contrast mermaid diagrams across all docs (README, CODEMAP,
  JLINALG_ARCHITECTURE, USER_GUIDE) with dark subgraph backgrounds and
  bright node fills
- Add three new diagrams to USER_GUIDE: GWAS pipeline flow, BLAS/eigendecomp
  dispatch, and memory safety architecture

## [5.1.0] - 2026-03-25

### Added

- Telemetry transparency: opt-out via `JAMMA_NO_TELEMETRY=1` or `DO_NOT_TRACK=1`,
  with docs and hardening for privacy-sensitive environments
- Safety gates for LP64 integer overflow, LOCO chromosome invariant, and ABI
  validation at import time
- GEMMA equivalence tests for full validation coverage

### Changed

- Rename pipeline methods to `_run_batch`/`_run_streaming` for clarity
- Remove dead `lmm_mode` parameter from `select_execution_mode`
- Remove dead backend dispatch types and simplify consumers
- Consolidate dev dependencies and clean up CI build matrix
- Fix incorrect `gwas()` API examples in README and USER_GUIDE
- Update CODEMAP.md after backend simplification

## [5.0.1] - 2026-03-25

### Fixed

- Fix CI smoke tests for v5.0 simplification: remove `daxpy` import from C extension (moved to numpy-only), handle missing vendor LAPACK gracefully in eigh smoke test

## [5.0.0] - 2026-03-25

### Changed

- **BREAKING**: Remove JAX backend — NumPy+C is now the only compute path
- Strip own-BLAS/LAPACK C implementations (dgemm, dsyrk, dsytrd, dstedc, dormtr); vendor-only dispatch
- Archive JAX runners, tests, and scripts to `legacy/`
- Simplify jlinalg to vendor-BLAS-only dispatch (ILP64 MKL/OpenBLAS/Accelerate → NumPy fallback)
- Add clang-format and cppcheck pre-commit hooks for C extensions
- Add SeededETA progress bars with model-predicted initial ETAs
- Net -21,900 lines removed

## [4.6.3] - 2026-03-24

### Changed

- Raise maximum covariate limit from 20 to 100 in C extension (MAX_N_CVT)

## [4.6.2] - 2026-03-23

### Changed

- Eigendecomp log now shows driver name (DSYEVD-inplace/DSYEVD/DSYEVR) instead
  of generic `jlinalg.eigh`, explains why that driver was chosen (e.g. "kinship
  in memory, overwriting in place"), and lists the relevant alternative with its
  memory cost (e.g. "DSYEVR fallback=126.3GB")

## [4.6.1] - 2026-03-23

### Fixed

- Prefer clang over GCC when linking libiomp5 — GCC's GOMP compatibility shim
  triggers assertion failures (`kmp_runtime.cpp` Error #13) after MKL LAPACK
  operations (e.g. DSYEVR). Clang natively generates `kmp_*` calls that
  libiomp5 handles correctly.
- Simplify clang OpenMP detection to avoid `omp.h` dependency and `-x none`
  parsing issues with libiomp5.so paths
- Add `JLINALG_NO_VENDOR_LAPACK` env var to skip MKL dsyevd/dsyevr in eigh,
  falling back to jlinalg-own LAPACK
- Respect `JLINALG_NO_VENDOR_LAPACK` in eigendecomp driver selection
- Replace OpenMP with pthreads in `compute_snp_stats_chunk` to avoid
  MKL/libiomp5 conflict — SNP stats is memory-bandwidth-bound, not compute-bound
- Auto-recompile jlinalg C extension on import failure (stale `.so`)

### Changed

- Centralize jlinalg thread control: new `jlinalg_threads()` context manager
  with RLock for thread-safe `set_n_threads()` scoping (replaces ad-hoc
  `blas_threads()` calls for jlinalg rotation in runners)
- Centralize C extension OpenMP detection: `get_c_extension_capabilities()`
  returns `(available, has_openmp)` tuple; `get_c_extension_thread_count()`
  consolidates thread sizing logic
- Chunk `compute_snp_stats()` in 10k-SNP slices to avoid full contiguous
  copy of large genotype matrices
- `detect_openmp_flags()` returns `cc_override` as third element when
  switching to clang for libiomp5 compatibility
- Fix pipeline thread logging for serial (no-OpenMP) C extension builds

## [4.6.0] - 2026-03-23

### Added

- `JAMMA_NO_OPENMP=1` environment variable to compile C extensions without
  OpenMP — completely avoids dual OpenMP runtime SIGABRT on Databricks where
  both Intel OMP (MKL) and GNU OMP (scipy) are pre-loaded by the kernel before
  any user code runs. Single-threaded C extensions are still much faster than
  pure-Python fallback.

## [4.5.3] - 2026-03-23

### Fixed

- Move `KMP_DUPLICATE_LIB_OK` to `jamma/__init__.py` (earliest import point) —
  on Databricks, `mkl._mklinit` and scipy are loaded by the kernel before
  `jlinalg/__init__.py` runs, so the v4.5.2 fix was too late

### Changed

- Consolidate OpenMP detection into `core.openmp_detect` — eliminates 3-way
  duplication across `_compile_accel.py`, `_compile_jlinalg.py`, and
  `hatch_build.py` (hatch_build.py keeps its own copy with a sync comment)

## [4.5.2] - 2026-03-23

### Fixed

- Set `KMP_DUPLICATE_LIB_OK=TRUE` before C extension import to prevent dual
  OpenMP runtime SIGABRT on Databricks — scipy (pre-loaded by kernel) brings
  libgomp while jlinalg/`_lmm_accel` link against MKL's libiomp5

## [4.5.1] - 2026-03-23

### Fixed

- Two-step compile+link for `_lmm_accel` to prevent dual OpenMP runtime SIGABRT
  on Linux with MKL numpy — GCC's `-fopenmp` implicitly links libgomp alongside
  libiomp5, causing `kmp_runtime.cpp` assertion failure

## [4.5.0] - 2026-03-23

### Added

- Split general Score/LRT C entry points (`compute_score_split_general_c`,
  `compute_lrt_split_general_c`) — accept SoA data directly, eliminating
  `reconstruct_uab_from_soa` for n_cvt>1 (~75 GB saved at n_cvt=2/100k samples)
- `out=` buffer reuse for general n_cvt in `batch_compute_uab_varying_soa_numpy` —
  zero per-chunk allocation for varying SoA across all covariate counts
- `logdet_from_row0` helper — deduplicates 3 inline identity Pab prepass blocks
- Fused general mode-4 dispatch for n_cvt≥2 — all 8 output arrays (Wald + Score +
  LRT) computed in a single workspace pass

### Fixed

- Mode-4 fused general availability guard now checks `_C_MODE4_FUSED_GENERAL_AVAILABLE`
  (previously used Wald-only flag)
- `out=` buffer validates dtype (float64) and C-contiguity
- OpenMP compile/link flag split to prevent dual-runtime SIGABRT (libgomp + libiomp5)
- Chunk-size accounting for n_cvt>1 Score/LRT reflects split C dispatch (no Uab
  reconstruction overhead)

## [4.4.2] - 2026-03-23

### Fixed

- Use actual inplace DSYEVD memory requirement for DSYEVR fallback decision instead
  of always using the non-inplace peak estimate
- Guard `out=` buffer allocation behind `n_cvt==1` in batch and streaming NumPy
  runners — `batch_compute_uab_varying_soa_numpy` only supports it for single-covariate
- Improve `dispatch_soa_split` error message for unreachable mode-4 path
- Simplify no-DSYEVR fallback branch — no longer silently downgrades inplace to
  conservative estimate

## [4.4.1] - 2026-03-22

### Changed

- Updated benchmark table with best NumPy+C numbers — Wald 879ms (12.5x vs GEMMA), All 16.0x

## [4.4.0] - 2026-03-21

### Added

- Early sample filtering via `valid_indices` — missing-phenotype samples are
  excluded before kinship accumulation rather than post-hoc, avoiding full n×n
  matrix materialisation (kinship streaming, LOCO NumPy, LOCO JAX, PipelineRunner)
- Input validation (`_validate_valid_indices`) for LOCO NumPy kinship streamer
- Filtered sample count in LOCO log messages for both NumPy and JAX backends

### Removed

- Secular equation solver and LOCO streaming modes (`S_CHR`, `X_C`,
  `X_C_SEQUENTIAL`) — superseded by streaming LOCO with better memory
  characteristics
- `--secular` CLI flag and `use_secular_update` config option
- `loco_eigen_update.py` (1090 lines) and associated tests (~2200 lines)

### Fixed

- Replace `assert` with `raise ValueError` for kinship shape validation in
  pipeline (assert stripped by `python -O`)
- Remove stale documentation references to deleted secular update feature

## [4.3.1] - 2026-03-21

### Added

- Pipeline machinery for NumPy streaming runner — overlaps DGEMM rotation of
  chunk N+1 with C extension compute of chunk N via ThreadPoolExecutor
  double-buffering, with adaptive core splitting and memory-aware chunk sizing

### Changed

- Swap utg_t layout to (n_snps, n_samples) for direct DGEMM TRANSA — eliminates
  post-rotation transpose in batch and streaming NumPy runners
- Add GEMMA Accelerate to backend comparison benchmark

### Fixed

- Avoid O(n²) eigenvector copy in streaming chunk loop
- Rename unused loop variable to satisfy linter

## [4.3.0] - 2026-03-21

### Added

- Fused general C kernels for arbitrary n_cvt Wald test — eliminates Python-level
  Uab reconstruction loop for multi-covariate models (n_cvt ≥ 2)
- Availability flags (`_C_FUSED_GENERAL_AVAILABLE`, `_C_MODE4_FUSED_GENERAL_AVAILABLE`)
  with workspace creation and dispatch functions
- Runner integration test for n_cvt=2 end-to-end fused vs non-fused validation

### Changed

- Batch and streaming runners auto-dispatch fused general path when n_cvt ≥ 2
  and C extension is available
- Updated PERFORMANCE.md and time estimates to v4.2.0 benchmarks (2h 29m at 125k)
- Removed DSYEVR time multiplier — empirically comparable to DSYEVD at scale

### Fixed

- Input validation hardening — bounds checks on table indices, var columns,
  n_snps in C kernels
- Mode-4 fused general disabled at dispatch level due to NaN lambda_mle bug

## [4.2.1] - 2026-03-20

### Fixed

- Link Intel OpenMP by full path in hatch_build.py — numpy bundles versioned
  names like `libiomp5-2f035e84.so` with no unversioned symlink, so `-liomp5`
  fails at link time

## [4.2.0] - 2026-03-20

### Changed

- Fused Uab compute — reduces peak memory for NumPy batch and streaming runners
  by computing Uab in a single pass instead of separate U.T @ W and U.T @ y steps
- Complete analytical dev2 for all n_cvt values (previously only n_cvt=1)
- Deduplicate cleanup_jax_caches and fix per-chromosome cache clearing in LOCO
- Extract shared PASS 1 + setup into _loco_chr_common for LOCO runners
- Extract try/finally bodies to _impl() helpers in JAX runners
- Remove private import aliasing in loco.py

### Fixed

- Decouple DSYEVR/DSYEVD attribute checks in pre-flight memory estimate
- Relax eigh inplace eigenvalue tolerance from 1e-12 to 5e-12 for CI stability
- Relax pve_se assertion for synthetic data with no signal
- Fix flaky memory test and add mode-4 threading parity test

### Removed

- Lazy eigendecomposition (phases 89, 89.1) — dstedc workspace (3N²) exceeds
  DSYEVR memory at scale, making the lazy path unviable for 100k+ samples

## [4.1.0] - 2026-03-19

### Changed

- `jlinalg.eigh` gains `inplace` keyword — when `inplace=True`, eigenvectors are
  written directly into the input K buffer, avoiding one N×N allocation (~125 GB
  savings at 125k samples). Requires vendor DSYEVD (ILP64 BLAS).
- `eigendecompose_kinship` automatically uses `inplace=True` when vendor DSYEVD is
  available and DSYEVD fits in memory
- Memory estimator (`check_memory_before_run`) accounts for in-place path, producing
  tighter estimates when vendor DSYEVD is available
- Add `_dsyevd_inplace_peak_gb` memory estimator for the in-place eigendecomp path

### Fixed

- Remove unused `null_inv_ww` variable in `compute_score_split_c` (_lmm_accel.c)
- Document FP tolerance rationale in streaming NumPy test

## [4.0.3] - 2026-03-18

### Fixed

- _lmm_accel compile summary now correctly reports "single-threaded" when
  OpenMP fallback was used, instead of always reporting "OpenMP"

## [4.0.2] - 2026-03-18

### Changed

- C extension compile scripts now default to quiet output — only errors and a
  one-line summary are printed. Pass `verbose=True` for full per-command detail.

## [4.0.1] - 2026-03-18

### Fixed

- Link Intel OpenMP (libiomp5) by full path in C extension compile scripts — numpy
  bundles versioned names like `libiomp5-2f035e84.so` with no unversioned symlink,
  so `-liomp5` fails at link time
- Add OpenMP link fallback in jlinalg compile — retries without OpenMP flags if
  linking fails, producing a single-threaded build instead of a hard error

### Changed

- **Eigendecomposition now uses jlinalg.eigh** — replaced the legacy `_eigen_accel`
  C extension and `numpy._umath_linalg.eigh_lo` gufunc cascade with unified
  `jlinalg.eigh`, which dispatches to vendor DSYEVD/DSYEVR or the jlinalg D&C
  pipeline depending on available BLAS backends
- Add DSYEVR vendor dispatch to jlinalg C layer — memory-pressure fallback with
  O(N) workspace vs O(N²) for DSYEVD, ILP64-only
- Wire `jlinalg.dsyrk` into kinship and `jlinalg.dgemm` into prepare
- Expose `jlinalg.blas_has_dsyevr` capability flag
- `jlinalg.eigh` now raises `numpy.linalg.LinAlgError` (not `RuntimeError`) on
  convergence failure
- Memory estimator simplified: removed `_inplace_eigen_available()` check since
  jlinalg.eigh always allocates separate eigenvectors
- DSYEVR availability check in `check_memory_before_run()` now queries
  `jlinalg.blas_has_dsyevr` instead of importing from `eigen.py`
- **Rename `jblas` package to `jlinalg`** — the package now covers BLAS, LAPACK,
  and LAPACKE dispatch (not just BLAS), so `jlinalg` ("JAMMA linear algebra")
  better reflects its scope. All imports, C function prefixes (`jlinalg_*`),
  macros (`JLINALG_*`), and file paths updated.

### Removed

- Delete legacy `_eigen_accel` C extension and `_secular_accel` C extension
  source + compile script (`_secular_accel.c`, `_compile_secular.py`);
  LOCO secular path now always uses Python fallback
- Remove `INPLACE_EIGEN_AVAILABLE` flag and `_eigh_inplace()` gufunc path
- Remove `_DSYEVR_AVAILABLE`, `_try_import_dsyevr()`, `_lazy_init_dsyevr()`,
  `_select_eigen_driver()`, `_eigh_dsyevr()` from `eigen.py`
- Remove `_inplace_eigen_available()` from `memory.py`

## [4.0.0] - 2026-03-18

### Added

- **NumPy streaming runner** — disk-streaming LMM association using the C
  extension, matching the JAX streaming runner's two-pass architecture
  (float32 stats pass then float64 compute pass) with incremental I/O
- Wire numpy-streaming into pipeline, CLI (`--backend numpy-streaming`),
  backend selection, and benchmark suite

### Fixed

- Thread-safe P_yy warning deduplication — replace global `bool` flags with
  `threading.local()` in `likelihood.py` and `likelihood_numpy.py`
- Add `get_last_run_timing()` accessor to `runner_jax.py` matching the
  pattern in streaming runners; pipeline uses accessor instead of directly
  importing the mutable module-level dict
- Inline `_calc_pab_general` into `calc_pab`, removing unnecessary
  indirection layer
- Use keyword arguments for `AccelImport` NamedTuple construction to prevent
  positional field mismatch in the 17-field type
- Narrow `_check_hwe_support` to numpy-batch only (was incorrectly guarding
  all numpy paths)

### Changed

- Exclude `tier3` marker from default pytest addopts — the 22-minute
  `test_secular_speedup_correctness_at_scale` was running on every invocation
- Mark eigendecomp symmetry check tests and LOCO eigen cache integration
  tests as `slow` (15–35s each)

## [3.5.1] - 2026-03-12

### Fixed

- Ship `_secular_accel.c` C extension in wheel — was missing from
  `hatch_build.py`, causing Databricks to fall back to Python rank-1 update
  which allocates n×n dense matrices (58 GB at n=85k) and segfaults
- Guard Python fallback rank-1 updates with `MemoryError` at n > 10k to fail
  fast with actionable message instead of silent segfault

## [3.5.0] - 2026-03-12

### Added

- Benchmark telemetry module (`core/telemetry.py`) — appends structured JSONL
  run records to `~/.jamma/benchmarks.jsonl` with `JAMMA_NO_TELEMETRY` opt-out
- `n_cvt`-aware backend selection — `select_execution_mode` accounts for
  covariate count in memory estimates and falls through to JAX when C general
  extension is unavailable for `n_cvt > 1`
- Telemetry emission from `PipelineRunner.run()` via `_emit_telemetry()` helper
  (both LOCO and standard paths)

## [3.4.1] - 2026-03-11

### Changed

- Make `deflated` a required parameter in blocked Cauchy multiply functions,
  preventing silent fallback to approximate 0/0 handling
- Remove redundant `n` parameter from `_check_and_reorthogonalize` helper
- Replace O(n) `argmin` with O(log n) `searchsorted` in deflated column detection
- Lazy `argsort` — check `np.diff >= 0` before sorting eigenvalues
- Deduplicate eigen write block and `batch_chr_set` in LOCO orchestrator

## [3.4.0] - 2026-03-11

### Added

- LOCO secular equation solver — O(n^2 * r_eff) eigenvalue perturbation path
  replacing O(n^3) `np.linalg.eigh` for leave-one-chromosome-out eigendecomposition.
  Enabled via `--secular` CLI flag or `PipelineConfig(secular=True)`
- C extension (`_secular_accel.c`) implementing LAPACK DLAED4-based rank-1
  eigenvalue solver with negative-rho handling via negation/reversal identity
- Delta-path eigenvector recomputation eliminating 55 GB `Q = np.eye(n)` allocation
  at n=83k. Two-pass algorithm with blocked Cauchy multiply and pre-allocated buffers
- `LocoStreamingMode` enum and `SequentialLocoResult` NamedTuple for type-safe
  streaming mode dispatch in LOCO pipeline
- `yield_x_c_sequential` streaming mode for one-chromosome-at-a-time secular processing
- Orthogonality monitoring (`check_orthogonality`) and `reorth_interval` parameter
  in secular solver for numerical stability tracking
- `bench_secular.py` benchmark script for secular solver performance profiling

### Changed

- `SecularImport` NamedTuple + named constants for secular solver clarity
- Extract `_cauchy_block` helper, deduplicating 6 call sites in eigenvector reconstruction
- Deflation guard in C eigenvector path, NaN check in delta forward pass

## [3.3.2] - 2026-03-10

### Fixed

- LOCO `save_kinship` log message showed `.txt` path but `write_kinship_matrix()`
  actually writes `.npy` (binary default since v2.11). Now logs the actual path written.

## [3.3.1] - 2026-03-10

### Fixed

- LOCO multi-pass batch sizing now reserves eigendecomposition workspace memory.
  Previously the batch sizer allocated too many S_chr matrices per pass, leaving
  insufficient memory for DSYEVR/DSYEVD when eigendecomp runs while the generator
  is suspended with remaining S_chr matrices alive (OOM on 85k+ samples, 40 chromosomes)
- Fix `single_pass_gb` formula in JAX LOCO path — was `matrix_gb * (1 + n_chr)`,
  now `(2 + n_chr)` to account for K_loco_buf
- Fix `min_required_gb` to include eigendecomp workspace and K_loco_buf in both
  JAX and NumPy paths

## [3.3.0] - 2026-03-10

### Added

- NaN diagnostic accumulation in streaming runner — tracks per-key NaN counts
  across chunks and logs warnings with actionable advice (degenerate genotypes,
  kinship quality)

### Changed

- Extract `_guarded_compute` helper to DRY up 8 duplicated try/except error-
  wrapping blocks in NumPy runner with operation-specific labels for diagnosis
- Add `dtype.kind` guard on NaN check to prevent diagnostic from crashing on
  non-float arrays
- LMM All (`-lmm 4`) NumPy+C: 5.5s → 1.4s on mouse_hs1940 (3.6x faster,
  14.3x vs GEMMA) — removing per-SNP exception frame overhead from hot loop

## [3.2.0] - 2026-03-09

### Added

- LOCO per-chromosome eigen cache (`--eigen-dir`) — saves eigendecomposition
  results per chromosome and reloads them on subsequent runs, skipping both
  kinship computation and eigendecomposition entirely
- `_find_loco_eigen_cache()` helper validates cache completeness before use;
  partial or missing caches fall back to full compute transparently
- `-eigen` flag now works with `-lmm -loco` to write per-chromosome eigen files
  (previously only supported with `-gk`)
- `write_eigen`, `eigen_dir`, `eigen_prefix` parameters on `run_lmm_loco()`
- Dimension validation on cached eigen load with chromosome-contextual errors
- `-d`/`-u` (pre-computed global eigen) now blocked with `-loco` with clear
  error message directing users to `--eigen-dir`

## [3.1.0] - 2026-03-07

### Added

- PVE standard error (`pve_se`) computed via delta method from REML second
  derivative — available in `LmmRunResult`, `LocoResult`, `PipelineResult`,
  and `GWASResult`
- `LocoResult` dataclass replaces raw tuple return from `run_lmm_loco()`,
  with named fields: `associations`, `n_tested`, `pve`, `pve_se`
- `finite_difference_dev2()` — numerical REML second derivative via central
  finite differences; used for `pve_se` computation for all covariate counts
- `reml_log_likelihood_dev2()` — partial analytical REML second derivative
  (intercept-only); delegates to `finite_difference_dev2` for n_cvt > 1
- `calc_ppab()` and `calc_pppab()` — second/third-order projected Pab
  recursions (ports of GEMMA's `CalcPPab`/`CalcPPPab`)
- Finite-difference tests validate second derivative for n_cvt=1,2,3,4
- `jax.clear_caches()` now runs in `finally` blocks across all runners,
  with defensive `try/except` to avoid masking original exceptions

### Fixed

- `reml_log_likelihood_dev2()` was missing the d²(logdet_hiw)/dλ² term,
  producing incorrect REML curvature for multi-covariate models (n_cvt > 1).
  `compute_and_log_pve()` now uses `finite_difference_dev2()` for all n_cvt

### Breaking

- `run_lmm_loco()` returns `LocoResult` dataclass instead of
  `tuple[list, int, float | None, float | None]`

## [3.0.1] - 2026-03-06

### Fixed

- Streaming memory estimates now distinguish disk chunk size (raw genotype buffer)
  from JAX sub-chunk size (rotation/Uab/grid buffers), producing accurate LMM
  phase estimates after per-subchunk flush

## [3.0.0] - 2026-03-06

### Breaking

- `LmmRunResult` no longer supports list-like access (`len()`, iteration,
  indexing, `bool()`). Use `.associations` explicitly:

  ```python
  # Before (2.x)
  results = run_lmm_association_numpy(...)
  for r in results: ...

  # After (3.0)
  run_result = run_lmm_association_numpy(...)
  for r in run_result.associations: ...
  ```

### Added

- `_chunk_result_to_numpy()` — transfers JAX sub-chunk results to host
  immediately instead of accumulating on device until disk chunk completes
- PVE capture in LOCO is now robust to filtered first chromosomes — falls back
  to the next chromosome with passing SNPs
- Warning logged when PVE cannot be computed (all chromosomes fully filtered)
- Regression tests for per-sub-chunk flushing (disk-write and in-memory paths)
- PVE cross-backend parity assertions in LOCO tests

### Changed

- Streaming and LOCO runners flush each JAX sub-chunk to host/disk immediately,
  reducing peak device memory from O(disk_chunk) to O(jax_chunk)

### Removed

- Dead code: `strip_and_append`, `_concat_jax_accumulators`,
  `_init_accumulators`

## [2.12.0] - 2026-03-06

### Added

- Invariant/varying Uab column split for general n_cvt — correctly classifies
  columns as lambda-invariant or lambda-varying based on covariate structure
- Consolidated pipeline startup logging into a single banner line

### Fixed

- JAX batch LMM memory estimate used max chunk size instead of actual chunk size,
  causing unnecessary chunking on smaller datasets
- JAX batch memory safety factor reduced from 1.5x to 1.25x to avoid over-conservative
  chunk splitting
- Pipeline banner logging hardened against missing backend diagnostics
- Technical RSS labels replaced with plain English in log messages

### Changed

- Extracted `_prepare_general_split_inputs` to deduplicate column setup across
  Uab split paths
- Simplified banner formatting code

### Documentation

- Added cross-references between LOCO test files
- Clarified `gwas()` as recommended API, Intel CPU optimization, platform BLAS details

## [2.11.2] - 2026-03-05

### Fixed

- Test expected text `.cXX.txt` kinship path but default output is now binary `.cXX.npy`

### Changed

- Consolidated pipeline startup logging into a single banner line showing runner, BLAS backend, eigen driver, C extension status, and thread count
- Updated project logo

## [2.11.1] - 2026-03-05

### Fixed

- Multi-phenotype runs crashed with eigenpair dimension mismatch when phenotype
  missingness differed across columns — now NaN-stamps samples outside the shared
  valid_mask intersection so runners compute a consistent mask
- `JAMMA_BACKEND` environment variable was ignored when `backend="auto"` — now
  resolved before auto-selection logic
- Backend logging falsely attributed selection to `JAMMA_BACKEND` when env var was
  set but not actually honored; removed misleading "JAX not installed" message for
  memory/C-extension-based NumPy selection
- NumPy multi-phenotype runs reloaded full genotype matrix per phenotype — now
  pre-loads PLINK data once
- Windows + JAX docs contradiction between README and User Guide
- "Full test suite" claim in PERFORMANCE.md now notes default marker exclusions

### Changed

- Extracted `compute_valid_mask()` to `prepare_common.py` — single source of truth
  for valid-sample mask logic (was duplicated in pipeline, prepare_common, loco)
- Added `get_last_run_timing()` accessor for thread-safe timing snapshot

## [2.11.0] - 2026-03-05

### Added

- Binary `.npy` as default output format for kinship matrices and eigendecomposition
  files — 10-100x faster I/O at scale. Use `--legacy-text` for GEMMA-compatible text format
- Multi-phenotype support: `-n "1 2 3"` or `-n "1,2,3"` processes multiple phenotype
  columns with a single eigendecomposition, saving hours at scale
- Shared `npy_cache` module for `.npy` sibling validation logic

### Changed

- Kinship output file extension changed from `.cXX.txt` to `.cXX.npy` by default
- Eigen output files changed from `.eigenD.txt`/`.eigenU.txt` to `.eigenD.npy`/`.eigenU.npy`
  by default

## [2.10.1] - 2026-03-03

### Fixed

- Golden section optimizer returned inconsistent (lambda, logl) pair — lambda
  at midpoint `(a+b)/2` but logl as `max(fc, fd)` from different points c and d.
  Now evaluates logl at the midpoint, matching the JAX path. This eliminates
  cross-backend p_lrt divergence (4.5e-4 → 1.05e-10 on gemma_synthetic)
- `compare_assoc_results` LRT mode used `pvalue_rtol` (1e-4) instead of
  `p_lrt_rtol` (5e-3) for p_lrt comparison
- C vs Python parity test compared C extension (generic golden section) against
  Python split-Uab optimizer — now calls generic optimizer directly
- `check_memory_before_run` passed defaults to `_compute_chunk_size` instead of
  `n_samples` and `pipeline_buffers=2`, causing overestimated memory
- `_compute_chunk_size_numpy` lacked `pipeline_buffers` type/range validation
- Exposed rotation time metric could exceed total rotation time due to
  GC/scheduling jitter — now capped at `rot_dur`

### Added

- MemoryError passthrough tests for both JAX batch and streaming runners
- `pipeline_buffers` TypeError tests (float/str/None) for all chunk sizers
  and memory estimators

## [2.10.0] - 2026-03-03

### Added

- Rotation-compute overlap pipelining — both JAX batch and streaming runners
  overlap BLAS rotation (U.T @ G) with XLA compute using a `ThreadPoolExecutor`
  background thread, achieving ~15% wall-time reduction on mouse_hs1940
- `pipeline_buffers` parameter for `_compute_chunk_size` and streaming memory
  estimators to account for double-buffered UtG arrays during overlap
- Input validation for `pipeline_buffers` (type check, >= 1 guard)
- `MemoryError` passthrough in both runners to avoid wrapping OOM as RuntimeError
- Background rotation failure propagation tests with exception chaining
- Multi-file-chunk `prev_compute_end` handoff test for streaming runner
- Rotation overlap effectiveness tests (timing invariants)

### Fixed

- Streaming runner `ThreadPoolExecutor` scope hoisted to span BED file-chunk
  boundaries, fixing `prev_compute_end` timing handoff across chunks
- Memory estimators in `check_memory_before_run` and streaming runner now pass
  `pipeline_buffers=2` for accurate double-buffer accounting

## [2.9.6] - 2026-03-03

### Added

- Device-memory-aware JAX chunk sizing — auto-scales to GPU/TPU memory budget
  with psutil fallback for CPU
- Filtered reads and threaded prefetch iterator for streaming runner —
  `snp_indices` column-selection in PLINK reader skips unneeded genotype columns
- Multi-pass chromosome batching for NumPy LOCO kinship — streams BED in
  multiple passes when all per-chromosome matrices don't fit in memory
- `SnpStatsCache` — caches global SNP statistics from kinship pass, eliminating
  redundant per-chromosome BED re-reads in the association phase
- Valid-indices threading — propagates phenotype-valid sample indices into
  kinship streaming so K_loco is built at n_valid × n_valid directly
- In-place K_loco buffer reused across chromosomes (caller must eigendecompose
  before advancing)
- `JAMMA_LOCO_WORKERS` env var for LOCO parallel execution control
- Imputation guard raises on >50% missing rate before centering
- 500+ lines of new tests: chunk tuning, split-Uab modes, LOCO aliasing,
  filtered reads, streaming edge cases, multi-pass equivalence, valid-sample
  subsetting

### Changed

- Split-Uab for all LMM modes — LRT/Score/All reconstruct full Uab from split
  SoA components with correct 9-col peak memory accounting
- Adaptive core split (`compute_pipeline_core_split`) replaces fixed 75/25 split
  with min-2 / fallback logic
- BLAS controllability detection gracefully falls back when Accelerate (macOS) is
  the BLAS backend
- DRY refactors in plink.py and runner_numpy.py, structured error handling in
  chunk.py
- Documentation: quote `'jamma[jax]'` in shell contexts (zsh glob fix), remove
  misleading GPU Support section

### Fixed

- K_loco aliasing bug — copy buffer before yielding to prevent all chromosomes
  sharing a single array
- SnpStatsCache stores `n_samples` (all-sample population denominator) —
  prevents inflated miss_rates when n_valid < n_samples
- `_s_full_accumulated` assert prevents S_full double-counting in LOCO
- Strict snp_indices validation (ascending + bounds), removes tail-chunk NaN
  padding

## [2.9.5] - 2026-03-02

### Added

- AVX2-optimized wheel build job in CI — builds with `-march=x86-64-v3 -mavx2`,
  verifies AVX2 instructions via `objdump`, attaches to GitHub releases
- `aligned_alloc` for C extension workspace arrays (32-byte AVX2 alignment)
- ABI mismatch detection — stale `.so` fallback logged via `loguru.warning`

### Changed

- Fused Wald computation into golden section optimizer — eliminates redundant
  `n_samples` pass to recompute `hi_eval` at `lambda_opt` by reusing the buffer
  from the final REML evaluation
- C extension build: `CFLAGS` passthrough, `-funroll-loops`,
  `-fno-finite-math-only` safety, C11 standard, `schedule(static)` for uniform
  SNP cost
- C vs Python parity test uses well-conditioned synthetic data (proper w×x
  cross-products) with calibrated tolerances from measured FP differences

### Fixed

- Degenerate SNP hardening — negative P_YY guard (Schur complement), early-return
  when every grid point is NaN (`REML_SENTINEL` pattern), explicit `is_valid`
  return from `wald_from_pab`, p-value clamping to [0,1]
- C extension validity checks hardened with input shape and scalar parameter
  validation
- README: removed false GPU acceleration claims, fixed architecture diagram

## [2.9.4] - 2026-03-02

### Changed

- `impute_and_center()` operates in-place on writable NumPy arrays, eliminating an
  O(N×M) copy during kinship computation (KIN-03)
- `impute_center_and_standardize()` uses `np.einsum('ij,ij->j')` for variance
  computation instead of materializing an O(N×M) `X**2` intermediate (KIN-06)
- `compute_loco_kinship()` rewritten in pure NumPy — no longer initializes JAX
  during in-memory LOCO kinship computation (KIN-01, KIN-04)
- `_ensure_float64()` skips copy when input is already float64 (KIN-02)
- Per-chromosome `block_until_ready()` calls added to streaming LOCO accumulation
  to prevent unbounded JAX async dispatch (KIN-05)
- `_compute_chunk_size()` simplified: removed vestigial `n_samples`/`bytes_per_element`
  parameters, uses `MAX_SAFE_CHUNK` cap directly

### Fixed

- Streaming LOCO `S_chr` matrices were not synchronized before subtraction, which
  could produce stale results under heavy JAX async dispatch

## [2.9.3] - 2026-03-01

### Added

- Runtime LAPACK discovery via dlopen — `_eigen_accel` no longer has link-time LAPACK
  dependency, making compiled wheels portable across numpy builds (OpenBLAS, MKL, Accelerate)
- `scipy_dsyevr_64_` symbol resolution for PyPI numpy wheels (scipy-openblas64 uses
  `scipy_` prefix on all LAPACK symbols)
- Intel OpenMP (`libiomp5`) detection for `_lmm_accel` — avoids libgomp/libiomp5
  dual-runtime conflict on MKL systems
- `EIGEN_ACCEL_DEBUG=1` environment variable for LAPACK discovery diagnostics
- `IS_ILP64` constant exported from `_eigen_accel` module

### Changed

- LAPACK discovery tries symbols in priority order: `dsyevr_64_` → `scipy_dsyevr_64_`
  → `dsyevr64_` (ILP64), then `dsyevr_` (LP64)
- `_eigen_accel` ABI version bumped to 2 (dlopen rewrite)
- Linux wheels no longer need system LAPACK — dlopen resolves from numpy's bundled BLAS

### Fixed

- Linux CI: `_eigen_accel` DSYEVR resolution failed because PyPI numpy bundles
  scipy-openblas64 with `scipy_` prefixed symbols
- Module init: replaced `PyRun_String` with C API calls — `__builtins__` is unavailable
  in globals dict during module init, causing silent import failures
- Linux dlopen: uses `/proc/self/maps` scan after forcing numpy BLAS load to find
  libraries opened with `RTLD_LOCAL` (invisible to `RTLD_DEFAULT`)

## [2.9.2] - 2026-03-01

### Added

- DSYEVR C extension for eigendecomposition — O(N) workspace vs O(N²) for DSYEVD,
  saving ~250GB at 125k samples; auto-compiled on first use with lazy recompilation
- LAPACK linkage for Linux wheels: auto-detects numpy's bundled OpenBLAS in numpy.libs/
  for C extension compilation (both hatch_build.py and post-install_compile_eigen.py)
- Negative n_samples validation in memory estimation functions
- ABI mismatch test for DSYEVR import probe

### Changed

- Eigendecomposition now prefers DSYEVD (1.2–1.5x faster) by default, falling back
  to DSYEVR only when DSYEVD workspace exceeds available memory
- Memory estimates default to DSYEVD (conservative); actual peak is lower if DSYEVR
  is triggered
- DSYEVR auto-recompilation deferred from module import to first eigendecomp call
  (avoids subprocess/compiler side effects during import)
- DSYEVR workspace query uses ceil() to prevent off-by-one undersized allocation
- DSYEVR type stub accepts lowercase UPLO values ('l', 'u')
- Memory comment corrected: DSYEVR saves ~250GB (not ~232GB) at 125k samples

### Fixed

- DSYEVR fallback when neither driver fits: now uses DSYEVR (smaller peak) with
  OOM warning instead of silently falling through to DSYEVD
- Matrix reader: MemoryError re-raised directly instead of wrapping in RuntimeError
- Matrix reader: temp dir fallback includes OS error message in warning
- Class-level import in TestLambdaBoundaryDiagnostics moved to method level
  (prevents import error when results module unavailable)

## [2.9.1] - 2026-03-01

### Fixed

- Platform-tagged wheels: set `pure_python=False` and `infer_tag=True` in hatch build
  hook so cibuildwheel produces platform-specific wheels (e.g. `cp311-cp311-manylinux_2_28_x86_64`)
  instead of `py3-none-any`
- Upgrade cibuildwheel v2.22.0 to v3.3.1 (fixes stale manylinux2014 image reference)
- Switch `CIBW_BEFORE_BUILD_LINUX` from `yum` to `dnf` (manylinux_2_28 uses AlmaLinux 8)
- macOS wheels compile C extension single-threaded (no OpenMP) to avoid delocate
  `MACOSX_DEPLOYMENT_TARGET` conflict with Homebrew libomp
- Replace inline `python -c` test with standalone smoke test script to avoid shell
  escaping and indentation issues in cibuildwheel containers

## [2.9.0] - 2026-03-01

### Added

- **C extension for NumPy LMM runner** (`_lmm_accel.c`) — OpenMP-parallelized Wald test
  replaces Python loop over SNPs. Includes workspace API (`create_workspace` /
  `compute_wald_stats_workspace`) that pre-allocates all per-thread buffers once per chunk,
  and SoA-native Uab generation with invariant precompute to eliminate redundant work
- **Split-Uab C extension** — SoA (struct-of-arrays) layout for split Uab computation
  with internal Iab precompute, avoiding Python-side Iab construction entirely
- **Parallel matrix text I/O** (`matrix_reader.py`) — multi-worker `.eigenD.txt` /
  `.eigenU.txt` reader with chunk-boundary scanning and `np.loadtxt` per chunk
- **Eigen I/O `.npy` sidecar cache** — binary cache with mtime-based invalidation for
  eigenvalue/eigenvector files (3s warm read vs 4min cold text parse at 50k samples)
- **CI wheel build workflow** (`build-wheels.yml`) — cibuildwheel for manylinux x86_64
  and macOS arm64 wheels with OpenMP support
- Static OpenMP schedule for deterministic thread assignment across chunks

### Changed

- NumPy runner auto-detects C extension availability and dispatches to accelerated path
  when `n_cvt=1`, falling back to pure Python otherwise
- Memory estimator (`estimate_streaming_memory`) accounts for C extension workspace
  allocation when the accelerated path is active
- BLAS thread coordination: rotation threads and compute threads are balanced to prevent
  oversubscription during OpenMP regions
- Publish workflow includes wheel artifacts from build-wheels CI

## [2.8.3] - 2026-02-27

### Changed

- Local pytest defaults to `-n 3`, `--no-cov`, and skips `slow`/`tier2` tests to reduce memory pressure on dev machines
- CI overrides `addopts` with `-o 'addopts='` to run full suite with coverage independently of local config

## [2.8.2] - 2026-02-27

### Fixed

- **Critical: NaN propagation in golden section optimizer** — if the first grid point
  returned NaN (degenerate kinship), the scalar optimizer stayed stuck at NaN forever,
  silently producing NaN results for the entire GWAS run. Now initializes `best_val=inf`
  and skips NaN grid points.
- **Critical: `argmax` on NaN-containing grids** — JAX/NumPy `argmax` could select NaN
  entries as "best", causing the golden section to refine around a garbage bracket.
  NaN entries are now replaced with `-inf` before `argmax` in both batch paths.
- **Negative eigenvalues now zeroed in `eigendecompose_kinship`** — previously only warned.
  Negative eigenvalues above the threshold (e.g. -1e-5) survived into likelihood computation
  where `np.abs(v_temp)` silently masked incorrect logdet values.
- Missing `kinship is None` guard in JAX runner (NumPy runner already had it)
- Missing `lmm_mode` validation in `_compute_lmm_chunk` (JAX compute dispatch)
- `block_chunk_result` could `AttributeError` on `None` values for unexpected modes

### Changed

- Batch `_guard_P_yy` now logs a warning when negative P_yy values are detected
  (previously silent, unlike the scalar `_clamp_p_yy` path)
- Scalar Pab recursion now logs debug message for degenerate `ps_ww=0` entries
- Runners now emit per-key NaN count warnings after processing all chunks
- Removed unused `lambda_null` parameter from `calc_score_test`

## [2.8.1] - 2026-02-27

### Performance

- **NumPy grid REML/MLE vectorized**: replaced Python `for` loop over 50 grid lambdas
  with single `np.tensordot` call. Since all SNPs share the same lambda at each grid point,
  `Hi_eval` is `(n_grid, n_samples)` not `(n_snps, n_samples)`, eliminating the dominant
  memory allocation at scale. Benchmark (mouse_hs1940): Wald 18.3s → 6.4s (2.9x),
  All 34.4s → 11.6s (3.0x)
- `_fill_pab_recursion` uses `...` indexing to support both 3D and 4D Pab arrays,
  enabling the grid vectorization without duplicating the recursion logic

### Changed

- Extracted `_guard_P_yy` helper to deduplicate the P_yy clamping pattern (4 call sites)
- Extracted `_batch_grid_pab_numpy` to share tensordot + Pab computation between
  REML and MLE grid functions
- LOCO NumPy progress import hoisted to single location (was duplicated in pass 1/2)

### Fixed

- **LOCO NumPy runtime crash**: `progress_iterator` was imported from `jamma.utils`
  (which doesn't export it) instead of `jamma.core.progress` — caused `ImportError`
  when `show_progress=True` (the default)

## [2.8.0] - 2026-02-27

### Added

- **NumPy LOCO kinship streaming**: `_compute_loco_kinship_streaming_numpy()` — pure NumPy
  LOCO kinship computation (no JAX dependency), enabling `--loco --backend numpy` workflows
- **`LazySnpMeta` in `schema.py`**: Single canonical source for lazy PLINK metadata wrapper
  (was duplicated in `loco.py` and `runner_streaming.py`)
- **Shared LOCO helpers**: `_collect_chr_snp_stats()` and `_filter_chr_snps()` extract
  duplicated pass-1 SNP statistics and filtering logic from JAX/NumPy chromosome runners
- Backend validation in `run_lmm_loco`: raises `ValueError` for invalid backend values
- Write-offset validation in NumPy LOCO path: raises `RuntimeError` if pre-allocated
  result arrays are not fully written
- Diagnostic error handling around NumPy LOCO computation loop (logs chromosome, chunk
  offset, and SNP count on failure)
- **GEMMA covariate validation tests**: 4 mouse_hs1940 covariate tests for NumPy backend
  (Wald, LRT, Score, All modes) validating beta, SE, p-values against GEMMA reference
- **Synthetic no-covariate GEMMA validation tests**: LRT, Score, and All mode tests
  completing the NumPy backend validation matrix

### Changed

- `_P_YY_MIN` constant (1e-8) propagated from `likelihood.py` to `likelihood_numpy.py`
  and `stats.py` (was hardcoded in 7 locations)
- `runner_streaming.py` imports `LazySnpMeta` from `schema.py` instead of defining its own copy

### Fixed

- LOCO backend dispatch: `backend="numpy"` now uses NumPy kinship streaming instead of
  unconditionally importing JAX kinship module
- `pipeline.py` XLA profiling catch restored `AttributeError` (JAX can raise this on
  some platforms when profiling is unavailable)
- `backend.py` logger text: "to suppress this warning" → "to suppress this error" (matches
  actual log level)
- `generate_loco_fixtures.sh`: corrected GEMMA version reference (0.96 → 0.98.5)
- `test_loco.py`: fixed `ModuleNotFoundError` from invalid conftest import

## [2.7.1] - 2026-02-27

### Added

- **GEMMA LOCO integration test**: 3-chromosome validation (beta, SE, p_wald, l_remle,
  logl_H1, rank correlation, top hits) against GEMMA LMM with JAMMA-computed LOCO kinship
- Fixture generation scripts: `generate_loco_synthetic.py` (PLINK data),
  `generate_loco_fixtures.sh` (Docker-based GEMMA reference outputs)
- `logl_H1` per-chromosome comparison test (LOCO-04b)
- Merge completeness assertion in LOCO test fixture (detects inner-join data loss)

### Changed

- `load_phenotypes_from_fam` extracted to `conftest.py` for reuse; simplified to
  `np.loadtxt(usecols=5)`
- CI: dropped Intel Mac job (macos-13 deprecated) and Windows job (pytest-xdist
  deadlock); added `--cov-fail-under` per matrix entry
- Causal SNP check in `generate_loco_fixtures.sh` is now a hard failure (was warning)
- Tolerance rationale comments added to LOCO integration tests

### Fixed

- CI: added per-matrix `--cov-fail-under` thresholds (80% JAX, 50% NumPy-only)
- `pytest.importorskip('jax')` added before JAX-only imports in all
  JAX-dependent test files (fixes NumPy-only CI)

## [2.7.0] - 2026-02-26

### Added

- **Pure-NumPy backend**: Full LMM association (Wald, LRT, Score, All modes) without
  JAX dependency — `jamma` now works out-of-the-box on any platform with just numpy
- **`--backend` CLI flag**: Explicit backend selection (`auto`, `jax`, `numpy`); `auto`
  prefers JAX when available, falls back to NumPy
- **`backend` parameter on `gwas()` API and `PipelineConfig`**: Programmatic backend control
- **`special.py` module**: Pure-stdlib `betainc()` (Lentz continued-fraction) and `chi2_sf()`
  implementations — eliminates scipy dependency for p-value computation
- **`prepare_common.py`**: Shared null-model setup (eigendecomposition, rotation, REML)
  extracted from JAX-specific code for reuse by both backends
- **`likelihood_numpy.py`**: Batch Uab/Pab/REML/MLE computation and Wald/LRT/Score
  statistics using pure NumPy — vectorized across grid/refinement steps
- **`compute_numpy.py`**: Mode-dispatch layer routing to NumPy likelihood functions
- **`runner_numpy.py`**: Streaming chunk-loop LMM runner using NumPy backend with
  identical output format to JAX runner
- **`detect_backend()` and `log_backend_selection()`**: Backend probing and diagnostic logging
- **Platform-smart JAX defaults**: `pip install jamma[jax]` auto-includes JAX on Linux
  and ARM Mac via PEP 508 markers; Windows/Intel Mac get NumPy-only by default
- **`requires_jax` pytest marker**: JAX-dependent tests auto-skip when JAX unavailable
- **Cross-backend CI matrix**: Tests run on Linux+JAX, Linux+NumPy, macOS+JAX,
  Windows+NumPy, and Linux+JAX(3.11) configurations
- **406 new tests** in `test_special.py` for `betainc`/`chi2_sf` edge cases
- **Typed backend literals**: `BackendRequest` and `BackendResolved` types for pipeline config

### Changed

- JAX moved from required to optional dependency (`jamma[jax]` extra)
- All `__init__.py` modules guard JAX imports behind `has_jax()` — `import jamma`
  succeeds without JAX installed
- `PipelineConfig.backend` uses `BackendRequest` literal type; `PipelineResult.backend`
  uses `BackendResolved` literal type
- `conftest.py` registers `requires_jax` marker and auto-applies to JAX-importing tests
- Dockerfile updated for layered `jamma[jax]` install
- `_compute_lmm_chunk` defaults aligned: `n_grid=50`, `n_refine=10` (was inconsistent
  between JAX and NumPy compute modules)
- `snp_filter.py` `np.errstate` scope narrowed to `invalid`/`divide` only (was `all`)

### Fixed

- **`has_jax()` swallowed `RuntimeError`/`OSError`**: JAX installation failures (broken
  CUDA, missing libraries) now log a warning instead of silently returning `False`
- **`runner_jax.py` crashed on `kinship=None`**: Type signature and guard updated to
  accept `None` when pre-computed eigendecomposition is provided
- **Missing eigenpair validation in `runner_jax.py`**: Added dimension checks matching
  `runner_numpy.py` — catches shape mismatches before LAPACK calls
- **`prepare.py` dropped `TypeError`**: `_setup_cpu_sharding` exception tuple restored
  to include `TypeError` alongside `RuntimeError`/`ValueError`
- **Silent invalid `lmm_mode` in `_compute_null_model_common`**: Mode 1 now returns
  `None` explicitly; invalid modes raise `ValueError` (was silently returning `None`)
- **`betainc` ArithmeticError catch unlogged**: CF non-convergence now logged at debug level
- **All-SNPs-filtered produced silent empty return in `runner_jax.py`**: Now logs warning
- **Memory estimate ran unconditionally in `runner_jax.py`**: Now gated behind `check_memory` flag
- **P_yy zero in Score test denominator**: Clamped to 1e-8 floor to prevent Inf F-statistic
- **`runner_numpy.py` missing early validation**: Raises `ValueError` when neither kinship
  nor eigendecomposition is provided
- **`_RESULT_FIELDS` import path**: `runner_jax.py` now imports from `schema.py` (was
  importing from deleted `results.py` path)

## [2.6.1] - 2026-02-26

### Fixed

- `test_lmm_jax_chunk_invariance` passed consumed kinship to second
  `run_lmm_association_jax` call (in-place eigendecomp overwrites K with
  eigenvectors; added `.copy()`)

## [2.6.0] - 2026-02-26

### Added

- **Runtime buffer mismatch detection**: If `eigh_lo` ignores the `out=` parameter
  (future numpy change), `INPLACE_EIGEN_AVAILABLE` flag is set False at runtime
  and memory estimates automatically correct to include separate eigenvector allocation
- Tests for buffer mismatch flag update, fallback memory estimates, guard clauses,
  safety margin cap, ImportError logging in `_inplace_eigen_available()`
- Tests for LOCO kinship bugs: aliasing, chromosome ordering, fallback normalization,
  n_filtered=0 guard, GeneratorExit partial retention, flush failure propagation,
  `_dsyevd_workspace_gb` formula (LIWORK uses 8-byte integers)

### Changed

- `chr_sort_key` extracted from `loco.py` to `utils/__init__.py` (DRY — used by
  both loco.py and kinship/compute.py); unknown chromosome sentinel raised from
  100 to 1000 (supports species with >99 numeric chromosomes)
- Memory safety margin capped at 10GB absolute (was unbounded 10%, which
  demanded 50GB+ headroom at scale)
- Memory estimates adapt to in-place vs fallback eigendecomp path at runtime
- `IncrementalAssocWriter.__exit__` cleans up partial output on any `Exception`
  subclass (was OSError-only); retains partial output on `KeyboardInterrupt`,
  `SystemExit`, and `MemoryError` (partial results are valid up to point of failure)
- Docstrings clarified: K "may be overwritten" / "treat as consumed" (was
  unconditional "OVERWRITTEN" which was inaccurate for fallback path)
- `_inplace_eigen_available()` ImportError logged at warning (was info) — indicates
  broken installation
- In-place eigendecomp fallback logged at warning (was info) — 320GB impact at scale
- Unknown chromosome names logged at info (was debug) — aids debugging LOCO issues
- `pipeline.py` XLA profiling catch narrowed to `(OSError, ImportError, AttributeError)`
  (was bare `except Exception`)
- `S_full_np` marked read-only after in-place division in `_yield_loco_matrices`
  to guard against accidental re-mutation
- LOCO `write_kinship_matrix` error includes chromosome name and path for diagnostics

### Fixed

- **`IncrementalAssocWriter.__exit__` flush failure silently deleted output** — now
  raises after cleanup so callers know the write failed (was `logger.warning` + return)
- `_format_duration` produced "2h 60m" for durations near hour boundaries and
  "60 min" at exactly 3599s due to `:.0f` rounding (now uses integer truncation
  throughout)
- README Low-level API example passed consumed kinship matrix to streaming runner
  (now correctly passes eigenvalues/eigenvectors); added missing `import numpy as np`
- `test_runner_jax.py` passed mutated K as kinship to runner (added `.copy()`)
- `_yield_full_kinship_fallback` held persistent `K_full` alongside `S_full_np`
  while consumer processed yielded matrix (3 n×n matrices live). Now divides
  `S_full_np` in-place once and yields `.copy()` per chromosome (2 matrices
  live at yield: modified `S_full_np` + the copy), matching the LOCO memory
  gate budget
- Stale field comments on `MemoryBreakdown.sufficient` and
  `StreamingMemoryBreakdown.sufficient` (referenced old `total * 1.1` formula)
- Inaccurate comments: plink.py "two boolean ops" (actually 3), eigen.py
  "re-imports each call" (reads module attribute), CHANGELOG fallback description

## [2.5.8] - 2026-02-25

### Changed

- In-place kinship accumulation (`K += np.matmul(...)`) eliminates one n×n temporary per batch
- Size-gated eigendecomp symmetry check: full `np.allclose` for n<10k, vectorized sampled check for n≥10k (avoids 80GB temporary at 100k samples)
- Single-pass eigenvalue post-processing with in-place thresholding (no `np.where` allocation)
- LOCO valid_mask guard: skips n×n subsetting copy when all samples are valid
- LOCO SNP-list restriction uses precomputed boolean mask instead of per-chromosome `np.isin`

## [2.5.7] - 2026-02-23

### Added

- Unit tests for likelihood_jax.py edge cases: negative P_yy, degenerate SNPs, near-zero eigenvalues, lambda bounds, JAX/NumPy consistency, covariate rank validation, kinship symmetry checks
- CI coverage threshold (`--cov-fail-under=80`) enforced on the full test suite

### Fixed

- JAX REML and MLE paths now guard negative P_yy → NaN (previously only the NumPy path had this guard)
- CLI rejects `--mem-budget <= 0` with a clear error instead of silently proceeding
- Covariate rank validation: rank-deficient covariate matrices now raise `ValueError` before LMM runs
- Kinship eigendecomposition warns when input matrix is asymmetric
- LOCO warns and uses full kinship for chromosomes with 0 ksnps (was silently skipping them)
- Out-of-place kinship accumulation (`K = K + matmul(...)`) for deterministic FP rounding
- Explicit `del chunk` in streaming stats loops to free memory between iterations
- Test tolerances aligned with GEMMA_EQUIVALENCE.md (kinship 1e-10 → 1e-8)
- CI: `test-slow` job skips coverage threshold (partial test runs can't meet 80%)

### Changed

- Dockerfile: consolidated RUN layers, added non-root user (`jamma`, uid 1000)
- CI: upgraded `astral-sh/setup-uv` v4 → v5
- Pinned `ruff>=0.15.0` in dev dependencies to match CI
- Kinship non-LOCO path converted from JAX to numpy (JAX not initialized during kinship phase)
- Extracted `DevicePlacement` and shared chunk preparation into `prepare.py`
- Deferred JAX backend initialization until LMM phase
- Wall clock time estimates for kinship, eigendecomp, and LMM phases

## [2.5.6] - 2026-02-22

### Fixed

- LMM rotation (`U.T @ G`) now uses all physical cores instead of `physical_cores // n_jax_devices` — same bug class as eigendecomp (v2.5.4), but in the per-chunk dgemm. On a 48-core machine with 24 JAX devices, rotation ran with 2 threads instead of 48 (~16x slowdown per chunk, ~4 hours instead of ~30 minutes for 125k samples)
- Applied fix to all three runners: `runner_jax.py`, `runner_streaming.py`, `loco.py`
- Extracted `get_physical_core_count()` helper in `core/threading.py` to consolidate physical core detection (replaces inline `psutil.cpu_count(logical=False)` in eigen.py)

## [2.5.5] - 2026-02-22

### Added

- Regression tests for worker cap (verifies 32-worker limit on high-core machines)
- Regression tests for eigendecomp threading (verifies all physical cores used, not divided by JAX devices)
- Chunk sizing tests at Databricks scale (125k samples, parametrized across 1-48 devices)
- Fast synthetic LOCO partition tests (no fixture dependency)
- Eigendecomp memory gate integration tests (verifies MemoryError before LAPACK runs)

## [2.5.4] - 2026-02-22

### Fixed

- Eigendecomposition now uses all physical cores instead of `physical_cores // n_jax_devices` — JAX isn't running during `eigh`, so the thread reduction was a ~16x slowdown on multi-device configs

## [2.5.3] - 2026-02-22

### Fixed

- Cap matrix writer workers at 32 (was unbounded cpu_count) — 96 workers on Databricks added process overhead with no I/O benefit
- Eliminate per-row `tuple()` allocation in worker formatting — was creating 125k Python float objects (~3 MB) per row per worker, causing GC thrashing
- Correct peak disk estimate to account for all chunks existing simultaneously during worker phase

## [2.5.2] - 2026-02-22

### Fixed

- Matrix writer no longer fills `/tmp` when writing large kinship matrices — temp files (memmap + chunks) are now created on the same filesystem as the output file
- Chunks are deleted eagerly during concatenation, reducing peak disk from 2x output size to ~1x
- Memmap is freed before concatenation starts, reclaiming matrix-sized temp space earlier
- Pre-flight disk space warning when free space looks insufficient for the write

## [2.5.1] - 2026-02-22

### Added

- PyPI keywords and classifiers for search discoverability
- Project URLs (Homepage, Repository, Documentation, Changelog, Issues) for PyPI verified details

## [2.5.0] - 2026-02-21

### Added

- **CPU device sharding**: JAX automatically partitions SNP batches across
  virtual CPU devices using `NamedSharding`. Auto-configures as
  `max(1, physical_cores // 2)` — no user action required. Override with
  `JAMMA_JAX_DEVICES` environment variable for custom tuning.
- **BLAS thread coordination**: BLAS thread count auto-reduces when multiple
  JAX devices are active to avoid oversubscription. Override with
  `JAMMA_BLAS_THREADS` environment variable.
- **`--profile-dir` CLI flag**: Capture XLA profiling traces for
  TensorBoard/Perfetto analysis. Degrades gracefully — profiling failures
  never prevent GWAS results.
- **Per-stage timing**: LMM runners now log timing breakdowns for
  eigendecomposition, DGEMM rotation, JAX compute, and result writing.
- **JAX profiler annotations**: `TraceAnnotation` labels on all pipeline
  stages for use with `--profile-dir`.
- **Benchmark harness**: `pytest-benchmark` pedantic-mode benchmarks for
  eigendecomp, DGEMM rotation, JAX optimization, and full pipeline on
  mouse_hs1940. Includes hardware context (CPU model, BLAS backend, device
  count) for cross-machine comparison.
- **Hardware context module**: `jamma.core.hardware.get_hardware_context()`
  collects CPU, BLAS, JAX, and platform info for benchmark reproducibility.
  `assert_x64_precision()` guard prevents silent float32 fallback in
  benchmark entry points.

### Fixed

- **Sharding divisibility fallback**: SNP counts not evenly divisible by the
  device count (e.g. 50,000 SNPs with 32 devices) no longer silently disable
  sharding. UtG arrays are zero-padded to the next device-count multiple and
  padded results are discarded.
- **Chunk device alignment**: `_compute_chunk_size` and `auto_tune_chunk_size`
  round chunk sizes to device-count multiples, preventing XLA from padding
  partial shards internally.

## [2.4.5] - 2026-02-20

### Fixed

- **Assert→RuntimeError**: Replaced `assert` with `if`/`raise` for write_offset
  check in JAX runner — `assert` is stripped under `python -O`, risking silent
  data truncation.
- **File handle leak**: Fixed `IncrementalAssocWriter.__exit__` skipping file
  close on non-OSError exceptions.
- **Off-by-one in retry count**: Error message now reports correct attempt number.
- **Exception propagation**: `verify_jax_installation()` re-raises original
  exceptions instead of wrapping in `RuntimeError`.

### Added

- **Eigenvector shape validation**: Raises `ValueError` when pre-computed
  eigenvectors don't match sample count after filtering.
- **JAX int32 overflow detection**: Streaming and LOCO runners catch and log
  diagnostic context for JAX buffer overflow errors.
- **Parameter validation**: `_compute_lmm_chunk()` validates `logl_H0` and
  `Hi_eval_null` are provided for modes that require them.
- **23 new tests**: Covers `_safe_sqrt` boundary behavior, `_clamp_p_yy`
  clamping, P_yy in log-likelihood, over-parameterization guard, and golden
  section optimizer.

### Changed

- **DRY P_yy clamping**: Extracted `_clamp_p_yy()` helper replacing 4 duplicate
  clamping blocks in likelihood.py.
- **Precomputed sample filter**: `needs_sample_filter` flag computed once before
  hot loops instead of `np.all(valid_mask)` per iteration.
- **Narrowed eigendecomp exception**: Catches `np.linalg.LinAlgError` before
  generic `Exception` with PSD-specific guidance.
- **Debug tracebacks**: CLI exception handlers log `exc_info=True` for verbose
  diagnostics.

## [2.4.4] - 2026-02-19

### Added

- **GEMMA-style startup banner**: Consolidated dataset summary logged at startup for
  both LMM and kinship modes — version, release date, total/analyzed individuals,
  covariates, phenotypes, and total SNPs.
- **Auto-derived release date**: Hatchling build hook (`hatch_build.py`) embeds the git
  commit date into the package at build time. No manual maintenance required — the date
  appears in the banner and `--version` output automatically.

## [2.4.3] - 2026-02-19

### Changed

- **File-to-file parallel writer**: Workers write formatted text to per-chunk temp
  files instead of returning ~1.2 GB bytes objects through the multiprocessing IPC
  pipe. Eliminates memory spike at end of write phase — at 100k×100k with 16 workers,
  old code buffered ~19 GB in the IPC queue; new code: ~0 bytes in IPC.
- **Removed 16-worker cap**: `write_matrix_parallel()` defaults to `cpu_count`
  instead of `min(cpu_count, 16)`. Per-worker memory is now ~150 MB process overhead
  (not 1.2 GB buffered bytes), so higher worker counts are safe.

## [2.4.2] - 2026-02-18

### Fixed

- **Loguru traceback in pool errors**: Use `logger.opt(exception=e)` instead of
  `exc_info=True` (stdlib pattern ignored by Loguru) for full traceback on worker failures.
- **Improved error handling**: Use `RuntimeError` for worker exception wrapping
  (fixes fragile `type(e)(...)` pattern), log warning on temp file cleanup failure,
  add `TMPDIR` hint for disk-full errors.
- **Pre-commit hooks**: Fix hook chain so ruff lint/format runs in CI and locally
  (was bypassed by beads `core.hooksPath` override).

### Changed

- **Code simplification**: List comprehensions in parallel matrix writer, extracted
  shared test fixture for temp dir isolation.

## [2.4.1] - 2026-02-18

### Fixed

- **Docker SIGBUS on large matrices**: Replaced `SharedMemory` (POSIX `shm_open()`)
  with file-backed `numpy.memmap` in `write_matrix_parallel()`. Docker defaults
  `/dev/shm` to 64 MB — a 100k×100k float64 matrix is ~75 GB, causing SIGBUS on
  access. The memmap approach uses filesystem-backed temp files instead, bypassing
  `/dev/shm` entirely. ([cpython#114390](https://github.com/python/cpython/issues/114390))

## [2.4.0] - 2026-02-18

### Added

- **Parallel matrix writer**: `write_matrix_parallel()` using `multiprocessing.Pool.imap`
  with SharedMemory to format matrix rows across CPU cores. Falls back to `np.savetxt`
  for small matrices (<500 rows). Byte-identical output to `np.savetxt` for all sizes.
  Reduces 100k×100k matrix write from ~30min to ~2-4min.
- **Unified output schema**: `schema.py` with `StatColumn`/`ModeSpec` frozen dataclasses
  as single source of truth for LMM output column definitions. `MODE_SPECS` frozen via
  `MappingProxyType`. Replaces 4 separate dispatch tables across 3 modules.
- **Fast PLINK line counting**: `_count_lines_fast()` uses binary `bytes.count(b'\n')`
  in 1MB chunks instead of text-mode `sum(1 for _ in f)`. Correctly handles files
  without trailing newline.
- **`write_arrays_batch` hot path**: Formats and writes results directly from numpy
  arrays, bypassing `AssocResult` construction. Validates stat array lengths and
  snp_info keys upfront.

### Changed

- Kinship and eigenvector writers now delegate to `write_matrix_parallel()`
- `IncrementalAssocWriter` retry logic consolidated into shared `_write_buf()` method
- `ACCUM_KEYS`, `RESULT_FIELDS`, `FORMAT_COLUMNS`, `HEADERS`, `TEST_TYPE_MAP` derived
  mechanically from `MODE_SPECS` (eliminates manual sync)
- File I/O functions in `kinship/io.py` and `lmm/eigen_io.py` log resolved paths

### Fixed

- Partial file cleanup on worker failure in parallel matrix writer
- `n_workers` validation (reject < 1) in `write_matrix_parallel`
- `StatColumn.fmt` included in string type validation
- `ModeSpec` validates duplicate header names across columns
- Worker error context includes chunk row range for debugging
- Pool errors logged before `pool.terminate()` for diagnostics

## [2.3.0] - 2026-02-18

### Fixed

- **Lambda bounds not plumbed to null MLE**: `l_min`/`l_max` now passed through
  `_compute_null_model` to `compute_null_model_mle` so null-model optimization
  respects user-configured lambda bounds
- **Memory check used raw PLINK dimensions**: Pipeline memory estimation now uses
  post-filter sample count (`n_valid`) and actual covariate count instead of raw
  `.fam`/`.bim` metadata dimensions
- **LOCO accumulated full-chromosome results**: Results now flushed per disk chunk
  instead of accumulating all JAX arrays for the entire chromosome before conversion
- **`chunk_size <= 0` in `stream_genotype_chunks`**: Guard against `chunk_size=0`
  (ZeroDivisionError) and negative values (infinite range)
- **Batch runner missing `jax.clear_caches()`**: Added after chunk loop for parity
  with streaming runner — prevents JIT trace accumulation across LOCO runs
- **Batch runner missing lambda boundary tracking**: Diagnostic warning for SNPs
  converging at lambda bounds (was only in streaming runner)
- **Output prefix path traversal**: `OutputConfig` and `PipelineConfig` now reject
  `output_prefix` containing path separators
- **Biological chromosome ordering in LOCO**: Chromosomes now sort 1..22, X, Y,
  XY, MT instead of lexicographic order (1, 10, 11, ..., 2, 20, ...)
- **CLI timing key access**: Bare `result.timing["key"]` replaced with `.get()`
  to prevent KeyError when timing keys are missing
- **CLI `n_covariates` display**: Pipeline now populates `n_covariates` in timing
  dict (was always showing default value)
- **LRT validation `all_passed`**: Beta/SE NaN validation now included in LRT
  comparison (was silently skipped)
- **Duplicate BIM SNP IDs**: `resolve_snp_list_to_indices()` now warns about
  duplicate SNP IDs and keeps first occurrence (was silently using last)
- **`donate_argnums` deprecation**: Removed deprecated JAX `donate_argnums` from
  golden section optimizers
- **Empty samples guard**: `run_lmm_association_jax()` now raises `ValueError`
  when all phenotypes are NaN/-9 (no valid samples remain)
- **Streaming runner empty samples guard**: Streaming runner raises `ValueError`
  on zero valid samples after filtering
- **`__main__.py` missing `__name__` guard**: Prevented double execution on import
- **`ensure_jax_configured` silent on conflicts**: Now raises `RuntimeError` on
  conflicting non-default args after JAX is locked (was silent warning)
- **Negative P_yy only logged at debug**: Elevated to `warning` with lambda context
  in 4 locations in `likelihood.py`
- **GPU fallback only logged at debug**: Elevated to `warning` in `prepare.py`
- **Empty results misclassification**: `compare.py` guarded `all()` on empty lists
- **Double eigendecomp in `test_hypothesis.py`**: Reduced to single `eigh` call

### Changed

- `_MAX_BUFFER_ELEMENTS` derived from `INT32_MAX` constant instead of magic number
- `_LazySnpMeta.__getitem__` supports slice indexing for list-like behavior
- Removed unused backward-compat re-exports from `runner_jax.py`
- Migrated `snp_filter.py` logging from `print()` to `loguru`
- SNP statistics in streaming runners use numpy arrays instead of `locals()` dict
- Replaced `np.random.seed()` with `np.random.default_rng()` in tests
- Dead code removed: unused `n_snps` param, unreachable shape check, unused
  `lambda_val` param, 8 duplicated `setup_jax` test fixtures
- 4 `format_assoc_line_*` → 1 table-driven function (`io.py`, -161 lines)
- 4 `_build_results_*` → 1 with `_RESULT_FIELDS` dispatch (`results.py`)
- `runner_jax.py` mode-to-arrays refactored to use `_RESULT_FIELDS` (DRY)
- Header selection unified to table-driven `_HEADERS` dict in `io.py`
- Input validation on dispatch keys in `io.py` and `results.py`
- Fixture paths use `Path(__file__).parent` instead of cwd-relative (6 files)
- CLI subprocess tests decoupled from `uv` runtime
- `ToleranceConfig` gains `p_lrt_rtol` field
- Pinned ruff to 0.15.x across local dev deps and CI pre-commit

### Added

- Streaming-vs-batch parity tests for degenerate SNP and empty-samples edge cases
- Hypothesis property tests for variance computation and SNP filtering
- `Raises` docstring for `run_lmm_association_jax()` ValueError
- `__main__.py` for `python -m jamma` execution
- LOCO integration tests: multi-pass batching, NaN covariates, MAF filtering
- Streaming LRT/Score mode tests, writer retry/rollback tests
- 7 new Hypothesis property tests for Score test and LRT invariants
- Tier markers on all 723 tests (392 tier0, 309 tier1, 22 tier2)
- 22 unit tests in `test_review_fixes.py` for dispatch validation and erfc

## [2.2.0] - 2026-02-17

### Added

- **Lambda bounds** (`-lmin`/`-lmax`): Configurable optimization bounds for lambda
  with boundary convergence warnings when SNPs cluster at bounds
- **Individual weights** (`-widv`): GEMMA-exact kinship pre-transformation
  K[i,j] /= sqrt(w_i * w_j) via memory-efficient two-pass scaling (O(n) memory)
- **Categorical covariates** (`-cat`): One-hot encode specified covariate columns
  with reference level dropped. JAMMA-specific feature (not GEMMA's -cat)
- `-wsnp` flag accepted (hidden, not yet implemented — clear error message)
- Eigen I/O validation: empty file checks, parse error wrapping with file paths,
  `atleast_1d`/`atleast_2d` for single-line files, square matrix validation

### Changed

- `IncrementalAssocWriter`: retries transient write failures with backoff,
  truncates partial writes before retry, cleans up partial files on final failure
- Replaced `click.echo` with `loguru` in I/O module (removes click dependency from io)
- Eigen file writers use `np.savetxt` instead of Python f-string loops
- Slow gwas_api integration tests marked `@pytest.mark.slow`, skipped by default

### Fixed

- Categorical single-level columns with NaN now keep a NaN marker column
  (previously deleted entirely, losing missingness signal for pipeline filtering)
- Weight file reader rejects multi-column files instead of silently flattening
  via `.ravel()` (prevented weight misalignment)
- Weight file reader rejects NaN values (bypassed all scaling logic due to
  NaN comparison semantics)
- `__exit__` cleanup now properly nulls `_file` on successful close
- Writer retry truncates partial writes to prevent duplicate lines on retry

## [2.1.0] - 2026-02-16

### Added

- **Multi-pass LOCO S_chr batching**: When all per-chromosome S_chr matrices don't
  fit in memory (e.g. 100k samples x 22 chromosomes), chromosomes are automatically
  batched across multiple disk passes — S_full computed once and reused
- **LOCO writer passthrough**: `_run_lmm_for_chromosome` streams results directly
  to disk via optional `writer` parameter, eliminating per-chromosome result
  accumulation in memory
- **In-memory mode warnings**: Log warning when running without `output_path` with
  >100k SNPs, recommending disk streaming
- Memory estimates now logged even when `check_memory=False`
- **CONTRIBUTING.md**: Development setup, testing, code style, and PR guidelines

### Changed

- **CLI: Typer → Click**: Flat GEMMA-compatible CLI — `jamma -gk 1 -bfile data` instead
  of `jamma gk -bfile data`. True drop-in replacement for GEMMA command lines
- **Dockerfile**: Uses uv for package management; documents `--platform linux/amd64`
  requirement for MKL (x86_64-only)
- All documentation updated to flat CLI syntax
- LOCO kinship: extracted `_stream_s_full_and_chr` and `_yield_loco_matrices`
  helpers to eliminate code duplication across single-pass and multi-pass paths
- Deduplicated `_yield_chunk_results` call in `_run_lmm_for_chromosome` — iterator
  created once, only consumption differs (writer vs list)
- Memory safety margin reduced from 50% to 10% for streaming kinship
- **Removed Databricks notebooks and Dockerfile**: Moved to separate `jamma-databricks`
  project — JAMMA repo now contains only the library and a general-purpose Dockerfile
- Minimum numpy bumped to 2.0+ (1.26 is EOL)

### Fixed

- JAX device array leak on write exception — `eigenvalues_jax`, `UtW_jax`,
  `Uty_jax` now freed via `try/finally` in `_run_lmm_for_chromosome`
- Multi-pass memory accounting underestimated first-pass peak by one `matrix_gb`
  (JAX and numpy S_full coexist briefly during conversion)
- Exception-safe writer lifecycle using `ExitStack` for LOCO writer
- Eigen file validation against covariate-filtered sample count
- Empty output and dead `logls_mle` accumulation removed from LMM runner
- `check_memory` flag now respected in `eigendecompose_kinship`

### Performance

- Two-pass chunked column iteration in LOCO replaces single full-matrix read
- Lazy SNP metadata loading and early cleanup of pass-1 statistics arrays
  (`all_vars`, `all_means`, `all_miss_counts`) immediately after deriving filters
- Free kinship matrix after `write_eigen` instead of holding until end
- Remove unnecessary `U.T` contiguous transpose copy
- Hoist `snps_indices` set conversion out of LOCO chromosome loop
- Skip `impute_and_center` in multi-pass when no target chromosomes in chunk

## [2.0.0] - 2026-02-12

### Added

- **LOCO kinship** (`-loco` flag): Leave-one-chromosome-out kinship via streaming
  subtraction approach — computes per-chromosome K_loco one at a time for memory
  efficiency. Eliminates proximal contamination in LMM association
- **Eigendecomposition reuse** (`-d`/`-u`/`-eigen` flags): Save and load pre-computed
  eigendecomposition for multi-phenotype workflows — skip O(n³) eigendecomp after first run
- **Phenotype selection** (`-n` flag): Select phenotype column from multi-phenotype
  .fam files (1-based indexing, matching GEMMA)
- **Standardized kinship** (`-gk 2`): GEMMA-compatible standardized relatedness matrix
  using (X - mean) / sqrt(p*(1-p)) normalization
- **SNP subset selection** (`-snps`/`-ksnps` flags): Restrict association testing and/or
  kinship computation to SNP lists (one RS ID per line)
- **HWE QC filtering** (`-hwe` flag): Hardy-Weinberg equilibrium chi-squared
  goodness-of-fit test — exclude SNPs below p-value threshold. Genotype counts
  piggyback on pass-1 streaming (no extra disk pass)
- **PLINK dimension validation**: Cross-validate .bed file size against .fam/.bim
  line counts before processing
- **Genotype value validation**: Warn on values outside expected range {0, 1, 2, NaN}
- **`apply_snp_list_mask()` helper**: DRY bounds-validated SNP mask application
  (replaces 3 duplicate code blocks in kinship and LMM runners)
- **SNP filter regression tests**: Verify searchsorted-based chunk filtering matches
  naive linear scan across edge cases (boundary SNPs, full/empty chunks, single-element)
- **Missingness test suite**: Heterogeneous missingness patterns, column-specific
  imputation accuracy, edge cases (all-missing, no-missing, single-sample)
- **Hypothesis property tests for v2.0 features**: 14 new tests covering HWE chi-squared
  (p-value bounds, allele swap symmetry, perfect equilibrium, degenerate inputs,
  vectorized/scalar equivalence), standardized kinship (symmetry, PSD, trace approximation,
  shape consistency), and eigen I/O round-trip (.10g format reconstruction, orthonormality,
  eigenvalue precision). Total: 42 hypothesis tests (up from 29)

### Changed

- **Streaming SNP filtering**: Replaced O(n) linear scan with `np.searchsorted` for
  chunk-level SNP range filtering — eliminates per-SNP Python overhead in streaming runners
- **Memory module comments**: Updated docstrings to reflect streaming architecture
  and actual component breakdown
- **HWE accumulators**: Upgraded int32 → int64 for overflow safety on large cohorts
- **HWE NaN handling**: Replaced `np.nan_to_num` with explicit `np.where` to avoid
  silent inf/neginf clobbering

### Fixed

- **HWE silently ignored in LOCO mode**: `-hwe` parameter was accepted but had no
  effect when `-loco` was active — now rejected with clear error message
- **CLI gk ksnps errors uncaught**: Missing/invalid ksnps file produced a traceback
  instead of user-friendly error — now wrapped in try/except
- **HWE threshold >1.0 accepted**: Out-of-range p-value threshold now validated

### Removed

- **Bioconda recipe**: Removed `bioconda/meta.yaml` and automated bioconda PR submission —
  bioconda's conda-forge numpy is LP64 only, which silently breaks for JAMMA's target
  users (>46k samples require ILP64 MKL). pip is the canonical install path.

## [1.5.1] - 2026-02-10

### Changed

- README logo and badge layout refinements

## [1.5.0] - 2026-02-10

### Added

- **PipelineRunner service**: Shared orchestration class eliminates duplicated pipeline
  logic between CLI and Python API — single source of truth for validate, parse, check
  memory, load kinship, load covariates, run LMM
- **Bioconda recipe**: `bioconda/meta.yaml` and automated PR submission to
  bioconda-recipes on each release
- **Memory/chunk coupling**: Memory estimation now uses computed chunk size from
  `_compute_chunk_size()` instead of hardcoded 10,000 — estimates match actual runtime
- **README badges**: Bioconda, JAX, NumPy, Hypothesis
- **Project logo** in README hero section

### Changed

- CLI `lmm` command delegates to `PipelineRunner` (256 → 78 lines)
- `gwas()` API delegates to `PipelineRunner` (164 → 28 lines)
- Removed import-time side effects — `configure_jax()` is now lazy via
  `ensure_jax_configured()` sentinel pattern
- CI restructured into 3 jobs: `lint`, `test-fast` (unmarked tests), `test-slow`
  (tier2/slow, master-only)
- Ruff pre-commit hook updated v0.8.6 → v0.15.0
- Publish workflow updated for live PyPI with automated bioconda PR submission

### Fixed

- Memory estimates used hardcoded chunk size (10,000) instead of the actual computed
  chunk size — could over/underestimate by 2-5x at different scales

## [1.4.3] - 2026-02-10

### Added

- **Production-scale GEMMA validation**: 85,000 real samples × 91,613 SNPs — 100%
  significance agreement, 100% effect direction agreement, Spearman rho 1.000000
- **Compare-only mode** for GEMMA comparison notebook — load pre-computed results
  from configurable source paths, skip all compute
- **OOM-safe kinship comparison**: Sampled Spearman (10M elements) + chunked row-by-row
  statistics for 85k+ matrices without materializing `np.triu_indices` (~58GB) or
  full rank arrays (~60GB)
- **Performance documentation** (`docs/PERFORMANCE.md`): Bottleneck breakdown,
  theoretical floor analysis, configuration guide, validation results
- **Top-level `gwas()` API**: Single-call entry point for full GWAS pipeline
  - `from jamma import gwas` — load data, compute kinship, run LMM, write results
  - Returns `GWASResult` dataclass with associations, timing, and summary stats
  - Supports pre-computed kinship, covariates, save-kinship mode
- **Phase-specific memory estimation**: `estimate_lmm_memory()` and
  `estimate_lmm_streaming_memory()` check only LMM-phase memory (not full pipeline peak)
- **Progress bar** for in-memory kinship computation
- **Method logging** for kinship computation (in-memory vs streaming)

### Changed

- LMM runners use phase-specific memory checks instead of total pipeline peak —
  fixes false `MemoryError` when eigendecomp is already complete (e.g., 100k sample
  benchmark: 300GB available, LMM needs ~96GB, was incorrectly demanding 320GB)
- `__version__` now reads from package metadata (`importlib.metadata`) instead of
  hardcoded string — stays in sync with `pyproject.toml` automatically
- JAX cache directory creation wrapped in `try/except OSError` — no longer crashes
  in restricted environments (read-only filesystems, containers)
- Memory safety margin reduced from 50% to 10% based on empirical benchmarks
- Extracted shared helpers in memory estimation (`_check_available`,
  `_streaming_component_sizes`) to reduce duplication
- Vectorized phenotype parsing in `gwas.py` (numpy ops instead of list comprehension)
- Vectorized per-SNP imputation in streaming runner (~2x faster)
- GEMMA comparison notebook writes output to local `/tmp/` instead of DBFS FUSE
- GEMMA comparison notebook accepts pre-existing GEMMA output files

### Fixed

- **LMM MemoryError at 100k samples**: LMM phase demanded 320GB (eigendecomp peak)
  against 300GB available, but only needed ~96GB. Now uses `estimate_lmm_memory()`
- **JAX async dispatch**: `block_until_ready()` in kinship compute loop — progress
  bars and timing now reflect actual compute, not async dispatch time
- **Progress bar lifecycle**: Bars complete cleanly (no hanging on final iteration)
- **Double `.bed` extension**: Fixed `.bed.bed` path construction in GEMMA comparison notebook
- Flaky `test_gwas_with_precomputed_kinship` timing assertion under pytest-xdist

## [1.3.0] - 2026-02-07

### Added

- **Golden section optimizer**: Replaced Brent's method (via scipy) with grid search +
  golden section refinement for lambda optimization — removes scipy runtime dependency
- Auto-select streaming kinship for large datasets (>10k samples)

### Changed

- **Removed scipy runtime dependency**: scipy is now dev-only (tests use `scipy.stats`).
  JAMMA uses `numpy.linalg.eigh` for eigendecomposition, which correctly uses ILP64
  when numpy is built with ILP64 MKL
- Deleted `optimize.py` — lambda optimization now lives in `likelihood_jax.py`
- Stripped numba from `likelihood.py`
- Split `runner_streaming.py` from `runner_jax.py` (separate module)
- Extracted shared utilities: `prepare.py`, `chunk.py`, `results.py`, `progress.py`,
  `snp_filter.py`
- Cached contiguous `U.T` in both LMM runners (perf)
- Replaced list accumulators with pre-allocated numpy arrays (perf)

### Removed

- `optimize.py` (Brent's method via scipy)
- Numba dependency in likelihood computation
- scipy as a runtime dependency

### Fixed

- `NotImplementedError` for kinship mode 2 (standardized) — now raises explicitly
  instead of producing wrong results

## [1.2.0] - 2026-02-05

### Added

- **Databricks benchmark notebook** (`notebooks/databricks_jamma_vs_gemma.py`):
  Widget-parameterized notebook comparing JAMMA vs GEMMA runtime and accuracy
- **Kinship matrix comparison**: Spearman rho, Frobenius norm, max/mean absolute/relative diff
- **CPU pinning for GEMMA**: `taskset --cpu-list 0-23` for eigendecomp in benchmark notebook

### Changed

- Skip JIT warmup for large datasets (>10k samples) to avoid double eigendecomp
- Auto-select streaming kinship for large datasets (>10k samples) with progress bar
- Expanded WHY_JAMMA.md with detailed GEMMA vs JAMMA speed comparison

### Fixed

- Double eigendecomposition in benchmark notebook (warmup was running full pipeline)

## [1.1.0] - 2026-02-05

### Added

- **Score test** (`-lmm 3`): Efficient screening test using null model lambda
- **Likelihood ratio test** (`-lmm 2`): MLE-based chi-square test
- **All tests mode** (`-lmm 4`): Combined Wald, LRT, and Score output
- **Covariate support**: `-c <file>` flag for covariate file input (GEMMA format)
- **Memory pre-flight checks**: Fail fast before OOM instead of silent crash
  - `--no-check-memory` to disable checks on both `gk` and `lmm` commands
  - `estimate_lmm_memory()` API for programmatic memory estimation
  - 50% safety margin based on empirical JAX overhead benchmarks
- **RSS memory logging**: Track memory usage at workflow boundaries
- **Incremental result writing**: Results written per-SNP/per-chunk to disk
  - `output_path` parameter in `run_lmm_association()`
  - JAX streaming runner writes per-file-chunk
- **Safe chunk size defaults**: `MAX_SAFE_CHUNK=50,000` prevents int32 overflow
- **Test tier system**: `tier0` (fast), `tier1` (parity), `tier2` (scale) markers

### Changed

- Memory now bounded by chunk size, not total SNP count
- CLI lmm command uses incremental writing by default
- Eigendecomposition uses numpy LAPACK (not scipy) for large matrix support

### Removed

- Rust/faer eigendecomposition backend (unreliable at scale, higher memory overhead)
- Multi-backend infrastructure (Backend type, `JAMMA_BACKEND` env var, `-be` CLI flag)

### Fixed

- Pre-flight memory check now accounts for full pipeline peak (eigendecomp), not just kinship
- Pre-flight check accounts for SNP count in non-streaming path (JAX genotype copy)
- Eigendecomposition memory check prevents OOM

## [1.0.0] - 2026-02-01

### Added

- **Kinship matrix computation** (`-gk 1`): Centered relatedness matrix XX'/p
- **LMM Wald test** (`-lmm 1`): Univariate linear mixed model association
- **Pre-computed kinship input** (`-k`): Load kinship from file
- **PLINK binary format**: `.bed/.bim/.fam` file support
- **Streaming I/O**: Handle 200k+ samples without loading full matrix
- **JAX acceleration**: CPU/GPU support via JAX backend
- **GEMMA-compatible output**: Identical `.assoc.txt` and `.cXX.txt` formats
- **Numerical equivalence**: Results match GEMMA (identical significance calls, rankings, directions)

### Performance

- 7x faster than GEMMA on kinship computation
- 4x faster than GEMMA on LMM association
- Streaming kinship for datasets exceeding memory

[5.1.4]: https://github.com/michael-denyer/jamma/compare/v5.1.3...v5.1.4
[5.1.3]: https://github.com/michael-denyer/jamma/compare/v5.1.2...v5.1.3
[5.1.2]: https://github.com/michael-denyer/jamma/compare/v5.1.1...v5.1.2
[5.1.1]: https://github.com/michael-denyer/jamma/compare/v5.1.0...v5.1.1
[5.1.0]: https://github.com/michael-denyer/jamma/compare/v5.0.1...v5.1.0
[5.0.1]: https://github.com/michael-denyer/jamma/compare/v5.0.0...v5.0.1
[5.0.0]: https://github.com/michael-denyer/jamma/compare/v4.1.0...v5.0.0
[4.1.0]: https://github.com/michael-denyer/jamma/compare/v4.0.3...v4.1.0
[4.0.3]: https://github.com/michael-denyer/jamma/compare/v4.0.2...v4.0.3
[4.0.2]: https://github.com/michael-denyer/jamma/compare/v4.0.1...v4.0.2
[4.0.1]: https://github.com/michael-denyer/jamma/compare/v4.0.0...v4.0.1
[4.0.0]: https://github.com/michael-denyer/jamma/compare/v3.5.1...v4.0.0
[3.5.1]: https://github.com/michael-denyer/jamma/compare/v3.5.0...v3.5.1
[3.5.0]: https://github.com/michael-denyer/jamma/compare/v3.4.1...v3.5.0
[3.4.1]: https://github.com/michael-denyer/jamma/compare/v3.4.0...v3.4.1
[3.4.0]: https://github.com/michael-denyer/jamma/compare/v3.3.2...v3.4.0
[3.3.2]: https://github.com/michael-denyer/jamma/compare/v3.3.1...v3.3.2
[3.3.1]: https://github.com/michael-denyer/jamma/compare/v3.3.0...v3.3.1
[3.3.0]: https://github.com/michael-denyer/jamma/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/michael-denyer/jamma/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/michael-denyer/jamma/compare/v3.0.1...v3.1.0
[3.0.1]: https://github.com/michael-denyer/jamma/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/michael-denyer/jamma/compare/v2.12.0...v3.0.0
[2.12.0]: https://github.com/michael-denyer/jamma/compare/v2.11.2...v2.12.0
[2.11.2]: https://github.com/michael-denyer/jamma/compare/v2.11.1...v2.11.2
[2.11.1]: https://github.com/michael-denyer/jamma/compare/v2.11.0...v2.11.1
[2.11.0]: https://github.com/michael-denyer/jamma/compare/v2.10.1...v2.11.0
[2.10.1]: https://github.com/michael-denyer/jamma/compare/v2.10.0...v2.10.1
[2.10.0]: https://github.com/michael-denyer/jamma/compare/v2.9.6...v2.10.0
[2.9.6]: https://github.com/michael-denyer/jamma/compare/v2.9.5...v2.9.6
[2.9.5]: https://github.com/michael-denyer/jamma/compare/v2.9.4...v2.9.5
[2.9.4]: https://github.com/michael-denyer/jamma/compare/v2.9.3...v2.9.4
[2.9.3]: https://github.com/michael-denyer/jamma/compare/v2.9.2...v2.9.3
[2.9.2]: https://github.com/michael-denyer/jamma/compare/v2.9.1...v2.9.2
[2.9.1]: https://github.com/michael-denyer/jamma/compare/v2.9.0...v2.9.1
[2.9.0]: https://github.com/michael-denyer/jamma/compare/v2.8.3...v2.9.0
[2.8.3]: https://github.com/michael-denyer/jamma/compare/v2.8.2...v2.8.3
[2.8.2]: https://github.com/michael-denyer/jamma/compare/v2.8.1...v2.8.2
[2.8.1]: https://github.com/michael-denyer/jamma/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/michael-denyer/jamma/compare/v2.7.1...v2.8.0
[2.7.1]: https://github.com/michael-denyer/jamma/compare/v2.7.0...v2.7.1
[2.7.0]: https://github.com/michael-denyer/jamma/compare/v2.6.1...v2.7.0
[2.6.1]: https://github.com/michael-denyer/jamma/compare/v2.6.0...v2.6.1
[2.6.0]: https://github.com/michael-denyer/jamma/compare/v2.5.8...v2.6.0
[2.5.8]: https://github.com/michael-denyer/jamma/compare/v2.5.7...v2.5.8
[2.5.7]: https://github.com/michael-denyer/jamma/compare/v2.5.6...v2.5.7
[2.5.6]: https://github.com/michael-denyer/jamma/compare/v2.5.5...v2.5.6
[2.5.5]: https://github.com/michael-denyer/jamma/compare/v2.5.4...v2.5.5
[2.5.4]: https://github.com/michael-denyer/jamma/compare/v2.5.3...v2.5.4
[2.5.3]: https://github.com/michael-denyer/jamma/compare/v2.5.2...v2.5.3
[2.5.2]: https://github.com/michael-denyer/jamma/compare/v2.5.1...v2.5.2
[2.5.1]: https://github.com/michael-denyer/jamma/compare/v2.5.0...v2.5.1
[2.5.0]: https://github.com/michael-denyer/jamma/compare/v2.4.5...v2.5.0
[2.4.5]: https://github.com/michael-denyer/jamma/compare/v2.4.4...v2.4.5
[2.4.4]: https://github.com/michael-denyer/jamma/compare/v2.4.3...v2.4.4
[2.4.3]: https://github.com/michael-denyer/jamma/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/michael-denyer/jamma/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/michael-denyer/jamma/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/michael-denyer/jamma/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/michael-denyer/jamma/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/michael-denyer/jamma/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/michael-denyer/jamma/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/michael-denyer/jamma/compare/v1.5.1...v2.0.0
[1.5.1]: https://github.com/michael-denyer/jamma/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/michael-denyer/jamma/compare/v1.4.3...v1.5.0
[1.4.3]: https://github.com/michael-denyer/jamma/compare/v1.3.0...v1.4.3
[1.3.0]: https://github.com/michael-denyer/jamma/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/michael-denyer/jamma/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/michael-denyer/jamma/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/michael-denyer/jamma/releases/tag/v1.0.0
