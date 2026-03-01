# Changelog

All notable changes to JAMMA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Eigendecomposition now prefers DSYEVD (1.2–1.5x faster) by default, falling back to DSYEVR only when DSYEVD workspace exceeds available memory
- Memory estimates default to DSYEVD (conservative); actual peak is lower if DSYEVR is triggered

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
- Test tolerances aligned with EQUIVALENCE.md (kinship 1e-10 → 1e-8)
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

[Unreleased]: https://github.com/michael-denyer/jamma/compare/v2.8.1...HEAD
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
