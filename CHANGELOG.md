# Changelog

All notable changes to JAMMA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/michael-denyer/jamma/compare/v2.4.4...HEAD
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
