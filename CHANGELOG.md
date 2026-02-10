# Changelog

All notable changes to JAMMA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/michael-denyer/jamma/compare/v1.5.0...HEAD
[1.5.0]: https://github.com/michael-denyer/jamma/compare/v1.4.3...v1.5.0
[1.4.3]: https://github.com/michael-denyer/jamma/compare/v1.3.0...v1.4.3
[1.3.0]: https://github.com/michael-denyer/jamma/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/michael-denyer/jamma/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/michael-denyer/jamma/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/michael-denyer/jamma/releases/tag/v1.0.0
