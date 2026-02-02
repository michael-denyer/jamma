# Changelog

All notable changes to JAMMA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Score test (`-lmm 3`) - planned
- Likelihood ratio test (`-lmm 2`) - planned
- All tests mode (`-lmm 4`) - planned

## [1.1.0] - 2026-02-02

### Added
- **Covariate support**: `-c <file>` flag for covariate file input (GEMMA format)
- **Memory pre-flight checks**: Fail fast before OOM instead of silent crash
  - `--mem-budget <GB>` to set memory limit
  - `--no-check-memory` to disable checks
  - `estimate_lmm_memory()` API for programmatic memory estimation
- **RSS memory logging**: Track memory usage at workflow boundaries
- **Incremental result writing**: Results written per-SNP/per-chunk to disk
  - `output_path` parameter in `run_lmm_association()`
  - JAX streaming runner writes per-file-chunk
- **Safe chunk size defaults**: `MAX_SAFE_CHUNK=50,000` prevents int32 overflow
- **Test tier system**: `tier0` (fast), `tier1` (parity), `tier2` (scale) markers

### Changed
- Memory now bounded by chunk size, not total SNP count
- CLI lmm command uses incremental writing by default

### Fixed
- Eigendecomposition memory check prevents scipy OOM
- Genotype loading memory check prevents full matrix OOM

## [1.0.0] - 2026-02-01

### Added
- **Kinship matrix computation** (`-gk 1`): Centered relatedness matrix XX'/p
- **LMM Wald test** (`-lmm 1`): Univariate linear mixed model association
- **Pre-computed kinship input** (`-k`): Load kinship from file
- **PLINK binary format**: `.bed/.bim/.fam` file support
- **Streaming I/O**: Handle 200k+ samples without loading full matrix
- **JAX acceleration**: CPU/GPU support via JAX backend
- **GEMMA-compatible output**: Identical `.assoc.txt` and `.cXX.txt` formats
- **Numerical equivalence**: Results match GEMMA (beta 1e-6, p-values 1e-8)

### Performance
- 7x faster than GEMMA on kinship computation
- 4x faster than GEMMA on LMM association
- Streaming kinship for datasets exceeding memory

[Unreleased]: https://github.com/michael-denyer/jamma/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/michael-denyer/jamma/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/michael-denyer/jamma/releases/tag/v1.0.0
