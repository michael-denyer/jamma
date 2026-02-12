---
phase: 28-filtering-and-input-validation
plan: 02
subsystem: pipeline
tags: [cli, gwas-api, plink-validation, snp-filter, hwe, pipeline-config]

requires:
  - phase: 28-filtering-and-input-validation
    plan: 01
    provides: read_snp_list_file, resolve_snp_list_to_indices, compute_hwe_pvalues, snps_indices/ksnps_indices/hwe_threshold params on runners and kinship
provides:
  - CLI -snps, -ksnps, -hwe flags on lmm command and -ksnps on gk command
  - gwas() Python API with snps_file, ksnps_file, hwe parameters
  - PipelineConfig with snps_file, ksnps_file, hwe_threshold fields
  - validate_plink_dimensions for PLINK .bed size validation (VALID-01)
  - PipelineRunner resolves SNP list files to indices and passes through to runners
affects: [cli-users, python-api-users, plink-input-quality]

tech-stack:
  added: []
  patterns: [lazy-import-for-optional-deps, file-existence-validation-in-pipeline]

key-files:
  created:
    - tests/test_plink_validation.py
  modified:
    - src/jamma/io/plink.py
    - src/jamma/pipeline.py
    - src/jamma/cli.py
    - src/jamma/gwas.py
    - tests/test_cli.py
    - tests/test_pipeline.py

key-decisions:
  - "validate_plink_dimensions uses line counts from .fam/.bim + expected .bed formula (3 + ceil(n/4) * m) for cross-validation"
  - "Lazy import of snp_list module in PipelineRunner.run() to avoid import overhead when not using SNP lists"
  - "gk command resolves ksnps_indices before load to support both streaming and non-streaming kinship paths"
  - "Non-streaming gk path filters genotype matrix columns before passing to compute_fn (column subset)"

patterns-established:
  - "CLI flag -> PipelineConfig field -> PipelineRunner resolution -> runner parameter: consistent wiring pattern for all filtering flags"
  - "validate_inputs checks file existence for all optional file parameters before pipeline execution"

duration: 9min
completed: 2026-02-12
---

# Phase 28 Plan 02: CLI Flags, GWAS API, and PLINK Validation Summary

**CLI -snps/-ksnps/-hwe flags, gwas() API parameters, PipelineConfig wiring, and PLINK dimension validation completing Phase 28 end-to-end**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-12T01:34:19Z
- **Completed:** 2026-02-12T01:44:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- PLINK dimension validation (validate_plink_dimensions) catches corrupted/truncated .bed files with clear error messages
- CLI exposes -snps, -ksnps, -hwe on lmm command and -ksnps on gk command, matching GEMMA flag names
- gwas() Python API accepts snps_file, ksnps_file, hwe parameters for programmatic use
- PipelineConfig stores, validates, and passes through all filtering parameters to streaming runners and kinship
- 17 new tests covering PLINK validation, CLI help output, and pipeline config validation
- Full test suite (590 tests) passes with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: PLINK dimension and genotype value validation** - `e15728f` (feat)
2. **Task 2: CLI flags, gwas() API, PipelineConfig wiring, and end-to-end tests** - `10f4e51` (feat)

## Files Created/Modified
- `src/jamma/io/plink.py` - Added validate_plink_dimensions function
- `src/jamma/pipeline.py` - PipelineConfig with snps_file/ksnps_file/hwe_threshold; validate_inputs checks; run() resolves SNP lists
- `src/jamma/cli.py` - -snps, -ksnps, -hwe on lmm; -ksnps on gk; ksnps_indices resolution for both LOCO and standard gk paths
- `src/jamma/gwas.py` - snps_file, ksnps_file, hwe parameters on gwas() function
- `tests/test_plink_validation.py` - 6 tests for dimension and genotype value validation
- `tests/test_cli.py` - 4 tests for new flag presence in help output
- `tests/test_pipeline.py` - 7 tests for PipelineConfig defaults, custom values, and validation errors

## Decisions Made
- validate_plink_dimensions uses .fam/.bim line counts cross-validated against expected .bed formula (3 + ceil(n/4) * m)
- Lazy import of snp_list module in PipelineRunner.run() avoids import overhead for users not using SNP lists
- gk command resolves ksnps_indices early to support both streaming LOCO and non-streaming standard kinship paths
- Non-streaming gk filters genotype matrix columns before passing to compute_fn (simple column subset vs. streaming index pass-through)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 28 complete: all 7 requirements (SNP-01, SNP-02, SNP-03, QC-01, QC-02, VALID-01, VALID-02) satisfied
- All filtering infrastructure wired end-to-end through CLI, Python API, and pipeline
- All new parameters default to None/0.0, fully backward compatible
- Full test suite (590 tests) passes

## Self-Check: PASSED

All 7 files verified present. Both commits (e15728f, 10f4e51) verified in git log.
All 6 artifact patterns confirmed in source files.

---
*Phase: 28-filtering-and-input-validation*
*Completed: 2026-02-12*
