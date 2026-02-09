---
phase: quick-2
plan: 01
subsystem: api
tags: [gwas, dataclass, orchestration, public-api]

# Dependency graph
requires:
  - phase: v1.3
    provides: streaming LMM runner, kinship computation, CLI orchestration
provides:
  - "gwas() single-call GWAS pipeline function"
  - "GWASResult dataclass with timing, counts, associations"
  - "Top-level jamma.gwas and jamma.GWASResult exports"
affects: [cli, notebooks, documentation]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Orchestrator function wrapping streaming internals"]

key-files:
  created:
    - src/jamma/gwas.py
    - tests/test_gwas_api.py
  modified:
    - src/jamma/__init__.py

key-decisions:
  - "gwas() returns empty associations list when output_path used (results on disk), matching streaming runner behavior"
  - "n_snps_tested uses metadata count (total SNPs in file), not post-filter count"

patterns-established:
  - "Public API pattern: gwas() as single entry point for notebook/library users"

# Metrics
duration: 6min 28s
completed: 2026-02-09
---

# Quick Task 2: Add Top-Level jamma.gwas() API Function Summary

**Single-call gwas() function orchestrating full GWAS pipeline (load, kinship, LMM, write) with GWASResult dataclass**

## Performance

- **Duration:** 6min 28s
- **Started:** 2026-02-09T15:35:05Z
- **Completed:** 2026-02-09T15:41:33Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created `gwas()` function that replaces 5+ manual function calls with a single entry point
- GWASResult dataclass provides structured output with timing breakdown, sample/SNP counts
- Full test coverage: 7 tests covering happy path, error handling, and import verification
- All 410 existing tests pass with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Create gwas.py with gwas() function and GWASResult dataclass** - `1caa9fe` (feat)
2. **Task 2: Add tests for gwas() API** - `3fd94cf` (test)

## Files Created/Modified
- `src/jamma/gwas.py` - gwas() function and GWASResult dataclass (~175 lines)
- `src/jamma/__init__.py` - Re-exports gwas and GWASResult, updated docstring example
- `tests/test_gwas_api.py` - 7 tests for gwas() API (114 lines)

## Decisions Made
- Used `time.perf_counter()` for timing (matching cli.py and streaming runner)
- gwas() validates PLINK files (.bed, .bim, .fam) upfront before any computation
- Memory check delegated to `estimate_streaming_memory` (same as CLI path)
- Covariates loaded via `read_covariate_file` with row count validation against phenotype count

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `from jamma import gwas` is the clean public API for notebook users
- Documentation (USER_GUIDE.md, README) could be updated to reference the new API
- Future: could add `gwas()` parameters for eigenvalue caching, chunk_size tuning

## Self-Check: PASSED

All files verified present. All commit hashes found in git log.

---
*Quick Task: 2-add-top-level-jamma-gwas-api-function*
*Completed: 2026-02-09*
