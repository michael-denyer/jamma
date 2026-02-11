---
phase: 26-eigendecomposition-reuse
plan: 02
subsystem: testing
tags: [eigendecomposition, gemma-compat, round-trip, lmm-equivalence, validation]

# Dependency graph
requires:
  - phase: 26-eigendecomposition-reuse
    plan: 01
    provides: "Eigen I/O module, pipeline integration, CLI flags"
provides:
  - "Comprehensive test suite for eigen I/O format, round-trip, and reuse"
  - "LMM equivalence proof: loaded-eigen matches fresh-eigen"
  - "Flag interaction validation tests"
affects: [27-phenotype-selection]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Valid-sample subset before eigendecomposition for write_eigen"
    - "LMM equivalence testing: fresh vs loaded eigen comparison"

key-files:
  created:
    - tests/test_eigen_io.py
  modified:
    - src/jamma/pipeline.py

key-decisions:
  - "write_eigen subsets kinship to valid-phenotype samples before eigendecomposition, ensuring files match analyzed sample count"
  - "Standard calibrated tolerances (rtol=1e-2 beta, rtol=1e-4 p_wald) sufficient for loaded-eigen equivalence -- no widening needed"

patterns-established:
  - "Eigen reuse testing: compute eigen from valid-sample-subsetted kinship, write to files, load and run LMM, compare against fresh pipeline"

# Metrics
duration: 9min
completed: 2026-02-11
---

# Phase 26 Plan 02: Eigen I/O Validation Test Suite Summary

**22 tests covering GEMMA-format round-trip precision (rtol=1e-9), loaded-vs-fresh LMM equivalence, flag interaction rules, and CLI flag visibility**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-11T14:10:46Z
- **Completed:** 2026-02-11T14:20:06Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- 13 format, round-trip, dimension, and edge case tests proving .10g precision preserves eigendecomposition within rtol=1e-9
- LMM equivalence test: loaded-eigen pipeline matches fresh-eigen pipeline within calibrated tolerances (beta rtol=1e-2, p_wald rtol=1e-4, se rtol=1e-5)
- 5 flag interaction tests: -d/-u pairing, -loco incompatibility, file existence, kinship-optional-with-eigen
- 2 CLI help tests: -d, -u, -eigen flags visible in lmm and gk help output
- Pipeline bug fix: write_eigen now subsets kinship to valid-phenotype samples before eigendecomposition
- All 495 existing tests pass with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Eigen I/O format and round-trip tests** - `5f41809` (test)
2. **Task 2: LMM equivalence and flag interaction tests** - `b85eb38` (test)

## Files Created/Modified
- `tests/test_eigen_io.py` - 22 tests across 7 test classes covering all eigen I/O validation
- `src/jamma/pipeline.py` - Fix write_eigen to subset kinship to valid samples before eigendecomp

## Decisions Made
- Standard calibrated tolerances from CLAUDE.md sufficient for loaded-eigen comparison -- .10g format preserves enough precision that no widening was needed
- write_eigen must subset kinship to valid-phenotype samples before eigendecomposition (full kinship produces dimensions that don't match the runner's internal subsetting)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Pipeline write_eigen dimension mismatch with missing phenotypes**
- **Found during:** Task 2 (LMM equivalence test)
- **Issue:** write_eigen eigendecomposed the full kinship matrix (1940x1940 for mouse_hs1940) but the runner internally subsets to valid-phenotype samples (1410). The pre-computed eigen dimensions (1940) mismatched the runner's covariate matrix dimensions (1410), causing `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0`.
- **Fix:** Added valid_mask computation and kinship subsetting in pipeline before eigendecomposition for write_eigen path, matching the runner's internal behavior.
- **Files modified:** src/jamma/pipeline.py
- **Verification:** test_write_eigen_flag_creates_files passes; test_loaded_eigen_matches_fresh_eigen_lmm passes; all 495 tests pass
- **Committed in:** b85eb38 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix required for correctness of write_eigen with datasets that have missing phenotypes. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 26 (eigendecomposition reuse) fully complete with both implementation and tests
- Eigen I/O, pipeline integration, CLI flags, and Python API all tested
- Ready for Phase 27 (phenotype selection and standardized kinship)

## Self-Check: PASSED

All files verified on disk. Both task commits (5f41809, b85eb38) verified in git log.

---
*Phase: 26-eigendecomposition-reuse*
*Completed: 2026-02-11*
