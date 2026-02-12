---
phase: 29-loco-integration-wiring
plan: 01
subsystem: lmm
tags: [loco, kinship, ksnps, validation, cli, pipeline]

# Dependency graph
requires:
  - phase: 28-filtering-and-input-validation
    provides: ksnps_indices resolution in pipeline, compute_loco_kinship_streaming ksnps param
  - phase: 25-loco-kinship
    provides: run_lmm_loco, compute_loco_kinship_streaming
provides:
  - ksnps_indices wired through LOCO LMM path end-to-end
  - gk command rejects -gk 2 -loco with clear error
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: [early-validation-in-cli, parameter-passthrough-wiring]

key-files:
  created: []
  modified:
    - src/jamma/lmm/loco.py
    - src/jamma/pipeline.py
    - src/jamma/cli.py
    - tests/test_loco.py

key-decisions:
  - "ksnps_indices passed directly to compute_loco_kinship_streaming from run_lmm_loco (no transformation needed)"
  - "-gk 2 -loco validation placed in CLI gk_command, not in pipeline (lmm command uses centered internally)"

patterns-established:
  - "Gap closure: audit-driven wiring fixes with targeted tests proving integration"

# Metrics
duration: 6min
completed: 2026-02-12
---

# Phase 29 Plan 01: LOCO Integration Wiring Summary

**Wire ksnps_indices through LOCO path and reject -gk 2 -loco with early CLI validation**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-12T02:15:02Z
- **Completed:** 2026-02-12T02:21:40Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- ksnps_indices flows from PipelineConfig.ksnps_file through pipeline LOCO branch through run_lmm_loco() to compute_loco_kinship_streaming()
- gk command rejects -gk 2 -loco with exit code 1 and descriptive error message
- 3 new tests prove both integration gaps are closed (2 ksnps wiring + 1 standardized-loco rejection)
- All 12 existing fast LOCO tests pass without modification (568 total non-slow tests pass)

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire ksnps_indices through LOCO path (GAP-1)** - `5bebdd2` (feat)
2. **Task 2: Reject -gk 2 -loco with validation error (GAP-2)** - `cc154a0` (feat)

Supplementary fix commit: `faa310a` (restore STATE.md removed by stale staging)

## Files Created/Modified
- `src/jamma/lmm/loco.py` - Added ksnps_indices parameter to run_lmm_loco(), passes to compute_loco_kinship_streaming()
- `src/jamma/pipeline.py` - LOCO branch passes ksnps_indices=ksnps_indices to run_lmm_loco()
- `src/jamma/cli.py` - Added -gk 2 -loco early validation with descriptive error message
- `tests/test_loco.py` - Added TestLocoKsnpsWiring (2 tests) and test_cli_gk_standardized_loco_rejected

## Decisions Made
- ksnps_indices passed directly to compute_loco_kinship_streaming from run_lmm_loco without transformation -- the streaming function already accepts and handles ksnps_indices
- -gk 2 -loco validation placed in CLI gk_command only, not in pipeline -- the lmm command does not have a -gk flag and always computes centered kinship internally for LOCO

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed pipeline test ksnps_file using first 100 SNPs (all on chr 1)**
- **Found during:** Task 1 (TestLocoKsnpsWiring)
- **Issue:** Using `meta["sid"][:100]` selected the first 100 SNPs which are all on chromosome 1. LOCO requires SNPs on multiple chromosomes, causing ValueError.
- **Fix:** Changed to `meta["sid"][::120]` to sample ~102 SNPs evenly across all chromosomes
- **Files modified:** tests/test_loco.py
- **Verification:** test_pipeline_loco_ksnps_wiring passes
- **Committed in:** 5bebdd2 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in test data selection)
**Impact on plan:** Test fix necessary for correctness. No scope creep.

## Issues Encountered
- Pre-existing staged deletion of .planning/STATE.md was included in Task 1 commit along with a modified .gitignore. Restored STATE.md in a follow-up commit (faa310a).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both LOCO integration gaps from the v2.0 milestone audit are now closed
- All v2.0 requirements (Phases 24-29) are implemented
- Ready for final milestone validation or release

---
*Phase: 29-loco-integration-wiring*
*Completed: 2026-02-12*
