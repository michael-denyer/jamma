---
phase: 25-loco-kinship
plan: 01
subsystem: kinship
tags: [loco, kinship, chromosome-partitioning, subtraction-approach, jax, streaming]

# Dependency graph
requires:
  - phase: 24-quality-cleanup
    provides: searchsorted SNP filtering, snp_filter shared utilities
provides:
  - get_chromosome_partitions() for BIM chromosome grouping
  - compute_loco_kinship() in-memory LOCO via subtraction approach
  - compute_loco_kinship_streaming() disk-streamed LOCO kinship
affects: [25-02 (LOCO tests), 25-03 (LOCO LMM orchestrator), 26-eigendecomp-reuse]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Subtraction-based LOCO: S_full - S_c / (p - p_c) avoids redundant computation"
    - "Generator-based yield for LOCO matrices to avoid holding all in memory"
    - "Global centering before chromosome partitioning to preserve subtraction identity"

key-files:
  created: []
  modified:
    - src/jamma/io/plink.py
    - src/jamma/io/__init__.py
    - src/jamma/kinship/compute.py
    - src/jamma/kinship/__init__.py

key-decisions:
  - "Generator (Iterator) return type for LOCO kinship functions -- yields one K_loco at a time to support large-scale processing without holding all matrices"
  - "Streaming LOCO accumulates all S_chr simultaneously in second pass -- O(n_chr * n^2) memory but avoids multi-pass complexity"
  - "Global SNP filtering applied before chromosome partitioning with chromosome array filtered in parallel"

patterns-established:
  - "LOCO subtraction: compute S_full once, derive K_loco_c = (S_full - S_c) / (p - p_c)"
  - "Chromosome array co-filtering: when filtering genotypes, filter the chromosome metadata array with the same mask"

# Metrics
duration: 10min
completed: 2026-02-11
---

# Phase 25 Plan 01: LOCO Kinship Computation Summary

**Chromosome partitioning and LOCO kinship via subtraction approach with in-memory and streaming paths producing machine-epsilon identical results**

## Performance

- **Duration:** 10 min
- **Started:** 2026-02-11T01:24:16Z
- **Completed:** 2026-02-11T01:35:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- `get_chromosome_partitions()` groups SNP indices by chromosome from BIM metadata via bed-reader
- `compute_loco_kinship()` implements efficient subtraction-based LOCO with global centering, yielding one K_loco at a time
- `compute_loco_kinship_streaming()` accumulates S_full and all per-chromosome S_chr in a single second pass from disk
- In-memory and streaming produce identical LOCO matrices within 2.22e-16 (machine epsilon) across all 19 mouse_hs1940 chromosomes
- All 481 existing tests pass with zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Chromosome partitioning and in-memory LOCO kinship** - `a79edca` (feat)
2. **Task 2: Streaming LOCO kinship computation** - `70fd4ef` (feat)

## Files Created/Modified
- `src/jamma/io/plink.py` - Added `get_chromosome_partitions()` for BIM chromosome grouping
- `src/jamma/io/__init__.py` - Export `get_chromosome_partitions`
- `src/jamma/kinship/compute.py` - Added `compute_loco_kinship()` and `compute_loco_kinship_streaming()`
- `src/jamma/kinship/__init__.py` - Export new LOCO functions and `get_chromosome_partitions`

## Decisions Made
- **Generator return type:** Both LOCO functions yield `(chr_name, K_loco)` tuples via `Iterator` rather than returning a dict. This allows callers to process and discard each matrix without holding all LOCO matrices in memory simultaneously. Critical for large-scale datasets where each K_loco is 80GB at 100k samples.
- **Simultaneous S_chr accumulation in streaming:** The streaming function accumulates all per-chromosome S_chr matrices in a single second pass rather than requiring one pass per chromosome. This trades O(n_chr * n^2) memory for O(1) disk passes. Acceptable for typical chromosome counts (19-22) but documented as a scaling concern for 100k+ samples.
- **Chromosome array co-filtering:** When `_filter_snps()` removes SNPs by MAF/missingness/monomorphism, the chromosome array must be filtered with the identical mask. The in-memory path recomputes the SNP stats and filter mask to get the boolean mask for this purpose.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Restored get_chromosome_partitions import after ruff removal**
- **Found during:** Task 2 (streaming LOCO kinship)
- **Issue:** Ruff's unused-import auto-fix in the Task 1 pre-commit hook removed `get_chromosome_partitions` from compute.py imports because the streaming function (which uses it) didn't exist yet at commit time
- **Fix:** Re-added `get_chromosome_partitions` to the import block in compute.py
- **Files modified:** src/jamma/kinship/compute.py
- **Verification:** `compute_loco_kinship_streaming()` runs successfully on mouse_hs1940
- **Committed in:** 70fd4ef (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minor import ordering issue from incremental commits. No scope creep.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- LOCO kinship functions are ready for test coverage (Plan 25-02)
- `compute_loco_kinship()` and `compute_loco_kinship_streaming()` provide the foundation for LOCO LMM orchestration (Plan 25-03)
- Generator-based API enables memory-efficient per-chromosome eigendecomposition in downstream pipeline

## Self-Check: PASSED

- All 4 modified files exist on disk
- Both task commits (a79edca, 70fd4ef) found in git log
- All must_have artifacts verified (get_chromosome_partitions, compute_loco_kinship, exports, key_links)
- 481/481 existing tests pass with zero regressions

---
*Phase: 25-loco-kinship*
*Completed: 2026-02-11*
