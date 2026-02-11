---
phase: 25-loco-kinship
plan: 02
subsystem: lmm
tags: [loco, lmm, orchestrator, cli, pipeline, gwas-api, per-chromosome-eigendecomp]

# Dependency graph
requires:
  - phase: 25-01
    provides: compute_loco_kinship_streaming(), get_chromosome_partitions()
provides:
  - run_lmm_loco() LOCO LMM orchestrator
  - _run_lmm_for_chromosome() per-chromosome LMM runner
  - write_loco_kinship_matrices() convenience I/O
  - PipelineConfig.loco and PipelineRunner LOCO dispatch
  - CLI -loco flag on gk and lmm commands
  - gwas(loco=True) Python API
affects: [25-03 (LOCO tests), 26-eigendecomp-reuse]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "LOCO orchestrator streams K_loco one at a time, eigendecomposes, runs LMM, discards"
    - "Per-chromosome LMM reads only that chromosome's SNPs in a single BED read"
    - "Shared IncrementalAssocWriter across all chromosomes for single output file"

key-files:
  created:
    - src/jamma/lmm/loco.py
  modified:
    - src/jamma/kinship/io.py
    - src/jamma/kinship/__init__.py
    - src/jamma/lmm/__init__.py
    - src/jamma/pipeline.py
    - src/jamma/cli.py
    - src/jamma/gwas.py

key-decisions:
  - "Per-chromosome single BED read instead of streaming: chromosome subsets are small enough to fit in memory"
  - "-k and -loco mutually exclusive in this version; LOCO computes kinship internally"
  - "LOCO branch in PipelineRunner.run() skips standard kinship loading entirely"
  - "CLI gk -loco uses compute_loco_kinship_streaming + write_loco_kinship_matrices for standalone kinship output"

patterns-established:
  - "LOCO orchestrator pattern: stream K_loco from generator, eigendecompose, run per-chromosome LMM, discard"
  - "Reuse streaming runner internals (_init_accumulators, _append_chunk_results) for per-chromosome JAX pipeline"

# Metrics
duration: 9min
completed: 2026-02-11
---

# Phase 25 Plan 02: LOCO LMM Orchestrator and Integration Summary

**LOCO LMM orchestrator with per-chromosome eigendecomp/association, integrated into CLI (-loco flag), pipeline, and gwas() Python API**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-11T01:38:41Z
- **Completed:** 2026-02-11T01:48:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- `run_lmm_loco()` orchestrates per-chromosome LOCO kinship streaming, eigendecomp, and LMM association
- `_run_lmm_for_chromosome()` reads only that chromosome's SNPs from BED file and runs full JAX LMM pipeline
- CLI `jamma lmm -loco` and `jamma gk -loco` flags with mutual exclusivity check against `-k`
- Python API `gwas(bfile, loco=True)` for programmatic LOCO analysis
- All 481 existing tests pass with zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: LOCO LMM orchestrator and kinship I/O** - `786777a` (feat)
2. **Task 2: Pipeline, CLI, and API integration** - `8a74a50` (feat)

## Files Created/Modified
- `src/jamma/lmm/loco.py` - LOCO LMM orchestrator (run_lmm_loco, _run_lmm_for_chromosome)
- `src/jamma/kinship/io.py` - Added write_loco_kinship_matrices() for per-chromosome output
- `src/jamma/kinship/__init__.py` - Export write_loco_kinship_matrices
- `src/jamma/lmm/__init__.py` - Export run_lmm_loco
- `src/jamma/pipeline.py` - PipelineConfig.loco field, LOCO branch in PipelineRunner.run()
- `src/jamma/cli.py` - -loco flag on gk and lmm commands
- `src/jamma/gwas.py` - loco parameter on gwas() function

## Decisions Made
- **Per-chromosome single BED read:** `_run_lmm_for_chromosome()` reads the entire chromosome's SNPs in one `bed.read(index=...)` call rather than streaming, because chromosome subsets are small enough (typically hundreds to a few thousand SNPs) to fit in memory.
- **Mutual exclusivity of -k and -loco:** Rather than supporting pre-computed LOCO kinship loading (which would require convention for multi-file loading), this version computes LOCO kinship internally. Pre-computed LOCO kinship loading deferred to Phase 26.
- **LOCO pipeline branch:** The PipelineRunner.run() method adds a full LOCO branch that skips standard kinship loading and delegates to run_lmm_loco(). The standard (non-LOCO) path is completely unchanged.
- **Reuse streaming runner internals:** `_run_lmm_for_chromosome()` reuses `_init_accumulators`, `_append_chunk_results`, and `_concat_jax_accumulators` from the streaming runner for consistent JAX pipeline behavior.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- LOCO LMM orchestrator ready for test coverage (Plan 25-03)
- CLI `-loco` flag functional for both `gk` and `lmm` commands
- Python API `gwas(loco=True)` ready for integration tests
- Standard pipeline unchanged, no regressions

## Self-Check: PASSED

- All 7 modified/created files exist on disk
- Both task commits (786777a, 8a74a50) found in git log
- All must_have artifacts verified (run_lmm_loco, loco in pipeline, -loco in CLI, loco in gwas, write_loco_kinship_matrices)
- All key_links verified (`compute_loco_kinship`, `_run_lmm_for_chromosome`, `eigendecompose_kinship`, `run_lmm_loco` in pipeline, loco patterns in CLI)
- 481/481 existing tests pass with zero regressions

---
*Phase: 25-loco-kinship*
*Completed: 2026-02-11*
