# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-06)

**Core value:** Exact GEMMA statistical results at 200k sample scale
**Current focus:** v1.3 Tech Debt Cleanup — Phase 17 complete, ready to plan Phase 18

## Current Position

Milestone: v1.3
Phase: 18 of 18 (Correctness & Performance)
Plan: 2 plans (18-01, 18-02) in 2 waves
Status: Ready to execute
Last activity: 2026-02-06 — Phase 18 planned (2 plans: CORR-01+PERF-01, PERF-02)

## Performance Metrics

**v1.0 Velocity:**

- Total plans completed: 21
- Average duration: 5min 47s
- Total execution time: 1.93 hours

**v1.1 Velocity:**

- Total plans completed: 39
- Average duration: ~6min
- Total execution time: ~3.9 hours

**v1.2 Velocity:**

- Total plans completed: 18
- Average duration: ~6min
- Total execution time: ~1hr 48min

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key decisions from v1.2 relevant to v1.3:

- [15-05]: NumPy runner removed, JAX streaming is sole execution path
- [15-05]: check_memory_before_run() uses estimate_streaming_memory
- [14-01]: JAX path passes raw (unfiltered) phenotypes/kinship; runner handles filtering
- [11-01]: build_index_table runs at Python level, not inside JIT
- [13-01]: Mode 4 execution order is Score -> MLE -> REML -> Wald
- [16-01]: Golden section matches JAX batch optimizer algorithm (grid + refinement), not Brent's method
- [16-01]: Pure Python math.log/math.exp for self-contained optimizer
- [16-02]: No replacement tests for deleted Brent/OptimizeLambda -- golden section tested via null model tests
- [17-01]: Drop underscore prefix on progress_iterator (now public utility in core/)
- [17-01]: Place snp_filter.py in core/ not a new snp/ package
- [17-02]: Re-export moved names from runner_jax.py with noqa: F401 for backward compatibility
- [17-02]: Monkeypatch test targets chunk module directly (re-exports don't propagate mutations)
- [17-03]: Place _grid_optimize_lambda_batched in prepare.py (shared by both runners)
- [17-03]: Use dict-based accumulators in streaming runner for compact mode dispatch
- [17-03]: likelihood.py (607) and likelihood_jax.py (883) left as-is (out of scope)

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-06
Stopped at: Phase 18 planned (2 plans, plan check passed)
Resume file: None
Next: `/gsd:execute-phase 18` (Correctness & Performance)

---

## v1.0 Summary

**Shipped:** 2026-02-01
**Phases:** 6 (21 plans)
**Requirements:** 14/14 complete
**Tests:** 243 passing

See: .planning/v1-MILESTONE-ARCHIVE.md

## v1.2 Summary

**Shipped:** 2026-02-05
**Phases:** 11-15 (18 plans)
**Requirements:** 12/12 complete
**Tests:** 400+ passing
