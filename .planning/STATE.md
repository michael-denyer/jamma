# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-06)

**Core value:** Exact GEMMA statistical results at 200k sample scale
**Current focus:** v1.4 Performance -- Phase 20: Thread Configuration Fix

## Current Position

Milestone: v1.4 Performance (Phases 19-22)
Phase: 20 of 22 (Thread Configuration Fix)
Plan: 0 of ~2 in current phase
Status: Not started
Last activity: 2026-02-06 -- Completed Phase 19 (Measure and Diagnose, 2/2 plans)

Progress: [██░░░░░░░░] 25% (v1.4 milestone: 2/~8 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 86
- Average duration: ~6 min
- Total execution time: ~8.8 hours

**By Milestone:**

| Milestone | Plans | Total | Avg/Plan |
|-----------|-------|-------|----------|
| v1.0 | 21 | 1.93h | 5m 47s |
| v1.1 | 39 | 3.9h | ~6m |
| v1.2 | 18 | 1.8h | ~6m |
| v1.3 | 7 | 56m | ~8m |
| v1.4 | 2 | 100m | 50m |

## Accumulated Context

### Decisions

All milestone decisions archived in:
- .planning/v1-MILESTONE-ARCHIVE.md (v1.0)
- .planning/milestones/v1.3-ROADMAP.md (v1.3)
- .planning/PROJECT.md Key Decisions table (cumulative)

v1.4-specific:
- Research confirms eigendecomp + UT@G rotation are >99% of compute
- **Thread pinning bug is INACTIVE on Databricks** -- MKL runs at 32 threads throughout
- `_pin_blas_threads(1)` is a no-op: loads after `import jax` which loads MKL
- Eigendecomp dominates at 54% (3,114s), kinship 25% (1,440s), LMM 21% (1,211s)
- scipy bundles its own OpenBLAS (64 threads) separate from numpy's MKL
- No approximate methods (preserves GEMMA equivalence)
- No new runtime dependencies
- Timing is always-on (no profiling flag), acceptable for profiling phase

### Pending Todos

None.

### Blockers/Concerns

- Phase 20 gains capped at ~7% (thread pinning already inactive)
- scipy.linalg.eigh would use OpenBLAS not MKL (segfault risk >50k)
- Phase 20-22 scope may need revision based on Phase 19 findings

## Session Continuity

Last session: 2026-02-06
Stopped at: Completed Phase 19 (both plans)
Resume file: None
Next: Plan Phase 20 (thread configuration fix -- scope revised by Phase 19 findings)

---

## Milestone History

| Version | Shipped | Phases | Plans |
|---------|---------|--------|-------|
| v1.0 MVP | 2026-02-01 | 1-4.2 | 21 |
| v1.1 Covariates | 2026-02-05 | 5-10 | 39 |
| v1.2 JAX Unification | 2026-02-05 | 11-15 | 18 |
| v1.3 Tech Debt | 2026-02-06 | 16-18 | 7 |

**Cumulative:** 88 plans across 19 phases in 4 milestones
