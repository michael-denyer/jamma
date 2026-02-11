# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-10)

**Core value:** Exact GEMMA statistical results at large scale
**Current focus:** v2.0 Production GWAS

## Current Position

Milestone: v2.0 Production GWAS
Phase: Phase 25 - LOCO Kinship — COMPLETE
Plan: 3/3 complete (25-01, 25-02, 25-03 done)
Status: Phase 25 complete, ready for Phase 26
Last activity: 2026-02-11 — 25-03 LOCO validation test suite

Progress: [████████░░░░░░░░░░░░] 40% (2/5 phases)

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

**v1.3 Velocity:**

- Total plans completed: 7
- Average duration: ~8min
- Total execution time: ~56min

**v1.4:** Executed outside GSD (direct commits for memory fixes, async fixes, validation notebook, docs)

**v1.5 (Phase 23):**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| 23-01 Import Side Effects | 6min | 2 | 5 |
| 23-02 CI Tiered Testing | 2min | 2 | 27 |
| 23-03 Memory/Chunk Coupling | — | 2 | 3 |
| 23-04 Pipeline Runner | 32min | 2 | 5 |

**v2.0 (Phase 24+):**

| Plan                                       | Duration | Tasks | Files |
|--------------------------------------------|----------|-------|-------|
| 24-01 SNP Filter Searchsorted              | 2min     | 2     | 3     |
| 24-02 Missingness Tests + Memory Comments  | 3min     | 2     | 2     |
| 25-01 LOCO Kinship Computation             | 10min    | 2     | 4     |
| 25-02 LOCO LMM Orchestrator + Integration  | 9min     | 2     | 7     |
| 25-03 LOCO Validation Test Suite           | 25min    | 2     | 1     |

## Accumulated Context

### Decisions

All milestone decisions archived in:
- .planning/v1-MILESTONE-ARCHIVE.md (v1.0)
- .planning/milestones/v1.3-ROADMAP.md (v1.3)
- .planning/PROJECT.md Key Decisions table (cumulative)

### v1.4 Key Results

- **85k GEMMA validation**: 91,613 SNPs, 100% significance agreement, Spearman rho 1.0
- **Memory fix**: Phase-specific LMM memory estimates (was demanding 320GB pipeline peak, needed 96GB)
- **JAX async fix**: block_until_ready() for accurate progress/timing
- **Eigendecomp at theoretical floor**: 90k at 32 cores ≈ 3,100s, matching ~310 GFLOPS effective
- **scipy reverted**: numpy.linalg.eigh used (scipy ILP64 not viable)

### v1.5 Decisions

- Loguru default stderr handler retained for library users; CLI setup_logging() handles reconfiguration
- ensure_jax_configured() wraps configure_jax() with sentinel, preserving backward compatibility
- Negative marker selection for test-fast: unmarked tests default to fast tier
- Pre-commit in CI lint job for exact parity with local hooks
- test-slow gated on master merge and workflow_dispatch only
- PipelineRunner raises exceptions; CLI catches and converts to typer.Exit (23-04)
- CLI retains kinship-required check; API can compute kinship from genotypes (23-04)
- Intercept warning moved into PipelineRunner.load_covariates for single source of truth (23-04)

### v2.0 Decisions

- np.searchsorted with side="left" for half-open [start, end) chunk filtering on sorted snp_indices (24-01)
- No functional changes to memory.py for comment updates -- docstrings only (24-02)
- Generator (Iterator) return type for LOCO kinship -- yields one K_loco at a time for memory efficiency (25-01)
- Streaming LOCO accumulates all S_chr simultaneously in single pass -- trades O(n_chr * n^2) memory for O(1) passes (25-01)
- Global SNP filtering before chromosome partitioning; chromosome array co-filtered with same mask (25-01)
- Per-chromosome single BED read in _run_lmm_for_chromosome; chromosome subsets small enough for memory (25-02)
- -k and -loco mutually exclusive; LOCO computes kinship internally (25-02)
- LOCO pipeline branch skips standard kinship loading entirely (25-02)
- rtol=1e-9 validated bound for LOCO subtraction identity (batched JAX FP accumulation) (25-03)
- Mathematical self-consistency as LOCO validation strategy (no GEMMA reference available) (25-03)

### v2.0 Roadmap Structure

**5 phases, 24 requirements:**

- Phase 24: Quality and Cleanup (4 reqs) - performance fix, test coverage, docs
- Phase 25: LOCO Kinship (4 reqs) - chromosome-specific kinship
- Phase 26: Eigendecomposition Reuse (4 reqs) - `-d` and `-u` flags
- Phase 27: Phenotype Selection and Standardized Kinship (5 reqs) - `-n` flag, `-gk 2`
- Phase 28: Filtering and Input Validation (7 reqs) - `-snps`, `-ksnps`, `-hwe`, PLINK checks

**Phase dependency chain:**
Phase 24 (no deps) → Phase 25 → Phase 26 → Phase 27 → Phase 28

### Pending Todos

None.

### Blockers/Concerns

None.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | Vectorize per-SNP imputation loop in streaming runner | 2026-02-09 | c222376 | [1-vectorize-per-snp-imputation-loop-in-str](./quick/1-vectorize-per-snp-imputation-loop-in-str/) |
| 2 | Add top-level jamma.gwas() API function | 2026-02-09 | 3fd94cf | [2-add-top-level-jamma-gwas-api-function](./quick/2-add-top-level-jamma-gwas-api-function/) |

## Session Continuity

Last session: 2026-02-11
Stopped at: Completed 25-03-PLAN.md (Phase 25 complete)
Resume file: None
Next: `/gsd:execute-phase 26` (Phase 26: Eigendecomposition Reuse)

---

## Milestone History

| Version | Shipped | Phases | Plans |
|---------|---------|--------|-------|
| v1.0 MVP | 2026-02-01 | 1-4.2 | 21 |
| v1.1 Covariates | 2026-02-05 | 5-10 | 39 |
| v1.2 JAX Unification | 2026-02-05 | 11-15 | 18 |
| v1.3 Tech Debt | 2026-02-06 | 16-18 | 7 |
| v1.4 Performance | 2026-02-10 | 19-22 | (direct commits) |
| v1.5 Tests & Architecture | 2026-02-10 | 23 | 4 |
| v2.0 Production GWAS | (planned) | 24-28 | TBD |

**Cumulative:** 89 GSD plans + v1.4 direct work across 23 phases in 6 milestones
