# Roadmap: JAMMA v3.0 Advanced Mixed Models

## Overview

v3.0 extends JAMMA beyond univariate LMM to four advanced statistical modes: gene-environment interaction (GxE), variance component estimation (VC), multiple kinship matrices (multi-K), and multivariate LMM (mvLMM). Build order follows dependency chains and risk ordering -- GxE validates design matrix augmentation, VC provides the estimator Multi-K needs for K weights, and mvLMM ships last as the deepest change with no downstream dependents.

## Phases

**Phase Numbering:**
- Continues from v2.0 (Phase 29 was last)
- Integer phases (30, 31, 32, ...): Planned milestone work
- Decimal phases (30.1, 30.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 29.1: Reduce LMM Memory Overhead** - Eliminate redundant n×n copies in LMM pipeline (INSERTED) ✓ 2026-02-15
- [ ] **Phase 29.2: LMM Pipeline Bug Fixes** - Correctness and robustness fixes from code review (INSERTED)
- [ ] **Phase 29.3: Pipeline Memory Lifecycle** - Free K after eigen, lazy SNP metadata, drop pass-1 arrays, fix chunk alignment (INSERTED)
- [ ] **Phase 30: GxE Interaction Testing** - Environment-specific genetic effects via design matrix augmentation
- [ ] **Phase 31: Variance Component Estimation** - Heritability estimation via REML and HE regression
- [ ] **Phase 32: Multiple Kinship Matrices** - Multi-kernel studies with VC-based weight estimation
- [ ] **Phase 33: mvLMM Null Model** - Multivariate EM/NR optimizer and matrix Uab/Pab foundation
- [ ] **Phase 34: mvLMM Association & Validation** - Per-SNP multivariate tests with GEMMA-exact output

## Phase Details

### Phase 29.1: Reduce LMM Memory Overhead (INSERTED)

**Goal**: Eliminate redundant n×n matrix copies in the LMM pipeline to reduce peak memory during association testing, closing the gap with GEMMA's memory footprint
**Depends on**: Nothing (pure optimization of existing v2.0 LMM pipeline)
**Requirements**: None (operational improvement, not a feature requirement)
**Success Criteria** (what must be TRUE):
  1. `UT = np.ascontiguousarray(U.T)` replaced with BLAS transpose-flag approach — no 125GB copy
  2. Kinship matrix freed immediately after eigendecomp returns, before any other allocations
  3. Explicit `gc.collect()` after eigendecomp prevents LAPACK workspace from overlapping with LMM phase
  4. All existing tests pass with identical numerical results (no tolerance changes)
  5. Peak RSS during LMM phase measurably lower on 125k-sample benchmark
**Plans**: 1 plan

Plans:
- [x] 29.1-01-PLAN.md — Remove UT contiguous copy, add kinship/gc cleanup, verify tests

### Phase 29.2: LMM Pipeline Bug Fixes (INSERTED)

**Goal**: Fix correctness and robustness issues identified during code review of the LMM pipeline
**Depends on**: Phase 29.1 (memory optimization changes the same files; bug fixes should apply on top)
**Requirements**: None (operational improvement, not a feature requirement)
**Success Criteria** (what must be TRUE):
  1. Precomputed eigen files are validated against covariate-filtered sample count, not just phenotype-filtered
  2. LOCO chromosome genotype reads use chunked streaming instead of single-read for memory scalability
  3. Streaming runner produces a valid (possibly empty) output file when all SNPs are filtered
  4. `snps_indices` set conversion hoisted out of per-chromosome loop in LOCO
  5. Unused `logls_mle` accumulation removed from modes that don't use it
  6. `lmm_mode` validation produces clean ValueError in runner APIs
  7. `test_jamma_vs_gemma_synthetic` skips gracefully when Docker is unavailable instead of raising FileNotFoundError
  8. All existing tests pass with identical numerical results
**Plans**: 3 plans

Plans:
- [ ] 29.2-01-PLAN.md — Runner hygiene: dead logls_mle removal, lmm_mode validation, empty output fix, Docker test skip
- [ ] 29.2-02-PLAN.md — Pipeline correctness: covariate-aware eigen validation, snps_set hoisting
- [ ] 29.2-03-PLAN.md — LOCO memory: two-pass chunked column iteration for chromosome genotype reads

### Phase 29.3: Pipeline Memory Lifecycle (INSERTED)

**Goal**: Reduce peak memory across the full pipeline by fixing object lifecycle issues — freeing K after eigen, lazifying SNP metadata, dropping pass-1 arrays before pass-2, removing unnecessary copies, and aligning memory estimates with runtime chunk sizes
**Depends on**: Phase 29.1 (UT copy removal and gc.collect must land first; 29.3 addresses remaining lifecycle gaps)
**Requirements**: None (operational improvement, not a feature requirement)
**Success Criteria** (what must be TRUE):
  1. Pipeline sets `K = None` before calling LMM when eigenvalues/eigenvectors are already computed — no extra n×n matrix alive during association
  2. SNP metadata in streaming runner uses lazy view over meta arrays instead of materializing one dict per SNP
  3. Kinship accumulator in compute.py drops JAX device array immediately after host conversion — no transient double-copy
  4. Pass-1 statistics arrays (`all_means`, `all_miss_counts`, `all_vars`) explicitly freed before pass-2 association loop
  5. Unnecessary `.copy()` in `runner_streaming.py` association pass removed (or justified if mutation semantics require it)
  6. Pipeline passes explicit `chunk_size` to streaming runner so memory estimate matches runtime allocation
  7. All existing tests pass with identical numerical results
**Plans**: ~2-3 plans (TBD during planning)

Plans:
- [ ] 29.3-01: TBD
- [ ] 29.3-02: TBD

### Phase 30: GxE Interaction Testing
**Goal**: Users can test for gene-environment interactions using existing LMM infrastructure with augmented design matrices
**Depends on**: Nothing (reuses existing v2.0 LMM pipeline)
**Requirements**: GXE-01, GXE-02, GXE-03, GXE-04, GXE-05, GXE-06
**Success Criteria** (what must be TRUE):
  1. User can run `jamma -gxe env.txt -lmm 1` and get per-SNP GxE interaction p-values
  2. User can provide a continuous environment file (one value per sample) and JAMMA validates dimensions
  3. GxE works with all LMM test modes (`-lmm 1/2/3/4`) producing correct test-specific output
  4. GxE output matches GEMMA `-gxe` results within validation tolerances on mouse_hs1940 data
**Plans**: ~2-3 plans (TBD during planning)

Plans:
- [ ] 30-01: TBD
- [ ] 30-02: TBD

### Phase 31: Variance Component Estimation
**Goal**: Users can estimate SNP heritability and variance components from kinship and phenotype data
**Depends on**: Phase 30 (GxE validates design matrix patterns; VC wraps null model independently)
**Requirements**: VC-01, VC-02, VC-03, VC-04, VC-05, VC-06
**Success Criteria** (what must be TRUE):
  1. User can run `jamma -vc 1` and get REML-based heritability estimate with standard errors
  2. User can run `jamma -vc 2` and get Haseman-Elston regression heritability estimate with standard errors
  3. Output reports h2, sigma_g^2, sigma_e^2, and their standard errors in GEMMA-compatible format
  4. VC output matches GEMMA `-vc 1` and `-vc 2` results within validation tolerances
**Plans**: ~3-4 plans (TBD during planning)

Plans:
- [ ] 31-01: TBD
- [ ] 31-02: TBD
- [ ] 31-03: TBD

### Phase 32: Multiple Kinship Matrices
**Goal**: Users can provide multiple kinship matrices for multi-ethnic or multi-kernel GWAS with automatic variance component weight estimation
**Depends on**: Phase 31 (Multi-K requires VC estimation for finding optimal K weights)
**Requirements**: MK-01, MK-02, MK-03, MK-04, MK-05, MK-06, MK-07
**Success Criteria** (what must be TRUE):
  1. User can run `jamma -mk k_files.txt -lmm 1` and get association results using a weighted combination of kinship matrices
  2. JAMMA estimates per-K variance component weights and reports them in output
  3. Multi-K handles sequential K loading so that 100k-sample studies with 5+ matrices do not OOM
  4. After K_effective formation, standard LMM pipeline produces identical results to single-K mode
  5. Multi-K output matches GEMMA `-mk` results within validation tolerances
**Plans**: ~4-5 plans (TBD during planning)

Plans:
- [ ] 32-01: TBD
- [ ] 32-02: TBD
- [ ] 32-03: TBD
- [ ] 32-04: TBD

### Phase 33: mvLMM Null Model
**Goal**: Users can fit a multivariate null model estimating d x d genetic (Vg) and residual (Ve) covariance matrices across multiple phenotypes
**Depends on**: Phase 30 (GxE validates augmented design matrix; mvLMM is independent of VC/Multi-K)
**Requirements**: MVLMM-04, MVLMM-05, MVLMM-06, MVLMM-07, MVLMM-10
**Success Criteria** (what must be TRUE):
  1. User can select multiple phenotype columns with `-n 1 2 3` and JAMMA loads a d-phenotype matrix
  2. EM algorithm converges to Vg/Ve estimates that match GEMMA's initialization and convergence criteria
  3. Newton-Raphson acceleration after EM burn-in produces identical final Vg/Ve to GEMMA
  4. Matrix Uab/Pab computation (d x d generalization) produces numerically correct intermediate values
  5. All mvLMM code lives in `src/jamma/mvlmm/` package (no branching in existing scalar likelihood)
**Plans**: ~4-5 plans (TBD during planning)

Plans:
- [ ] 33-01: TBD
- [ ] 33-02: TBD
- [ ] 33-03: TBD
- [ ] 33-04: TBD

### Phase 34: mvLMM Association & Validation
**Goal**: Users can run multivariate association tests (Wald, LRT) with GEMMA-exact output for publication-ready multi-phenotype GWAS
**Depends on**: Phase 33 (requires null model Vg/Ve and matrix Uab/Pab infrastructure)
**Requirements**: MVLMM-01, MVLMM-02, MVLMM-03, MVLMM-08, MVLMM-09, MVLMM-11, MVLMM-12
**Success Criteria** (what must be TRUE):
  1. User can run `jamma -mvlmm 1` for multivariate Wald test and `jamma -mvlmm 2` for multivariate LRT
  2. User can run `jamma -mvlmm 4` to get both Wald and LRT results in a single pass
  3. mvLMM handles individuals with missing phenotypes for a subset of traits without crashing or silent bias
  4. Output includes genetic correlation between all phenotype pairs
  5. mvLMM output matches GEMMA `-mvlmm` results within validation tolerances on reference data
**Plans**: ~5-6 plans (TBD during planning)

Plans:
- [ ] 34-01: TBD
- [ ] 34-02: TBD
- [ ] 34-03: TBD
- [ ] 34-04: TBD
- [ ] 34-05: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 29.1 -> 29.2 -> 29.3 -> 30 -> 31 -> 32 -> 33 -> 34

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 29.1. Reduce LMM Memory Overhead (INSERTED) | 1/1 | ✓ Complete | 2026-02-15 |
| 29.2. LMM Pipeline Bug Fixes (INSERTED) | 0/3 | Not started | - |
| 29.3. Pipeline Memory Lifecycle (INSERTED) | 0/TBD | Not started | - |
| 30. GxE Interaction Testing | 0/TBD | Not started | - |
| 31. Variance Component Estimation | 0/TBD | Not started | - |
| 32. Multiple Kinship Matrices | 0/TBD | Not started | - |
| 33. mvLMM Null Model | 0/TBD | Not started | - |
| 34. mvLMM Association & Validation | 0/TBD | Not started | - |

## Coverage

31/31 v3.0 requirements mapped:

| Requirement | Phase |
|-------------|-------|
| GXE-01 | 30 |
| GXE-02 | 30 |
| GXE-03 | 30 |
| GXE-04 | 30 |
| GXE-05 | 30 |
| GXE-06 | 30 |
| VC-01 | 31 |
| VC-02 | 31 |
| VC-03 | 31 |
| VC-04 | 31 |
| VC-05 | 31 |
| VC-06 | 31 |
| MK-01 | 32 |
| MK-02 | 32 |
| MK-03 | 32 |
| MK-04 | 32 |
| MK-05 | 32 |
| MK-06 | 32 |
| MK-07 | 32 |
| MVLMM-01 | 34 |
| MVLMM-02 | 34 |
| MVLMM-03 | 34 |
| MVLMM-04 | 33 |
| MVLMM-05 | 33 |
| MVLMM-06 | 33 |
| MVLMM-07 | 33 |
| MVLMM-08 | 34 |
| MVLMM-09 | 34 |
| MVLMM-10 | 33 |
| MVLMM-11 | 34 |
| MVLMM-12 | 34 |

---
*Roadmap created: 2026-02-12*
*Last updated: 2026-02-15*
