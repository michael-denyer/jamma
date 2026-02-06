# Roadmap: JAMMA

## Milestones

- v1.0 MVP - Phases 1-4.2 (shipped 2026-02-01)
- v1.1 Covariates & Extended Tests - Phases 5-10 (shipped 2026-02-05)
- v1.2 JAX Runner Unification - Phases 11-15 (shipped 2026-02-05)
- v1.3 Tech Debt Cleanup - Phases 16-18 (shipped 2026-02-06)
- v1.4 Performance - Phases 19-22 (in progress)
- v2.0 Extended Features - Phases 23+ (planned)

## Overview

JAMMA (JAX-Accelerated Mixed Model Association) delivers GEMMA-equivalent statistical output at 200k sample scale on single-node cloud VMs. v1.0-v1.3 built the complete, validated, clean codebase (394 tests, 7366 lines). v1.4 targets the performance ceiling: minimize wall-clock time for eigendecomposition and LMM association on CPU, targeting 90k samples x 90k SNPs on a Databricks VM with MKL ILP64. Current runtime is ~150 min; theoretical floor is ~50-90 min depending on core count.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

<details>
<summary>v1.0 MVP (Phases 1-4.2) - SHIPPED 2026-02-01</summary>

### Phase 1: Foundation
**Goal**: Establish project infrastructure with PLINK binary input parsing, JAX setup, and validation framework
**Depends on**: Nothing (first phase)
**Requirements**: IO-01, IO-02, IO-03, IO-06, VAL-01, VAL-02
**Success Criteria** (what must be TRUE):
  1. User can load PLINK binary files (.bed/.bim/.fam) via `-bfile` flag
  2. Output files are written to specified directory with specified prefix
  3. Log file captures execution parameters and timing
  4. JAX environment configured with XLA compilation working
  5. Validation framework can compare output files to GEMMA reference within tolerance
  6. CI runs automated tests on every commit
**Plans**: 6 plans

Plans:
- [x] 01-01-PLAN.md — Project structure and legacy migration
- [x] 01-02-PLAN.md — PLINK I/O module with bed-reader
- [x] 01-03-PLAN.md — Validation framework and tolerances
- [x] 01-04-PLAN.md — CLI with Typer and logging
- [x] 01-05-PLAN.md — JAX configuration and verification
- [x] 01-06-PLAN.md — CI pipeline and integration tests

### Phase 2: Kinship
**Goal**: Compute centered relatedness matrix using JAX, validate against GEMMA output
**Depends on**: Phase 1
**Requirements**: CORE-01, CORE-04, IO-04
**Success Criteria** (what must be TRUE):
  1. User can compute kinship matrix with `-gk 1` (centered XX'/p)
  2. Kinship matrix output (.cXX.txt) matches GEMMA within 1e-8 relative tolerance
  3. Kinship computation handles missing genotype values identically to GEMMA
  4. JAX implementation is JIT-compiled and faster than naive NumPy
  5. Works on test dataset (n=1k, p=10k) with CI verification
  6. **Decision gate**: Document JAX vs GEMMA divergence — if >1e-6, evaluate SciPy fallback
**Plans**: 4 plans

Plans:
- [x] 02-01-PLAN.md — Kinship computation core (JAX with missing data)
- [x] 02-02-PLAN.md — Kinship I/O and CLI integration
- [x] 02-03-PLAN.md — Validation and benchmarks
- [x] 02-04-PLAN.md — Decision gate and CI integration

### Phase 3: LMM Association
**Goal**: Run univariate LMM Wald test matching GEMMA's exact output
**Depends on**: Phase 2
**Requirements**: CORE-02, CORE-03, IO-05
**Success Criteria** (what must be TRUE):
  1. User can load pre-computed kinship matrix via `-k` flag
  2. User can run univariate LMM with `-lmm 1` (Wald test)
  3. Association results (.assoc.txt) match GEMMA: beta within 1e-6, p-values within 1e-8
  4. Complete `-gk 1` followed by `-lmm 1` workflow produces identical results to GEMMA
**Plans**: 5 plans

Plans:
- [x] 03-01-PLAN.md — Core math: scipy dependency, eigendecomposition, REML likelihood
- [x] 03-02-PLAN.md — Optimization and statistics: lambda via Brent, Wald test
- [x] 03-03-PLAN.md — I/O and public API: .assoc.txt output, run_lmm_association()
- [x] 03-04-PLAN.md — CLI integration: -k flag, full -lmm 1 workflow
- [x] 03-05-PLAN.md — Validation and tests: compare_assoc_results, unit tests

### Phase 4: Scale (Superseded)
**Goal**: Handle 200k samples x 95k SNPs on 512GB cloud VM without OOM
**Depends on**: Phase 3
**Requirements**: SCALE-01, SCALE-02
**Status**: Superseded by Phases 4.1 + 4.2

### Phase 4.1: Streaming I/O Pipeline (INSERTED)
**Goal**: Stream genotypes from disk in chunks to avoid materializing full n*p matrix
**Depends on**: Phase 4
**Requirements**: SCALE-01
**Success Criteria** (what must be TRUE):
  1. Genotypes read in SNP chunks via bed-reader windowed reads (no full matrix load)
  2. Peak memory dominated by kinship (n^2) not genotypes (n*p)
  3. Memory check uses actual chunk size and validates before any allocation
  4. 200k*95k workflow runs on 512GB VM (same as Phase 4 target)
**Plans**: 3 plans

Plans:
- [x] 04.1-01-PLAN.md — Streaming iterator and memory estimation foundation
- [x] 04.1-02-PLAN.md — Streaming kinship computation
- [x] 04.1-03-PLAN.md — Streaming LMM association and integration tests

### Phase 4.2: JAX Runtime Optimization (INSERTED)
**Goal**: Eliminate JAX recompilation, optimize device memory, and align memory checks with actual runtime behavior
**Depends on**: Phase 4.1
**Requirements**: SCALE-01
**Success Criteria** (what must be TRUE):
  1. Memory checks account for actual runtime arrays: Uab + (n_grid * chunk_size) + rotation buffers
  2. Last chunk padded to fixed size to avoid JAX recompilation for smaller tail
  3. Shared arrays (U, UtW, Uty, eigenvalues) kept on device throughout run (no repeated device_put)
  4. Single-chunk vs multi-chunk equivalence test validates chunking produces identical results
  5. Memory-budget test verifies chunk_size reduces when available memory is mocked low
**Plans**: 3 plans

Plans:
- [x] 04.2-01-PLAN.md — Accurate memory model with n_grid and runtime buffers
- [x] 04.2-02-PLAN.md — Fixed-shape chunks and device residency optimization
- [x] 04.2-03-PLAN.md — Memory budget and chunk equivalence tests

</details>

<details>
<summary>v1.1 Covariates & Extended Tests (Phases 5-10) - SHIPPED 2026-02-05</summary>

### Phase 5: Covariate Infrastructure
**Goal**: Enable covariate file input and integrate covariates through the full LMM pipeline
**Depends on**: Phase 4.2
**Requirements**: COV-01, COV-02, COV-03, COV-04
**Plans**: 4 plans

Plans:
- [x] 05-01-PLAN.md — Covariate file I/O
- [x] 05-02-PLAN.md — LMM covariate integration
- [x] 05-03-PLAN.md — CLI -c flag
- [x] 05-04-PLAN.md — Validation & GEMMA reference

### Phase 5.1: Memory Reliability (INSERTED)
**Goal**: Prevent OOM failures at scale
**Depends on**: Phase 5
**Plans**: 4 plans

Plans:
- [x] 05.1-01-PLAN.md — Pre-flight eigendecomp memory check
- [x] 05.1-02-PLAN.md — RSS memory logging at workflow boundaries
- [x] 05.1-03-PLAN.md — Incremental result writer
- [x] 05.1-04-PLAN.md — Test tier markers

### Phase 5.2: LMM Memory Completeness (INSERTED)
**Goal**: Complete memory pre-flight checks for all LMM code paths
**Depends on**: Phase 5.1
**Plans**: 6 plans

Plans:
- [x] 05.2-01-PLAN.md — estimate_lmm_memory() standalone helper
- [x] 05.2-02-PLAN.md — Safe chunk size default
- [x] 05.2-03-PLAN.md — CLI memory pre-flight check
- [x] 05.2-04-PLAN.md — NumPy runner output_path
- [x] 05.2-05-PLAN.md — CLI output_path integration
- [x] 05.2-06-PLAN.md — JAX streaming per-chunk write

### Phase 5.3: Allele Semantics & Beta Sign (INSERTED)
**Goal**: GEMMA-exact allele orientation
**Depends on**: Phase 5.2
**Plans**: 2 plans

Plans:
- [x] 05.3-01-PLAN.md — Fix AF output
- [x] 05.3-02-PLAN.md — Validation comparison fix

### Phase 6: Score Test
**Goal**: Implement efficient Score test using null model lambda
**Depends on**: Phase 5.3
**Requirements**: TEST-02, OUT-02
**Plans**: 3 plans

Plans:
- [x] 06-01-PLAN.md — Core Score test math
- [x] 06-02-PLAN.md — Runner + CLI integration
- [x] 06-03-PLAN.md — GEMMA reference and validation

### Phase 6.1: Rust Eigendecomposition Backend (INSERTED)
**Goal**: Drop-in Rust/faer eigendecomposition
**Depends on**: Phase 6
**Plans**: 4 plans

Plans:
- [x] 06.1-01-PLAN.md — Rust crate setup
- [x] 06.1-02-PLAN.md — PyO3 bindings
- [x] 06.1-03-PLAN.md — Python integration
- [x] 06.1-04-PLAN.md — Validation and CI

### Phase 6.2: Dual Backend Cleanup (INSERTED)
**Goal**: Clarify backend naming
**Depends on**: Phase 6.1
**Plans**: 4 plans

Plans:
- [x] 06.2-01-PLAN.md — Backend type refactoring
- [x] 06.2-02-PLAN.md — CLI -be flag
- [x] 06.2-03-PLAN.md — Test updates
- [x] 06.2-04-PLAN.md — Backend documentation

### Phase 7: Likelihood Ratio Test
**Goal**: Implement LRT using MLE likelihood with chi-squared statistics
**Depends on**: Phase 6
**Requirements**: TEST-01, OUT-01
**Plans**: 3 plans

Plans:
- [x] 07-01-PLAN.md — Core MLE math
- [x] 07-02-PLAN.md — Runner + CLI integration
- [x] 07-03-PLAN.md — Validation tests

### Phase 7.1: Null Model Discrepancy Fix (INSERTED)
**Goal**: Resolve GEMMA CalcPVEnull vs JAMMA REML discrepancy
**Depends on**: Phase 7
**Plans**: 2 plans

Plans:
- [x] 07.1-01-PLAN.md — Null model likelihood fix
- [x] 07.1-02-PLAN.md — Remove xfail markers

### Phase 8: All-Tests Mode & Validation
**Goal**: Combined output and GEMMA equivalence validation
**Depends on**: Phase 7
**Requirements**: TEST-03, OUT-03, VAL-03, VAL-04
**Plans**: 3 plans

Plans:
- [x] 08-01-PLAN.md — Core -lmm 4 implementation
- [x] 08-02-PLAN.md — Fixture generation and validation framework
- [x] 08-03-PLAN.md — Validation tests

### Phase 10: Remove Rust/faer Backend
**Goal**: Remove unreliable Rust eigendecomp, simplify to single backend
**Depends on**: Phase 8
**Plans**: 2 plans

Plans:
- [x] 10-01-PLAN.md — Simplify Python source
- [x] 10-02-PLAN.md — Delete Rust files and clean up

### Phase 9: GEMMA Equivalence Validation Framework
**Goal**: Rigorous equivalence testing with canonical fixtures
**Depends on**: Phase 8
**Plans**: TBD

Plans:
- [ ] TBD

</details>

<details>
<summary>v1.2 JAX Runner Unification (Phases 11-15) - SHIPPED 2026-02-05</summary>

### Phase 11: JAX Covariate Generalization
**Goal**: JAX likelihood functions and runner handle arbitrary covariate counts, not just intercept-only
**Depends on**: Phase 10
**Requirements**: JAX-04, JAX-05
**Plans**: 4 plans

Plans:
- [x] 11-01-PLAN.md — Index tables, Uab, and Pab generalization
- [x] 11-02-PLAN.md — REML, MLE, Wald, and optimizer generalization
- [x] 11-03-PLAN.md — Runner covariate integration
- [x] 11-04-PLAN.md — Validation tests and regression suite

### Phase 12: JAX Score & LRT Tests
**Goal**: JAX runner computes Score test and LRT statistics using batch processing
**Depends on**: Phase 11
**Requirements**: JAX-01, JAX-02
**Plans**: 3 plans

Plans:
- [x] 12-01-PLAN.md — Score stats and MLE optimizer JAX functions
- [x] 12-02-PLAN.md — Runner mode dispatch for Score and LRT
- [x] 12-03-PLAN.md — Validation tests for Score and LRT parity

### Phase 13: JAX All-Tests Mode
**Goal**: JAX runner produces all three test statistics (Wald + LRT + Score) in a single pass
**Depends on**: Phase 12
**Requirements**: JAX-03
**Plans**: 2 plans

Plans:
- [x] 13-01-PLAN.md — Mode 4 implementation in both JAX runners
- [x] 13-02-PLAN.md — Validation tests for JAX all-tests mode parity

### Phase 14: CLI JAX Integration
**Goal**: CLI uses streaming JAX runner as the default execution path with NumPy fallback
**Depends on**: Phase 13
**Requirements**: JAX-07, JAX-08, JAX-09
**Plans**: 2 plans

Plans:
- [x] 14-01-PLAN.md — Rewire CLI to streaming JAX runner with --backend flag
- [x] 14-02-PLAN.md — Update tests for JAX default and backend flag coverage

### Phase 15: JAX-GEMMA Validation & NumPy Removal
**Goal**: Validate JAX runner output directly against GEMMA reference fixtures across the full option matrix, then remove the NumPy runner code path
**Depends on**: Phase 14
**Requirements**: JAX-06, JAX-10, JAX-11, JAX-12
**Plans**: 5 plans

Plans:
- [x] 15-01-PLAN.md — Download mouse_hs1940 PLINK data and generate all 9 missing GEMMA fixtures
- [x] 15-02-PLAN.md — Rewrite test_lmm_validation.py (40+ calls) + add mouse_hs1940 GEMMA tests
- [x] 15-03-PLAN.md — Rewrite test_runner_jax + 4 other test files
- [x] 15-04-PLAN.md — Rewrite test_scale + test_hypothesis + chunk-size invariance test
- [x] 15-05-PLAN.md — Remove NumPy runner, CLI --backend flag, and dead code

</details>

<details>
<summary>v1.3 Tech Debt Cleanup (Phases 16-18) - SHIPPED 2026-02-06</summary>

- [x] Phase 16: Dead Code Removal (2/2 plans) -- completed 2026-02-06
- [x] Phase 17: Module Restructuring (3/3 plans) -- completed 2026-02-06
- [x] Phase 18: Correctness & Performance (2/2 plans) -- completed 2026-02-06

See: .planning/milestones/v1.3-ROADMAP.md for full details.

</details>

### v1.4 Performance (In Progress)

**Milestone Goal:** Minimize eigendecomp + LMM association wall-clock time on CPU. Target: 90k samples x 90k SNPs on Databricks VM with MKL ILP64. Current ~150 min total; theoretical floor ~50-90 min.

**Key insight:** Two operations dominate: eigendecomp (~40% FLOPS) and UT@G rotation (~60% FLOPS). Everything else is <1%. A confirmed thread-pinning bug may be forcing single-threaded MKL execution, meaning the fix could deliver 8-32x speedup -- or the bug may be inactive on Databricks, capping gains at ~7%. Phase 19 resolves this unknown before committing to optimization scope.

- [x] **Phase 19: Measure and Diagnose** - Instrument profiling, get baseline timings, determine actual MKL thread state (completed 2026-02-06)
- [ ] **Phase 20: Thread Configuration Fix** - Fix MKL thread pinning, benchmark optimal thread count
- [ ] **Phase 21: Scipy Eigendecomp Switch** - Explicit LAPACK driver, memory optimization, adaptive driver selection
- [ ] **Phase 22: Validation and Micro-Optimization** - Post-optimization profiling, equivalence verification, documentation

## Phase Details

### Phase 19: Measure and Diagnose
**Goal**: Establish empirical baseline -- actual MKL thread count, per-phase wall-clock breakdown, and LAPACK driver identification at 90k scale on Databricks
**Depends on**: Phase 18 (v1.3 complete)
**Requirements**: PROF-01, PROF-02, PROF-03, PROF-04
**Success Criteria** (what must be TRUE):
  1. Running `jamma lmm` at 90k scale produces per-phase timing in the log: eigendecomp, UT@G rotation, JAX compute, I/O read, SNP stats, and result write as separate wall-clock measurements
  2. Actual MKL thread count after JAMMA import on Databricks is known and documented (threadpool_info output)
  3. Which LAPACK driver numpy.linalg.eigh invokes on the target VM is confirmed (dsyevd vs dsyevr via MKL_VERBOSE=1)
  4. Baseline wall-clock numbers at 90k exist for A/B comparison with subsequent phases
**Plans**: 2 plans

Plans:
- [x] 19-01-PLAN.md — Add per-phase timing instrumentation at 6 measurement points in runner_streaming.py
- [x] 19-02-PLAN.md — Databricks diagnostic notebook (thread count, LAPACK driver, baseline 90k timing)

### Phase 20: Thread Configuration Fix
**Goal**: MKL BLAS operations (eigendecomp and UT@G rotation) run multi-threaded, while JAX phases remain pinned to 1 thread
**Depends on**: Phase 19
**Requirements**: THRD-01, THRD-02, THRD-03
**Success Criteria** (what must be TRUE):
  1. Eigendecomp and UT@G rotation execute with MKL using multiple threads (verified via threadpool_info inside context manager)
  2. JAX JIT-compiled functions still execute with BLAS pinned to 1 thread (no JAX recompilation or numerical drift)
  3. Optimal thread count for the target VM is determined via empirical sweep and documented
  4. Wall-clock time at 90k is measurably faster than Phase 19 baseline (with measured speedup ratio)
  5. All 394 tests pass unchanged (no tolerance relaxation)
**Plans**: TBD

Plans:
- [ ] 20-01: Add threadpool_limits context managers and restructure thread management
- [ ] 20-02: Databricks thread count benchmark sweep (1/4/8/16/32/64) and speedup measurement

### Phase 21: Scipy Eigendecomp Switch
**Goal**: Eigendecomposition uses scipy with explicit LAPACK driver selection, saving ~65 GB peak memory and enabling adaptive algorithm choice
**Depends on**: Phase 20
**Requirements**: EIGEN-01, EIGEN-02, EIGEN-03, EIGEN-04
**Success Criteria** (what must be TRUE):
  1. Eigendecomp calls scipy.linalg.eigh with driver='evd', overwrite_a=True, check_finite=False instead of numpy.linalg.eigh
  2. On VMs where dsyevd workspace does not fit in RAM, eigendecomp automatically falls back to dsyevr driver
  3. estimate_eigendecomp_memory() reports workspace matching the actual LAPACK driver (within 10% of true allocation)
  4. K matrix is explicitly deleted after eigendecomp, reducing LMM-phase peak memory by n^2 * 8 bytes (~64.8 GB at 90k)
  5. All 394 tests pass with eigenvalues and association results within existing calibrated tolerances
**Plans**: TBD

Plans:
- [ ] 21-01: Replace numpy.linalg.eigh with scipy.linalg.eigh and adaptive driver selection
- [ ] 21-02: Fix memory estimator and add post-eigendecomp memory reclamation

### Phase 22: Validation and Micro-Optimization
**Goal**: Confirm total speedup, verify GEMMA equivalence end-to-end, and document the performance floor
**Depends on**: Phase 21
**Requirements**: VALID-01, VALID-02, VALID-03
**Success Criteria** (what must be TRUE):
  1. Post-optimization 90k run produces per-phase timing that can be compared side-by-side with Phase 19 baseline
  2. All 394 tests pass with zero tolerance changes from pre-v1.4 values
  3. A performance summary documents: achieved speedup (total and per-phase), remaining bottleneck breakdown, and theoretical floor distance
**Plans**: TBD

Plans:
- [ ] 22-01: Post-optimization Databricks profiling run and GEMMA equivalence sweep
- [ ] 22-02: Performance summary documentation and optional micro-optimizations

## Progress

**Execution Order:**
Phases execute in numeric order: 19 -> 20 -> 21 -> 22

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v1.0 MVP | 1-4.2 | 21 | Complete | 2026-02-01 |
| v1.1 Covariates & Extended Tests | 5-10 | 39 | Complete | 2026-02-05 |
| v1.2 JAX Runner Unification | 11-15 | 18 | Complete | 2026-02-05 |
| v1.3 Tech Debt Cleanup | 16-18 | 7 | Complete | 2026-02-06 |
| v1.4 Performance | 19-22 | ~8 | In Progress | - |
| v2.0 Extended Features | 23+ | TBD | Planned | - |

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 19. Measure and Diagnose | 2/2 | Complete | 2026-02-06 |
| 20. Thread Configuration Fix | 0/2 | Not started | - |
| 21. Scipy Eigendecomp Switch | 0/2 | Not started | - |
| 22. Validation and Micro-Optimization | 0/2 | Not started | - |

---
*Roadmap created: 2026-01-31*
*v1.0 shipped: 2026-02-01*
*v1.1 shipped: 2026-02-05*
*v1.2 shipped: 2026-02-05*
*v1.3 shipped: 2026-02-06*
*v1.4 roadmap added: 2026-02-06*
