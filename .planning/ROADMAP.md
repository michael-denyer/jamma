# Roadmap: JAMMA

## Milestones

- v1.0 MVP - Phases 1-4.2 (shipped 2026-02-01)
- v1.1 Covariates & Extended Tests - Phases 5-10 (shipped 2026-02-05)
- v1.2 JAX Runner Unification - Phases 11-15 (shipped 2026-02-05)
- v1.3 Tech Debt Cleanup - Phases 16-18 (shipped 2026-02-06)
- v1.5 Tests and Architecture - Phase 23 (shipped 2026-02-10)
- v2.0 Production GWAS - Phases 24-28 (planned)

## Overview

JAMMA (JAX-Accelerated Mixed Model Association) delivers GEMMA-equivalent statistical output at 200k sample scale on single-node cloud VMs. v1.0 established core LMM (Wald test) with exact numerical equivalence. v1.1 completed statistical test coverage (LRT, Score, all-tests) and covariate support. v1.2 unifies the fast JAX runner as the production default for all test modes and covariates, closing the gap between the advertised JAX performance and the NumPy-based CLI. v1.3 eliminates dead code left from the NumPy-to-JAX migration, restructures oversized modules, and adds targeted performance and correctness improvements. v1.5 eliminates import-time side effects, enforces CI test tiers, and extracts shared pipeline orchestration. v2.0 adds production GWAS features that researchers expect for real analyses.

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
**Success Criteria** (what must be TRUE):
  1. `compute_uab_jax` and `batch_compute_uab` accept UtW with arbitrary column count (not just 1)
  2. `calc_pab_jax` produces correct projections for n_cvt=1,2,3 (Uab/Pab shapes scale with `(n_cvt+3)(n_cvt+2)/2` columns and `n_cvt+2` rows)
  3. JAX runner accepts covariate matrix and constructs multi-column W (intercept + user covariates)
  4. Wald test via JAX runner with covariates produces results within tolerance of NumPy runner with same covariates (beta 1e-6, p_wald 1e-8)
  5. Existing intercept-only JAX tests produce identical results (no regression)
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
**Success Criteria** (what must be TRUE):
  1. JAX runner with `lmm_mode=3` computes null model MLE lambda once, then batch Score statistics (no per-SNP optimization)
  2. JAX runner with `lmm_mode=2` computes null MLE once, then per-SNP MLE optimization using batch golden section on MLE likelihood
  3. Score p-values from JAX runner match NumPy runner within 1e-8 tolerance
  4. LRT p-values from JAX runner match NumPy runner within 2e-3 tolerance (chi-squared magnification)
  5. Both modes work with covariates (n_cvt > 1) and produce correct output columns
**Plans**: 3 plans

Plans:

- [x] 12-01-PLAN.md — Score stats and MLE optimizer JAX functions
- [x] 12-02-PLAN.md — Runner mode dispatch for Score and LRT
- [x] 12-03-PLAN.md — Validation tests for Score and LRT parity

### Phase 13: JAX All-Tests Mode
**Goal**: JAX runner produces all three test statistics (Wald + LRT + Score) in a single pass
**Depends on**: Phase 12
**Requirements**: JAX-03
**Success Criteria** (what must be TRUE):
  1. JAX runner with `lmm_mode=4` outputs beta, se, logl_H1, l_remle, l_mle, p_wald, p_lrt, p_score in one pass
  2. Null MLE computed once before SNP loop, shared by LRT and Score paths
  3. Per-SNP REML optimization (Wald) and MLE optimization (LRT) both run in batch
  4. All-tests output matches NumPy runner all-tests output within existing tolerances
**Plans**: 2 plans

Plans:
- [x] 13-01-PLAN.md — Mode 4 implementation in both JAX runners
- [x] 13-02-PLAN.md — Validation tests for JAX all-tests mode parity

### Phase 14: CLI JAX Integration
**Goal**: CLI uses streaming JAX runner as the default execution path with NumPy fallback
**Depends on**: Phase 13
**Requirements**: JAX-07, JAX-08, JAX-09
**Success Criteria** (what must be TRUE):
  1. `jamma lmm -lmm 1` uses streaming JAX runner by default (not NumPy `run_lmm_association`)
  2. `jamma lmm -lmm 2/3/4` passes mode through to JAX runner and produces correct output
  3. `jamma lmm --backend numpy` (or similar flag) falls back to NumPy runner for debugging
  4. Streaming JAX runner handles covariate file from CLI (`-c` flag) through to JAX covariate path
  5. Existing CLI integration tests pass with JAX runner as default
**Plans**: 2 plans

Plans:
- [x] 14-01-PLAN.md — Rewire CLI to streaming JAX runner with --backend flag
- [x] 14-02-PLAN.md — Update tests for JAX default and backend flag coverage

### Phase 15: JAX-GEMMA Validation & NumPy Removal

**Goal**: Validate JAX runner output directly against GEMMA reference fixtures across the full option matrix (all modes x covariate configs x datasets), then remove the NumPy runner code path
**Depends on**: Phase 14
**Requirements**: JAX-06, JAX-10, JAX-11, JAX-12

**Success Criteria** (what must be TRUE):

  1. All 16 cells in the test matrix above have GEMMA reference fixtures (generated via Docker)
  2. For every cell: JAX runner output matches GEMMA fixture within calibrated tolerances (see `ToleranceConfig`)
  3. Chunk-size invariance: changing chunk_size does not change JAX results beyond floating-point noise
  4. NumPy runner (`run_lmm_association`, `stats.py` NumPy path) removed from codebase
  5. `--backend numpy` CLI flag removed; single JAX execution path
  6. All existing tests updated to validate JAX directly against GEMMA reference (not NumPy runner)
  7. CI gates all GEMMA parity checks on every commit

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

- [x] Phase 16: Dead Code Removal (2/2 plans) — completed 2026-02-06
- [x] Phase 17: Module Restructuring (3/3 plans) — completed 2026-02-06
- [x] Phase 18: Correctness & Performance (2/2 plans) — completed 2026-02-06

See: .planning/milestones/v1.3-ROADMAP.md for full details.

</details>

<details>
<summary>v1.5 Tests and Architecture (Phase 23) - SHIPPED 2026-02-10</summary>

### Phase 23: Tests and Architecture

**Goal:** Enforce CI test tiers, eliminate import-time side effects, align CI quality gates with pre-commit, couple memory estimates to runtime chunking, and extract shared pipeline orchestration from CLI/API duplication
**Depends on:** Phase 22
**Requirements:** Beads jamma-52p, jamma-h1m, jamma-7xq, jamma-71z, jamma-7jh
**Success Criteria** (what must be TRUE):
  1. CI runs fast (unit) and slow (integration/scale) test tiers in separate jobs
  2. Importing `jamma` has zero side effects (no handler removal, no JAX config)
  3. CI enforces ruff check + format + all pre-commit hooks
  4. Memory estimation uses the same chunk size that the JAX runner actually allocates
  5. CLI (`cli.py`) and Python API (`gwas.py`) share a single `PipelineRunner` service — no duplicated orchestration
**Plans:** 4 plans

Plans:
- [x] 23-01-PLAN.md — Eliminate import-time side effects (loguru + JAX)
- [x] 23-02-PLAN.md — CI test tiers and pre-commit parity
- [x] 23-03-PLAN.md — Couple memory estimation to runtime chunk size
- [x] 23-04-PLAN.md — Extract PipelineRunner service from CLI/API duplication

</details>

## v2.0 Production GWAS (Phases 24-28)

### Phase 24: Quality and Cleanup — COMPLETE (2026-02-11)
**Goal**: Fix streaming performance, expand test coverage, and update documentation
**Depends on**: Phase 23
**Requirements**: PERF-01, TEST-01, TEST-02, DOC-01
**Plans**: 2 plans

Plans:
- [x] 24-01-PLAN.md — Replace linear SNP filtering with searchsorted in streaming modules
- [x] 24-02-PLAN.md — Missingness pattern tests and memory model comment cleanup

### Phase 25: LOCO Kinship
**Goal**: Enable leave-one-chromosome-out kinship computation for reduced proximal contamination
**Depends on**: Phase 24
**Requirements**: LOCO-01, LOCO-02, LOCO-03, LOCO-04
**Success Criteria** (what must be TRUE):
  1. User can compute LOCO kinship with `-loco` flag that generates one kinship matrix per chromosome
  2. Chromosome annotations are read from .bim file and used to partition SNPs by chromosome
  3. Each chromosome-specific kinship matrix excludes all SNPs from that chromosome
  4. LOCO kinship satisfies mathematical self-consistency (subtraction identity, symmetry, PSD, manual computation equivalence)
  5. LMM association with `-loco` uses the appropriate chromosome-specific kinship for each SNP
**Plans**: 3 plans

Plans:
- [ ] 25-01-PLAN.md — Chromosome partitioning and LOCO kinship computation (subtraction approach)
- [ ] 25-02-PLAN.md — LOCO LMM orchestrator, pipeline, and CLI integration
- [ ] 25-03-PLAN.md — LOCO validation tests (mathematical self-consistency and integration)

### Phase 26: Eigendecomposition Reuse
**Goal**: Enable loading pre-computed eigendecomposition for multi-phenotype workflows
**Depends on**: Phase 25
**Requirements**: EIGEN-01, EIGEN-02, EIGEN-03, EIGEN-04
**Success Criteria** (what must be TRUE):
  1. User can load pre-computed eigenvalues from file via `-d eigenvalues.txt`
  2. User can load pre-computed eigenvectors from file via `-u eigenvectors.txt`
  3. Eigendecomposition outputs (.eigenD.txt, .eigenU.txt) written in GEMMA-compatible space-separated text format
  4. Loading pre-computed eigen with `-d` and `-u` skips eigendecomposition entirely (kinship matrix not required)
  5. LMM results with pre-computed eigen match results from fresh kinship+eigen workflow
**Plans**: TBD

### Phase 27: Phenotype Selection and Standardized Kinship
**Goal**: Enable phenotype column selection and standardized kinship computation
**Depends on**: Phase 26
**Requirements**: PHENO-01, PHENO-02, KIN-01, KIN-02, KIN-03
**Success Criteria** (what must be TRUE):
  1. User can select phenotype column from .fam file via `-n 2` (1-based column index)
  2. Multi-phenotype .fam files with columns beyond standard 6 are parsed correctly
  3. User can compute standardized kinship with `-gk 2`
  4. Standardized kinship applies formula (X - mean) / sqrt(p*(1-p)) before outer product
  5. Standardized kinship output matches GEMMA `-gk 2` within calibrated tolerances
**Plans**: TBD

### Phase 28: Filtering and Input Validation
**Goal**: Enable SNP subset selection, HWE filtering, and PLINK dimension validation
**Depends on**: Phase 27
**Requirements**: SNP-01, SNP-02, SNP-03, QC-01, QC-02, VALID-01, VALID-02
**Success Criteria** (what must be TRUE):
  1. User can restrict association testing to SNP list via `-snps snplist.txt`
  2. User can restrict kinship computation to SNP list via `-ksnps snplist.txt`
  3. SNP list files contain one RS ID per line (whitespace-trimmed)
  4. User can exclude SNPs failing HWE test via `-hwe 1e-6` (chi-squared p-value threshold)
  5. HWE p-value computed from chi-squared test on observed vs expected genotype counts
  6. PLINK loader validates genotype matrix dimensions match .fam sample count and .bim SNP count
  7. Genotype loader warns when values fall outside expected range (0, 1, 2, NaN for missing)
**Plans**: TBD

## Progress

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v1.0 MVP | 1-4.2 | 21 | Complete | 2026-02-01 |
| v1.1 Covariates & Extended Tests | 5-10 | 39 | Complete | 2026-02-05 |
| v1.2 JAX Runner Unification | 11-15 | 18 | Complete | 2026-02-05 |
| v1.3 Tech Debt Cleanup | 16-18 | 7 | Complete | 2026-02-06 |
| v1.5 Tests and Architecture | 23 | 4 | Complete | 2026-02-10 |
| v2.0 Production GWAS | 24-28 | TBD | Planned | - |

---
*Roadmap created: 2026-01-31*
*v1.0 shipped: 2026-02-01*
*v1.1 shipped: 2026-02-05*
*v1.2 shipped: 2026-02-05*
*v1.3 shipped: 2026-02-06*
*v1.5 shipped: 2026-02-10*
*v2.0 roadmap extended: 2026-02-10*
