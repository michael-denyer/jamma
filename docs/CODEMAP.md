# JAMMA Code Map

Architectural overview with bidirectional links between diagram nodes and source code.

## System Overview

```mermaid
flowchart TB
    subgraph L1["Entry Points [1]"]
        CLI["CLI (click) [1a]"]
        GWAS["gwas() API [1b]"]
        PIPE["PipelineRunner [1c]"]
    end

    subgraph L2["I/O Layer [2]"]
        PLINK["PLINK Reader [2a]"]
        COVAR["Covariate Reader [2b]"]
        KIO["Kinship I/O [2c]"]
        RIO["Result Writer [2d]"]
        SNPLIST["SNP List I/O [2e]"]
        EIGIO["Eigen I/O [2f]"]
        MATIO["Matrix Writer [2g]"]
    end

    subgraph L3["Core Computation [3]"]
        KINSHIP["Kinship Compute [3a]"]
        MISSING["Missing Imputation [3b]"]
        EIGEN["Eigendecomposition [3c]"]
        LIKE["REML Likelihood [3d]"]
        OPT["Lambda Optimizer [3e]"]
        STATS["Test Statistics [3f]"]
        SNPF["SNP Filters [3g]"]
        PREPCOM["Shared Preparation [3h]"]
        SPECIAL["Special Functions [3i]"]
    end

    subgraph L4["JAX Backend [4]"]
        LIKEJAX["Batch Likelihood [4a]"]
        RUNNER["JAX Runner [4b]"]
        STREAM["Streaming Runner [4c]"]
        LOCO["LOCO Runner [4d]"]
        PREP["Device Preparation [4e]"]
        CHUNK["Chunk Sizing [4f]"]
        COMPUTE["Chunk Compute [4g]"]
        SCHEMA["Output Schema [4h]"]
        RESULTS["Result Building [4i]"]
    end

    subgraph L4N["NumPy Backend [4N]"]
        LIKENP["Batch Likelihood [4Na]"]
        RUNNERNP["NumPy Runner [4Nb]"]
        COMPUTENP["Chunk Compute [4Nc]"]
    end

    subgraph L5["Infrastructure [5]"]
        CONFIG["Output Config [5a]"]
        JAX["JAX Config [5b]"]
        MEM["Memory Manager [5c]"]
        LOG["Logging [5d]"]
        THREAD["Threading [5e]"]
        HW["Hardware Context [5f]"]
        PROG["Progress [5g]"]
    end

    subgraph L6["Validation [6]"]
        TOL["Tolerance Config [6a]"]
        CMP["GEMMA Comparator [6b]"]
    end

    CLI --> PIPE
    GWAS --> PIPE

    CLI --> PLINK
    CLI --> KIO
    CLI --> SNPLIST
    CLI --> CONFIG
    CLI --> LOG

    PIPE --> PLINK
    PIPE --> COVAR
    PIPE --> KIO
    PIPE --> RIO
    PIPE --> SNPLIST
    PIPE --> EIGIO
    PIPE --> KINSHIP
    PIPE --> EIGEN
    PIPE --> STREAM
    PIPE --> RUNNERNP
    PIPE --> LOCO
    PIPE --> MEM

    KINSHIP --> MISSING
    KINSHIP --> PLINK
    KINSHIP --> MEM
    KINSHIP --> SNPF

    CLI --> KINSHIP

    LOCO --> KINSHIP
    LOCO --> EIGEN
    LOCO --> STREAM

    %% JAX backend connections
    RUNNER --> LIKEJAX
    RUNNER --> STATS
    RUNNER --> RIO
    RUNNER --> PREP
    RUNNER --> CHUNK
    RUNNER --> COMPUTE
    STREAM --> LIKE
    STREAM --> OPT
    STREAM --> STATS
    STREAM --> RIO
    STREAM --> SNPF
    STREAM --> PREP
    STREAM --> CHUNK
    STREAM --> COMPUTE
    COMPUTE --> LIKEJAX
    COMPUTE --> STATS
    PREP --> THREAD

    %% NumPy backend connections
    RUNNERNP --> LIKENP
    RUNNERNP --> PREPCOM
    RUNNERNP --> RIO
    COMPUTENP --> LIKENP
    LIKENP --> SPECIAL

    %% Shared connections
    PREPCOM --> EIGEN
    LIKEJAX --> LIKE
    OPT --> LIKE
    STATS --> LIKE

    EIGEN --> MEM
    EIGEN --> THREAD

    KIO --> MATIO

    CMP --> TOL
```

---

### [1] Entry Points

Two user-facing entry points: the `gwas()` API for programmatic use and the CLI for command-line use. Both delegate to `PipelineRunner` for LMM orchestration.

| ID | Component | Description | File:Line |
|----|-----------|-------------|-----------|
| 1a | `main()` | Click command — all flags (`-gk`, `-lmm`, `-bfile`, `-o`, `-outdir`) | [cli.py](../src/jamma/cli.py) |
| 1a | `_run_gk()` | Kinship computation (`-gk 1`) | [cli.py](../src/jamma/cli.py) |
| 1a | `_run_lmm()` | LMM association (`-lmm 1/2/3/4`) | [cli.py](../src/jamma/cli.py) |
| 1b | `gwas()` | One-call GWAS pipeline (load → kinship → LMM → results) | [gwas.py:40](../src/jamma/gwas.py#L40) |
| 1b | `GWASResult` | Pipeline result dataclass (associations, timing, counts) | [gwas.py:22](../src/jamma/gwas.py#L22) |
| 1c | `PipelineRunner` | Shared orchestration (validate → parse → memory → kinship → LMM) | [pipeline.py](../src/jamma/pipeline.py) |
| 1c | `PipelineConfig` | Pipeline configuration dataclass (all CLI flags) | [pipeline.py](../src/jamma/pipeline.py) |

---

### [2] I/O Layer

Reads PLINK binary genotypes, covariates, and kinship matrices. Writes GEMMA-compatible output.

| ID | Component | Description | File:Line |
|----|-----------|-------------|-----------|
| 2a | `PlinkData` | Genotype container (n_samples × n_snps float32) | [plink.py:20](../src/jamma/io/plink.py#L20) |
| 2a | `load_plink_binary()` | Full-load PLINK .bed/.bim/.fam | [plink.py:53](../src/jamma/io/plink.py#L53) |
| 2a | `stream_genotype_chunks()` | Windowed reads from .bed (O(n×chunk)) | [plink.py:139](../src/jamma/io/plink.py#L139) |
| 2a | `get_plink_metadata()` | Dimensions + metadata without loading genotypes | [plink.py:92](../src/jamma/io/plink.py#L92) |
| 2b | `read_covariate_file()` | Whitespace-delimited covariate matrix | [covariate.py:19](../src/jamma/io/covariate.py#L19) |
| 2c | `read_kinship_matrix()` | Load GEMMA `.cXX.txt` format | [kinship/io.py:8](../src/jamma/kinship/io.py#L8) |
| 2c | `write_kinship_matrix()` | Write `.cXX.txt` via parallel writer | [kinship/io.py:42](../src/jamma/kinship/io.py#L42) |
| 2d | `IncrementalAssocWriter` | Per-SNP disk writer (no memory accumulation) | [lmm/io.py:172](../src/jamma/lmm/io.py#L172) |
| 2d | `format_assoc_line()` | Table-driven output row formatting | [lmm/io.py:12](../src/jamma/lmm/io.py#L12) |
| 2d | `write_assoc_results()` | Batch write from list | [lmm/io.py:150](../src/jamma/lmm/io.py#L150) |
| 2e | `read_snp_list_file()` | Parse SNP list file (one RS ID per line) | [io/snp_list.py](../src/jamma/io/snp_list.py) |
| 2e | `resolve_snp_list_to_indices()` | Map SNP IDs to dataset indices | [io/snp_list.py](../src/jamma/io/snp_list.py) |
| 2f | `read_eigen_files()` | Load eigenvalue/eigenvector files | [lmm/eigen_io.py](../src/jamma/lmm/eigen_io.py) |
| 2f | `write_eigen_files()` | Write eigendecomposition to disk | [lmm/eigen_io.py](../src/jamma/lmm/eigen_io.py) |
| 2g | `write_matrix_parallel()` | Parallel matrix writer using file-backed memmap | [io/matrix_writer.py:91](../src/jamma/io/matrix_writer.py#L91) |

---

### [3] Core Computation

GEMMA algorithm reimplementation: kinship → eigendecomp → REML → test statistics. These modules are shared across both backends.

| ID | Component | Description | File:Line |
|----|-----------|-------------|-----------|
| 3a | `compute_centered_kinship()` | K = (1/p) × Xc × Xc' in batches of 10k SNPs | [compute.py:86](../src/jamma/kinship/compute.py#L86) |
| 3a | `compute_kinship_streaming()` | 2-pass streaming (stats → accumulate) | [compute.py:216](../src/jamma/kinship/compute.py#L216) |
| 3a | `_filter_snps()` | MAF, missing rate, monomorphism filters | [compute.py:53](../src/jamma/kinship/compute.py#L53) |
| 3b | `impute_and_center()` | NaN → mean, then center (JIT-compiled) | [missing.py:23](../src/jamma/kinship/missing.py#L23) |
| 3c | `eigendecompose_kinship()` | numpy.linalg.eigh with BLAS thread control | [eigen.py:28](../src/jamma/lmm/eigen.py#L28) |
| 3d | `reml_log_likelihood()` | REML ℓ(λ) for variance component estimation | [likelihood.py:262](../src/jamma/lmm/likelihood.py#L262) |
| 3d | `mle_log_likelihood()` | MLE ℓ(λ) for LRT | [likelihood.py:533](../src/jamma/lmm/likelihood.py#L533) |
| 3d | `compute_Uab()` | Element-wise products of rotated vectors | [likelihood.py:74](../src/jamma/lmm/likelihood.py#L74) |
| 3d | `calc_pab()` | Recursive Schur complement projection (GEMMA CalcPab) | [likelihood.py:173](../src/jamma/lmm/likelihood.py#L173) |
| 3d | `get_ab_index()` | GEMMA GetabIndex — 1-based upper triangular | [likelihood.py:53](../src/jamma/lmm/likelihood.py#L53) |
| 3d | `compute_null_model_lambda()` | Null model REML for Score test | [likelihood.py:493](../src/jamma/lmm/likelihood.py#L493) |
| 3d | `compute_null_model_mle()` | Null model MLE for LRT | [likelihood.py:583](../src/jamma/lmm/likelihood.py#L583) |
| 3e | `golden_section_optimize_lambda()` | REML optimization per SNP (Wald) | [likelihood_jax.py:495](../src/jamma/lmm/likelihood_jax.py#L495) |
| 3e | `golden_section_optimize_lambda_mle()` | MLE optimization per SNP (LRT) | [likelihood_jax.py:833](../src/jamma/lmm/likelihood_jax.py#L833) |
| 3f | `AssocResult` | Per-SNP result dataclass (all test fields) | [stats.py:40](../src/jamma/lmm/stats.py#L40) |
| 3f | `calc_wald_test()` | β, SE, p_wald from Pab matrix | [stats.py:98](../src/jamma/lmm/stats.py#L98) |
| 3f | `calc_score_test()` | p_score using null model lambda | [stats.py:232](../src/jamma/lmm/stats.py#L232) |
| 3f | `calc_lrt_test()` | p_lrt via chi-squared CDF | [stats.py:202](../src/jamma/lmm/stats.py#L202) |
| 3f | `f_sf()` | F-distribution survival via JAX betainc | [stats.py:67](../src/jamma/lmm/stats.py#L67) |
| 3g | `compute_hwe_pvalues()` | Chi-squared HWE test via JAX | [core/snp_filter.py](../src/jamma/core/snp_filter.py) |
| 3g | `apply_snp_list_mask()` | DRY bounds-validated SNP mask application | [core/snp_filter.py](../src/jamma/core/snp_filter.py) |
| 3h | `_build_covariate_matrix()` | Pure-NumPy covariate setup (shared by both backends) | [prepare_common.py](../src/jamma/lmm/prepare_common.py) |
| 3h | `_eigendecompose_or_reuse()` | Handles kinship decomp or reuses pre-computed eigen | [prepare_common.py](../src/jamma/lmm/prepare_common.py) |
| 3h | `_compute_null_model_common()` | Null model fitting (shared by both backends) | [prepare_common.py](../src/jamma/lmm/prepare_common.py) |
| 3i | `betainc()` | Regularized incomplete beta (pure-stdlib, no scipy) | [special.py](../src/jamma/lmm/special.py) |
| 3i | `chi2_sf()` | Chi-squared survival function (pure-stdlib) | [special.py](../src/jamma/lmm/special.py) |

---

### [4] JAX Backend

Batch SNP processing with JIT compilation and vmap vectorization. Requires JAX (`pip install jamma[jax]`). Supports LOCO, HWE filtering, disk streaming, and CPU device sharding.

| ID | Component | Description | File:Line |
|----|-----------|-------------|-----------|
| 4a | `compute_uab_jax()` | JIT-compiled Uab for n_cvt=1 | [likelihood_jax.py:113](../src/jamma/lmm/likelihood_jax.py#L113) |
| 4a | `calc_pab_jax()` | JIT-compiled Pab projection | [likelihood_jax.py:149](../src/jamma/lmm/likelihood_jax.py#L149) |
| 4a | `batch_compute_uab()` | vmap across SNP dimension | [likelihood_jax.py:346](../src/jamma/lmm/likelihood_jax.py#L346) |
| 4a | `golden_section_optimize_lambda()` | Grid search + 20 golden section iterations | [likelihood_jax.py:495](../src/jamma/lmm/likelihood_jax.py#L495) |
| 4b | `run_lmm_association_jax()` | Full-load JAX batch runner | [runner_jax.py](../src/jamma/lmm/runner_jax.py) |
| 4c | `run_lmm_association_streaming()` | Streaming from disk, O(n² + n×chunk) | [runner_streaming.py:92](../src/jamma/lmm/runner_streaming.py#L92) |
| 4d | `run_lmm_loco()` | LOCO: per-chromosome kinship → eigen → LMM | [lmm/loco.py](../src/jamma/lmm/loco.py) |
| 4e | `DevicePlacement` | CPU/GPU device + sharding configuration | [lmm/prepare.py:216](../src/jamma/lmm/prepare.py#L216) |
| 4e | `resolve_device_placement()` | Select device and set up NamedSharding | [lmm/prepare.py:249](../src/jamma/lmm/prepare.py#L249) |
| 4e | `prepare_utg_chunk()` | Rotate genotype chunk: U.T @ G with device transfer | [lmm/prepare.py:274](../src/jamma/lmm/prepare.py#L274) |
| 4f | `_compute_chunk_size()` | JAX int32-safe chunk sizing with device alignment | [lmm/chunk.py:24](../src/jamma/lmm/chunk.py#L24) |
| 4f | `auto_tune_chunk_size()` | Chunk size auto-tuning for memory/performance | [lmm/chunk.py:98](../src/jamma/lmm/chunk.py#L98) |
| 4g | `_compute_lmm_chunk()` | Per-chunk Wald/LRT/Score computation | [lmm/compute.py:107](../src/jamma/lmm/compute.py#L107) |
| 4g | `block_chunk_result()` | Call `block_until_ready()` on JAX arrays | [lmm/compute.py:238](../src/jamma/lmm/compute.py#L238) |
| 4h | `StatColumn` | Frozen dataclass for output column definitions | [lmm/schema.py:17](../src/jamma/lmm/schema.py#L17) |
| 4h | `ModeSpec` | Per-mode column specification (single source of truth) | [lmm/schema.py:43](../src/jamma/lmm/schema.py#L43) |
| 4i | `_build_results()` | Table-driven result building from numpy arrays | [lmm/results.py:44](../src/jamma/lmm/results.py#L44) |
| 4i | `count_lambda_boundary_hits()` | Diagnostic: count SNPs at lambda bounds | [lmm/results.py:177](../src/jamma/lmm/results.py#L177) |

---

### [4N] NumPy Backend

Pure-NumPy LMM implementation with zero JAX dependency. Works on all platforms (Intel Mac, Windows, Linux). Uses `np.vectorize` for batch operations and stdlib-only special functions for p-value computation.

| ID | Component | Description | File:Line |
|----|-----------|-------------|-----------|
| 4Na | `batch_calc_wald_stats_numpy()` | Vectorized Wald: REML optimize → β, SE, p_wald | [likelihood_numpy.py](../src/jamma/lmm/likelihood_numpy.py) |
| 4Na | `batch_calc_score_stats_numpy()` | Vectorized Score: null λ → p_score | [likelihood_numpy.py](../src/jamma/lmm/likelihood_numpy.py) |
| 4Na | `batch_calc_lrt_stats_numpy()` | Vectorized LRT: MLE optimize → p_lrt | [likelihood_numpy.py](../src/jamma/lmm/likelihood_numpy.py) |
| 4Nb | `run_lmm_association_numpy()` | In-memory batch runner (full genotype load) | [runner_numpy.py](../src/jamma/lmm/runner_numpy.py) |
| 4Nc | `compute_lmm_chunk_numpy()` | Per-chunk dispatch for NumPy backend | [compute_numpy.py](../src/jamma/lmm/compute_numpy.py) |

---

### [5] Infrastructure

Configuration, memory management, threading, and logging.

| ID | Component | Description | File:Line |
|----|-----------|-------------|-----------|
| 5a | `OutputConfig` | Output directory + prefix + verbose flag | [config.py:12](../src/jamma/core/config.py#L12) |
| 5b | `configure_jax()` | Enable x64, set platform, XLA cache | [jax_config.py:27](../src/jamma/core/jax_config.py#L27) |
| 5b | `get_jax_info()` | JAX version, backend, devices | [jax_config.py:76](../src/jamma/core/jax_config.py#L76) |
| 5c | `MemoryBreakdown` | Estimated memory per workflow stage | [memory.py:54](../src/jamma/core/memory.py#L54) |
| 5c | `estimate_workflow_memory()` | Full pipeline memory estimate (pre-flight) | [memory.py:97](../src/jamma/core/memory.py#L97) |
| 5c | `estimate_lmm_memory()` | LMM-phase-only memory estimate | [memory.py:188](../src/jamma/core/memory.py#L188) |
| 5c | `estimate_lmm_streaming_memory()` | LMM streaming phase memory estimate | [memory.py:373](../src/jamma/core/memory.py#L373) |
| 5c | `check_memory_before_run()` | Raise MemoryError if insufficient | [memory.py:598](../src/jamma/core/memory.py#L598) |
| 5c | `get_memory_snapshot()` | Current RSS, VMS, available | [memory.py:478](../src/jamma/core/memory.py#L478) |
| 5c | `cleanup_memory()` | GC + clear JAX caches | [memory.py:529](../src/jamma/core/memory.py#L529) |
| 5d | `setup_logging()` | Loguru console + optional file | [logging.py:16](../src/jamma/utils/logging.py#L16) |
| 5d | `write_gemma_log()` | GEMMA-compatible `.log.txt` | [logging.py:51](../src/jamma/utils/logging.py#L51) |
| 5d | `log_rss_memory()` | RSS snapshot at phase boundaries | [logging.py:120](../src/jamma/utils/logging.py#L120) |
| 5e | `get_physical_core_count()` | Physical core detection (consolidated helper) | [threading.py:23](../src/jamma/core/threading.py#L23) |
| 5e | `blas_threads()` | Context manager for BLAS thread control | [threading.py:99](../src/jamma/core/threading.py#L99) |
| 5f | `get_hardware_context()` | CPU, BLAS, JAX, platform info for benchmarks | [hardware.py:21](../src/jamma/core/hardware.py#L21) |
| 5f | `assert_x64_precision()` | Guard against silent float32 fallback | [hardware.py:65](../src/jamma/core/hardware.py#L65) |
| 5g | `progress_iterator()` | Progress bar wrapper for iterables | [progress.py:13](../src/jamma/core/progress.py#L13) |

---

### [6] Validation

Tolerance-based comparison infrastructure for GEMMA parity testing.

| ID | Component | Description | File:Line |
|----|-----------|-------------|-----------|
| 6a | `ToleranceConfig` | Per-field tolerance dataclass (strict/default/relaxed) | [tolerances.py:40](../src/jamma/validation/tolerances.py#L40) |
| 6b | `ComparisonResult` | Pass/fail with max diffs and worst location | [compare.py:20](../src/jamma/validation/compare.py#L20) |
| 6b | `AssocComparisonResult` | Per-column comparison results | [compare.py:501](../src/jamma/validation/compare.py#L501) |
| 6b | `compare_assoc_results()` | Full association comparison across test types | [compare.py:536](../src/jamma/validation/compare.py#L536) |
| 6b | `compare_kinship_matrices()` | Symmetric matrix comparison | [compare.py:143](../src/jamma/validation/compare.py#L143) |
| 6b | `load_gemma_assoc()` | Parse GEMMA `.assoc.txt` | [compare.py:205](../src/jamma/validation/compare.py#L205) |
| 6b | `load_gemma_kinship()` | Parse GEMMA `.cXX.txt` | [compare.py:181](../src/jamma/validation/compare.py#L181) |

---

## Data Flow: Genotypes → Results

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as CLI [1a]
    participant IO as PLINK I/O [2a]
    participant K as Kinship [3a]
    participant E as Eigendecomp [3c]
    participant L as Likelihood [3d]
    participant O as Optimizer [3e]
    participant S as Statistics [3f]
    participant W as Writer [2d]

    U->>CLI: jamma -lmm 1 -bfile data -k K.txt
    CLI->>IO: load_plink_binary()
    IO-->>CLI: PlinkData (genotypes, metadata)
    CLI->>K: read_kinship_matrix()
    K-->>CLI: K (n×n)
    CLI->>E: eigendecompose_kinship(K)
    E-->>CLI: eigenvalues, eigenvectors (U)

    Note over CLI: Rotate: Uy = U'y, UtW = U'W

    loop For each SNP chunk
        CLI->>L: compute_Uab(UtW, Uty, Utx)
        L-->>CLI: Uab matrix
        CLI->>O: optimize_lambda(REML, Uab)
        O->>L: reml_log_likelihood(λ)
        L-->>O: ℓ(λ)
        O-->>CLI: λ*
        CLI->>L: calc_pab(Hi_eval, Uab)
        L-->>CLI: Pab matrix
        CLI->>S: calc_wald_test(Pab)
        S-->>CLI: β, SE, p_wald
        CLI->>W: write(AssocResult)
    end

    W-->>U: .assoc.txt + .log.txt
```

---

## LMM Test Modes

```mermaid
flowchart LR
    subgraph Input["Shared Computation"]
        UAB["Uab [3d]"]
    end

    subgraph Wald["-lmm 1: Wald"]
        REML1["REML λ* per SNP [3e]"]
        W1["β, SE, p_wald [3f]"]
    end

    subgraph LRT["-lmm 2: LRT"]
        MLE1["MLE λ* per SNP [3e]"]
        MLE0["MLE λ* null [3d]"]
        L1["p_lrt = χ²(ℓ₁−ℓ₀) [3f]"]
    end

    subgraph Score["-lmm 3: Score"]
        NULL["Null REML λ₀ [3d]"]
        SC["p_score (no per-SNP opt) [3f]"]
    end

    UAB --> REML1 --> W1
    UAB --> MLE1 --> L1
    MLE0 --> L1
    UAB --> NULL --> SC
```

---

## Memory Architecture

```mermaid
flowchart TD
    subgraph Preflight["Pre-flight Check [5c]"]
        EST["estimate_lmm_memory()"]
        CHK["check_memory_before_run()"]
    end

    subgraph Peak["Memory Peak"]
        KM["K matrix: 8n² bytes"]
        UM["U matrix: 8n² bytes"]
        WS["LAPACK workspace: ~8n² bytes"]
    end

    subgraph Runtime["Runtime Controls"]
        INC["IncrementalAssocWriter [2d]"]
        STR["Streaming chunks [4c]"]
        CLN["cleanup_memory() [5c]"]
    end

    EST --> CHK
    CHK -->|"insufficient"| FAIL["MemoryError (fail fast)"]
    CHK -->|"sufficient"| Peak
    Peak --> Runtime
    INC -->|"per-SNP to disk"| DISK["No list accumulation"]
    STR -->|"O(n×chunk)"| LOW["Bounded memory"]
```

---

## Backend Architecture

`PipelineRunner` selects a backend at startup via `detect_backend()` and routes all LMM computation through that backend. Both backends produce identical `AssocResult` outputs.

```mermaid
flowchart TD
    PIPE["PipelineRunner"]
    DET["detect_backend()"]
    PREP["prepare_common.py<br>(covariates, eigen, null model)"]

    subgraph JAX["JAX Backend (requires JAX)"]
        direction TB
        RJ["runner_jax / runner_streaming"]
        CJ["compute.py"]
        LJ["likelihood_jax.py"]
        PJ["prepare.py<br>(device sharding)"]
    end

    subgraph NP["NumPy Backend (no JAX)"]
        direction TB
        RN["runner_numpy"]
        CN["compute_numpy.py"]
        LN["likelihood_numpy.py"]
        SP["special.py<br>(stdlib betainc/chi2)"]
    end

    PIPE --> DET
    DET -->|"jax"| JAX
    DET -->|"numpy"| NP
    PIPE --> PREP
    PREP --> JAX
    PREP --> NP
    RJ --> CJ --> LJ
    RJ --> PJ
    RN --> CN --> LN
    LN --> SP
```

### Backend Selection

Priority order: `JAMMA_BACKEND` env var → `--backend` CLI flag → auto-detect (try JAX, fall back to NumPy).

### Feature Parity

| Feature | JAX | NumPy |
|---------|-----|-------|
| Wald test (`-lmm 1`) | Yes | Yes |
| LRT (`-lmm 2`) | Yes | Yes |
| Score test (`-lmm 3`) | Yes | Yes |
| All tests (`-lmm 4`) | Yes | Yes |
| LOCO (`-loco`) | Yes | No |
| HWE filtering (`-hwe`) | Yes | No |
| Disk streaming | Yes | No (full load) |
| CPU device sharding | Yes | N/A |
| GPU acceleration | Yes | N/A |

### File Naming Convention

| Pattern | Purpose | Examples |
|---------|---------|---------|
| `*_jax.py` | JAX-specific implementation | `likelihood_jax.py`, `runner_jax.py` |
| `*_numpy.py` | Pure-NumPy implementation | `likelihood_numpy.py`, `runner_numpy.py`, `compute_numpy.py` |
| `*_common.py` | Shared by both backends | `prepare_common.py` |
| No suffix | Base algorithms or shared code | `likelihood.py`, `stats.py`, `eigen.py` |

---

## Quick Navigation

| Area | Entry Point |
|------|-------------|
| gwas() API | [gwas.py:40](../src/jamma/gwas.py#L40) |
| PipelineRunner | [pipeline.py](../src/jamma/pipeline.py) |
| CLI dispatch | [cli.py:54](../src/jamma/cli.py#L54) |
| Load genotypes | [plink.py:53](../src/jamma/io/plink.py#L53) |
| SNP list I/O | [io/snp_list.py](../src/jamma/io/snp_list.py) |
| Eigen I/O | [lmm/eigen_io.py](../src/jamma/lmm/eigen_io.py) |
| Matrix writer | [io/matrix_writer.py:91](../src/jamma/io/matrix_writer.py#L91) |
| Kinship compute | [compute.py:86](../src/jamma/kinship/compute.py#L86) |
| Eigendecomposition | [eigen.py:28](../src/jamma/lmm/eigen.py#L28) |
| REML likelihood | [likelihood.py:262](../src/jamma/lmm/likelihood.py#L262) |
| Lambda optimization | [likelihood_jax.py:495](../src/jamma/lmm/likelihood_jax.py#L495) |
| Wald/Score/LRT tests | [stats.py:98](../src/jamma/lmm/stats.py#L98) |
| SNP filters (HWE) | [core/snp_filter.py](../src/jamma/core/snp_filter.py) |
| Output schema | [lmm/schema.py:17](../src/jamma/lmm/schema.py#L17) |
| Chunk preparation | [lmm/prepare.py:216](../src/jamma/lmm/prepare.py#L216) |
| Chunk sizing | [lmm/chunk.py:24](../src/jamma/lmm/chunk.py#L24) |
| Chunk compute | [lmm/compute.py:107](../src/jamma/lmm/compute.py#L107) |
| Result building | [lmm/results.py:44](../src/jamma/lmm/results.py#L44) |
| JAX batch runner | [runner_jax.py](../src/jamma/lmm/runner_jax.py) |
| Streaming runner | [runner_streaming.py:92](../src/jamma/lmm/runner_streaming.py#L92) |
| NumPy batch runner | [runner_numpy.py](../src/jamma/lmm/runner_numpy.py) |
| NumPy likelihood | [likelihood_numpy.py](../src/jamma/lmm/likelihood_numpy.py) |
| NumPy chunk compute | [compute_numpy.py](../src/jamma/lmm/compute_numpy.py) |
| Shared preparation | [prepare_common.py](../src/jamma/lmm/prepare_common.py) |
| Special functions | [special.py](../src/jamma/lmm/special.py) |
| Backend detection | [backend.py](../src/jamma/core/backend.py) |
| LOCO runner | [lmm/loco.py](../src/jamma/lmm/loco.py) |
| Result writer | [lmm/io.py:172](../src/jamma/lmm/io.py#L172) |
| Memory estimation | [memory.py:97](../src/jamma/core/memory.py#L97) |
| Threading | [threading.py:23](../src/jamma/core/threading.py#L23) |
| Hardware context | [hardware.py:21](../src/jamma/core/hardware.py#L21) |
| Validation comparison | [compare.py:536](../src/jamma/validation/compare.py#L536) |
| Equivalence proof | [EQUIVALENCE.md](EQUIVALENCE.md) |
