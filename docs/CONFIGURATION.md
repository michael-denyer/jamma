# JAMMA Configuration Reference

This document covers all configuration surfaces in JAMMA: environment variables,
CLI flags, `pyproject.toml` tool settings, BLAS backend dispatch, and the Docker
image for containerised runs.

## Environment Variables

JAMMA reads these environment variables at runtime. None are required for normal
use — the defaults are appropriate for most analyses.

| Variable | Default | Description |
|---|---|---|
| `JAMMA_BACKEND` | `auto` | Force the compute backend: `auto`, `numpy`, or `numpy-streaming`. Auto-detect selects the C+NumPy runner, falling back to streaming when memory is insufficient. |
| `JAMMA_BLAS_THREADS` | Physical core count | Thread count for NumPy BLAS operations (eigendecomposition, matmul). Controls MKL/OpenBLAS via `threadpoolctl`. **Linux/Windows only** — has no effect on macOS Accelerate. |
| `JAMMA_LOCO_WORKERS` | `1` | Parallel chromosome workers for LOCO analysis. Each worker holds a full K_loco matrix (`n_samples² × 8` bytes), so increase with caution. |
| `JAMMA_NO_TELEMETRY` | *(unset)* | Set to any non-empty value to disable local benchmark telemetry. |
| `DO_NOT_TRACK` | *(unset)* | Universal convention: set to `1` to disable JAMMA telemetry. |
| `JLINALG_NO_VENDOR_LAPACK` | *(unset)* | Set to any non-empty value (not `0`) to force `np.linalg.eigh` instead of vendor LAPACK (DSYEVD/DSYEVR) for eigendecomposition only (scope: `lmm/eigen.py`). Useful for debugging numerical differences. |
| `JLINALG_DISPATCH_DEBUG` | *(unset)* | Set to `1` to print jlinalg BLAS dispatch diagnostics (backend detection, ILP64 status, library path) from the `jlinalg` C layer. Debug aid only. |
| `JAMMA_FORCE_NUMPY_FALLBACK` | *(unset)* | Set to any non-empty value (not `0`) to force the **entire jlinalg layer** onto its NumPy fallback path even when vendor BLAS is loaded. Wider scope than `JLINALG_NO_VENDOR_LAPACK`: also affects `dgemm`, `dsyrk`, `dsyr2k`, `qr`, `svd`. Used by the weekly sanitizer workflow and by full numerical-divergence debugging. |
| `JAMMA_NO_OPENMP` | *(unset)* | Set to any non-empty value (not `0`) to disable OpenMP when compiling the C extension. The extension will be single-threaded. |
| `OMP_NUM_THREADS` | *(system default)* | OpenMP thread count for C extension kernels (`_lmm_accel`, `_jlinalg`). Separate from `JAMMA_BLAS_THREADS`, which controls BLAS only. |
| `JAMMA_SANITIZE` | *(unset)* | **Build-time only.** Comma-separated sanitizer list (e.g. `address,undefined`) injected into compile and link flags by `_build_support/compile_and_link.py`. Used by `.github/workflows/sanitizers.yml`. See `docs/TESTING.md` §1.10 for local repro. |
| `JAMMA_SENTINEL_UB` | *(unset)* | **Build-time only.** When set to `1`, `_compile_accel.py` injects `-DJAMMA_SENTINEL_UB`, which compiles a known heap-OOB into `_lmm_accel.c`. Used by the sanitizer workflow's `asan-sentinel-meta-test` job to verify ASAN is actually catching bugs (distinguishes a clean run from an unwired sanitizer). |

```bash
# Example: 4 BLAS threads, 2 LOCO workers, no telemetry
export JAMMA_BLAS_THREADS=4
export JAMMA_LOCO_WORKERS=2
export JAMMA_NO_TELEMETRY=1
jamma -lmm 1 -bfile data/my_study -loco -o output
```

**Threading note:** `JAMMA_BLAS_THREADS` controls BLAS libraries (MKL, OpenBLAS)
via `threadpoolctl` and does not affect OpenMP threads. On macOS with Apple Accelerate,
`JAMMA_BLAS_THREADS` has no effect — Accelerate provides no public thread-count API.
JAMMA detects this automatically and halves OpenMP threads to avoid oversubscription.

## CLI Flags

JAMMA's CLI is Click-based and mirrors GEMMA's flat flag interface. One of `-gk`
or `-lmm` is required.

### Input and mode selection

| Flag | Type | Default | Description |
|---|---|---|---|
| `-bfile` | path | *(required)* | PLINK binary file prefix (`.bed`/`.bim`/`.fam` without extension) |
| `-gk` | int | — | Kinship mode: `1` = centered, `2` = standardized. Mutually exclusive with `-lmm`. |
| `-lmm` | int | — | LMM association mode: `1` = Wald, `2` = LRT, `3` = Score, `4` = All. Mutually exclusive with `-gk`. |
| `-k` | path | — | Pre-computed kinship matrix file. Required for `-lmm` unless using `-loco` or pre-computed eigen files. |
| `-c` | path | — | Covariate file (whitespace-delimited, no header) |
| `-n` | str | `1` | Phenotype column(s) in `.fam` file, 1-based. Single value or space/comma-separated list: `-n 1` or `-n '1 2 3'` or `-n '1,2,3'`. |

### SNP filtering

| Flag | Type | Default | Description |
|---|---|---|---|
| `-maf` | float | `0.01` (lmm) / `0.0` (gk) | Minor allele frequency threshold. Default is `0.0` in `-gk` mode to match GEMMA kinship behavior. |
| `-miss` | float | `0.05` (lmm) / `1.0` (gk) | Missing rate threshold. Default is `1.0` in `-gk` mode. |
| `-hwe` | float | `0.0` | HWE p-value threshold (0 = no filtering). Requires `numpy-streaming` backend. |
| `-snps` | path | — | SNP list file for association testing |
| `-ksnps` | path | — | SNP list file for kinship computation |

### Lambda optimization bounds

| Flag | Type | Default | Description |
|---|---|---|---|
| `-lmin` | float | `1e-5` | Minimum lambda for variance component optimization. Must be > 0. |
| `-lmax` | float | `1e5` | Maximum lambda for variance component optimization. Must be > `-lmin`. |

### Output

| Flag | Type | Default | Description |
|---|---|---|---|
| `-o` | str | `result` | Output file prefix |
| `-outdir` | path | `output` | Output directory path |
| `--legacy-text` | flag | off | Write kinship/eigen files in GEMMA text format instead of binary `.npy` |

### Eigen files

| Flag | Type | Default | Description |
|---|---|---|---|
| `-d` | path | — | Pre-computed eigenvalue file (`.eigenD.npy` or `.txt`) |
| `-u` | path | — | Pre-computed eigenvector file (`.eigenU.npy` or `.txt`) |
| `-eigen` | flag | off | Write eigendecomposition files alongside kinship output |
| `--eigen-dir` | path | — | Directory for LOCO per-chromosome eigen cache. With `-lmm -loco`, reads cached files to skip re-computation. With `-eigen`, writes them here. Only valid with `-loco`. |

> **Cache validation.** The eigen cache is keyed by a content + parameter hash
> over its determinants (the genotype files, MAF and missingness thresholds, any
> `-ksnps` restriction, and the analysed-sample set), stored in
> `<prefix>.loco.cache_manifest.json`. When any determinant changes the cache is
> recomputed rather than silently reused. A cache written before the manifest
> existed has no key and is recomputed once.

### Analysis modes

| Flag | Type | Default | Description |
|---|---|---|---|
| `-loco` | flag | off | Leave-one-chromosome-out analysis. Mutually exclusive with `-k`. |
| `-cat` | str | — | Categorical covariate columns, 1-indexed, space-separated (e.g., `-cat '1 3'`). JAMMA-specific extension. |
| `-widv` | path | — | Per-individual weight file (one weight per line) |

### Performance and memory

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backend` | choice | `auto` | Compute backend: `auto`, `numpy`, or `numpy-streaming`. Overridden by `JAMMA_BACKEND` env var. |
| `--check-memory` / `--no-check-memory` | flag | on | Enable/disable pre-flight memory check before eigendecomposition. |
| `--mem-budget` | float | — | Hard memory budget in GB. Overrides auto-detected available memory for backend selection. |
| `-v` / `--verbose` | flag | off | Verbose logging output |
| `--no-telemetry` | flag | off | Disable benchmark telemetry for this run (equivalent to `JAMMA_NO_TELEMETRY=1`) |
| `--version` | flag | — | Print version and backend info, then exit |

## pyproject.toml Settings

### Project metadata and dependencies

`pyproject.toml` is the single configuration file for project metadata, build
settings, and tool configuration.

```toml
[project]
name = "jamma"
version = "5.3.0"
requires-python = ">=3.11"
```

Runtime dependencies: `bed-reader>=1.0.0`, `numpy>=2.0.0`, `psutil>=5.9.0`,
`threadpoolctl>=3.0.0`, `click>=8.0.0`, `loguru>=0.7.0`, `progressbar2>=4.2.0`.

**scipy is intentionally excluded from runtime dependencies.** Installing scipy
would overwrite an ILP64 numpy-mkl installation with LP64 numpy, breaking large-scale
(>46k sample) eigendecomposition. scipy is available as a dev-only dependency for
test use only.

### pytest configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-n 3 --randomly-seed=last --benchmark-skip --timeout=120 -m 'not slow and not tier2' --no-cov"
```

Key settings:

- `-n 3` — parallelism capped at 3 workers. Do not override with `-n auto`; it spawns too many workers and contaminates BLAS-threaded tests.
- `--randomly-seed=last` — repeatable random ordering for debugging.
- Tests are tiered: `tier0` (fast unit), `tier1` (GEMMA parity), `tier2` (scale, runs in `test-slow.yml`).

### Ruff linter and formatter

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

Run with `uv run ruff check .` (lint) and `uv run ruff format .` (format).

## BLAS Backend Dispatch

JAMMA dispatches to vendor BLAS/LAPACK through the `jlinalg` C extension. The
selection order is:

| Priority | Backend | Condition |
|---|---|---|
| 1 | ILP64 MKL / OpenBLAS-ILP64 / Accelerate-ILP64 | ILP64 symbols detected by jlinalg |
| 2 | NumPy fallback (`np.linalg`, `np.matmul`) | No ILP64 vendor BLAS detected |
| — | LP64 backends | Detected but **not wired** — different FP accumulation causes result drift |

LP64 backends (e.g. standard Accelerate on macOS, standard MKL) are detected but
intentionally not used for matrix operations. LP64 uses different floating-point
accumulation order compared to ILP64, which propagates through lambda optimisation,
Pab computation, and ultimately p-values, breaking GEMMA equivalence.

**Why ILP64 matters at scale:** LP64 BLAS uses 32-bit integers for matrix
dimension parameters. At ~46,340 samples, a kinship matrix has `46340² ≈ 2.1 billion`
elements, overflowing `int32` and causing silent corruption or segfaults.
ILP64 uses 64-bit integers and handles arbitrarily large matrices.

### Eigendecomposition driver selection

Within a run, `jlinalg.eigh` selects a driver in priority order:

1. **DSYEVD** (in-place, vendor LAPACK) — fastest; requires `O(N²)` workspace (~240 GB for 100k samples)
2. **DSYEVR** (vendor LAPACK) — lower peak memory (`O(N)` workspace, ~160 GB for 100k samples)
3. **`np.linalg.eigh`** — used when no vendor LAPACK is available, or when `JLINALG_NO_VENDOR_LAPACK` is set

Set `JLINALG_NO_VENDOR_LAPACK=1` to force the NumPy fallback for debugging.

### Checking the active backend

```bash
jamma --version
# prints: JAMMA version 5.3.0 (...)
#         Backend: numpy
```

```python
from jamma.jlinalg import blas_backend, blas_is_ilp64
print(blas_backend)    # e.g. "mkl-ilp64", "openblas-ilp64", "numpy-fallback"
print(blas_is_ilp64)   # 1 if ILP64, 0 if not
```

## ILP64 numpy Installation (Linux/Windows)

Standard numpy uses LP64 BLAS, which overflows at ~46k samples. For large-scale
GWAS, install the ILP64-backed numpy from `michael-denyer/numpy-mkl`.

**Install order matters** — numpy must be installed before JAMMA to prevent a
normal `pip install jamma` from overwriting the ILP64 build:

```bash
# 1. Install runtime dependencies (no numpy yet)
pip install psutil loguru threadpoolctl click progressbar2 bed-reader

# 2. Install ILP64 numpy (force-reinstall to replace any existing LP64 build)
pip install numpy \
  --index-url https://michael-denyer.github.io/numpy-mkl \
  --force-reinstall --upgrade

# 3. Install JAMMA without deps to preserve the ILP64 numpy
pip install jamma --no-deps
```

Omitting `--no-deps` on the final step will pull standard numpy as a transitive
dependency and overwrite the ILP64 build.

macOS does not require this procedure — Apple Accelerate provides ILP64 BLAS
natively since macOS 13.3 and JAMMA auto-detects it.

## Docker Configuration

The provided `Dockerfile` builds a slimline image with ILP64 numpy (MKL) for
large-scale GWAS. MKL is x86_64-only — always build and run with
`--platform linux/amd64`.

```bash
# Build
docker build --platform linux/amd64 -t jamma .

# Kinship computation
docker run --platform linux/amd64 \
  -v $(pwd)/data:/data \
  jamma -gk 1 -bfile /data/study -o /data/output

# LMM association
docker run --platform linux/amd64 \
  -v $(pwd)/data:/data \
  jamma -lmm 1 -bfile /data/study -k /data/k.cXX.txt -o /data/output
```

The Docker build installs dependencies in this order to preserve the ILP64 numpy:

1. `mkl` — Intel MKL runtime libraries
2. `numpy` with `--index-url https://michael-denyer.github.io/numpy-mkl` — ILP64 build
3. Runtime dependencies (`psutil`, `loguru`, `threadpoolctl`, `click`, `progressbar2`, `bed-reader`)
4. `jamma --no-deps` — JAMMA without dependency resolution (preserves ILP64 numpy)

ILP64 is verified at build time by asserting `ilp64` appears in the BLAS name
from `numpy.show_config()`. Builds that fail this check do not produce an image.

The container runs as a non-root user (`jamma`, UID 1000) for security. The
`/data` volume is pre-created and owned by this user.

## Telemetry

JAMMA writes **local-only** benchmark data to `~/.jamma/benchmarks.jsonl` after
each `-lmm` run. No data is transmitted to any external server.

Each line records: timestamp, JAMMA version, sample/SNP counts, backend, timings
(kinship, LMM, rotation, eigendecomp), peak memory, BLAS backend, and platform.
No genotype data, phenotype data, or file paths are recorded.

To opt out:

```bash
# Per-run (CLI flag)
jamma --no-telemetry -lmm 1 -bfile data/study -k kinship.cXX.txt

# Persistent (environment variable)
export JAMMA_NO_TELEMETRY=1

# Universal convention (also disables telemetry in other tools)
export DO_NOT_TRACK=1
```

`JAMMA_NO_TELEMETRY` disables telemetry for any non-empty value.
`DO_NOT_TRACK=1` opts out; `DO_NOT_TRACK=0` explicitly opts in.
Kinship-only mode (`-gk`) never emits telemetry regardless of these settings.
