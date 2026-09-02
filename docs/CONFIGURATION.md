# JAMMA Configuration Reference

This document covers all configuration surfaces in JAMMA: environment variables,
CLI flags, `pyproject.toml` tool settings, BLAS backend dispatch, and the Docker
image for containerised runs.

## Environment Variables

JAMMA reads these environment variables at runtime. None are required for normal
use — the defaults are appropriate for most analyses.

Seven of them (`JAMMA_BLAS_THREADS`, `JAMMA_LOCO_WORKERS`, `JAMMA_BACKEND`,
`JAMMA_FORCE_NUMPY_FALLBACK`, `JAMMA_NO_TELEMETRY`, `JAMMA_NO_OPENMP`,
`JAMMA_SENTINEL_UB`) plus `JAMMA_SANITIZE` are parsed once per read through
`jamma.core.constants.Env.current()`, rather than each call site spelling
out its own `os.environ.get(...)`. `Env` reads fresh on every call instead of
caching a module-level singleton, so `monkeypatch.setenv`/`delenv` in tests
and a CI job's `env:` block both still take effect. `DO_NOT_TRACK` stays a
direct read in `telemetry.py`: it follows a different truthiness rule (only
`"1"` opts out) than every `JAMMA_*` toggle's presence-based one, so folding
it into `Env` would misrepresent it. `JAMMA_SANITIZE` and `JAMMA_SENTINEL_UB`
(build-time, resolved by `_build_support/build_models.py` through the shared
build facade), `JAMMA_NO_OPENMP`
(`_build_support/openmp_detect.py`), and `CC` stay direct reads too — those
modules run under PEP 517 build isolation, or standalone before the package
is installed, and cannot import the runtime `jamma` package that `Env` lives
in without pulling in the full numpy/loguru stack they are built to avoid.

| Variable | Default | Description |
|---|---|---|
| `JAMMA_BACKEND` | `auto` | Force the compute backend: `auto`, `numpy`, or `numpy-streaming`. Auto-detect selects the C+NumPy runner, falling back to streaming when memory is insufficient. |
| `JAMMA_BLAS_THREADS` | Physical core count | Thread count for NumPy BLAS operations (eigendecomposition, matmul). Controls MKL/OpenBLAS via `threadpoolctl`. **Linux/Windows only** — has no effect on macOS Accelerate. |
| `JAMMA_LOCO_WORKERS` | `1` | Parallel chromosome workers for LOCO analysis. Each worker holds a full K_loco matrix (`n_samples² × 8` bytes), so increase with caution. |
| `JAMMA_NO_TELEMETRY` | *(unset)* | Set to any non-empty value to disable local benchmark telemetry. Merged with the CLI's `--no-telemetry` / the Python API's `no_telemetry` argument onto `PipelineConfig.no_telemetry` in `pipeline.py`; `cli.py` no longer writes this variable into `os.environ` to reach `telemetry.py`, and `append_benchmark_record` takes the resolved value as an explicit argument rather than reading the variable itself. |
| `DO_NOT_TRACK` | *(unset)* | Universal convention: set to `1` to disable JAMMA telemetry. |
| `JLINALG_NO_VENDOR_LAPACK` | *(unset)* | Set to any non-empty value (not `0`) to force `np.linalg.eigh` instead of vendor LAPACK (DSYEVD/DSYEVR) for eigendecomposition only (scope: `lmm/eigen.py`). Useful for debugging numerical differences. |
| `JLINALG_NO_VENDOR_DGEMM` | *(unset)* | Set to any non-empty value (not `0`) to leave vendor `dgemm` unwired, so `blas_has_dgemm` reports `0` while the C extension stays loaded and the rest of dispatch (`dsyrk`, DSYEVD/DSYEVR) is untouched. That is the permanent state of an LP64-only host — distro or conda numpy — which CI never reaches because PyPI numpy ships ILP64 `scipy_openblas64`. Narrower than `JAMMA_FORCE_NUMPY_FALLBACK`, which skips the `.so` import entirely. Used by `tests/test_jlinalg_dispatch.py::TestDgemmVendorGate`. |
| `JLINALG_DISPATCH_DEBUG` | *(unset)* | Set to `1` to print jlinalg BLAS dispatch diagnostics (backend detection, ILP64 status, library path) from the `jlinalg` C layer. Debug aid only. |
| `JAMMA_FORCE_NUMPY_FALLBACK` | *(unset)* | Set to any non-empty value (not `0`) to force the **entire jlinalg layer** onto its NumPy fallback path even when vendor BLAS is loaded. Wider scope than `JLINALG_NO_VENDOR_LAPACK`: also affects `dgemm`, `dsyrk`. Used by the weekly sanitizer workflow and by full numerical-divergence debugging. |
| `JAMMA_NO_OPENMP` | *(unset)* | Set to any non-empty value (not `0`) to disable OpenMP when compiling the C extension. The extension will be single-threaded. |
| `OMP_NUM_THREADS` | *(system default)* | OpenMP thread count for C extension kernels (`_lmm_accel`, `_jlinalg`). Separate from `JAMMA_BLAS_THREADS`, which controls BLAS only. |
| `JAMMA_SANITIZE` | *(unset)* | **Build-time only.** Comma-separated sanitizer list (e.g. `address,undefined`) injected into compile and link flags by `_build_support/build_models.py`. Used by `.github/workflows/sanitizers.yml`. See `docs/TESTING.md` §1.10 for local repro. |
| `JAMMA_SENTINEL_UB` | *(unset)* | **Build-time only.** When set to `1`, the shared build model injects `-DJAMMA_SENTINEL_UB`, which compiles a known heap-OOB into the `_lmm_accel` module-registration unit. Used by the sanitizer workflow's `asan-sentinel-meta-test` job to verify ASAN is actually catching bugs (distinguishes a clean run from an unwired sanitizer). |
| `JAMMA_FINGERPRINT_OUT` | *(unset)* | **Test-harness only.** Where the test suite writes the C accelerator's bit-exactness fingerprint. `scripts/run-fingerprint.sh` sets it, and `.github/workflows/fingerprint.yml` runs that script on both sides of a PR touching the accelerator. |

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
The C extension still uses every physical core for its OpenMP threads.

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

> **Cache validation.** The eigen cache is keyed by a SHA-256 over its
> determinants: the `.bim` is content-hashed, the `.bed` is fingerprinted by
> size + modification time, plus the MAF and missingness thresholds, any
> `-ksnps` restriction, the analysed-sample set, and the manifest
> `schema_version` (bumping it invalidates all prior caches), stored in
> `<prefix>.loco.cache_manifest.json`. The `.bed` is fingerprinted rather than
> fully hashed to avoid re-reading large genotype files every run; regenerating
> genotypes changes the `.bed` size or timestamp and invalidates the cache.
> When any determinant changes the cache is recomputed rather than silently
> reused. A cache written before the manifest existed has no key; it is
> rejected and recomputed on every read run until regenerated with `-eigen`,
> which writes a manifest.

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
| `--mem-budget` | float | — | Ceiling in GB. Narrows the chunk size the batch/streaming preflight and LOCO/`-gk` size against (so a tight budget can shrink the plan's chunk rather than only reject it), and still raises `MemoryError` if the plan cannot fit within it. |
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
version = "7.2.0"
requires-python = ">=3.11"
```

Runtime dependencies: `bed-reader>=1.0.0`, `numpy>=2.4.6`, `psutil>=5.9.0`,
`threadpoolctl>=3.0.0`, `click>=8.0.0`, `loguru>=0.7.0`, `progressbar2>=4.2.0`.

The numpy floor matches the `numpy==2.4.6` pin in `[build-system].requires`, so
a wheel never builds against newer headers than it runs on.

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
select = [
    "E", "F", "I", "UP", "B",
    "SIM", "RUF", "C4", "PIE", "PERF", "NPY", "PT", "PTH", "BLE",
]
```

Run with `uv run ruff check .` (lint) and `uv run ruff format .` (format). See
`[tool.ruff.lint].ignore` in `pyproject.toml` for the per-rule exemptions and
the reason each one is there.

### Other tool configuration

`pyproject.toml` also configures [refurb](https://github.com/dosisod/refurb)
(`[tool.refurb]`) and [pyrefly](https://pyrefly.org) (`[tool.pyrefly]`). Where
refurb and ruff implement the same rule, the decision lives once in
`[tool.refurb]` rather than in an inline `# noqa`, because ruff flags an
unrecognised code in a noqa it does not own. pyrefly type-checks `src`, `tests`,
and `scripts` against Python 3.11, the floor of `requires-python`. Its gate is
absolute. Any error fails the build.

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

`core.eigen_plan.plan_eigen_driver` picks a driver from the available memory
and vendor capability flags, in priority order:

1. **DSYEVD** (in-place, vendor LAPACK) — fastest; requires `O(N²)` workspace (~240 GB for 100k samples)
2. **DSYEVR** (vendor LAPACK) — lower peak memory (`O(N)` workspace, ~160 GB for 100k samples)
3. **`np.linalg.eigh`** — used when no vendor LAPACK is available, or when `JLINALG_NO_VENDOR_LAPACK` is set

`eigendecompose_kinship` (`lmm/eigen.py`) passes that choice through to
`jlinalg.eigh(K, driver=...)`: `driver="dsyevr"` when the plan picked DSYEVR,
`driver="auto"` otherwise. `jlinalg_eigh_c` honours it directly -- when
`driver="dsyevr"` it skips the DSYEVD attempt outright rather than trying
DSYEVD first and falling back to DSYEVR only on an allocation failure, so a
memory-constrained run never touches pages the plan did not reserve. `eigh`
returns the driver that actually ran as `status.driver_used`, and
`eigendecompose_kinship` logs that value (`Eigendecomp: dsyevr`), not the
planned one, since a DSYEVD allocation failure can still fall through to
DSYEVR even under `driver="auto"`.

Set `JLINALG_NO_VENDOR_LAPACK=1` to force the NumPy fallback for debugging.

### Checking the active backend

```bash
jamma --version
# prints: JAMMA version 7.2.0 (...)
#         Backend: numpy
```

```python
from jamma.jlinalg import blas_backend, blas_is_ilp64
print(blas_backend)    # e.g. "mkl-ilp64", "openblas-ilp64", "numpy-fallback"
print(blas_is_ilp64)   # 1 if ILP64, 0 if not
```

## Bit-Exactness Fingerprint Reproduce Recipe

`.github/workflows/fingerprint.yml` gates every PR touching `src/jamma/lmm/_lmm_*.c`,
`_lmm_*.h`, `_build_support/**`, or the fingerprint scripts themselves. It builds the
C accelerator on both sides of the PR (head and merge base) on one CI runner, runs
`scripts/run-fingerprint.sh` against each build, and diffs the two record files with
`scripts/compare_fingerprints.py`. There is no committed baseline, because digests
depend on the compiler and the CPU, and dev builds use `-march=native`.

To reproduce the same comparison locally:

```bash
# 1. Fingerprint the current worktree (HEAD).
rm -f src/jamma/lmm/_lmm_accel*.so
uv run python -m jamma.lmm._compile_accel
bash scripts/run-fingerprint.sh /tmp/head.txt

# 2. Fingerprint the commit you want to diff against (e.g. the merge base).
git checkout --detach <base-sha>
rm -f src/jamma/lmm/_lmm_accel*.so
uv run python -m jamma.lmm._compile_accel
bash scripts/run-fingerprint.sh /tmp/base.txt
git checkout --force -  # back to the branch

# 3. Rebuild the extension for the branch you're actually working on.
uv run python -m jamma.lmm._compile_accel

# 4. Compare.
uv run python scripts/compare_fingerprints.py /tmp/base.txt /tmp/head.txt
```

`run-fingerprint.sh` runs `tests/lmm_accel/` under the recorder plugin with `-n0`
and a fixed `--randomly-seed=1234`, so both sides drive the accelerator with
identical inputs. Keys present on only one side (a deleted or renamed entry point)
do not fail the comparison; a key present on both sides whose result digest differs
does. Step 2 checks out `<base-sha>` without staging the head's
`scripts/lmm_accel_fingerprint.py` first, unlike the CI job, because a local
reproduce compares two full commits rather than a head diffed against history it
does not have; if the base predates the fingerprint harness there is nothing to
compare against, same as the CI job's check.

A differing digest is the gate's finding, not by itself a defect. A change that
intends to move the last bits fails this job by design. Such a PR states the
intent, names the entry points expected to change, gives the
measured bound against the NumPy path or a high-precision reference, and shows
the GEMMA parity tier passing at the documented tolerances. `fingerprint.yml`
is not in the required-checks ruleset, so the merge is not blocked, and the
reviewer reads the red job as confirmation that only the intended keys moved.

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

The provided `Dockerfile` uses a pinned Python 3.11 Bookworm builder and a
pinned slim-Bookworm runtime with ILP64 NumPy (MKL). The builder compiles this
checkout's native extensions before `/usr/local` is copied into the non-root
runtime image. MKL is x86_64-only — always build and run with
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

The Docker build installs dependencies in this order to preserve ILP64 NumPy:

1. Exact runtime and MKL packages from `docker/requirements-container.txt`
   with dependency resolution disabled
2. Exact `numpy` and `mkl-service` wheels from the custom ILP64 index
3. The current checkout with `--no-deps`, preserving the selected NumPy and
   ensuring the image corresponds to the source being built

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
