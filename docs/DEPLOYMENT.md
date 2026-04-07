<!-- generated-by: gsd-doc-writer -->
# JAMMA Deployment

JAMMA is a Python CLI tool distributed as a PyPI package and a Docker image. This document
covers the two primary deployment targets — Docker and PyPI — along with the CI/CD pipeline
that builds and publishes release artifacts.

## Deployment Targets

| Target | Config file | Purpose |
|---|---|---|
| Docker (ILP64) | `Dockerfile` | Containerised large-scale GWAS with MKL ILP64 numpy pre-installed |
| PyPI | `.github/workflows/build-wheels.yml` | Standard package distribution via `pip install jamma` |

### Docker

The `Dockerfile` builds a slim `python:3.11-slim` image with MKL-backed ILP64 numpy for
large-scale GWAS (>46k samples). MKL is x86_64-only — always build and run with
`--platform linux/amd64`.

**Build:**

```bash
docker build --platform linux/amd64 -t jamma .
```

The build installs packages in strict order to preserve the ILP64 numpy build:

1. `mkl` — MKL runtime libraries
2. `numpy` from `https://michael-denyer.github.io/numpy-mkl` — ILP64 MKL-backed numpy
3. Runtime deps (`psutil loguru threadpoolctl click progressbar2 bed-reader`)
4. `jamma --no-deps` — prevents numpy downgrade back to LP64

ILP64 is verified at build time with an assertion on the BLAS name; the build fails if
LP64 numpy is active.

**Run:**

```bash
# Kinship matrix computation
docker run --platform linux/amd64 -v $(pwd)/data:/data jamma \
  -gk 1 -bfile /data/study -o /data/output

# LMM association analysis (Wald test)
docker run --platform linux/amd64 -v $(pwd)/data:/data jamma \
  -lmm 1 -bfile /data/study -k /data/k.cXX.txt -o /data/output
```

Mount your data directory to `/data` and use `/data/...` paths inside the container.
The container runs as a non-root `jamma` user (uid 1000); `/data` is pre-created and
owned by that user.

**Note:** Databricks-specific notebooks and a Databricks-targeted Dockerfile live in
`../jamma-databricks/` (a separate repository, not part of this repo).

### PyPI

JAMMA is published to PyPI as both platform wheels and a source distribution. See the
[Build pipeline](#build-pipeline) section for how artifacts are produced.

**Standard install (macOS, ARM Linux, small datasets on Linux/Windows):**

```bash
pip install jamma
```

**ILP64 install (Linux/Windows, >46k samples):**

The install order is critical. Installing jamma before its deps, or after numpy without
`--no-deps`, will overwrite the ILP64 numpy build with LP64 and silently break
large-scale eigendecomposition.

```bash
pip install psutil loguru threadpoolctl click progressbar2 bed-reader
pip install numpy \
  --index-url https://michael-denyer.github.io/numpy-mkl \
  --force-reinstall --upgrade
pip install jamma --no-deps
```

## Build Pipeline

The `build-wheels.yml` workflow triggers on every GitHub Release (published event) and
on manual dispatch.

### Baseline wheels (`build_wheels` job)

Runs on `ubuntu-latest` and `macos-latest`.

| Setting | Value |
|---|---|
| Python versions | CPython 3.11, 3.12, 3.13 |
| Linux arch | x86_64 (manylinux, not musllinux or 32-bit) |
| macOS arch | arm64 only (`MACOSX_DEPLOYMENT_TARGET=14.0`) |
| Wheel repair (Linux) | `auditwheel repair` |
| Wheel repair (macOS) | `delocate-wheel` |
| Smoke tests | `smoke_test_c_extension.py`, `smoke_test_eigen_extension.py`, `smoke_test_jlinalg.py` |

### AVX2-optimised wheels (`build_avx2_wheels` job)

Linux x86_64 only, compiled with `-march=x86-64-v3 -mavx2`. These wheels carry the
same platform tags as baseline wheels and are **not uploaded to PyPI** — they would
conflict. Instead they are attached to the GitHub Release as additional assets. Users
who want AVX2 can download the `.whl` directly and install with:

```bash
pip install jamma-<version>-cp311-cp311-manylinux_x86_64_avx2.whl --force-reinstall
```

### Source distribution (`build_sdist` job)

Built with `uv build --sdist` on `ubuntu-latest`.

### PyPI upload (`upload_pypi` job)

Uploads baseline wheels (Linux + macOS) plus the sdist to PyPI using GitHub trusted
publishing (OIDC — no API tokens required). The `pypi` environment gate must approve
the deployment.

**Steps to cut a release:**

1. Bump `version` in `pyproject.toml`.
2. Update `CHANGELOG.md` — move Unreleased items into a new version section.
3. Commit and push to `master`.
4. Create a GitHub release:
   ```bash
   gh release create v<X.Y.Z> --title "v<X.Y.Z>" --notes "..."
   ```
5. The `build-wheels.yml` workflow fires automatically on the published event and
   uploads to PyPI.

## Environment Setup

JAMMA has no required environment variables. Optional variables that affect runtime
behaviour are documented in [CONFIGURATION.md](CONFIGURATION.md#environment-variables).

For production Docker runs, the variables most likely to need tuning are:

| Variable | Recommendation |
|---|---|
| `JAMMA_BLAS_THREADS` | Set to the number of physical cores available to the container |
| `JAMMA_LOCO_WORKERS` | Increase only if RAM budget allows (each worker holds a full K_loco matrix) |
| `OMP_NUM_THREADS` | Match to container CPU limit to avoid oversubscription |

## Rollback Procedure

**PyPI:** PyPI does not support deleting or replacing a published version. To roll back:

1. Yank the bad version on PyPI (<!-- VERIFY: PyPI yank via project settings dashboard -->):
   ```bash
   pip install twine
   # or use the PyPI web UI to yank
   ```
2. Publish a patch release with the fix as `v<X.Y.Z+1>`.

**Docker:** Redeploy using the previous image tag. If using a registry:

```bash
docker pull jamma:<previous-tag>
docker run --platform linux/amd64 jamma:<previous-tag> ...
```

<!-- VERIFY: Docker registry URL or image hosting location if images are published to a registry -->

## Monitoring

No monitoring libraries (`@sentry/*`, `dd-trace`, `newrelic`, `@opentelemetry/*`) are
included as dependencies. JAMMA logs to stderr via `loguru`. For production deployments,
capture stderr and route it to your preferred log aggregation platform.

<!-- VERIFY: Any centralised logging or alerting infrastructure for published releases -->
