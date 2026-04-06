<!-- generated-by: gsd-doc-writer -->
# Testing

JAMMA uses pytest with parallel execution (`pytest-xdist`), test randomization
(`pytest-randomly`), and property-based testing (`hypothesis`). Tests are organized
into tiers to balance CI speed against thorough numerical validation.

## Test Framework and Setup

| Tool | Version | Purpose |
|------|---------|---------|
| `pytest` | `>=8.0.0` | Test runner |
| `pytest-xdist` | `>=3.5.0` | Parallel test execution (`-n 3`) |
| `pytest-randomly` | `>=3.15.0` | Test randomization to detect order dependencies |
| `pytest-cov` | `>=4.0.0` | Coverage reporting |
| `pytest-benchmark` | `>=5.0.0` | Microbenchmarks (run separately) |
| `hypothesis` | `>=6.100.0` | Property-based tests |
| `scipy` | `>=1.10.0` | Test-only dependency for `scipy.stats` (dev group only) |

**Important:** `scipy` is a dev-only dependency. It must never be added as a runtime
dependency — installing scipy overwrites the ILP64 numpy build and breaks large-scale
eigendecomposition. See `docs/ARCHITECTURE.md` for background.

Install all dev dependencies with:

```bash
uv sync
```

C extensions must be compiled before running tests:

```bash
uv run python -m jamma.lmm._compile_accel
uv run python -m jamma.jlinalg._compile_jlinalg
```

## Running Tests

### Default test run (tier0 + tier1, parallel)

The default configuration in `pyproject.toml` runs fast and parity tests in parallel,
skips benchmarks, and randomizes test order:

```bash
uv run pytest tests/ -x
```

This picks up `addopts` automatically:
`-n 3 --randomly-seed=last --benchmark-skip -m 'not slow and not tier2 and not tier3 and not benchmark' --no-cov`

**Never use `-n auto`** — it spawns too many workers and contaminates BLAS-threaded
tests. Always use `-n 3` (or `-n0` for benchmarks).

### Run by tier

```bash
# Fast unit tests only (~30s total)
uv run pytest -m tier0 -x

# Fast + GEMMA parity tests
uv run pytest -m "tier0 or tier1" -x

# Exclude slow/memory tests explicitly
uv run pytest -m "not tier2" -x

# Scale tests (run locally — requires large memory)
uv run pytest -m tier2 -v -o 'addopts='
```

### Run a single file or test

```bash
uv run pytest tests/test_likelihood_numpy.py -x
uv run pytest tests/test_kinship_validation.py -k "test_kinship_matches_gemma" -x
```

### Reproduce a specific random order

```bash
# --randomly-seed=last reuses the seed from the previous run (pyproject.toml default)
uv run pytest tests/ -x

# Specify an explicit seed
uv run pytest tests/ --randomly-seed=12345

# Disable randomization temporarily
uv run pytest tests/ -p no:randomly
```

### Watch mode

pytest-xdist does not include a built-in watch mode. Use `pytest-watch` or re-run
manually during development.

## Test Tier System

JAMMA uses a three-tier system to balance CI speed with thorough validation.

| Marker | Speed | Description | Runs in CI |
|--------|-------|-------------|-----------|
| `tier0` | < 5s each | Pure computation, no I/O, no GEMMA reference | Every push/PR |
| `tier1` | < 60s each | Numerical parity against GEMMA reference fixtures | Every push/PR |
| `tier2` | Minutes | Large sample counts (10k+), memory-constrained scenarios | On push to master (Slow Tests workflow) |
| `tier3` | > 60s | Heavy scale tests, local only | Never in CI |
| `slow` | — | Alias for tier2 | On push to master |
| `benchmark` | — | `pytest-benchmark` microbenchmarks | Manually only |

Markers are declared in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Writing New Tests

### File naming and location

All tests live in `tests/`. File names follow the pattern `test_<module>.py`.

```
tests/
├── conftest.py                  # Shared fixtures
├── fixtures/                    # Reference data
│   ├── gemma_synthetic/         # Synthetic PLINK data + GEMMA reference output
│   ├── gemma_all_tests/
│   ├── gemma_covariate/
│   ├── gemma_loco/
│   ├── gemma_score/
│   ├── kinship/
│   ├── lmm/
│   └── mouse_hs1940/            # Real mouse dataset
├── test_likelihood_numpy.py
├── test_kinship_validation.py
├── test_hypothesis.py           # Hypothesis property tests
└── ...
```

### Shared fixtures (`conftest.py`)

Key fixtures available to all tests:

| Fixture | Description |
|---------|-------------|
| `sample_plink_data` | Path prefix for synthetic PLINK files (`tests/fixtures/gemma_synthetic/test`) |
| `output_dir` | Temporary output directory (wraps `tmp_path`) |
| `tolerance_config` | Default `ToleranceConfig` for numerical comparisons |
| `synthetic_covariate_data_ncvt2` | Rotated data with 2 covariates (200 samples, 50 SNPs) |
| `synthetic_covariate_data_ncvt4` | Rotated data with 4 covariates (200 samples, 50 SNPs) |

### Marking tests

Always mark tests with an appropriate tier. Unmarked tests run under the default filter:

```python
import pytest

@pytest.mark.tier0
def test_pab_computation():
    ...

@pytest.mark.tier1
def test_assoc_matches_gemma(sample_plink_data, tolerance_config):
    ...

@pytest.mark.tier2
def test_streaming_large_dataset():
    ...
```

### Property-based tests (Hypothesis)

Property tests live in `tests/test_hypothesis.py`. Use `@given` with custom strategies
for genetic data:

```python
from hypothesis import given, settings
from hypothesis import strategies as st

@given(st.integers(min_value=10, max_value=100))
def test_kinship_symmetry(n_samples):
    ...
```

Run hypothesis tests specifically:

```bash
uv run pytest tests/test_hypothesis.py -x
```

### Numerical comparison tolerances

Use `ToleranceConfig` from `src/jamma/validation/tolerances.py` for all comparisons
against GEMMA reference output. The default configuration is calibrated from formal
error propagation analysis:

```python
from jamma.validation import ToleranceConfig

config = ToleranceConfig()
np.testing.assert_allclose(result, reference, rtol=config.pvalue_rtol, atol=config.atol)
```

## Coverage Requirements

Coverage is measured with `slipcover` (not `pytest-cov`) in CI. The threshold is
enforced in the coverage workflow:

```bash
uv run slipcover --source src/jamma --fail-under 80 -m pytest \
  -m "not tier2 and not tier3 and not slow and not benchmark" -v -n0 -o 'addopts='
```

No per-module thresholds are configured — only the overall 80% line coverage threshold
is enforced.

## Benchmarks

Benchmarks use `pytest-benchmark` and must be run separately from the normal suite,
always with `-n0` (no parallelism) to avoid cross-test timing interference.

### Microbenchmarks (per-stage)

```bash
uv run pytest tests/test_jlinalg_dgemm.py tests/test_jlinalg_dsyrk.py tests/test_lmm_accel.py \
  -v -n0 --benchmark-only -m benchmark
```

### End-to-end backend comparison

```bash
uv run python scripts/bench_all_backends.py
```

This benchmarks GEMMA and the NumPy+C backend on the `mouse_hs1940` dataset. Use
`--runs N` for best-of-N averaging. Results should be compared against the Performance
table in `README.md` after any change to runner logic, chunk sizing, BLAS threading,
C extensions, or likelihood computation.

## CI Integration

### CI workflow (`ci.yml`)

Trigger: push and pull requests to `main` / `master`.

| Job | Matrix | Test command |
|-----|--------|-------------|
| `lint` | ubuntu/Python 3.12 | `prek run --all-files` |
| `test` | Linux 3.11, Linux 3.12, ARM Mac 3.12, Linux MKL ILP64 | `pytest -m "not tier2 and not tier3 and not slow and not benchmark" -v -n 3` |
| `coverage` | ubuntu/Python 3.12 | `slipcover --fail-under 80 -m pytest ... -n0` |

The test job overrides `addopts` via `-o 'addopts='` so CI controls markers and
parallelism explicitly, independent of `pyproject.toml` defaults.

### Slow Tests workflow (`test-slow.yml`)

Trigger: push to `main` / `master`, or manual dispatch.

Runs tier2 and `slow`-marked tests (excluding tier3):

```bash
uv run pytest -m "(tier2 or slow) and not tier3" -v -o 'addopts=' --no-cov
```

### MKL ILP64 matrix variant

One CI matrix entry installs MKL ILP64 numpy from `michael-denyer/numpy-mkl` before
running the standard test suite, verifying BLAS dispatch correctness under ILP64.
