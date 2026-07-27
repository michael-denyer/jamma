# Getting Started with JAMMA

**JAMMA** (Highly-Accelerated Multi-method Mixed-Model Association) is a Python and C
reimplementation of [GEMMA](https://github.com/genetics-statistics/GEMMA) for large-scale
genome-wide association studies (GWAS).

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | `>= 3.11` | 3.11, 3.12, and 3.13 supported |
| NumPy | `>= 2.4.6` | Bundled as a dependency |
| pip | any recent | or `uv` (recommended for development) |

No other system tools are required for standard usage. PLINK binary files (`.bed/.bim/.fam`)
are read natively — PLINK itself is not needed.

**For datasets over 46k samples (Linux and Windows only):** ILP64 numpy from
[michael-denyer/numpy-mkl](https://github.com/michael-denyer/numpy-mkl) is required.
Standard numpy uses 32-bit BLAS integers that overflow at ~46k samples. macOS Accelerate
provides ILP64 natively; no extra steps needed on macOS.

## Installation

### macOS (13.3+)

```bash
pip install jamma
```

That's it. macOS Accelerate BLAS provides native ILP64 support for large matrices.

### Linux and Windows (Intel/AMD x86_64) — datasets under 46k samples

```bash
pip install jamma
```

### Linux and Windows (Intel/AMD x86_64) — datasets over 46k samples

Install ILP64 numpy first, then JAMMA with `--no-deps` to prevent numpy being overwritten:

```bash
pip install psutil loguru threadpoolctl click progressbar2 bed-reader
pip install numpy \
  --index-url https://michael-denyer.github.io/numpy-mkl \
  --force-reinstall --upgrade
pip install jamma --no-deps
```

> **Why `--no-deps`?** `pip install jamma` pulls in `numpy>=2.4.6`, which overwrites the
> ILP64 build with standard LP64 numpy. Installing deps first and using `--no-deps` for
> JAMMA preserves the ILP64 build.

### Docker (Linux/amd64, ILP64 pre-configured)

A pre-built Docker image includes ILP64 numpy. MKL is x86_64-only, so always use
`--platform linux/amd64`:

```bash
docker build --platform linux/amd64 -t jamma .
docker run --platform linux/amd64 -v $(pwd)/data:/data jamma --help
```

### From Source (development)

```bash
git clone https://github.com/michael-denyer/jamma.git
cd jamma
uv sync
```

`uv sync` installs all runtime and dev dependencies including test tools. After this, use
`uv run jamma` instead of `jamma` to run within the managed virtualenv.

## First Run

Verify the installation and check your BLAS backend:

```bash
jamma --help
```

Then run a quick test against the bundled synthetic fixture data:

```bash
# 1. Compute kinship matrix
jamma -gk 1 -bfile tests/fixtures/gemma_synthetic/test -o kinship -outdir /tmp/jamma_test

# 2. Run LMM association (Wald test)
jamma -lmm 1 \
  -bfile tests/fixtures/gemma_synthetic/test \
  -k /tmp/jamma_test/kinship.cXX.npy \
  -o assoc \
  -outdir /tmp/jamma_test

# 3. Inspect results
head /tmp/jamma_test/assoc.assoc.txt
```

Expected output columns: `chr rs ps n_miss allele1 allele0 af beta se logl_H1 l_remle p_wald`

## Common Setup Issues

### Wrong Python version

JAMMA requires Python 3.11 or newer. Check your version:

```bash
python --version
```

If you have multiple Python versions installed, use `python3.11 -m pip install jamma` or
use `uv` which handles version selection automatically.

### NumPy version too old

JAMMA requires NumPy 2.4.6 or newer. If you see import errors about missing symbols, upgrade:

```bash
pip install "numpy>=2.4.6" --upgrade
```

### ILP64 overwritten after pip install

If you followed the ILP64 install steps and then installed another package that depends on
numpy, the ILP64 build may have been replaced. Verify with:

```bash
python -c "import numpy as np; cfg = np.show_config(mode='dicts'); print(cfg['Build Dependencies']['blas'])"
```

If the BLAS name does not contain `ilp64`, re-run the ILP64 install steps. Always install
other packages before ILP64 numpy, and always use `pip install jamma --no-deps`.

### C extensions not compiled (development installs)

After cloning from source, both C extensions must be compiled. `uv sync` does
not build them:

```bash
uv sync
uv run python -m jamma.lmm._compile_accel
uv run python -m jamma.jlinalg._compile_jlinalg
```

JAMMA falls back to pure Python without them. On mouse_hs1940 that fallback runs
LMM roughly 5x to 7x slower, and the streaming runner is unavailable entirely.

### Missing environment variables

JAMMA has no required environment variables for basic use. The optional `JAMMA_BACKEND`
variable forces a specific compute backend (`numpy`, `numpy-streaming`). Omit it to use
auto-detection.

## Next Steps

- [ARCHITECTURE.md](ARCHITECTURE.md) — How JAMMA is structured internally
- [CONFIGURATION.md](CONFIGURATION.md) — CLI flags, environment variables, and output formats
- [USER_GUIDE.md](USER_GUIDE.md) — Full CLI reference, Python API, and LOCO analysis
- [DEVELOPMENT.md](DEVELOPMENT.md) — Local dev setup, build commands, and code style
- [TESTING.md](TESTING.md) — Running tests and validation against GEMMA
