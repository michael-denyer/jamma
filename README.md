# JAMMA

[![CI](https://github.com/michael-denyer/jamma/actions/workflows/ci.yml/badge.svg)](https://github.com/michael-denyer/jamma/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**JAX-Accelerated Mixed Model Association** — A modern Python reimplementation of [GEMMA](https://github.com/genetics-statistics/GEMMA) for genome-wide association studies (GWAS).

## Features

- **GEMMA-compatible**: Drop-in replacement with identical CLI flags
- **Numerical equivalence**: Results match GEMMA within floating-point tolerance
- **Fast**: 4-5x faster than GEMMA on CPU, with GPU acceleration support
- **Pure Python**: JAX + NumPy stack, no C++ compilation required

## Installation

```bash
pip install jamma
```

Or with uv:

```bash
uv add jamma
```

## Quick Start

```bash
# Compute kinship matrix
jamma -o output gk -bfile data/my_study -gk 1

# Run LMM association
jamma -o results lmm -bfile data/my_study -k output.cXX.txt -lmm 1
```

## Performance

Benchmark on mouse_hs1940 (1,940 samples × 12,226 SNPs):

| Operation | GEMMA | JAMMA | Speedup |
|-----------|-------|-------|---------|
| Kinship (`-gk 1`) | 6.5s | 0.9s | **7.5x** |
| LMM (`-lmm 1`) | 19.5s | 4.7s | **4.2x** |
| **Total** | 26.0s | 5.5s | **4.7x** |

## Documentation

- [User Guide](docs/USER_GUIDE.md) — Installation, usage examples, CLI reference
- [Mathematical Equivalence](docs/MATHEMATICAL_EQUIVALENCE.md) — Validation against GEMMA

## Requirements

- Python 3.11+
- JAX (CPU or GPU)
- NumPy, SciPy

## License

MIT
