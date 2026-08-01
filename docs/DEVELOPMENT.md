# Development Guide

This guide covers local setup, build commands, code style, and the PR process for contributing to JAMMA.

## Local Setup

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), [prek](https://prek.j178.dev), and a C compiler (gcc or clang) for the C extensions.

```bash
git clone https://github.com/michael-denyer/jamma.git
cd jamma
uv sync
prek install
```

`uv sync` installs all runtime and dev dependencies (including scipy, which is dev-only). `prek install` sets up the hooks listed in `.pre-commit-config.yaml`: ruff lint and format, pyrefly, clang-format, cppcheck, markdownlint, `maid` (mermaid syntax), lychee, typos, actionlint, zizmor, shellcheck, vulture, refurb, and the JAMMA-specific gates, `check_doc_anchors.py` among them. Two more run at pre-push, because prek only inspects staged files and would miss them. Those are a repo-wide `ruff format --check` and the C-extension freshness check.

`check_doc_anchors.py` verifies that every `path#Lnnn` link in the docs still lands on the symbol it names, which neither lychee nor markdownlint can do because the *file* resolves and only the line is wrong. Read its guarantee narrowly: for CODEMAP's tables it has to guess which symbol a row means, so a green run says no anchor is provably wrong rather than every anchor is provably right. The module docstring has the details under "What a passing run does not prove".

### Compile C Extensions

The C extensions are not compiled automatically during `uv sync`. Compile them before running tests:

```bash
uv run python -m jamma.lmm._compile_accel
uv run python -m jamma.jlinalg._compile_jlinalg
```

JAMMA falls back to pure Python if extensions are absent, but compiled extensions are required for meaningful test coverage and performance.

**Important:** Native build policy is centralised in `src/jamma/_build_support/build_models.py`; toolchain execution lives in `build_execution.py`; and `compile_and_link.py` composes both behind `run_build` / `compile_extension`. `hatch_build.py` (wheel builds) calls `run_build` directly; `_compile_jlinalg.py` and `_compile_accel.py` (dev-mode and runtime recompile) are thin shims that bind `compile_extension` to their `BuildSpec` and, in their `__main__` block, prove the freshly compiled `.so` imports in a fresh subprocess rather than in-process. Add new sources or flags to `build_models.py`, not the entry points. LAPACK sources inside `jlinalg/src/` are compiled with strict IEEE 754 flags (`-O2 -fno-fast-math`); a pre-commit hook (`scripts/check_compile_flag_literals.py`) rejects bare flag literals (`-O3`, `-fno-fast-math`, etc.) outside `_build_support/`.

After modifying C source, recompile in place:

```bash
uv run python -c "from jamma.jlinalg._compile_jlinalg import compile_extension; compile_extension()"
```

## Build Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies (runtime + dev) |
| `uv run jamma` | Run the CLI |
| `uv run pytest tests/ -x` | Run the test suite (stops on first failure) |
| `uv run pytest tests/ -x -k lmm` | Run only tests matching `lmm` |
| `uv run ruff check .` | Lint all Python source |
| `uv run ruff format .` | Format all Python source |
| `uv run pyrefly check` | Static type check (gate is zero errors) |
| `prek run --all-files` | Run all pre-commit hooks across all files |
| `uv build` | Build sdist and wheel |
| `uv run python -m jamma.lmm._compile_accel` | Compile the LMM C extension |
| `uv run python -m jamma.jlinalg._compile_jlinalg` | Compile the jlinalg BLAS extension |
| `uv run python scripts/bench_all_backends.py` | End-to-end backend comparison benchmark |
| `uv run pytest tests/test_jlinalg_dgemm.py tests/test_jlinalg_dsyrk.py tests/lmm_accel/ -v -n0 --benchmark-only -m benchmark` | Microbenchmarks (no parallelism) |

## Code Style

**Linter and formatter:** [ruff](https://docs.astral.sh/ruff/) — configured in `pyproject.toml`.

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

This covers pyflakes (`F`), pycodestyle (`E`), isort import sorting (`I`), pyupgrade (`UP`), flake8-bugbear (`B`), flake8-simplify (`SIM`), ruff-specific checks (`RUF`), flake8-comprehensions (`C4`), flake8-pie (`PIE`), perflint (`PERF`), numpy-specific rules (`NPY`), flake8-pytest-style (`PT`), flake8-use-pathlib (`PTH`), and flake8-blind-except (`BLE`). See `pyproject.toml` for the per-rule `ignore` list.

Pre-commit hooks run ruff on staged files automatically. A pre-push hook runs `ruff format --check .` across all files to catch drift in unstaged files. To fix locally before pushing:

```bash
uv run ruff check .        # Lint (shows violations)
uv run ruff format .       # Format in place
uv run pyrefly check       # Static types (must be zero errors)
prek run --all-files       # Every hook, over every file
```

**Type checking:** [pyrefly](https://pyrefly.org) is pinned exactly in `pyproject.toml`, because the gate is "zero errors" and a minor bump that adds a check turns green into red with no code change. It type-checks `src`, `tests`, and `scripts` against Python 3.11, the floor of `requires-python`, so 3.12-only syntax is caught. There is no baseline file. Group work by root cause rather than by file, since one loose declaration scatters errors across every caller.

**C code style:** prek runs clang-format (v19.1.7) over `.c` files under `src/jamma/jlinalg/src/`, and cppcheck over both C trees, `src/jamma/jlinalg/src/` and `src/jamma/lmm/`.

**Conventions:**

- Docstrings: Google style, required on all public functions
- Type hints: Required on all public function signatures
- Line length: 88 characters
- Imports: sorted by ruff (isort rules, `I` ruleset)

## Branch Conventions

Branch from `master`. Use the following prefixes:

| Prefix | Use case |
|--------|----------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `perf/` | Performance improvements |
| `refactor/` | Refactoring without behaviour change |

Commit messages follow conventional commits:

```text
feat: add multi-pass LOCO kinship batching
fix: streaming runner chunk boundary handling
perf: skip impute_and_center when no target chromosomes in chunk
refactor: extract _yield_loco_matrices helper
```

## PR Process

1. Ensure all tests pass and hooks are clean:

   ```bash
   uv run pytest tests/ -x
   prek run --all-files
   ```

2. Open a pull request against `master`.

3. PR description must include:
   - **Summary**: What changed and why
   - **Test plan**: How to verify the changes

4. CI runs the following checks automatically (`.github/workflows/ci.yml`):
   - **lint** job: `uv lock --check`, `ruff check --no-fix`, `ruff format --check`, `uv run pyrefly check`, then `prek run --all-files` on Python 3.12 (Ubuntu)
   - **test** job: pytest on Linux (3.11, 3.12), ARM macOS (3.12), and Linux with MKL ILP64 numpy
   - **package-smoke** job: builds the sdist and wheel, asserts both ship `_build_support/`, then installs the wheel in a clean venv and imports it
   - **coverage** job: slipcover with `--fail-under 80` (single-threaded, tier0/tier1 only), plus per-subsystem floors from `scripts/check_subsystem_coverage.py`
   - **link-check** job: lychee in `--offline` mode over every `.md`

5. Other workflows gate a PR without ever running locally:

   | Workflow | What it catches |
   |----------|-----------------|
   | `test-slow.yml` | `tier2 or slow`, which the default `addopts` filter skips (push to master) |
   | `fingerprint.yml` | Bit-level drift in the C accelerator, on any PR touching `_lmm_*.c`, `_lmm_*.h`, or `_build_support/` |
   | `sanitizers.yml` | ASAN and UBSAN over the C extensions, weekly |
   | `flaky-detect.yml` | Repeated runs under five seeds, to surface flaky tests |
   | `codeql.yml`, `security.yml` | Static analysis and dependency scanning |
   | `link-check-external.yml` | Online link check, weekly; opens an issue instead of blocking merge |

## Benchmarks

After any change that could affect performance (runner logic, chunk sizing, BLAS threading, C extension, likelihood computation), run the full backend comparison benchmark **without parallelism** to avoid timing contamination:

```bash
uv run python scripts/bench_all_backends.py
```

Use `--runs N` for best-of-N timing and `--gemma-path` to specify a custom GEMMA binary (auto-detects `~/.local/bin/gemma`). Update the README performance table if numbers change significantly.

For per-stage microbenchmarks, always use `-n0` to disable pytest-xdist:

```bash
uv run pytest tests/test_jlinalg_dgemm.py tests/test_jlinalg_dsyrk.py tests/lmm_accel/ -v -n0 --benchmark-only -m benchmark
```

To compare large-N performance changes, use the drift-aware stage benchmark.
It interleaves two source trees in ABBA/BAAB order, hashes each stage result,
and reports kinship, eigendecomposition, rotation, and mode-4 timings
separately. Start with a small same-tree smoke run, then use dimensions that
fit the target machine:

```bash
uv run python scripts/bench_large_n_stages.py \
  --a-root . --b-root . --samples 256 --snps 128 --blocks 1

uv run python scripts/bench_large_n_stages.py \
  --a-root /path/to/base --b-root /path/to/candidate \
  --samples 10000 --snps 1000 --blocks 4
```

Treat a result as inconclusive when paired block deltas change sign. The
eigendecomposition stage grows rapidly with sample count, so choose dimensions
from the memory budget rather than copying the example blindly.

## Publishing

PyPI publishing uses GitHub trusted publishing (no API tokens needed locally).

1. Bump `version` in `pyproject.toml`
2. Run `uv lock` and commit the updated `uv.lock`
3. Update `CHANGELOG.md` (move Unreleased items to the new version section)
4. Commit and push to `master`
5. Create a GitHub release: `gh release create v<X.Y.Z> --title "v<X.Y.Z>" --notes "..."`
6. The `.github/workflows/build-wheels.yml` workflow builds wheels for Linux x86_64 and macOS arm64 (CPython 3.11–3.14) and uploads them to PyPI automatically on release

AVX2-optimised wheels are also built and attached to the GitHub release as assets (not uploaded to PyPI — they share platform tags with baseline wheels and would conflict).
