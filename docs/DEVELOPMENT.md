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

`uv sync` installs all runtime and dev dependencies (including scipy, which is dev-only). `prek install` sets up pre-commit hooks for ruff lint, ruff format, clang-format, cppcheck, and a pre-push ruff format check across all files.

### Compile C Extensions

The C extensions are not compiled automatically during `uv sync`. Compile them before running tests:

```bash
uv run python -m jamma.lmm._compile_accel
uv run python -m jamma.jlinalg._compile_jlinalg
```

JAMMA falls back to pure Python if extensions are absent, but compiled extensions are required for meaningful test coverage and performance.

**Important:** Compile flags, source lists, and link flags are centralised in `src/jamma/_build_support/compile_and_link.py` and consumed by all three entry points — `hatch_build.py` (wheel builds), `_compile_jlinalg.py`, and `_compile_accel.py` (dev-mode and runtime recompile). Add new sources or flags there, not in the entry points. LAPACK sources inside `jlinalg/src/` are compiled with strict IEEE 754 flags (`-O2 -fno-fast-math`); a pre-commit hook (`scripts/check-compile-flag-literals.py`) rejects bare flag literals (`-O3`, `-fno-fast-math`, etc.) outside `_build_support/`.

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
| `prek run --all-files` | Run all pre-commit hooks across all files |
| `uv build` | Build sdist and wheel |
| `uv run python -m jamma.lmm._compile_accel` | Compile the LMM C extension |
| `uv run python -m jamma.jlinalg._compile_jlinalg` | Compile the jlinalg BLAS extension |
| `uv run python scripts/bench_all_backends.py` | End-to-end backend comparison benchmark |
| `uv run pytest tests/test_jlinalg_dgemm.py tests/test_jlinalg_dsyrk.py tests/test_lmm_accel.py -v -n0 --benchmark-only -m benchmark` | Microbenchmarks (no parallelism) |

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
prek run --all-files       # All hooks: yaml, merge-conflict, ruff, clang-format, cppcheck
```

**C code style:** clang-format (v19.1.7) is enforced by prek on all `.c` files under `src/jamma/jlinalg/src/`. cppcheck static analysis also runs on these files.

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
   - **lint** job: `prek run --all-files` on Python 3.12 (Ubuntu)
   - **test** job: pytest on Linux (3.11, 3.12), ARM macOS (3.12), and Linux with MKL ILP64 numpy
   - **coverage** job: slipcover with `--fail-under 80` (single-threaded, tier0/tier1 only)

5. Slow tests (tier2) run separately on push to master via `.github/workflows/test-slow.yml`.

## Benchmarks

After any change that could affect performance (runner logic, chunk sizing, BLAS threading, C extension, likelihood computation), run the full backend comparison benchmark **without parallelism** to avoid timing contamination:

```bash
uv run python scripts/bench_all_backends.py
```

Use `--runs N` for best-of-N timing and `--gemma-path` to specify a custom GEMMA binary (auto-detects `~/.local/bin/gemma`). Update the README performance table if numbers change significantly.

For per-stage microbenchmarks, always use `-n0` to disable pytest-xdist:

```bash
uv run pytest tests/test_jlinalg_dgemm.py tests/test_jlinalg_dsyrk.py tests/test_lmm_accel.py -v -n0 --benchmark-only -m benchmark
```

## Publishing

PyPI publishing uses GitHub trusted publishing (no API tokens needed locally).

1. Bump `version` in `pyproject.toml`
2. Run `uv lock` and commit the updated `uv.lock`
3. Update `CHANGELOG.md` (move Unreleased items to the new version section)
4. Commit and push to `master`
5. Create a GitHub release: `gh release create v<X.Y.Z> --title "v<X.Y.Z>" --notes "..."`
6. The `.github/workflows/build-wheels.yml` workflow builds wheels for Linux x86_64 and macOS arm64 (CPython 3.11–3.13) and uploads them to PyPI automatically on release

AVX2-optimised wheels are also built and attached to the GitHub release as assets (not uploaded to PyPI — they share platform tags with baseline wheels and would conflict).
