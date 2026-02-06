# Phase 18: Correctness & Performance - Research

**Researched:** 2026-02-06
**Domain:** Kinship mode validation, JAX runner optimization
**Confidence:** HIGH

## Summary

This phase addresses one correctness gap and two performance inefficiencies in the JAMMA codebase. All three issues are well-scoped and localized to specific files with clear before/after states.

**CORR-01 (Kinship mode 2):** The CLI `gk` command at line 148 of `cli.py` silently falls back to mode 1 when mode 2 is requested, printing a warning but still computing the centered (mode 1) kinship. The fix is to raise `NotImplementedError` in the CLI command before any computation occurs. The `compute_centered_kinship` and `compute_kinship_streaming` functions in `kinship/compute.py` are mode-agnostic (they only implement mode 1 math) and don't need modification. An existing CLI test (`test_cli_lmm_mode_2_accepted`) tests LMM mode 2, not kinship mode 2 -- no naming collision.

**PERF-01 (Cache U.T):** Both `runner_jax.py` and `runner_streaming.py` compute `U.T` multiple times: once for `UtW` and `Uty` setup (lines 174-175 / 269-270), and again inside `_prepare_chunk` for every chunk (line 245 / 348). The transpose `U.T` should be computed once and reused. This is a `numpy.ndarray.T` operation that creates a view (zero-copy), but the subsequent `U.T @ geno_chunk` forces a full matrix multiplication each time with the transposed layout. Pre-computing `UT = U.T` and storing it would eliminate repeated property access and make the code clearer, though the main performance benefit is in signaling intent -- numpy `.T` is already a view. The more meaningful optimization is computing `UT` once as a contiguous array via `np.ascontiguousarray(U.T)` so that the BLAS `matmul` in `_prepare_chunk` operates on contiguous memory, improving cache behavior for the `UT @ geno_chunk` multiplication that runs once per chunk.

**PERF-02 (Pre-allocate result arrays):** In `runner_jax.py`, results accumulate via `list.append()` of JAX array slices (lines 203-227, 374-403), then `jnp.concatenate` at lines 411-436. In `runner_streaming.py`, the pattern uses dict-based accumulators (`_init_accumulators` / `_append_chunk_results`) with the same list-of-JAX-arrays pattern, plus `all_results.append(result)` for in-memory mode (line 475). The list-of-JAX-arrays pattern in `runner_jax.py` can be replaced with pre-allocated numpy arrays indexed by position. The streaming runner's in-memory path (`all_results: list[AssocResult]`) also uses append, but this is less impactful because the streaming runner already supports `IncrementalAssocWriter` for disk output.

**Primary recommendation:** All three changes are independent and can be done in any order. CORR-01 is a one-line fix with a test. PERF-01 and PERF-02 are internal refactors that don't change any public API or test assertions.

## Standard Stack

No new libraries needed. All changes use existing numpy/JAX patterns.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | existing | Array pre-allocation, contiguous transpose | Already in use |
| jax | existing | Device arrays, concatenation | Already in use |

## Architecture Patterns

### CORR-01: Where to Raise NotImplementedError

The error should be raised in `cli.py:gk_command()` at line 148, replacing the current warning+fallback:

```python
# Current (line 148-152 of cli.py):
if mode == 2:
    typer.echo(
        "Warning: Mode 2 (standardized) not yet implemented, "
        "using mode 1 (centered)"
    )

# Required:
if mode == 2:
    raise NotImplementedError(
        "Kinship mode 2 (standardized) is not yet implemented. "
        "Use -gk 1 for centered relatedness matrix."
    )
```

**Location analysis:**
- `cli.py:gk_command()` line 148 -- the ONLY place where `-gk 2` is handled
- `kinship/compute.py` -- has no mode parameter at all; `compute_centered_kinship` and `compute_kinship_streaming` only implement mode 1 math
- `kinship/__init__.py` -- re-exports only mode 1 functions
- No kinship mode parameter exists in the LMM runner path (LMM mode 2 is LRT, which is completely different)

**Confidence:** HIGH -- verified by reading all kinship and CLI code paths.

### PERF-01: U.T Computation Sites

Every `U.T` usage in both runners:

**runner_jax.py:**
1. Line 174: `UtW = U.T @ W` (setup, once)
2. Line 175: `Uty = U.T @ phenotypes` (setup, once)
3. Line 245: `UtG_chunk = np.ascontiguousarray(U.T @ geno_chunk)` (inside `_prepare_chunk`, once per chunk)

**runner_streaming.py:**
1. Line 269: `UtW = U.T @ W` (setup, once)
2. Line 270: `Uty = U.T @ phenotypes` (setup, once)
3. Line 348: `UtG_chunk = np.ascontiguousarray(U.T @ geno_jax_chunk)` (inside `_prepare_jax_chunk`, once per JAX chunk per file chunk)

**Pattern for fix:**
```python
# After eigendecomposition, compute UT once as contiguous array
UT = np.ascontiguousarray(U.T)  # (n_samples, n_samples), contiguous for BLAS

# Use UT everywhere
UtW = UT @ W
Uty = UT @ phenotypes

# In _prepare_chunk (closure captures UT):
UtG_chunk = np.ascontiguousarray(UT @ geno_chunk)
```

**Note on `_prepare_chunk` as a closure:** In `runner_jax.py`, `_prepare_chunk` is defined at line 229 as a nested function that captures `U` from the enclosing scope. The fix changes it to capture `UT` instead. Same pattern in `runner_streaming.py` with `_prepare_jax_chunk` at line 332.

**Performance impact analysis:**
- `U` is `(n_samples, n_samples)` numpy array. `.T` returns a non-contiguous view (Fortran-order when original is C-order).
- `np.ascontiguousarray(U.T)` creates a contiguous C-order copy once. Subsequent BLAS matmuls with contiguous inputs avoid internal copy/transpose penalties.
- For n_samples=1940 (mouse dataset), U is 30MB -- negligible copy cost once.
- For n_samples=200k (target scale), U is 320GB -- the eigendecomposition itself dominates. But the `UT @ geno_chunk` runs once per chunk (potentially hundreds of times), so contiguous memory layout matters.

**Confidence:** HIGH -- verified all `U.T` usage sites by grep.

### PERF-02: Result Accumulation Analysis

**runner_jax.py current pattern:**

```python
# Lines 202-227: Initialize lists per mode
all_lambdas = []    # mode 1, 4
all_logls = []      # mode 1, 4
all_betas = []      # mode 1, 3, 4
all_ses = []        # mode 1, 3, 4
all_pwalds = []     # mode 1, 4
all_lambdas_mle = []  # mode 2, 4
all_logls_mle = []    # mode 2, 4
all_p_lrts = []       # mode 2, 4
all_p_scores = []     # mode 3, 4

# Lines 374-403: Append per chunk (JAX array slices)
all_lambdas.append(best_lambdas[:slice_len])

# Lines 411-436: Concatenate and transfer to host
best_lambdas_np = np.asarray(jnp.concatenate(all_lambdas))
```

**Key insight:** These `all_*` lists accumulate **JAX arrays** (device-resident), not Python scalars. The `jnp.concatenate` at the end does a single device-to-host transfer. This is actually a reasonable pattern for JAX because:
1. Pre-allocating on-device arrays would require knowing chunk count upfront
2. JAX array slicing is zero-copy on device
3. The final `jnp.concatenate + np.asarray` is one bulk transfer

**However**, the requirement says "pre-allocate result arrays to full SNP count." This means:
- Pre-allocate numpy arrays of size `n_filtered` before the chunk loop
- Write results by index (`results_array[start:end] = values`) instead of appending
- Eliminates the `jnp.concatenate` step and list overhead

**Proposed pattern for runner_jax.py:**
```python
# Pre-allocate before chunk loop
n_filtered = len(snp_indices)
if lmm_mode in (1, 4):
    lambdas_out = np.empty(n_filtered, dtype=np.float64)
    logls_out = np.empty(n_filtered, dtype=np.float64)
if lmm_mode in (1, 3, 4):
    betas_out = np.empty(n_filtered, dtype=np.float64)
    ses_out = np.empty(n_filtered, dtype=np.float64)
# ... etc per mode

# In chunk loop, write by index:
chunk_start_idx = ... # track position
chunk_end_idx = chunk_start_idx + actual_chunk_len
lambdas_out[chunk_start_idx:chunk_end_idx] = np.asarray(best_lambdas[:slice_len])
# ... etc

# After loop: arrays are already numpy, no concatenate needed
```

**Trade-off:** This changes the transfer pattern from "accumulate JAX arrays, one bulk transfer" to "transfer per chunk into pre-allocated numpy." For GPU, the current pattern of keeping arrays on device until final transfer is preferable. For CPU (the common case), pre-allocation avoids list growth overhead and eliminates the concatenation step.

**runner_streaming.py accumulation:**
- Uses `_init_accumulators` / `_append_chunk_results` (dict-of-lists pattern, lines 60-89)
- Per-file-chunk: creates fresh `accum` dict, accumulates JAX sub-chunks, then `_concat_jax_accumulators` + `_yield_chunk_results` converts to `AssocResult` objects
- In-memory path: `all_results.append(result)` (line 475) accumulates `AssocResult` dataclass instances
- Disk path: `writer.write(result)` writes immediately (line 473)

**For streaming runner:** The dict-of-lists pattern is localized to each file chunk (not the full run), so the append overhead is bounded. The `all_results.append(result)` for in-memory mode is a Python list of dataclass instances -- pre-allocation here would mean pre-allocating a list of `None` and assigning by index, or switching to structured numpy arrays. The streaming runner is designed for the disk-write path where `all_results` is not used. The requirement likely targets `runner_jax.py` primarily.

**Confidence:** HIGH -- verified all accumulation sites.

### Anti-Patterns to Avoid
- **Modifying `_build_results_*` signatures:** These functions in `results.py` take numpy arrays and return `list[AssocResult]`. They are used at the end of `runner_jax.py` and by `_yield_chunk_results` in the streaming path. Don't change their signatures -- they're the final conversion step.
- **Pre-allocating JAX device arrays:** Don't use `jnp.zeros(n_filtered)` as pre-allocated device arrays. JAX's functional style makes in-place updates awkward (`at[].set()` creates copies). Use numpy pre-allocation on host.
- **Breaking the double-buffering pattern:** The `_prepare_chunk` / async `device_put` pattern overlaps CPU and device work. Don't break this by moving the host transfer into the critical path.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Contiguous transpose | Manual transpose loops | `np.ascontiguousarray(U.T)` | numpy handles memory layout |
| Array pre-allocation | Custom buffer management | `np.empty(n, dtype=np.float64)` | Standard numpy pattern |

## Common Pitfalls

### Pitfall 1: Confusing LMM mode 2 (LRT) with kinship mode 2 (standardized)
**What goes wrong:** `-lmm 2` is LRT test mode (fully implemented). `-gk 2` is standardized kinship (not implemented). Easy to confuse in tests and code.
**Why it happens:** Both use "mode 2" but in completely different contexts.
**How to avoid:** Test names should explicitly say "kinship mode 2" or "gk mode 2", not just "mode 2".
**Warning signs:** A test named `test_mode_2_*` without context is ambiguous.

### Pitfall 2: NotImplementedError vs typer.Exit in CLI
**What goes wrong:** `NotImplementedError` propagates as an unhandled exception in Typer, showing a stack trace to users.
**Why it happens:** Typer expects `typer.Exit(code=1)` for clean error handling.
**How to avoid:** The requirement says "raise NotImplementedError", so do that. But consider whether the test should catch it via `pytest.raises(NotImplementedError)` directly or via Typer's CLI runner which will show it as an error. In Typer's CliRunner, an unhandled exception results in `result.exception` being set and `result.exit_code == 1`.
**Recommendation:** Raise `NotImplementedError` as specified. The Typer CliRunner test can check `isinstance(result.exception, NotImplementedError)`.

### Pitfall 3: UT contiguity and BLAS behavior
**What goes wrong:** `U.T` is a non-contiguous view. BLAS `matmul` with non-contiguous inputs may internally copy to contiguous before computing.
**Why it happens:** numpy's `.T` returns a view with swapped strides, not a contiguous array.
**How to avoid:** `np.ascontiguousarray(U.T)` forces a contiguous copy once. All subsequent matmuls benefit.
**Warning signs:** `np.iscontiguous` (or `arr.flags['C_CONTIGUOUS']`) returning False on the transpose.

### Pitfall 4: Pre-allocation index tracking
**What goes wrong:** Off-by-one errors when writing chunks into pre-allocated arrays.
**Why it happens:** The padding logic means `actual_chunk_len` may differ from `chunk_size`.
**How to avoid:** Track a running `write_offset` that increments by `actual_chunk_len` (not `chunk_size`). Assert `write_offset == n_filtered` after the loop.
**Warning signs:** Array has trailing zeros or NaN values at the end.

### Pitfall 5: Streaming runner accumulator scope
**What goes wrong:** The streaming runner creates fresh `accum` dicts per file chunk, not per full run. Pre-allocating here would need to account for the nested loop structure (file chunks containing JAX sub-chunks).
**Why it happens:** The streaming runner has two levels of chunking: file-level (disk I/O) and JAX-level (buffer overflow prevention).
**How to avoid:** For PERF-02, focus on `runner_jax.py` first. The streaming runner's per-file-chunk accumulation is already bounded and less impactful.

## Code Examples

### CORR-01: CLI fix location
```python
# cli.py, gk_command(), replace lines 148-152
if mode == 2:
    raise NotImplementedError(
        "Kinship mode 2 (standardized) is not yet implemented. "
        "Use -gk 1 for centered relatedness matrix."
    )
```

### PERF-01: Caching UT in runner_jax.py
```python
# After line 172 (eigendecompose_or_reuse)
eigenvalues_np, U = _eigendecompose_or_reuse(
    kinship, eigenvalues, eigenvectors, show_progress, "lmm_jax"
)

# Cache contiguous transpose once
UT = np.ascontiguousarray(U.T)

# Replace lines 174-175
UtW = UT @ W
Uty = UT @ phenotypes

# In _prepare_chunk (line 245), replace U.T with UT
UtG_chunk = np.ascontiguousarray(UT @ geno_chunk)
```

### PERF-02: Pre-allocated arrays in runner_jax.py (mode 1 example)
```python
# Replace lines 202-207 (mode 1 accumulators)
n_filtered = len(snp_indices)
lambdas_out = np.empty(n_filtered, dtype=np.float64)
logls_out = np.empty(n_filtered, dtype=np.float64)
betas_out = np.empty(n_filtered, dtype=np.float64)
ses_out = np.empty(n_filtered, dtype=np.float64)
pwalds_out = np.empty(n_filtered, dtype=np.float64)
write_offset = 0

# In chunk loop (replace lines 374-380):
slice_len = actual_chunk_len if needs_padding else len(best_lambdas)
end = write_offset + slice_len
lambdas_out[write_offset:end] = np.asarray(best_lambdas[:slice_len])
logls_out[write_offset:end] = np.asarray(best_logls[:slice_len])
betas_out[write_offset:end] = np.asarray(betas[:slice_len])
ses_out[write_offset:end] = np.asarray(ses[:slice_len])
pwalds_out[write_offset:end] = np.asarray(p_walds[:slice_len])
write_offset = end

# After loop (replace lines 411-416):
# Arrays are already numpy, no concatenate needed
assert write_offset == n_filtered, f"Expected {n_filtered}, wrote {write_offset}"
# Use lambdas_out, logls_out, etc. directly as best_lambdas_np, etc.
```

## Dependency Analysis

### Independence of Changes

| Change | Depends On | Touches Files | Conflicts With |
|--------|-----------|---------------|----------------|
| CORR-01 | Nothing | `cli.py`, test file | Nothing |
| PERF-01 | Nothing | `runner_jax.py`, `runner_streaming.py` | PERF-02 (same function) |
| PERF-02 | Nothing | `runner_jax.py` (primary), possibly `runner_streaming.py` | PERF-01 (same function) |

**PERF-01 and PERF-02 both modify `runner_jax.py:run_lmm_association_jax()`** and `runner_streaming.py:run_lmm_association_streaming()`. They should be implemented in sequence (not in parallel) to avoid merge conflicts, but they are logically independent.

**Recommended order:** CORR-01 first (isolated), then PERF-01 (smallest change to runners), then PERF-02 (largest change to runners).

## Test Impact Analysis

### CORR-01 Tests
**Existing tests affected:**
- No existing test covers `gk -gk 2` behavior (grep confirmed zero hits in kinship test files)
- `test_cli_lmm_mode_2_accepted` tests LMM mode 2 (LRT), NOT kinship mode 2 -- no conflict

**New tests needed:**
- `test_cli_gk_mode_2_not_implemented` in `test_cli.py`: invoke `gk -gk 2`, assert `NotImplementedError` is raised
- Consider: `test_kinship_mode_2_raises` unit test if a mode parameter is added to `compute_centered_kinship` (but the requirement only says CLI)

### PERF-01 Tests
**Existing tests affected:** None directly. The `U.T` caching is a pure internal optimization.
**Verification approach:** All existing validation tests (test_lmm_validation.py, test_runner_jax.py, test_streaming.py, test_lmm_lrt.py, test_lmm_score.py) exercise the full pipeline and will catch any regression. No test assertions reference `U.T` directly.
**Optional new test:** Assert `UT` is contiguous via `assert UT.flags['C_CONTIGUOUS']`.

### PERF-02 Tests
**Existing tests affected:** None directly. The result arrays are internal accumulators.
**Verification approach:** All existing validation tests compare final `AssocResult` values against GEMMA reference data. If pre-allocation introduces any error, these tests will catch it.
**Optional new test:** Assert `write_offset == n_filtered` after the chunk loop (defensive check).

### Test Files to Watch
| Test File | What It Covers | Phase 18 Relevance |
|-----------|---------------|-------------------|
| `tests/test_cli.py` | CLI commands | CORR-01: add gk mode 2 test |
| `tests/test_lmm_validation.py` | GEMMA reference validation | Regression guard for PERF-01/02 |
| `tests/test_runner_jax.py` | JAX runner modes | Regression guard for PERF-01/02 |
| `tests/test_lmm_lrt.py` | LRT mode | Regression guard for PERF-02 |
| `tests/test_lmm_score.py` | Score mode | Regression guard for PERF-02 |
| `tests/test_streaming.py` | Streaming runner | Regression guard for PERF-01 |

## Open Questions

1. **PERF-02 scope for streaming runner:** The requirement says "pre-allocate result arrays." Should this apply to `runner_streaming.py`'s in-memory path (`all_results: list[AssocResult]`)? The streaming runner's primary use case is disk-write mode where `all_results` is empty. Recommendation: Focus on `runner_jax.py` only; note streaming runner's in-memory path as a future optimization.

2. **PERF-02 GPU path impact:** Pre-allocating numpy arrays and doing per-chunk `np.asarray()` transfers changes the device-to-host transfer pattern from "one bulk at end" to "per chunk." For CPU (default, most common), this is fine. For GPU, the current pattern of keeping JAX arrays on device until final transfer is more efficient. Recommendation: For CPU mode (the current default), pre-allocation is strictly better. Add a comment noting that GPU users may want to revisit this.

## Sources

### Primary (HIGH confidence)
- Direct source code inspection of all files listed in Architecture Context
- `src/jamma/cli.py` lines 87-189 (gk_command with mode 2 fallback at line 148)
- `src/jamma/lmm/runner_jax.py` full file (accumulator pattern lines 202-436)
- `src/jamma/lmm/runner_streaming.py` full file (dict-based accumulator lines 40-89, 351-476)
- `src/jamma/lmm/prepare.py` full file (eigendecompose_or_reuse)
- `src/jamma/lmm/results.py` full file (result building functions)
- `src/jamma/kinship/compute.py` full file (no mode parameter)
- `tests/test_cli.py` full file (no gk mode 2 test exists)

## Metadata

**Confidence breakdown:**
- CORR-01 (kinship mode 2): HIGH -- single location, simple fix, verified by code reading
- PERF-01 (cache U.T): HIGH -- all usage sites identified by grep, well-understood numpy behavior
- PERF-02 (pre-allocate arrays): HIGH -- accumulation pattern fully traced, standard refactoring
- Test impact: HIGH -- all affected test files identified, no signature changes

**Research date:** 2026-02-06
**Valid until:** 2026-03-06 (stable internal refactoring, no external dependencies)
