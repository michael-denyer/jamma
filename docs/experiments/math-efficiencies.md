# Mathematical efficiency experiments

This branch tests four numerical changes without proposing them for merge. Each
change receives a `VERIFIED`, `NOT VERIFIED`, or `INCONCLUSIVE` verdict. A change
is verified only when it preserves the calibrated GEMMA tolerances, passes the
relevant test suite, and improves its target metric in repeated measurements.

## Baseline

Captured from commit `a136586` on an 18-core Apple Silicon Mac with NumPy 2.4.3.
The command was `uv run python scripts/bench_all_backends.py --runs 3`.

| Operation | JAMMA NumPy | JAMMA NumPy+C | JAMMA stream |
|---|---:|---:|---:|
| Kinship | 211 ms | 211 ms | n/a |
| Wald | 2.6 s | 528 ms | 630 ms |
| All tests | 3.9 s | 677 ms | 803 ms |
| Wald with four covariates | 6.5 s | 980 ms | 1.1 s |

## Experiment results

### Caller-owned DSYRK output

Verdict: `VERIFIED` for peak memory. End-to-end runtime was neutral.

The old `K += dsyrk(X)` pattern reached 510,033,920 bytes peak RSS for a
5,000-by-1,000 proxy. `dsyrk(X, out=K, beta=1.0)` reached 309,280,768 bytes.
The 200,753,152-byte reduction matches one removed 5,000-by-5,000 `float64`
temporary within measurement overhead. Both paths returned the same probe
value.

The repeated mouse benchmark measured kinship at 213 ms after the change,
compared with 211 ms before it. That difference is noise, so no runtime gain is
claimed. The value of this change is removing an `8N²` peak allocation from
batched and LOCO accumulation paths.

Coverage includes the native vendor path, the forced NumPy fallback, output
buffer validation, arbitrary beta, zero beta, zero-width batches, exact
symmetry, kinship, streaming kinship, and LOCO. The native Unity harness also
passes. The focused Python fallback report covers every line in the new helper.
The final DSYRK gate passed 190 focused tests with coverage and 2,162 repository
tests, with 10 skips and 3 expected failures.

## Units

1. Add an output buffer and `beta` to DSYRK so batched kinship calculation does
   not allocate an extra sample-by-sample matrix for every batch.
2. Share REML and MLE coarse-grid weighted reductions in all-tests mode.
3. Compare the packed Pab recursion with a block Gram and Schur-complement
   formulation for arbitrary covariate counts.
4. Compare the full eigendecomposition with an exact thin spectral
   representation when the kinship genotype count is below the sample count.

The first two units may change experimental production paths. The latter two
start as isolated exactness and cost probes. They move into a runner only if the
probe clears both accuracy and performance gates.

## Verification

Run the arithmetic audit with:

```bash
uv run python scripts/audit_math_efficiencies.py
```

Run end-to-end timings sequentially with:

```bash
uv run python scripts/bench_all_backends.py --runs 3
```

The final gate is `uv run pytest tests/ -x`, followed by the repository's
pre-commit and push checks.
