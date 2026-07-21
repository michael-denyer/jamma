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
