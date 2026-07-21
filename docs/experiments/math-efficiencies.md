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

### Shared all-tests coarse grid

Verdict: `VERIFIED` for the one-covariate split/SoA mode-4 kernel.

REML and MLE use the same weighted `wx`, `xx`, and `xy` reductions at each
coarse-grid lambda. The experimental kernel now performs those reductions once,
selects both brackets, then runs the existing independent golden-section
refinements unchanged.

A same-machine A/B used the same compiled flags and deterministic arrays with
1,410 samples and 10,768 SNPs. Across 15 runs, the pre-change median was
156.870 ms and the shared-grid median was 141.287 ms, a 9.9% reduction. Best
times were 146.682 ms and 136.823 ms, a 6.7% reduction. Every output array had
the same SHA-256 digest in both implementations. The raw timing samples and
digest are committed in `mode4-shared-grid-benchmark.tsv`.

The full best-of-five benchmark measured the NumPy+C all-tests path at 634 ms.
Because Wald timing also varied between full runs, the focused A/B is the
primary attribution evidence. Thirty-four native mode-4 and LRT parity tests
passed before the full-suite gate, including bitwise split-versus-fused checks,
multithreading, and degenerate SNPs.

A focused regression also proves that the fixture contains REML and MLE optima
separated by more than two grid steps, then compares all eight shared-grid
outputs with the independent composed path. This exercises independent bracket
selection rather than only cases where both likelihoods could choose the same
grid point.

The final repository gate passed 2,166 tests, with 10 skips and 3 expected
failures. All repository hooks passed.

The general-covariate kernel is not changed by this experiment. Its packed Pab
work grows cubically with covariate count, so sharing only the outer weighted
scan needs a separate cost and scratch-memory design before it can claim a
useful end-to-end win.

### Block Gram and thin spectral probes

Verdict: `VERIFIED` for the isolated exactness identities and `INCONCLUSIVE` for
runner integration.

The block Gram/Schur formulation matches packed Pab across 1, 2, 4, 8, and 16
covariates. The exact thin spectral inverse matches a dense solve at both lambda
endpoints, `1e-5` and `1e5`, plus random interior values, using rank-deficient
genotype matrices. These probes now run in pytest as well as the standalone
audit. Neither candidate changes a production runner on this branch because an
end-to-end memory layout and performance gate has not yet been cleared.

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
