# Mathematical validation

This program compares declared JAMMA runs with immutable GEMMA v0.98.5 outputs
and an independent dense oracle. Its verdict applies to the recorded inputs,
configuration and numerical backend. Passing these cases does not prove all
inputs or hardware correct.

## Output and weight contracts

Mode 1 emits normalized REML likelihood. Modes 2 and 4 emit the alternative
MLE likelihood as `logl_H1`. Mode 4 reports both `l_remle` and `l_mle`;
mode 2 reports `l_mle`. The parser also accepts mode-2 files without `logl_H1`.

Individual weights require both transformations in GEMMA's model. For selected,
centered kinship K and positive diagonal weights D, decompose
`D^-1/2 K D^-1/2`, then rotate `D^1/2 y`, `D^1/2 W` and `D^1/2 x`.
The pipeline scales the eigenvector rows once before association. Nonpositive
weights produce zero kinship rows and columns and zero observation scales, as
in GEMMA. Saved eigenvectors remain raw and orthonormal. The weighted tests
cover selected samples, computed and supplied kinship, batch and streaming
execution, and multiple
phenotypes without double scaling. The formula follows
[GEMMA v0.98.5](https://github.com/genetics-statistics/GEMMA/blob/v0.98.5/src/gemma.cpp).

The positive-weight case requires every applicable field to pass with
`n_refine=30`. The default 20-step result is recorded separately. With computed
kinship, the native route can exceed the configured `2e-5` MLE lambda tolerance
for this case; that result is marked **NOT VERIFIED** in the bundle.

## Declared coverage

| Case family | Inputs and execution |
| --- | --- |
| [Supplied kinship manifest](../tests/math_validation/manifest.json) | 12 cases: modes 1 through 4 with one, two or three covariates |
| [Raw pipeline manifest](../tests/math_validation/pipeline_manifest.json) | Four modes; disjoint missing phenotype and covariate rows; MAF and missingness thresholds; imputation and monomorphic exclusion; computed kinship; save on/off; batch and streaming |
| [LOCO manifest](../tests/math_validation/loco_manifest.json) | Four modes; three chromosomes including a singleton; external kinship excluding the tested chromosome; cold cache creation and observed warm reuse |
| [Weight contract](../tests/math_validation/weight_contract.py) | Mode 4; nonuniform weights with selected samples; supplied/computed kinship; batch/streaming; independent fixed-lambda and optimized dense checks |
| [Likelihood and boundary checks](../tests/math_validation/phase1.py) | Separate mode-4 fields, mode-2 MLE and mode-1 REML identity; constrained lambda classes, objective gaps and curvature; native and NumPy |
| [Shared-input parity](../tests/test_pab_diagnostic.py) | Public native and NumPy results against the dense oracle |

References contain raw PLINK inputs, external outputs, executable hashes,
commands and provenance. Generation runs the external GEMMA executable and is
separate from pytest and comparison. Missing or changed references fail.
The inventory command compares declared cases with actual tier1 collection;
it rejects missing, extra and duplicate parameter cases.

The [dense oracle](../tests/math_validation/dense_oracle.py) constructs
`H = I + lambda K`, solves GLS directly, and optimizes MLE and normalized REML.
REML includes `log|Z'Z|`. The oracle imports no JAMMA code, schema, Pab mapping,
optimizer or existing reference helper. An AST check and a subprocess import
barrier enforce that separation. SciPy remains a test dependency only.

Comparisons use explicit external column names and ordered SNP identities.
They check alleles and counted-allele frequency, including AF above 0.5, so an
allele flip cannot disappear through conversion to MAF. Numerical comparisons
use the association comparator and its configured field tolerances. Matching
boundary lambdas also receive dense-oracle objective checks with the predeclared
threshold `max(1e-8, 128 * machine_epsilon * objective_scale)`. Bundles retain
numerical distance, boundary class, objective values and local curvature.
Interior lambdas still require the ordinary relative comparison.

## Negative controls and fingerprints

[Production mutations](../tests/math_validation/mutations.json) alter real
implementations in isolated copies. The runner first requires the untouched
named tests to pass, applies an exact patch, and requires the named assertion
to fail. An import error, unmatched patch or unrelated exception is inconclusive.
The original working tree is hash checked before and after each run.

The cases cover sample filtering, centering, weight order, saved-matrix subsetting,
MLE/REML output identity, packed Pab indexing, the Score null lambda, chunk
boundaries, stale LOCO caches, allele direction, the numeric comparator and an
incorrect interior optimum. Comparator and observation controls also
corrupt individual fields, Pab cells, objectives and route records.

Native fingerprints compare result bytes per field. A field present in both
revisions fails the comparison if its digest changes. Added or removed fields
are reported as coverage changes. CI builds both revisions on the same runner.

The split Pab route requires phenotype and covariate inputs shared across SNPs.
The parity tests compare its results with the public native entry point and
the dense oracle.

## Reproduce and inspect evidence

Run from the repository root with development dependencies and a native build.
Each output directory must be new.

```sh
uv run python scripts/check_mathematical_inventory.py --output /tmp/math/inventory.json
uv run python scripts/mathematical_validation.py compare --output /tmp/math/fixtures
uv run python scripts/mathematical_validation.py pipeline --output /tmp/math/pipeline
uv run python scripts/mathematical_validation.py loco --output /tmp/math/loco
uv run python scripts/mathematical_validation.py weights --output /tmp/math/weights
uv run python scripts/mathematical_validation.py phase1 --output /tmp/math/phase1
uv run python scripts/mathematical_mutations.py --all --output /tmp/math/mutations.json

# Force the NumPy numerical route in a separate process and destination.
JAMMA_FORCE_NUMPY_FALLBACK=1 uv run python scripts/mathematical_validation.py compare --output /tmp/math-numpy/fixtures

# Generate fresh external references without overwriting committed evidence.
uv run python scripts/mathematical_validation.py generate --gemma "$HOME/.local/bin/gemma" --output /tmp/new-reference
uv run python scripts/mathematical_validation.py generate-pipeline --gemma "$HOME/.local/bin/gemma" --output /tmp/new-pipeline-reference
uv run python scripts/mathematical_validation.py generate-loco --gemma "$HOME/.local/bin/gemma" --output /tmp/new-loco-reference
uv run python scripts/mathematical_validation.py generate-weights --gemma "$HOME/.local/bin/gemma" --output /tmp/new-weight-reference
```

The driver records software versions, actual BLAS/LAPACK identity, compiler,
source and binary hashes, configuration, raw outputs and field errors.
CI uploads bundles for each test-matrix platform and the forced NumPy job.
The always-running `numerical-validation` context requires the platform jobs
and production mutation smoke checks to pass. The slow workflow runs all
production mutations.

## Remaining scope

The declared matrix does not cover standardized internal kinship, HWE and
SNP-list interactions, empty LOCO partitions, changed-input cache invalidation against fresh external output,
all chunk widths, every in-memory output route, or rank-deficient negative cases
through the complete external pipeline. Existing focused regression tests cover
some of these behaviors; they do not replace missing end-to-end evidence.

Backend verdicts apply to the software and hardware recorded in each bundle.
The ordinary parity suite includes mouse fixtures. Production-scale validation
requires a separate run with recorded inputs, configuration, software versions
and artifact hashes.
