# Mathematical validation

This program compares declared JAMMA runs with immutable GEMMA v0.98.5 outputs
and an independent dense oracle. Its verdict applies to the recorded inputs,
configuration and numerical backend. Passing these cases does not prove all
inputs or hardware correct.

## Contracts and repairs

Mode 1 emits normalized REML likelihood. Modes 2 and 4 emit the alternative
MLE likelihood as `logl_H1`; their `l_remle` and `l_mle` columns retain their
separate meanings. Mode 2 previously discarded the MLE likelihood it had already
computed. Its canonical output now includes that field. The parser still reads
older files without the column.

Individual weights require both transformations in GEMMA's model. For selected,
centered kinship K and positive diagonal weights D, decompose
`D^-1/2 K D^-1/2`, then rotate `D^1/2 y`, `D^1/2 W` and `D^1/2 x`.
JAMMA previously applied only the kinship transformation. The pipeline now
scales the eigenvector rows once before association. As in GEMMA, nonpositive
weights produce zero kinship rows/columns and zero observation scales. It writes raw orthonormal
eigenvectors before that scaling. The weighted tests cover selected samples,
computed and supplied kinship, batch and streaming execution, and multiple
phenotypes without double scaling. The formula follows
[GEMMA v0.98.5](https://github.com/genetics-statistics/GEMMA/blob/v0.98.5/src/gemma.cpp).

The weighted fixture passes every applicable field with `n_refine=30`. At the
default 20 steps, one MLE lambda exceeds the configured `2e-5` relative tolerance
on the local native route. That default configuration remains **NOT VERIFIED**
for this fixture. No tolerance or default optimizer setting was changed to make
the refined case pass.

## Declared coverage

| Case family | Inputs and execution |
| --- | --- |
| [Supplied kinship manifest](../tests/math_validation/manifest.json) | 12 cases: modes 1 through 4 with one, two or three covariates, including the original tiny Wald case |
| [Raw pipeline manifest](../tests/math_validation/pipeline_manifest.json) | Four modes; disjoint missing phenotype and covariate rows; MAF and missingness thresholds; imputation and monomorphic exclusion; computed kinship; save on/off; batch and streaming |
| [LOCO manifest](../tests/math_validation/loco_manifest.json) | Four modes; three chromosomes including a singleton; external kinship excluding the tested chromosome; cold cache creation and observed warm reuse |
| [Weight contract](../tests/math_validation/weight_contract.py) | Mode 4; nonuniform weights with selected samples; supplied/computed kinship; batch/streaming; independent fixed-lambda and optimized dense checks |
| [Phase 1](../tests/math_validation/phase1.py) | Separate mode-4 fields, mode-2 MLE and mode-1 REML identity; constrained lambda classes, objective gaps and curvature; native and NumPy |
| [Pab diagnosis](../tests/math_validation/pab_trace.py) | Invalid historical sharing, reduced reproduction, valid shared inputs, observed native calls and intermediate projection products |

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
consume the existing R7 comparator and unchanged field tolerances. Matching
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
incorrect interior optimum. Comparator and observation controls additionally
corrupt individual fields, Pab cells, objectives and route records.

Native fingerprints remain a separate exact-bit gate. Dictionary results are
recorded per field so adding mode-2 likelihood output does not hide changes to
existing arrays. Against integration base `6fcf9a7`, all 413 shared local
fingerprint records are bit-identical; six new records cover mode-2 likelihoods.
CI builds both revisions on the same runner and repeats that comparison.

## Historical Pab result

The original benchmark supplied different `w` and `y` for each SNP while the
split route reused SNP 0's invariant products. The first incorrect stage was
that mixing, before Pab calculation. Its function labelled C actually called
the NumPy optimizer. The corrected benchmark calls the native entry point with
shared inputs, and its assertions also run in ordinary tests.

The invalid four-sample, two-SNP reproduction remains a strict expected failure.
Its parity claim is **NOT VERIFIED**; the input-contract diagnosis is verified.
Three samples leave one residual contrast and a flat REML profile, which the
oracle records. The [historical summary](../tests/math_validation/evidence/summary.json)
and [original failure](../tests/math_validation/evidence/original-red.txt) retain
the earlier software identity and measurements. They are not current-platform
results. Current Pab bundles observe the cached-grid selector directly. Native
final bracket endpoints are not exposed and are reported as unavailable.

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
uv run python scripts/mathematical_validation.py pab --output /tmp/math/pab
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
production mutations. Native Pab symbol observation requires an exporting build;
a missing symbol fails that command.

## Remaining scope

The declared matrix is still smaller than the full validation plan. It does not
establish standardized internal kinship, HWE and SNP-list interactions, empty
LOCO partitions, changed-input cache invalidation against fresh external output,
all chunk widths, every in-memory output route, or rank-deficient negative cases
through the complete external pipeline. Existing focused regression tests cover
some of these behaviors; they do not replace missing end-to-end evidence.

Linux and other backend verdicts belong to the artifacts from their actual CI
runs. A local macOS pass does not establish them. Existing mouse fixtures remain
in the ordinary parity suite. Historical production-scale comparisons are not
a fresh validation of this revision, and this program has not rerun a
production-shape dataset. Broader equivalence claims require those additional
runs and their immutable provenance.
