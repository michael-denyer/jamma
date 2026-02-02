#!/bin/bash
# Generate GEMMA reference with covariates for validation

set -e

FIXTURE_DIR="tests/fixtures/gemma_covariate"
mkdir -p "$FIXTURE_DIR"

# Use gemma_synthetic as base data
PLINK_PREFIX="tests/fixtures/gemma_synthetic/test"
KINSHIP="tests/fixtures/gemma_synthetic/gemma_kinship.cXX.txt"

# Count samples from .fam file
N_SAMPLES=$(wc -l < "$PLINK_PREFIX.fam" | tr -d ' ')

echo "Generating covariate file for $N_SAMPLES samples..."

# Generate covariate file (intercept + 1 continuous covariate)
# Deterministic values based on sample index for reproducibility
python3 -c "
import numpy as np
np.random.seed(42)
n = $N_SAMPLES
# Intercept column + one continuous covariate
covariates = np.column_stack([
    np.ones(n),
    np.random.randn(n) * 0.5  # Scaled random values
])
np.savetxt('$FIXTURE_DIR/covariates.txt', covariates, fmt='%.6f', delimiter='\t')
print(f'Generated {n} x 2 covariate file')
"

echo "Running GEMMA with covariates..."

# Run GEMMA with covariates
docker run --rm -v "$(pwd):/data" -w /data gemma \
    -bfile "$PLINK_PREFIX" \
    -k "$KINSHIP" \
    -c "$FIXTURE_DIR/covariates.txt" \
    -lmm 1 \
    -o gemma_covariate

# Move output to fixture directory
mv output/gemma_covariate.assoc.txt "$FIXTURE_DIR/"
mv output/gemma_covariate.log.txt "$FIXTURE_DIR/"

echo "Generated covariate reference in $FIXTURE_DIR"
echo "Files:"
ls -la "$FIXTURE_DIR"
