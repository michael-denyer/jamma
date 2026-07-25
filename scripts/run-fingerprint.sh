#!/usr/bin/env bash
# Run the accel suite under the fingerprint recorder, writing records to $1.
#
# Both sides of the bit-exactness comparison must use identical pytest
# arguments. The recorder keys each record by (entry point, digest of the
# arguments it was called with), so a different seed or a different selection
# produces a different set of keys, and the comparison then finds nothing to
# compare while still looking like it ran. Keeping the invocation in one file
# makes that structural rather than something two workflow steps have to agree
# on by hand.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <output-file>" >&2
    exit 2
fi
out=$1

JAMMA_FINGERPRINT_OUT="$out" uv run python -m pytest \
    tests/lmm_accel/ \
    -n0 \
    --randomly-seed=1234 \
    -p scripts.lmm_accel_fingerprint

# An empty or missing file means the recorder never installed itself. That
# would sail through the comparison as "no shared records" rather than as the
# setup failure it is, so catch it here.
if [ ! -s "$out" ]; then
    echo "ERROR: no fingerprint records written to $out" >&2
    exit 1
fi

echo "fingerprint: $(wc -l < "$out" | tr -d ' ') records -> $out"
