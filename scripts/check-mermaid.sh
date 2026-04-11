#!/usr/bin/env bash
# Wrapper around @probelabs/maid that fails only on hard parse errors,
# treating warnings as advisory.
#
# Why a wrapper:
#   maid returns exit 1 for any issue — warnings included. JAMMA hits
#   several maid warnings that are false positives on legitimate mermaid
#   features (FL-STYLE-TARGET-UNKNOWN against subgraph names, which
#   mermaid itself accepts). Gating on warnings would either break the
#   diagrams or require turning the hook off entirely.
#
#   Gating on errorCount from maid's JSON output gives us the parse-error
#   protection we care about (the thing that actually breaks rendering on
#   GitHub) without the warning false positives.
#
# Usage:
#   scripts/check-mermaid.sh [paths...]
#
# Defaults to scanning the whole repo if no paths are given. Uses the
# shared --exclude list so pre-commit and CI stay in sync.

set -eu

MAID_VERSION="0.0.29"
EXCLUDE=".venv/**,.planning/**,.beads/**,.claude/**,.code-review-graph/**,node_modules/**,dist/**,build/**,target/**,LICENSE.md,CLAUDE.md"

if [ "$#" -eq 0 ]; then
    TARGETS=(".")
else
    TARGETS=("$@")
fi

# maid exits non-zero on any issue (including warnings), so we can't use
# pipefail here. Capture the JSON into a tempfile and pass it to python
# explicitly. Errors-only gating happens in the python block below.
TMP_JSON=$(mktemp)
trap 'rm -f "${TMP_JSON}"' EXIT

npx --yes "@probelabs/maid@${MAID_VERSION}" \
    --format json \
    --exclude "${EXCLUDE}" \
    "${TARGETS[@]}" \
    >"${TMP_JSON}" || true

python3 - "${TMP_JSON}" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
files = data.get("files", [])
n_err = sum(f.get("errorCount", 0) for f in files)
n_warn = sum(f.get("warningCount", 0) for f in files)

print(f"maid: {n_err} error(s), {n_warn} warning(s)")

if n_err:
    for f in files:
        for err in f.get("errors", []):
            line = err.get("line", "?")
            code = err.get("code", "")
            msg = err.get("message", "").split("\n")[0]
            path = f["file"]
            print(f"  E {path}:{line} {code} {msg}")
    sys.exit(1)

if n_warn:
    # Advisory only — don't fail the hook. Show them at info level so
    # maintainers see the drift without getting blocked.
    for f in files:
        for w in f.get("warnings", []):
            line = w.get("line", "?")
            code = w.get("code", "")
            msg = w.get("message", "").split("\n")[0]
            path = f["file"]
            print(f"  W {path}:{line} {code} {msg}", file=sys.stderr)
PY
