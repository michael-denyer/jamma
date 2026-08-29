#!/bin/bash
# Regenerate the committed GEMMA reference fixtures under tests/fixtures/.
#
# MANUAL EXECUTION ONLY. Needs GEMMA 0.98.5, either as a local binary or as a
# docker image. CI has neither.
#
# Every fixture is one row of the CELLS table below: a dataset, an output
# directory, an output prefix, and the GEMMA arguments. The arguments are
# reproduced byte for byte from the runs that produced the committed files,
# recorded in each fixture's .log.txt "Command Line Input" line.
#
# The gemma_covariate row deliberately omits -outdir. GEMMA then writes to
# ./output/, and the row is moved into place afterwards, matching how the
# committed file was produced.
#
# Provenance note: five committed outputs came from GEMMA 0.96 (gemma_lrt,
# gemma_score, gemma_covariate, gemma_all, gemma_all_covar). 0.98.5 adds a
# logl_H1 column, so four of them (all but gemma_score, whose Score-mode
# output has no such column) will not reproduce byte-identically; every
# shared column does. Regenerating them changes the committed bytes; refresh
# the manifest with `uv run python scripts/check_fixture_manifest.py --write`
# when that is intended.
#
# Usage:
#   bash scripts/generate_gemma_fixtures.sh [options]
#
# Options:
#   --list                 Print the cell table and exit.
#   --dry-run              Print each GEMMA command without running it.
#   --only <glob>          Run only cells whose name matches the glob.
#   --gemma-path <path>    Use this GEMMA binary instead of docker or PATH.
#   --outroot <dir>        Write under this root instead of the repository.
#
# Environment:
#   GEMMA        Path to a local GEMMA binary.
#   GEMMA_IMAGE  Docker image name (default: gemma). The LOCO rows were
#                historically produced with an image named gemma-loco.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEMMA_IMAGE="${GEMMA_IMAGE:-gemma}"
GEMMA_BIN="${GEMMA:-}"
OUT_ROOT="$PROJECT_ROOT"
ONLY="*"
DRY_RUN=false
LIST_ONLY=false

while [ $# -gt 0 ]; do
    case "$1" in
        --list) LIST_ONLY=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --gemma-path) GEMMA_BIN="$2"; shift 2 ;;
        --outroot) OUT_ROOT="$(cd "$2" && pwd)"; shift 2 ;;
        -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# ─── The cell table ──────────────────────────────────────────────────────────
#
# Columns, pipe-separated:
#   name | outdir (relative to the root) | prefix | GEMMA args
#
# %ROOT% expands to the data root: the repository when running locally, /data
# inside the container. %OUTDIR% expands to the row's output directory under
# that root. A row whose args carry no -outdir leaves GEMMA writing to ./output.
#
# The rows themselves come from tests/fixtures/MANIFEST.toml's generation_cmd
# field on each fixture's .log.txt entry, so this table cannot drift from the
# provenance already recorded there. scripts/_gemma_fixture_cells.py is the
# single place that reads MANIFEST.toml and rewrites each generation_cmd back
# into %ROOT%/%OUTDIR% form. It excludes fixtures recorded as generated from a
# no-longer-regenerable source tree (a stray /data/legacy or /data/input path,
# or a jamma-binary provenance record rather than a gemma one) and the three
# gemma_loco_chr* fixtures, whose .log.txt entries carry no generation_cmd at
# all; those are appended by the chromosome loop below instead.

CELLS="$(uv run python3 "$PROJECT_ROOT/scripts/_gemma_fixture_cells.py" "$PROJECT_ROOT/tests/fixtures/MANIFEST.toml")"

# The LOCO rows share one shape, so they expand from a chromosome loop rather
# than being typed out three times. They depend on the kinship and SNP-list
# files that generate_loco_synthetic.py --loco-kinship writes first.
for CHR in 1 2 3; do
    CELLS+=$'\n'"gemma_loco_chr${CHR}|tests/fixtures/gemma_loco|gemma_loco_chr${CHR}|-bfile %ROOT%/tests/fixtures/gemma_loco/test -k %ROOT%/tests/fixtures/gemma_loco/loco_chr${CHR}_kinship.cXX.txt -snps %ROOT%/tests/fixtures/gemma_loco/chr${CHR}_snps.txt -lmm 1 -o gemma_loco_chr${CHR} -outdir %OUTDIR%"
done

# ─── Selection and listing ───────────────────────────────────────────────────

selected_rows() {
    local name rest
    while IFS='|' read -r name rest; do
        [ -n "$name" ] || continue
        # shellcheck disable=SC2254  # ONLY is a glob pattern by design
        case "$name" in
            $ONLY) printf '%s|%s\n' "$name" "$rest" ;;
        esac
    done <<< "$CELLS"
}

if [ "$LIST_ONLY" = true ]; then
    printf '%-26s %-32s %s\n' NAME OUTDIR ARGS
    while IFS='|' read -r name outdir _prefix args; do
        printf '%-26s %-32s %s\n' "$name" "$outdir" "$args"
    done < <(selected_rows)
    exit 0
fi

# ─── Runner selection ────────────────────────────────────────────────────────

if [ -z "$GEMMA_BIN" ] && command -v gemma &> /dev/null; then
    GEMMA_BIN="$(command -v gemma)"
fi

if [ -n "$GEMMA_BIN" ]; then
    RUNNER="local"
    DATA_ROOT="$OUT_ROOT"
elif command -v docker &> /dev/null && docker image inspect "$GEMMA_IMAGE" &> /dev/null; then
    RUNNER="docker"
    DATA_ROOT="/data"
elif [ "$DRY_RUN" = true ]; then
    RUNNER="local"
    GEMMA_BIN="gemma"
    DATA_ROOT="$OUT_ROOT"
else
    echo "Error: no GEMMA available." >&2
    echo "" >&2
    echo "Provide one of:" >&2
    echo "  a local binary   : --gemma-path /path/to/gemma, GEMMA=/path/to/gemma," >&2
    echo "                     or 'gemma' on PATH (GEMMA 0.98.5)" >&2
    echo "  a docker image   : docker pull quay.io/biocontainers/gemma:0.98.5--ha36d3ea_0" >&2
    echo "                     docker tag quay.io/biocontainers/gemma:0.98.5--ha36d3ea_0 $GEMMA_IMAGE" >&2
    echo "" >&2
    echo "Source: https://github.com/genetics-statistics/GEMMA" >&2
    exit 1
fi

# --platform linux/amd64 is required because GEMMA ships x86-only builds. It is
# harmless on an x86 host and mandatory on Apple silicon.
run_gemma() {
    if [ "$RUNNER" = "docker" ]; then
        docker run --rm --platform linux/amd64 -v "$OUT_ROOT:/data" "$GEMMA_IMAGE" gemma "$@"
    else
        "$GEMMA_BIN" "$@"
    fi
}

# ─── LOCO kinship prerequisite ───────────────────────────────────────────────

loco_selected() {
    selected_rows | grep -q '^gemma_loco_chr'
}

if loco_selected; then
    LOCO_DIR="$OUT_ROOT/tests/fixtures/gemma_loco"
    if [ "$DRY_RUN" = true ]; then
        echo "+ uv run python scripts/generate_loco_synthetic.py --loco-kinship" \
            "$PROJECT_ROOT/tests/fixtures/gemma_loco/test $LOCO_DIR"
    else
        echo "=== LOCO kinship (JAMMA, subtraction formula) ==="
        (cd "$PROJECT_ROOT" && uv run python scripts/generate_loco_synthetic.py \
            --loco-kinship "$PROJECT_ROOT/tests/fixtures/gemma_loco/test" "$LOCO_DIR")
    fi
fi

# ─── Run the selected cells ──────────────────────────────────────────────────

while IFS='|' read -r name outdir prefix args; do
    echo ""
    echo "=== $name -> $outdir/$prefix ==="

    abs_outdir="$OUT_ROOT/$outdir"
    mkdir -p "$abs_outdir"

    expanded="${args//%ROOT%/$DATA_ROOT}"
    expanded="${expanded//%OUTDIR%/$DATA_ROOT/$outdir}"
    read -r -a cell_args <<< "$expanded"

    if [ "$DRY_RUN" = true ]; then
        if [ "$RUNNER" = "docker" ]; then
            echo "+ docker run --rm --platform linux/amd64 -v $OUT_ROOT:/data $GEMMA_IMAGE gemma ${cell_args[*]}"
        else
            echo "+ $GEMMA_BIN ${cell_args[*]}"
        fi
        continue
    fi

    (cd "$OUT_ROOT" && run_gemma "${cell_args[@]}")

    # Rows without -outdir land in ./output; move them where they belong.
    if [ -f "$OUT_ROOT/output/$prefix.assoc.txt" ]; then
        mv "$OUT_ROOT/output/$prefix.assoc.txt" "$abs_outdir/"
        mv "$OUT_ROOT/output/$prefix.log.txt" "$abs_outdir/"
        rmdir "$OUT_ROOT/output" 2> /dev/null || true
    fi
done < <(selected_rows)

if [ "$DRY_RUN" = true ]; then
    exit 0
fi

# ─── LOCO intermediates ──────────────────────────────────────────────────────

if loco_selected; then
    rm -f "$OUT_ROOT/tests/fixtures/gemma_loco/loco_chr"*.cXX.txt
    rm -f "$OUT_ROOT/tests/fixtures/gemma_loco/chr"*"_snps.txt"
    echo ""
    echo "Removed LOCO kinship and SNP-list intermediates."
fi

echo ""
echo "Done. Refresh the fixture manifest before committing:"
echo "  uv run python scripts/check_fixture_manifest.py --write"
