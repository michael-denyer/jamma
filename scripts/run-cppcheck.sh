#!/usr/bin/env bash
# Static analysis over both C trees: src/jamma/jlinalg/src and src/jamma/lmm.
#
# NPY_INTP_FMT is a NumPy macro. cppcheck cannot resolve it without the NumPy
# headers, and undefined it reports unknownMacro and stops analysing the LMM
# accelerator units — files that then look clean because nothing was checked.
# Defining it here rather than suppressing the id keeps real parse
# failures visible elsewhere.
#
# This lives in a script rather than the hook's `entry:` line because
# pre-commit splits that line into arguments without a shell, which strips the
# inner quotes and leaves the macro expanding to a bare `ld` token.
set -euo pipefail

exec cppcheck \
    --std=c11 \
    --suppress=missingIncludeSystem \
    --enable=warning \
    --error-exitcode=1 \
    -DNPY_INTP_FMT='"ld"' \
    "$@"
