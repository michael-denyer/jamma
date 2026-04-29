# Vulture allowlist — names that are unused as variables but required
# by external API contracts (decorator signatures, dunder protocols,
# ABI flags) and therefore cannot be removed.
#
# Format: each name is referenced via ``_.<name>`` so vulture treats it
# as "used". Add a comment per group explaining the contract.

# ---------------------------------------------------------------------------
# click callback signature: ``def cb(ctx, param, value)``
# ---------------------------------------------------------------------------
_.param  # src/jamma/cli.py — click eager-callback signature

# ---------------------------------------------------------------------------
# __exit__ context-manager protocol: ``def __exit__(self, exc_type, exc_value, exc_tb)``
# ---------------------------------------------------------------------------
_.exc_tb  # src/jamma/lmm/io.py — __exit__ traceback parameter

# ---------------------------------------------------------------------------
# ABI presence flag: declared at module load to record optional capability
# (mode-4 fused workspace path) detected from the .so. Read by tests via
# importlib reload + introspection rather than direct import.
# ---------------------------------------------------------------------------
_.fused_mode4  # src/jamma/lmm/runner_numpy.py — ABI presence flag, set at import

# ---------------------------------------------------------------------------
# Local boolean predicate that documents the conditional branch but is
# only used implicitly through the if-expression. Kept as named variable
# for readability — restating the condition inline would obscure intent.
# ---------------------------------------------------------------------------
_.has_kinship  # src/jamma/core/memory.py — predicate documenting branch
