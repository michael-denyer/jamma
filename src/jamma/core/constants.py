"""Domain constants shared across JAMMA modules."""

# GEMMA encodes missing phenotypes as -9 in .fam files.
PHENOTYPE_MISSING: float = -9.0


def env_flag(name: str) -> bool:
    """Return whether an environment variable is set to a truthy value.

    Presence-based, matching ``docs/CONFIGURATION.md``: any value other than
    unset, ``""``, or ``"0"`` counts as on. That deliberately includes
    ``"false"``, ``"no"``, and ``"off"`` — set the variable to ``0`` or leave it
    unset to turn a flag off.

    Every JAMMA toggle shares this rule, so it lives in one place rather than
    being re-spelled per call site where the accepted values could drift apart.
    ``jamma._build_support`` cannot call this: it runs under PEP 517 build
    isolation with no runtime dependencies importable.
    """
    import os

    return os.environ.get(name, "").strip() not in ("", "0")
