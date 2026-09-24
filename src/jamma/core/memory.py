"""Memory gate and sizing primitives for large-scale GWAS operations.

``lmm.association_plan.ExecutableAssociationPlan.price`` owns the run's
quote. This module holds what every layer shares: the machine read
(``available_ram_gb``), the gate (``fits``, ``require``) that compares a
price against it, and the shape-only size (``array_gb``) the quote is
built from.
"""

import psutil


def array_gb(*shape: int) -> float:
    """Memory (GB) for a float64 array of the given shape."""
    total = 8
    for dim in shape:
        total *= dim
    return total / 1e9


def block_working_set_gb(n_rows: int, n_cols: int) -> float:
    """Three float64 blocks and two bool masks of shape ``(n_rows, n_cols)``.

    Kinship preprocessing and the NumPy SNP statistics kernel each peak here:
    their input, two same-shape float temporaries, and two missingness masks.
    """
    return (3 + 2 / 8) * array_gb(n_rows, n_cols)


def available_ram_gb() -> float:
    """Available system RAM in GB, the one read every memory decision uses.

    Every gate, the chunk planner's budget, the eigendecomposition driver
    plan, and LOCO's pass planner take their ``available_gb`` from here, so a
    test pins the whole run's view of the machine with one
    ``monkeypatch.setattr(memory, "available_ram_gb", ...)``.
    ``memory_snapshot.py`` reads psutil separately for its logging snapshot;
    nothing decides on that figure.
    """
    return psutil.virtual_memory().available / 1e9


def margin_gb(peak_gb: float) -> float:
    """Safety margin: 10% of *peak_gb*, capped at 10GB absolute.

    The single spelling of the margin. At large scale (500GB+) an uncapped
    10% (50GB) is excessive: OS and process overhead do not scale with the
    eigendecomposition workspace.
    """
    return min(peak_gb * 0.1, 10.0)


def fits(required_gb: float, available_gb: float) -> bool:
    """Whether *required_gb* plus the margin fits in *available_gb*.

    The one inequality every memory gate spells: strict, with the margin
    taken of the requirement, never of the machine.
    """
    return (required_gb + margin_gb(required_gb)) < available_gb


def headroom_gb(available_gb: float) -> float:
    """The largest requirement whose margin still fits in *available_gb*.

    The inverse of ``required + margin_gb(required)``, so a caller sizing a
    batch against a fixed budget can subtract its fixed costs from this
    figure and divide, and land where ``fits`` agrees. Taking
    ``margin_gb(available_gb)`` off the machine instead reserves 10% of the
    machine, which is more than 10% of the requirement below the 10GB cap.
    """
    if available_gb > 110.0:
        return available_gb - 10.0
    return available_gb / 1.1


def require(
    required_gb: float,
    available_gb: float,
    operation: str = "operation",
    *,
    budget_gb: float | None = None,
) -> None:
    """Raise ``MemoryError`` when *required_gb* does not fit the run's ceiling.

    The sole place JAMMA constructs and raises a ``MemoryError``. Every
    memory gate (the pipeline preflight, the LMM batch runner, the kinship
    accumulator, the eigendecomposition driver, and LOCO's pass planner)
    calls this instead of raising its own, so every insufficient-memory
    error looks the same and the message lives in one place.

    Two independent ceilings, checked in order:

    1. ``budget_gb``, a user-set ``--mem-budget`` ceiling, when given.
    2. ``available_gb``, what the machine reports free, via ``fits``.

    Args:
        required_gb: Estimated peak requirement.
        available_gb: What the system reports free.
        operation: Description of what needed the memory, for the message.
        budget_gb: User-set ceiling in GB, or None for no ceiling.

    Raises:
        MemoryError: Naming which ceiling failed, and how to override.
    """
    over_budget = budget_gb is not None and required_gb > budget_gb
    insufficient = not fits(required_gb, available_gb)
    if not over_budget and not insufficient:
        return

    if over_budget:
        message = (
            f"Estimated memory ({required_gb:.1f}GB) for {operation} exceeds "
            f"budget ({budget_gb}GB). Use --no-check-memory to override."
        )
    else:
        margin = margin_gb(required_gb)
        message = (
            f"Insufficient memory for {operation}. "
            f"Need {required_gb:.1f}GB (+{margin:.1f}GB margin = "
            f"{required_gb + margin:.1f}GB), but only {available_gb:.1f}GB "
            f"available. Use --no-check-memory to override, or use a machine "
            f"with more RAM."
        )
    raise MemoryError(message)
