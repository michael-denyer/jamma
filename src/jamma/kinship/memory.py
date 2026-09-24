"""Price streaming kinship memory from its input and output dimensions."""

from loguru import logger

from jamma.core.memory import array_gb, block_working_set_gb


def _dsyrk_scratch_gb(n_samples: int) -> float:
    """Scratch the active dsyrk backend holds during kinship accumulation.

    Zero on the native path, which accumulates in place. The NumPy fallback
    blocks its accumulation and holds one block-by-n product, so budgeting only
    the accumulator would approve a run the fallback then OOMs. jlinalg owns the
    block size, so it reports the figure rather than this module re-deriving it.

    Zero too when jlinalg will not import: kinship accumulation goes through
    ``jlinalg.dsyrk``, so there is no dsyrk phase left to budget for. The
    pre-flight must still produce an estimate rather than raise.
    """
    try:
        from jamma.jlinalg import dsyrk_scratch_bytes  # deferred: jlinalg is heavy
    except ImportError:
        logger.debug("Could not import jlinalg; assuming no dsyrk scratch.")
        return 0.0

    return dsyrk_scratch_bytes(n_samples) / 1e9


def kinship_chunk_gb(n_input_samples: int, width: int) -> float:
    """Peak GB of one kinship chunk of ``width`` variants over every input row.

    Preprocessing holds three float64 blocks, for decoded data, selected
    columns, and either transform output or the contiguous input copy made
    by dsyrk, plus two boolean masks. The standardized transform preserves
    its input, so it can hold all three float blocks while reducing means.
    The read before it is smaller for every encoding: a BGEN block at the
    decoder's 16-bit ceiling holds at most 23 bytes per cell with its decode
    buffers, under the working set's 26.
    """
    return block_working_set_gb(n_input_samples, width)


def estimate_kinship_memory(
    *,
    n_input_samples: int,
    n_output_samples: int,
    n_snps: int,
    chunk_size: int,
) -> float:
    """Price streaming kinship in GB from its input and output dimensions.

    Preprocessing uses all input rows; the accumulator and backend scratch
    use output rows. A short file never allocates the full requested block.
    """
    return (
        array_gb(n_output_samples, n_output_samples)
        + kinship_chunk_gb(n_input_samples, min(chunk_size, n_snps))
        + _dsyrk_scratch_gb(n_output_samples)
    )
