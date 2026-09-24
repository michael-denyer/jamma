"""Price streaming kinship memory from its input and output dimensions."""

from loguru import logger

from jamma.core.memory import array_gb
from jamma.genotype.dataset import GenotypeEncoding


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


def estimate_kinship_memory(
    *,
    n_input_samples: int,
    n_output_samples: int,
    n_snps: int,
    chunk_size: int,
    genotype_encoding: GenotypeEncoding = GenotypeEncoding.HARD_CALLS,
) -> float:
    """Price streaming kinship in GB from its input and output dimensions.

    Preprocessing uses all input rows; the accumulator and backend scratch
    use output rows. A short file never allocates the full requested block.
    Three float64 blocks cover decoded data, selected columns, and either
    transform output or the contiguous input copy made by dsyrk. Two boolean
    blocks cover preprocessing masks. The standardized transform preserves
    its input, so it can hold all three float blocks while reducing means.
    """
    chunk_gb = array_gb(n_input_samples, min(chunk_size, n_snps))
    return (
        array_gb(n_output_samples, n_output_samples)
        + (3 + 2 / 8) * chunk_gb
        + genotype_encoding.read_workspace_bytes(
            n_input_samples, min(chunk_size, n_snps)
        )
        / 1e9
        + _dsyrk_scratch_gb(n_output_samples)
    )
