"""Allocation contract shared by association planning and kernel creation."""

from __future__ import annotations

from dataclasses import dataclass

from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.schema import LmmMode

_DOUBLE = 8


def _output_columns(mode: LmmMode) -> int:
    return {1: 5, 2: 2, 3: 3, 4: 8}[mode]


@dataclass(frozen=True, slots=True)
class WorkspaceSpec:
    """Dimensions and conservative live-byte bounds for one association kernel.

    ``persistent_bytes`` includes arrays retained by the Python invariants and
    arrays owned by a native workspace. ``per_thread_bytes`` is multiplied by
    ``max_threads`` because the general workspace allocates its capacity once.
    ``bytes_per_snp`` covers result arrays; rotation and fallback Uab/Iab
    buffers remain part of the chunk geometry where they are allocated. The
    native query derives C and Python table transport from their dimensions,
    then adds a fixed bound for object headers and allocator metadata on top
    of the dominant array payloads.
    """

    dispatch: DispatchPath
    lmm_mode: LmmMode
    n_samples: int
    n_input_samples: int
    n_cvt: int
    n_grid: int
    n_refine: int
    max_threads: int
    persistent_bytes: int
    per_thread_bytes: int
    transient_per_thread_bytes: int
    bytes_per_snp: int

    @property
    def fixed_bytes(self) -> int:
        return self.persistent_bytes + self.max_threads * (
            self.per_thread_bytes + self.transient_per_thread_bytes
        )

    @classmethod
    def build(
        cls,
        dispatch: DispatchPath,
        lmm_mode: LmmMode,
        n_samples: int,
        n_input_samples: int,
        n_cvt: int,
        n_grid: int,
        n_refine: int,
        max_threads: int,
    ) -> WorkspaceSpec:
        if max_threads < 1:
            raise ValueError(f"max_threads must be >= 1, got {max_threads}")
        output_bytes = _output_columns(lmm_mode) * _DOUBLE
        if dispatch is DispatchPath.NUMPY_FALLBACK:
            fixed_bytes = 0
            bytes_per_snp = output_bytes
            if lmm_mode in (1, 2, 4):
                idx = (n_cvt + 3) * (n_cvt + 2) // 2
                rows = n_cvt + 2
                # _batch_grid_pab_numpy holds v_temp and Hi_eval_grid, then
                # Pab and the tensordot result for the whole SNP chunk.
                fixed_bytes = 2 * n_grid * n_samples * _DOUBLE
                bytes_per_snp += n_grid * (rows * idx + idx) * _DOUBLE
                if lmm_mode in (1, 4):
                    # Interior REML refinement: the masked Uab input,
                    # h/dh, trace temporaries, compensated reductions, and
                    # Pab/derivative recursion arrays coexist.
                    bytes_per_snp += (
                        n_samples * idx + 6 * n_samples + 2 * rows * idx + 6 * idx
                    ) * _DOUBLE
            return cls(
                dispatch,
                lmm_mode,
                n_samples,
                n_input_samples,
                n_cvt,
                n_grid,
                n_refine,
                1,
                fixed_bytes,
                0,
                0,
                bytes_per_snp,
            )

        from jamma.lmm import accel

        persistent, per_thread, transient, output_bytes = (
            accel.require().workspace_sizes_c(
                n_samples, n_cvt, n_grid, lmm_mode, max_threads
            )
        )
        return cls(
            dispatch,
            lmm_mode,
            n_samples,
            n_input_samples,
            n_cvt,
            n_grid,
            n_refine,
            max_threads,
            persistent,
            per_thread,
            transient,
            output_bytes,
        )
