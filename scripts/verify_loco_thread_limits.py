#!/usr/bin/env python3
"""Verify LOCO BLAS scope ownership against a real controllable library.

On Linux, the installed NumPy BLAS usually supplies the controller. On macOS,
pass --blas-library /path/to/libopenblas.dylib to load a controllable library
alongside Accelerate. Numerical solves still use JAMMA's selected backend.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blas-library", help="Path to an installed BLAS shared library"
    )
    args = parser.parse_args()
    if args.blas_library:
        ctypes.CDLL(args.blas_library)

    from loguru import logger

    from jamma.lmm.eigen import (
        eigendecompose_kinship_in_scope,
        plan_eigen_driver_for_machine,
    )
    from jamma.lmm.loco_config import LocoConfig
    from jamma.lmm.loco_eigen import _computed_eigen_pairs

    logger.remove()
    os.environ["JAMMA_BLAS_THREADS"] = "1"
    libraries = [entry for entry in threadpool_info() if entry["user_api"] == "blas"]
    if not libraries:
        parser.error("No controllable BLAS loaded; supply --blas-library")

    def counts() -> list[int]:
        return [
            entry["num_threads"]
            for entry in threadpool_info()
            if entry["user_api"] == "blas"
        ]

    def verify(mode: str) -> dict:
        barrier = threading.Barrier(3) if mode != "input_error" else None
        observed: list[list[int]] = []

        def stream():
            for i in range(6):
                if mode == "input_error" and i == 1:
                    raise RuntimeError("input failed")
                yield str(i), np.eye(32) * (i + 1)

        def solve(K, **kwargs):
            observed.append(counts())
            if barrier is not None:
                barrier.wait(timeout=10)
            result = eigendecompose_kinship_in_scope(K, **kwargs)
            observed.append(counts())
            return result

        plan = plan_eigen_driver_for_machine(
            32, 100, budget_gb=None, inplace_eligible=True
        )
        pairs = _computed_eigen_pairs(
            stream(),
            [str(i) for i in range(6)],
            valid_mask=np.ones(32, dtype=bool),
            n_valid=32,
            pre_subset=True,
            all_samples_valid=True,
            partitions={str(i): np.arange(1) for i in range(6)},
            check_memory=False,
            show_progress=False,
            loco=LocoConfig(),
            cache_write=None,
            eigen_plan=plan,
            mem_budget=None,
            workers=3,
            solve=solve,
        )
        with threadpool_limits(limits=8, user_api="blas"):
            original = counts()
            consumed = 0
            try:
                for _name, _values, _U in pairs:
                    assert all(n == 1 for n in counts()), "eigen scope not open"
                    with threadpool_limits(limits=4, user_api="blas"):
                        assert all(n == 4 for n in counts()), "nested scope not applied"
                    assert all(n == 1 for n in counts()), (
                        "nested scope broke eigen scope"
                    )
                    consumed += 1
                    if mode == "close":
                        break
            except RuntimeError as error:
                if mode != "input_error" or str(error) != "input failed":
                    raise
            finally:
                pairs.close()
            assert counts() == original, "BLAS limits were not restored"
            assert bool(observed) == (mode != "input_error")
            assert all(all(n == 1 for n in sample) for sample in observed)
            assert consumed == {"complete": 6, "close": 1, "input_error": 0}[mode]
            return {"consumed": consumed, "restored": counts(), "observed": observed}

    results = {mode: verify(mode) for mode in ("complete", "close", "input_error")}
    print(json.dumps({"libraries": libraries, "results": results}, indent=2))


if __name__ == "__main__":
    main()
