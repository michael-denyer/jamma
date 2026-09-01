"""DGEMM extension-load and thread-runtime tests."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import numpy.testing as npt
import pytest

from jamma.jlinalg import HAS_C_EXTENSION

pytestmark = pytest.mark.tier0


class TestDgemmInit:
    """The DGEMM facade and native export load together."""

    def test_import_succeeds(self) -> None:
        from jamma.jlinalg import dgemm

        assert callable(dgemm)

    def test_c_extension_has_dgemm(self) -> None:
        if not HAS_C_EXTENSION:
            pytest.skip("C extension not compiled")
        from jamma.jlinalg import _jlinalg  # type: ignore[import]

        assert hasattr(_jlinalg, "dgemm")


class TestDgemmThreadSafety:
    """DGEMM results remain consistent across OpenMP thread counts."""

    def test_single_vs_multi_thread(self) -> None:
        script = textwrap.dedent("""
            import sys
            import numpy as np
            from jamma.jlinalg import dgemm

            rng = np.random.default_rng(12345)
            A = rng.standard_normal((500, 500))
            B = rng.standard_normal((500, 500))
            C = dgemm(A, B)
            sys.stdout.buffer.write(C.tobytes())
        """)
        result_single = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            timeout=60,
        )
        result_multi = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            env={**os.environ, "OMP_NUM_THREADS": "4"},
            timeout=60,
        )

        assert result_single.returncode == 0, result_single.stderr.decode()
        assert result_multi.returncode == 0, result_multi.stderr.decode()
        single = np.frombuffer(result_single.stdout, dtype=np.float64).reshape(500, 500)
        multi = np.frombuffer(result_multi.stdout, dtype=np.float64).reshape(500, 500)
        npt.assert_allclose(single, multi, rtol=1e-12, atol=1e-12)
