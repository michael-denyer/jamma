"""Runtime identity and JSON evidence serialization."""

import io
import json
import os
import platform
import subprocess
import sys
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tests.math_validation.reference import ROOT, digest


@contextmanager
def capture_logs():
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        yield messages
    finally:
        logger.remove(sink)


def serialize_config(config) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def bundle_status(statuses) -> str:
    values = list(statuses)
    return (
        "VERIFIED"
        if values and all(s == "VERIFIED" for s in values)
        else "NOT VERIFIED"
    )


def run_pipeline(source: Path, out: Path, *, output_prefix="jamma", **overrides):
    """Run the standard pipeline while capturing logs and serialized config."""
    from jamma.pipeline import PipelineRunner
    from jamma.pipeline_config import PipelineConfig

    config = PipelineConfig(
        bfile=source / "tiny",
        output_dir=out,
        output_prefix=output_prefix,
        legacy_text=True,
        check_memory=False,
        show_progress=False,
        no_telemetry=True,
        **overrides,
    )

    with capture_logs() as messages:
        result = PipelineRunner(config).run()
    return result, messages, serialize_config(config)


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def environment():
    import scipy

    from jamma import jlinalg
    from jamma.lmm import accel

    expected_blas = os.environ.get("EXPECTED_BLAS_BACKEND")
    if expected_blas and jlinalg.blas_backend != expected_blas:
        raise RuntimeError(
            f"active BLAS {jlinalg.blas_backend!r} "
            f"differs from expected {expected_blas!r}"
        )

    with redirect_stdout(io.StringIO()) as config:
        np.show_config()
    compiler = subprocess.run(
        ["cc", "--version"], capture_output=True, text=True, check=True
    ).stdout
    native = accel.require() if accel.available() else None
    native_path = None if native is None else native.__file__
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compiler": compiler,
        "numpy_config": config.getvalue(),
        "active_blas": jlinalg.blas_backend,
        "expected_blas": expected_blas,
        "ilp64": bool(jlinalg.blas_is_ilp64),
        "lapack_dsyevd": bool(jlinalg.blas_has_dsyevd),
        "lapack_dsyevr": bool(jlinalg.blas_has_dsyevr),
        "forced_numpy": os.environ.get("JAMMA_FORCE_NUMPY_FALLBACK") == "1",
        "native_binary": native_path,
        "native_sha256": digest(native_path) if native_path is not None else None,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": {
            str(p.relative_to(ROOT)): digest(p)
            for parent in [ROOT / "src/jamma", ROOT / "tests/math_validation"]
            for p in sorted(parent.rglob("*"))
            if p.suffix in {".py", ".c", ".h", ".json"} and "evidence" not in p.parts
        },
    }
