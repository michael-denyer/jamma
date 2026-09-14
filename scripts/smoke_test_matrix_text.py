"""Require the installed formatter and compare adversarial values with Python %g.

Run directly in wheel tests. This deliberately imports the extension itself,
so a missing binary cannot pass by rebuilding or using the Python writer.
"""

import hashlib

import numpy as np


def precision_corpus() -> np.ndarray:
    """Random binary64 patterns, notation boundaries, decimal ties, and specials."""
    rng = np.random.default_rng(376)
    random_bits = rng.integers(0, 2**64, 500_000, dtype=np.uint64).view(np.float64)
    powers = 10.0 ** np.arange(-323.0, 309.0)
    near_powers = np.concatenate(
        [powers, np.nextafter(powers, 0), np.nextafter(powers, np.inf)]
    )
    ties = np.arange(1_000_000_000, 1_000_050_000, dtype=np.float64) * 10 + 5
    boundary = np.array(
        [
            0.0,
            -0.0,
            np.nan,
            -np.nan,
            np.inf,
            -np.inf,
            np.nextafter(0.0, 1.0),
            np.finfo(float).tiny,
            np.finfo(float).max,
            12345678905.0,
            99999999995.0,
        ]
    )
    return np.concatenate(
        [random_bits, near_powers, -near_powers, ties, -ties, boundary]
    )


def verify() -> str:
    from jamma.io._matrix_text import ABI_VERSION, BYTES_PER_VALUE, format_into

    assert ABI_VERSION == 1
    values = precision_corpus()
    digest = hashlib.sha256()
    for start in range(0, len(values), 10_000):
        batch = values[start : start + 10_000].reshape(-1, 1)
        buffer = bytearray(batch.size * BYTES_PER_VALUE)
        used = format_into(batch, buffer)
        actual = buffer[:used]
        fmt = "%.10g\n"
        expected = b"".join((fmt % value).encode("ascii") for value in batch[:, 0])
        assert actual == expected, (
            f"formatter mismatch in values {start}:{start + len(batch)}"
        )
        digest.update(actual)
    return f"{len(values):,} values byte-identical; SHA256 {digest.hexdigest()}"


if __name__ == "__main__":
    print(verify())
