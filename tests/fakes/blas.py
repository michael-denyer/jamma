"""Observable fake for the process-wide BLAS controller. Numerical work stays real."""

from contextlib import contextmanager


def fake_blas_controller(active_limit: list[int], transitions: list[tuple[str, int]]):
    @contextmanager
    def control(limit: int):
        previous = active_limit[0]
        transitions.append(("enter", limit))
        active_limit[0] = limit
        try:
            yield
        finally:
            active_limit[0] = previous
            transitions.append(("restore", previous))

    return control
