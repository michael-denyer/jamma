"""Thread-pool scopes that finish borrowed work before propagating interrupts."""

from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from threading import Condition
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


class JoiningThreadPoolExecutor(ThreadPoolExecutor):
    """Drain submitted work even when Ctrl-C interrupts shutdown's join.

    A callable's completion establishes that it has stopped using its
    arguments. Thread.is_alive() cannot establish that after an interrupted
    join on affected CPython versions. Count callable completions separately so
    callers can safely release input buffers and restore process-wide limits.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._completion = Condition()
        self._outstanding: dict[object, str] = {}

    def submit(
        self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> Future[_T]:
        token = object()

        def settle(*, cancel=False):
            with self._completion:
                if cancel and self._outstanding.get(token) == "running":
                    return
                self._outstanding.pop(token, None)
                self._completion.notify_all()

        def run() -> _T:
            with self._completion:
                if token not in self._outstanding:
                    raise CancelledError
                self._outstanding[token] = "running"
            try:
                return fn(*args, **kwargs)
            finally:
                settle()

        try:
            with self._completion:
                self._outstanding[token] = "pending"
            future = super().submit(run)
            future.add_done_callback(
                lambda completed: settle(cancel=True) if completed.cancelled() else None
            )
            return future
        except BaseException:
            # submit can enqueue work before starting a thread raises. Mark an
            # unstarted callable settled so it cannot later borrow its inputs.
            # A callable already running settles itself when it finishes.
            settle(cancel=True)
            raise

    def shutdown(self, wait=True, *, cancel_futures=False):
        if not wait:
            return super().shutdown(wait=False, cancel_futures=cancel_futures)

        interruption = None
        while True:
            try:
                if cancel_futures:
                    with self._completion:
                        # Settle queued wrappers before the library cancels
                        # their Futures. SIGINT can otherwise interrupt
                        # Future.cancel after its state changes but before
                        # its completion callbacks run.
                        self._outstanding = {
                            token: state
                            for token, state in self._outstanding.items()
                            if state == "running"
                        }
                        self._completion.notify_all()
                # Cancelled queued Futures also run their completion callbacks.
                super().shutdown(wait=False, cancel_futures=cancel_futures)
                with self._completion:
                    while self._outstanding:
                        self._completion.wait()
                super().shutdown(wait=True, cancel_futures=cancel_futures)
                break
            except KeyboardInterrupt as exc:
                interruption = exc
        if interruption is not None:
            raise interruption
