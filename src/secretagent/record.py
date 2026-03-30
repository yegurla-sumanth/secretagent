"""Context manager that will keep track of what Interfaced are called
while it is active, and also collect llm usage statistics.

Example:

  with record.recorder() as rollout:
    result = sports_understanding_workflow("DeMar DeRozan was called for the goal tend.")
    
rollout now is a list of Interfaces that were called, in order of
completion, along with usage information for each.

Recording state is **per-thread** (with a stack per thread so nested
``recorder()`` contexts work). That allows multiple threads to call
into measured interfaces concurrently without corrupting usage stats.
"""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Any

_tls = threading.local()


def _stack() -> list[list[dict[str, Any]]]:
    if not hasattr(_tls, "stack"):
        _tls.stack = []
    return _tls.stack


def reset_thread_state() -> None:
    """Clear recorder stack for the current thread (mainly for tests)."""
    if hasattr(_tls, "stack"):
        del _tls.stack


def __getattr__(name: str):
    if name == "RECORDING":
        st = getattr(_tls, "stack", None)
        return bool(st)
    if name == "RECORD":
        st = getattr(_tls, "stack", None)
        return st[-1] if st else []
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@contextmanager
def recorder():
    """Start recording subagent actions.

    Returns a list of dicts, each dict describing a subagent call.
    """
    buf: list[dict[str, Any]] = []
    _stack().append(buf)
    try:
        yield buf
    finally:
        _stack().pop()


def record(**kw):
    st = _stack()
    if st:
        st[-1].append({**kw})
