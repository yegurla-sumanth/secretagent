"""Smoke test: Oolong phase-2 measurements use concurrent workers when configured."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from secretagent import config
from secretagent.dataset import Case, Dataset

_ROOT = Path(__file__).resolve().parent.parent
_OOLONG = _ROOT / "benchmarks" / "oolong"
if str(_OOLONG) not in sys.path:
    sys.path.insert(0, str(_OOLONG))

import expt  # noqa: E402


def _peak_concurrency_during_sleeps(*, n_cases: int, workers: int, sleep_s: float) -> tuple[int, float]:
    """Run phase-2 measurements with a slow mock interface; return (peak_in_flight, elapsed_s)."""
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def mock_interface(_q: str, _labels: list, _recs: list) -> str:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(sleep_s)
        with lock:
            in_flight -= 1
        return "smoke"

    cases = [
        Case(
            name=f"c{i}",
            input_args=("q", [], []),
            expected_output="x",
            metadata={"datapoint": None},
        )
        for i in range(n_cases)
    ]
    dataset = Dataset(name="smoke", split="s", cases=cases)
    evaluator = expt.OolongEvaluator()
    t0 = time.perf_counter()
    list(evaluator.measurements(dataset, mock_interface))
    elapsed = time.perf_counter() - t0
    return peak, elapsed


def test_phase2_parallel_reaches_multiple_concurrent_workers():
    """With max_workers=4 and 16 slow calls, several must overlap (peak_in_flight >= 4)."""
    n = 16
    sleep_s = 0.04
    workers = 4
    with config.configuration(oolong={"max_workers": workers}):
        peak, elapsed = _peak_concurrency_during_sleeps(
            n_cases=n, workers=workers, sleep_s=sleep_s
        )
    assert peak >= workers, f"expected at least {workers} concurrent, got peak={peak}"
    seq_floor = n * sleep_s * 0.85
    assert elapsed < seq_floor, (
        f"expected parallel wall time < {seq_floor:.2f}s, got {elapsed:.2f}s "
        "(if this fails, measurements may be sequential)"
    )


def test_phase2_sequential_has_peak_one():
    with config.configuration(oolong={"max_workers": 1}):
        peak, _elapsed = _peak_concurrency_during_sleeps(n_cases=6, workers=1, sleep_s=0.01)
    assert peak == 1
