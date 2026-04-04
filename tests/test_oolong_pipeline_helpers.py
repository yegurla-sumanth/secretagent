"""Unit tests for Oolong classify batching (binary split + leaf retries)."""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

import pytest

_OOLONG = Path(__file__).resolve().parent.parent / "benchmarks" / "oolong"
if str(_OOLONG) not in sys.path:
    sys.path.insert(0, str(_OOLONG))

import pipeline_helpers as ph


@pytest.fixture
def no_sleep():
    with unittest.mock.patch.object(ph.time, "sleep"):
        yield


@pytest.fixture
def quiet_tqdm():
    with unittest.mock.patch.object(ph, "tqdm", lambda it, **kw: it):
        yield


def _ok_record(b_start: int, line: str, label: str = "spam") -> dict:
    return {"idx": b_start, "label": label, "entry_text": line}


def test_classify_binary_split_when_multi_batch_fails(no_sleep, quiet_tqdm):
    """Multi-line classify fails; recursion classifies each line as a singleton."""

    def classify_fn(
        *,
        entry_lines: list[str],
        label_set: list[str],
        batch_start_idx: int,
        num_entry_lines: int,
    ):
        assert len(entry_lines) == num_entry_lines
        if len(entry_lines) > 1:
            raise ValueError("model cannot batch")
        lbl = label_set[0]
        return {"records": [_ok_record(batch_start_idx, entry_lines[0], lbl)]}

    records, errors = ph.classify_batches_with_retry(
        classify_fn,
        ["L0", "L1", "L2"],
        ["spam"],
        batch_size=10,
        retries=3,
        backoff=1.0,
    )
    assert errors == []
    assert [r["idx"] for r in records] == [0, 1, 2]
    assert [r["entry_text"] for r in records] == ["L0", "L1", "L2"]
    assert all(r["label"] == "spam" for r in records)


def test_classify_multi_succeeds_first_try(no_sleep, quiet_tqdm):
    lines = ["a", "b"]

    def classify_fn(
        *,
        entry_lines: list[str],
        label_set: list[str],
        batch_start_idx: int,
        num_entry_lines: int,
    ):
        recs = [
            _ok_record(batch_start_idx + j, entry_lines[j], label_set[0])
            for j in range(len(entry_lines))
        ]
        return {"records": recs}

    records, errors = ph.classify_batches_with_retry(
        classify_fn,
        lines,
        ["spam"],
        batch_size=10,
        retries=2,
        backoff=1.0,
    )
    assert errors == []
    assert len(records) == 2


def test_single_line_retries_then_succeeds(no_sleep, quiet_tqdm):
    n_calls = 0

    def classify_fn(
        *,
        entry_lines: list[str],
        label_set: list[str],
        batch_start_idx: int,
        num_entry_lines: int,
    ):
        nonlocal n_calls
        assert len(entry_lines) == 1
        n_calls += 1
        if n_calls < 3:
            raise RuntimeError("transient")
        return {
            "records": [_ok_record(batch_start_idx, entry_lines[0], label_set[0])]
        }

    records, errors = ph.classify_batches_with_retry(
        classify_fn,
        ["only"],
        ["ham"],
        batch_size=1,
        retries=5,
        backoff=1.0,
    )
    assert errors == []
    assert records == [
        {"idx": 0, "entry_text": "only", "label": "ham"},
    ]
    assert n_calls == 3


def test_single_line_exhausts_retries(no_sleep, quiet_tqdm):
    def classify_fn(**kwargs):
        raise ValueError("permanent")

    records, errors = ph.classify_batches_with_retry(
        classify_fn,
        ["x"],
        ["spam"],
        batch_size=1,
        retries=2,
        backoff=1.0,
    )
    assert records == []
    assert len(errors) == 1
    assert errors[0]["b_start"] == 0
    assert "permanent" in errors[0]["error"]


def test_max_split_depth_exceeded(no_sleep, quiet_tqdm):
    def classify_fn(
        *,
        entry_lines: list[str],
        label_set: list[str],
        batch_start_idx: int,
        num_entry_lines: int,
    ):
        if len(entry_lines) > 1:
            raise ValueError("fail multi")
        return {"records": [_ok_record(batch_start_idx, entry_lines[0], label_set[0])]}

    with unittest.mock.patch.object(ph, "_MAX_SPLIT_DEPTH", 0):
        records, errors = ph.classify_batches_with_retry(
            classify_fn,
            ["a", "b"],
            ["spam"],
            batch_size=2,
            retries=3,
            backoff=1.0,
        )
    assert records == []
    assert len(errors) == 1
    assert "split depth" in errors[0]["error"].lower()


def test_two_coarse_chunks_both_split(no_sleep, quiet_tqdm):
    """batch_size=2 with 4 lines => 2 outer chunks; each chunk splits to singletons."""

    def classify_fn(
        *,
        entry_lines: list[str],
        label_set: list[str],
        batch_start_idx: int,
        num_entry_lines: int,
    ):
        if len(entry_lines) > 1:
            raise ValueError("no multi")
        return {"records": [_ok_record(batch_start_idx, entry_lines[0], label_set[0])]}

    records, errors = ph.classify_batches_with_retry(
        classify_fn,
        ["a", "b", "c", "d"],
        ["spam"],
        batch_size=2,
        retries=2,
        backoff=1.0,
    )
    assert errors == []
    assert [r["idx"] for r in records] == [0, 1, 2, 3]


def test_filesystem_slug_sanitizes_model_id():
    assert ph.filesystem_slug("together_ai/deepseek-ai/DeepSeek-V3") == "together_ai_deepseek-ai_DeepSeek-V3"


def test_window_cache_path_model_namespace():
    from pathlib import Path

    root = Path("/tmp/wc")
    p0 = ph.window_cache_path(root, "validation", 1024, 7, model_slug=None)
    assert p0 == Path("/tmp/wc/validation/1024/7.json")
    p1 = ph.window_cache_path(root, "validation", 1024, 7, model_slug="m_a")
    assert p1 == Path("/tmp/wc/m_a/validation/1024/7.json")


def test_classification_accuracy_vs_gold_notebook_parity():
    ctx = (
        "header\n"
        "Date: x || User: 1 || Instance: foo || Label: spam\n"
        "Date: y || User: 2 || Instance: bar || Label: ham\n"
    )
    compact = [{"idx": 0, "label": "spam"}, {"idx": 1, "label": "HAM"}]
    m = ph.classification_accuracy_vs_gold(compact, ctx)
    assert m["gold_records"] == 2
    assert m["matched_records"] == 2
    assert m["correct_on_matched"] == 2
    assert m["accuracy_on_matched"] == 1.0
    assert m["accuracy_over_gold"] == 1.0

    m2 = ph.classification_accuracy_vs_gold([{"idx": 0, "label": "ham"}], ctx)
    assert m2["correct_on_matched"] == 0
    assert m2["matched_records"] == 1
    assert m2["accuracy_over_gold"] == 0.0
