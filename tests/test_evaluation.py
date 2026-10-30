"""Tests for hoi_imitation.evaluation."""

import json

import pytest

from hoi_imitation.evaluation.selection import is_clearly_better, select
from hoi_imitation.evaluation.summary import Summary, summarize


def _write_rollout(directory, index, auc, coarse, fine, mppe):
    metrics = {"add_auc": auc, "spider_sr_uncentered": coarse, "maniptrans_sr": fine, "mppe_cm": mppe}
    (directory / f"run_{index}.json").write_text(json.dumps({"metrics": metrics}))


def test_summarize_averages_rollouts(tmp_path):
    _write_rollout(tmp_path, 1, 0.4, 1.0, 0.0, 4.0)
    _write_rollout(tmp_path, 2, 0.6, 0.0, 1.0, 6.0)
    summary = summarize(tmp_path)
    assert summary.rollouts == 2
    assert summary.auc == pytest.approx(0.5)
    assert (summary.coarse_success, summary.fine_success, summary.mppe_cm) == (0.5, 0.5, 5.0)


def test_summarize_without_rollouts_returns_none(tmp_path):
    assert summarize(tmp_path) is None


BASELINE = Summary(rollouts=6, auc=0.40, coarse_success=0.5, fine_success=0.0, mppe_cm=5.0)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (Summary(6, 0.50, 0.5, 0.0, 5.0), True),
        (Summary(6, 0.42, 1.0, 1.0, 3.0), False),  # AUC gain within noise
        (Summary(6, 0.60, 0.3, 1.0, 3.0), False),  # coarse success dropped
        (Summary(6, 0.60, 0.5, 1.0, 6.0), False),  # MPPE more than 10% worse
    ],
)
def test_is_clearly_better(candidate, expected):
    assert is_clearly_better(candidate, BASELINE) is expected


def test_select_keeps_baseline_unless_a_candidate_is_clearly_better():
    candidates = {
        "0": [("base0", BASELINE), ("weak", Summary(6, 0.41, 0.5, 0.0, 5.0))],
        "1": [
            ("base1", BASELINE),
            ("good", Summary(6, 0.50, 0.5, 0.0, 5.0)),
            ("best", Summary(6, 0.70, 1.0, 1.0, 2.0)),
        ],
        "2": [("base2", None), ("other", Summary(6, 0.90, 1.0, 1.0, 1.0))],
    }
    assert select(candidates) == {"0": "base0", "1": "best", "2": "base2"}
