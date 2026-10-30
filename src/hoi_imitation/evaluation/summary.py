"""Average the metrics of a checkpoint's evaluation rollouts.

Each rollout directory holds ``run_<i>.json`` files written by FlashCHORD's ``evaluate_policy.py``.
Success rates follow two common hand-object tracking criteria: coarse (mean position error <= 10 cm and
orientation error <= 0.5 rad) and fine (< 3 cm and < 30 deg). AUC is the area under the ADD accuracy curve
between 1 and 9 cm; MPPE is the mean per-point position error of six object keypoints.

Usage: python -m hoi_imitation.evaluation.summary <evaluation dir>...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EVALUATOR_KEYS = {
    "auc": "add_auc",
    "coarse_success": "spider_sr_uncentered",
    "fine_success": "maniptrans_sr",
    "mppe_cm": "mppe_cm",
}


@dataclass(frozen=True)
class Summary:
    rollouts: int
    auc: float
    coarse_success: float
    fine_success: float
    mppe_cm: float


def summarize(evaluation_dir: Path) -> Summary | None:
    """Return the mean metrics over all rollouts in ``evaluation_dir``, or None if it has none."""
    metrics = [json.loads(path.read_text())["metrics"] for path in sorted(Path(evaluation_dir).glob("run_*.json"))]
    if not metrics:
        return None
    means = {name: float(np.mean([m[key] for m in metrics])) for name, key in EVALUATOR_KEYS.items()}
    return Summary(rollouts=len(metrics), **means)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evaluation_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    print(f"{'evaluation':40s} {'n':>3s} {'AUC':>6s} {'coarse':>7s} {'fine':>6s} {'MPPE cm':>8s}")
    for directory in args.evaluation_dirs:
        summary = summarize(directory)
        if summary is None:
            print(f"{str(directory):40s} no rollouts")
            continue
        print(
            f"{str(directory):40s} {summary.rollouts:3d} {summary.auc:6.3f} {summary.coarse_success:7.2f} "
            f"{summary.fine_success:6.2f} {summary.mppe_cm:8.2f}"
        )


if __name__ == "__main__":
    main()
