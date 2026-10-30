"""Choose one checkpoint per episode from evaluated candidates.

The first candidate listed for an episode is its baseline. Another candidate replaces it only if it is
clearly better on the evaluation rollouts: AUC higher by more than ``min_auc_gain``, coarse success rate
not lower, and MPPE at most ``max_mppe_ratio`` times the baseline's. Among such candidates the one with the
highest AUC wins.

Usage: python -m hoi_imitation.evaluation.selection <candidates.csv> <checkpoints.json>
The CSV has the columns episode, checkpoint, evaluation_dir.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hoi_imitation.evaluation.summary import Summary, summarize


def is_clearly_better(
    candidate: Summary, baseline: Summary, min_auc_gain: float = 0.03, max_mppe_ratio: float = 1.1
) -> bool:
    return (
        candidate.auc > baseline.auc + min_auc_gain
        and candidate.coarse_success >= baseline.coarse_success
        and candidate.mppe_cm <= max_mppe_ratio * baseline.mppe_cm
    )


def select(candidates: dict[str, list[tuple[str, Summary | None]]], **criteria) -> dict[str, str]:
    """Map each episode to its chosen checkpoint; ``candidates`` lists (checkpoint, summary) baseline first."""
    chosen = {}
    for episode, options in candidates.items():
        baseline_checkpoint, baseline = options[0]
        best_checkpoint, best = baseline_checkpoint, baseline
        if baseline is not None:
            for checkpoint, summary in options[1:]:
                if summary is not None and is_clearly_better(summary, baseline, **criteria) and summary.auc > best.auc:
                    best_checkpoint, best = checkpoint, summary
        chosen[episode] = best_checkpoint
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-auc-gain", type=float, default=0.03)
    parser.add_argument("--max-mppe-ratio", type=float, default=1.1)
    args = parser.parse_args()

    candidates: dict[str, list[tuple[str, Summary | None]]] = {}
    with args.candidates.open() as handle:
        for row in csv.DictReader(handle):
            candidates.setdefault(row["episode"], []).append(
                (row["checkpoint"], summarize(Path(row["evaluation_dir"])))
            )
    chosen = select(candidates, min_auc_gain=args.min_auc_gain, max_mppe_ratio=args.max_mppe_ratio)
    args.output.write_text(json.dumps(chosen, indent=1))
    replaced = [episode for episode, options in candidates.items() if chosen[episode] != options[0][0]]
    print(f"{len(replaced)} of {len(chosen)} episodes use a non-baseline checkpoint: {replaced}")


if __name__ == "__main__":
    main()
