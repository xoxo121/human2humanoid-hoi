"""Close false-negative gaps in retargeted hand-object contact labels.

soma_to_g1.py labels a hand in contact when its palm is within 2 cm of the object mesh. During a
sustained grasp the palm often sits farther away than that, so the label flickers off while the
object is still being carried, and every contact-gated objective stops rewarding the grasp. A gap is
closed when it is short, the reference object keeps moving through it, and the other hand is not the
one holding the object (which keeps genuine handovers and releases intact).

Usage: python -m hoi_imitation.data.contact_labels <motion root> <output motion root> [--max-gap-s 4]
Each sequence partition is copied, with only `hand_contact_active` rewritten; support stages are copied too.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def contact_gaps(active: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index ranges of False runs bounded by True on both sides."""
    gaps, start = [], None
    for index in range(1, len(active)):
        if active[index - 1] and not active[index]:
            start = index
        elif start is not None and not active[index - 1] and active[index]:
            gaps.append((start, index))
            start = None
    return gaps


def repair(
    contact: np.ndarray,
    object_position: np.ndarray,
    fps: float,
    max_gap_s: float,
    min_speed: float,
    other_hand_tolerance: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Return repaired (hands, T) contact labels and the number of frames filled."""
    contact = contact.astype(bool).copy()
    speed = np.r_[0.0, np.linalg.norm(np.diff(object_position, axis=0), axis=1) * fps]
    filled = 0
    for hand in range(contact.shape[0]):
        other = (
            np.delete(contact, hand, axis=0).any(axis=0) if contact.shape[0] > 1 else np.zeros(contact.shape[1], bool)
        )
        for start, end in contact_gaps(contact[hand]):
            if (end - start) > max_gap_s * fps:
                continue
            if speed[start:end].mean() < min_speed:
                continue
            if other[start:end].mean() > other_hand_tolerance:
                continue
            contact[hand, start:end] = True
            filled += end - start
    return contact, filled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="motion root containing whole_body/soma/sequence_id=*/")
    parser.add_argument("target", type=Path)
    parser.add_argument("--max-gap-s", type=float, default=4.0)
    parser.add_argument("--min-speed", type=float, default=0.05, help="m/s the object must move during a gap")
    parser.add_argument("--sequences", nargs="*", help="sequence-id substrings to process (default: all)")
    args = parser.parse_args()

    stage_in = args.source / "whole_body" / "reconstructed_stage"
    if stage_in.is_dir():
        shutil.copytree(stage_in, args.target / "whole_body" / "reconstructed_stage", dirs_exist_ok=True)
    for partition in sorted((args.source / "whole_body" / "soma").glob("sequence_id=*")):
        if args.sequences and not any(key in partition.name for key in args.sequences):
            continue
        out = args.target / "whole_body" / "soma" / partition.name
        shutil.copytree(partition, out, dirs_exist_ok=True)
        path = out / "robot_name=g1" / "data.parquet"
        table = pq.read_table(path)
        row = table.slice(0, 1).to_pylist()[0]
        contact = np.asarray(row["hand_contact_active"], dtype=bool)
        position = np.asarray(row["object_body_position"], dtype=np.float64)[:, 0]
        repaired, filled = repair(contact, position, float(row["fps"]), args.max_gap_s, args.min_speed)
        column = table.schema.get_field_index("hand_contact_active")
        values = pa.array([repaired.tolist()], type=table.schema.field(column).type)
        pq.write_table(table.set_column(column, table.schema.field(column), values), path)
        print(
            f"{partition.name.removeprefix('sequence_id=')}: filled {filled} hand-frames "
            f"({contact.sum()} -> {repaired.sum()} labelled)"
        )


if __name__ == "__main__":
    main()
