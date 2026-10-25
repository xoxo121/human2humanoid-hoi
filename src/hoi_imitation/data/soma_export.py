"""Export LeRobot human-object interaction episodes to the folder layout read by ``soma_to_g1.py``.

The dataset stores the SOMA-X exporter outputs per frame: human parameters in the SOMA world
frame (Y up) and object poses in the OpenCV world frame. ``soma_to_g1.py`` reads the same
quantities from

    <out>/<sequence_id>/soma_params.npz
    <out>/<sequence_id>/poses.npy                           (T, 4, 4) object poses, OpenCV world
    <out>/<sequence_id>/reconstructed_mesh/output_aligned.glb
    <out>/<sequence_id>/ground_plane.json                   ax+by+cz+d=0, OpenCV world

so this module only reshapes and validates; it applies no coordinate transform.

Usage:
    python -m hoi_imitation.data.soma_export <dataset_root> <out_root> --episodes 16
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hoi_imitation.data.dataset import OBJECT_POSE_COLUMN, Episode, load_episodes, pose_to_matrices

POSE_COLUMN = "observation.human.pose"
POSE_AXES = (".rx", ".ry", ".rz")
QUAT_NORM_TOLERANCE = 1e-3


def joint_names(dataset_root: Path) -> list[str]:
    """Return SOMA joint names from the ``<joint>.r{x,y,z}`` pose feature names."""
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    names = info["features"][POSE_COLUMN]["names"]
    joints = []
    for start in range(0, len(names), 3):
        triple = names[start : start + 3]
        joint = triple[0].removesuffix(POSE_AXES[0])
        if [f"{joint}{axis}" for axis in POSE_AXES] != triple:
            raise ValueError(f"{POSE_COLUMN} names are not rotation-vector triples: {triple}")
        joints.append(joint)
    return joints


def _stack(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(frame[column].to_numpy()).astype(np.float64)


def build_soma_params(frame: pd.DataFrame, joints: list[str]) -> dict[str, np.ndarray]:
    """Arrange the human columns as the SOMA exporter's ``save_soma_npz`` arrays."""
    poses = _stack(frame, POSE_COLUMN)
    identity = _stack(frame, "observation.human.identity_coeffs")
    scale = _stack(frame, "observation.human.scale_params")
    for name, values in (("identity_coeffs", identity), ("scale_params", scale)):
        if not np.allclose(values, values[0]):
            raise ValueError(f"{name} varies over time; read_soma uses frame 0 only")
    return {
        "poses": poses.reshape(len(frame), len(joints), 3).astype(np.float32),
        "transl": _stack(frame, "observation.human.translation").astype(np.float32),
        "joint_names": np.asarray(joints),
        "identity_model_type": np.asarray("mhr"),
        "identity_coeffs": identity.astype(np.float32),
        "scale_params": scale.astype(np.float32),
        "unit": np.asarray("meters"),
        "keep_root": np.asarray(False),
        "rotation_repr": np.asarray("rotvec"),
        "absolute_pose": np.asarray(False),
    }


def object_matrices(frame: pd.DataFrame) -> np.ndarray:
    """Convert the object pose column to ``(T, 4, 4)`` matrices, rejecting non-unit quaternions."""
    pose = _stack(frame, OBJECT_POSE_COLUMN)
    norms = np.linalg.norm(pose[:, 3:], axis=1)
    if np.any(np.abs(norms - 1.0) > QUAT_NORM_TOLERANCE):
        worst = int(np.argmax(np.abs(norms - 1.0)))
        raise ValueError(f"object quaternion at frame {worst} has norm {norms[worst]:.6f}")
    return pose_to_matrices(pose)


def check_timing(frame: pd.DataFrame, fps: float) -> None:
    """Require contiguous frames on the dataset's fixed clock."""
    frames = frame["frame_index"].to_numpy()
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("frame_index is not contiguous from 0")
    if not np.allclose(frame["timestamp"].to_numpy(), frames / fps, atol=1e-4):
        raise ValueError(f"timestamps are not on a {fps} Hz clock")


def drop_invisible_tail(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop trailing frames whose object is marked invisible and carries no valid pose.

    Some episodes end with all-zero, invisible frames that carry no pose. Invalid poses anywhere
    else are left in place and rejected later.
    """
    pose = _stack(frame, OBJECT_POSE_COLUMN)
    invalid = np.abs(np.linalg.norm(pose[:, 3:], axis=1) - 1.0) > QUAT_NORM_TOLERANCE
    droppable = invalid & ~frame["observation.object.visible"].to_numpy().astype(bool)
    end = len(frame)
    while end > 0 and droppable[end - 1]:
        end -= 1
    if end < len(frame):
        print(f"  dropping {len(frame) - end} invisible frame(s) with no object pose from the end")
    return frame.iloc[:end]


def export_episode(episode: Episode, joints: list[str], fps: float, out_root: Path) -> Path:
    """Write one episode folder and return its path."""
    frame = pd.read_parquet(episode.data).sort_values("frame_index", ignore_index=True)
    check_timing(frame, fps)
    frame = drop_invisible_tail(frame)
    folder = out_root / episode.sequence_id
    (folder / "reconstructed_mesh").mkdir(parents=True, exist_ok=True)
    np.savez(folder / "soma_params.npz", **build_soma_params(frame, joints))
    np.save(folder / "poses.npy", object_matrices(frame))
    shutil.copyfile(episode.mesh, folder / "reconstructed_mesh" / "output_aligned.glb")
    shutil.copyfile(episode.ground_plane, folder / "ground_plane.json")
    visible = frame["observation.object.visible"].to_numpy().astype(bool)
    print(f"episode {episode.index:06d} -> {folder} " f"({len(frame)} frames, object visible in {visible.mean():.1%})")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_root", type=Path, help="LeRobot dataset directory containing meta/ and data/")
    parser.add_argument("out_root", type=Path, help="directory that receives one folder per sequence")
    parser.add_argument("--episodes", type=int, nargs="+", help="episode indices (default: all)")
    args = parser.parse_args()

    episodes = load_episodes(args.dataset_root)
    selected = args.episodes if args.episodes else sorted(episodes)
    unknown = sorted(set(selected) - set(episodes))
    if unknown:
        parser.error(f"unknown episode indices: {unknown}")
    fps = float(json.loads((args.dataset_root / "meta" / "info.json").read_text())["fps"])
    joints = joint_names(args.dataset_root)
    for index in selected:
        export_episode(episodes[index], joints, fps, args.out_root)


if __name__ == "__main__":
    main()
