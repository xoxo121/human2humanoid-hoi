"""Index a LeRobot human-object interaction dataset.

Each episode has one parquet of per-frame human (SOMA-X) and object poses, a textured object mesh, a ground
plane, and an exocentric RGB video. Object poses are ``world_T_object`` as ``[x, y, z, qw, qx, qy, qz]`` in the
OpenCV world frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

OBJECT_POSE_COLUMN = "observation.object.pose"
VIDEO_KEY = "observation.images.exo_camera"


@dataclass(frozen=True)
class Episode:
    index: int
    sequence_id: str
    object_name: str
    num_frames: int
    mesh: Path
    ground_plane: Path
    data: Path
    video: Path


def load_episodes(root: Path) -> dict[int, Episode]:
    """Index ``meta/episodes_metadata.jsonl`` by episode."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    episodes = {}
    for line in (root / "meta" / "episodes_metadata.jsonl").read_text().splitlines():
        meta = json.loads(line)
        index = int(meta["episode_index"])
        chunk = index // int(info["chunks_size"])
        episodes[index] = Episode(
            index=index,
            sequence_id=meta["sequence_id"],
            object_name=meta["object"],
            num_frames=int(meta["frames"]),
            mesh=root / meta["mesh"],
            ground_plane=root / meta["ground_plane"],
            data=root / info["data_path"].format(episode_chunk=chunk, episode_index=index),
            video=root / info["video_path"].format(episode_chunk=chunk, video_key=VIDEO_KEY, episode_index=index),
        )
    return episodes


def pose_to_matrices(pose: np.ndarray) -> np.ndarray:
    """Convert ``(T, 7)`` ``[x, y, z, qw, qx, qy, qz]`` poses to ``(T, 4, 4)`` homogeneous matrices."""
    pose = np.asarray(pose, dtype=np.float64)
    matrices = np.tile(np.eye(4), (len(pose), 1, 1))
    matrices[:, :3, :3] = Rotation.from_quat(pose[:, 3:], scalar_first=True).as_matrix()
    matrices[:, :3, 3] = pose[:, :3]
    return matrices


def first_frame_object_pose(episode: Episode) -> np.ndarray:
    """Return the object's ``world_T_object`` at frame 0 as a 4x4 matrix; no other frame is read."""
    frame = pd.read_parquet(episode.data, columns=["frame_index", OBJECT_POSE_COLUMN])
    row = frame.loc[frame["frame_index"] == 0, OBJECT_POSE_COLUMN]
    if len(row) != 1:
        raise ValueError(f"{episode.data} has no unique frame 0")
    return pose_to_matrices(np.asarray(row.iloc[0])[None])[0]
