"""Tests for hoi_imitation.data.soma_export on a synthetic LeRobot dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hoi_imitation.data import soma_export
from hoi_imitation.data.dataset import first_frame_object_pose, load_episodes

JOINTS = ["Hips", "Spine1", "Chest"]
FPS = 30.0
SEQUENCE = "2026-01-01_00-00-00_cube_lift_01"
QUARTER_TURN_Z_WXYZ = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]


def _make_dataset(root: Path, *, frames: int = 4, object_pose=None, visible=None, identity=None) -> Path:
    """Write a one-episode dataset with the LeRobot layout."""
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "mesh" / "cube").mkdir(parents=True)
    (root / "ground_plane").mkdir()
    (root / "mesh" / "cube" / "cube.glb").write_bytes(b"glb")
    (root / "ground_plane" / f"{SEQUENCE}.json").write_text(json.dumps({"plane": [0.0, -1.0, 0.0, 1.0]}))
    info = {
        "fps": FPS,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"observation.human.pose": {"names": [f"{j}{a}" for j in JOINTS for a in (".rx", ".ry", ".rz")]}},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    metadata = {
        "episode_index": 0,
        "sequence_id": SEQUENCE,
        "object": "cube",
        "frames": frames,
        "mesh": "mesh/cube/cube.glb",
        "ground_plane": f"ground_plane/{SEQUENCE}.json",
    }
    (root / "meta" / "episodes_metadata.jsonl").write_text(json.dumps(metadata) + "\n")

    if object_pose is None:
        object_pose = [[0.1 * t, 0.2, 3.0, *QUARTER_TURN_Z_WXYZ] for t in range(frames)]
    pose = np.arange(frames * len(JOINTS) * 3, dtype=np.float32).reshape(frames, -1) / 100.0
    identity = np.ones((frames, 45)) if identity is None else identity
    pd.DataFrame(
        {
            "observation.human.pose": list(pose),
            "observation.human.translation": list(np.tile([0.5, -0.3, -2.0], (frames, 1))),
            "observation.human.identity_coeffs": list(identity),
            "observation.human.scale_params": list(np.zeros((frames, 68))),
            "observation.object.pose": list(np.asarray(object_pose, dtype=np.float64)),
            "observation.object.visible": visible if visible is not None else [True] * frames,
            "timestamp": np.arange(frames) / FPS,
            "frame_index": np.arange(frames),
        }
    ).to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")
    return root


def _export(dataset: Path, out: Path) -> Path:
    episode = soma_export.load_episodes(dataset)[0]
    return soma_export.export_episode(episode, soma_export.joint_names(dataset), FPS, out)


def test_export_writes_soma_to_g1_layout(tmp_path):
    folder = _export(_make_dataset(tmp_path / "dataset"), tmp_path / "out")

    assert folder == tmp_path / "out" / SEQUENCE
    params = np.load(folder / "soma_params.npz")
    assert params["poses"].shape == (4, len(JOINTS), 3)
    np.testing.assert_allclose(params["poses"][1, 2], np.array([15, 16, 17]) / 100.0, rtol=1e-6)
    assert list(params["joint_names"]) == JOINTS
    assert str(params["identity_model_type"]) == "mhr"
    assert (folder / "reconstructed_mesh" / "output_aligned.glb").read_bytes() == b"glb"
    assert json.loads((folder / "ground_plane.json").read_text())["plane"] == [0.0, -1.0, 0.0, 1.0]


def test_object_quaternion_is_read_scalar_first(tmp_path):
    folder = _export(_make_dataset(tmp_path / "dataset"), tmp_path / "out")

    poses = np.load(folder / "poses.npy")
    assert poses.shape == (4, 4, 4)
    np.testing.assert_allclose(poses[2, :3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(poses[2, :3, 3], [0.2, 0.2, 3.0])


def test_trailing_invisible_frame_without_pose_is_dropped(tmp_path):
    pose = [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]] * 3 + [[0.0] * 7]
    dataset = _make_dataset(tmp_path / "dataset", object_pose=pose, visible=[True, True, True, False])

    folder = _export(dataset, tmp_path / "out")

    assert np.load(folder / "poses.npy").shape[0] == 3
    assert np.load(folder / "soma_params.npz")["poses"].shape[0] == 3


def test_invalid_pose_inside_episode_is_rejected(tmp_path):
    pose = [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]] * 4
    pose[1] = [0.0] * 7
    dataset = _make_dataset(tmp_path / "dataset", object_pose=pose, visible=[True, False, True, True])

    with pytest.raises(ValueError, match="frame 1"):
        _export(dataset, tmp_path / "out")


def test_time_varying_identity_is_rejected(tmp_path):
    identity = np.ones((4, 45))
    identity[2, 0] = 2.0

    with pytest.raises(ValueError, match="identity_coeffs"):
        _export(_make_dataset(tmp_path / "dataset", identity=identity), tmp_path / "out")


def test_first_frame_object_pose_reads_frame_zero(tmp_path):
    dataset = _make_dataset(tmp_path / "dataset")
    pose = first_frame_object_pose(load_episodes(dataset)[0])
    np.testing.assert_allclose(pose[:3, 3], [0.0, 0.2, 3.0])
    np.testing.assert_allclose(pose[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
