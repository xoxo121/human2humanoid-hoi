"""Tests for the camera-to-world alignment in hoi_imitation.reconstruction.to_soma."""

import numpy as np
from scipy.spatial.transform import Rotation

from hoi_imitation.reconstruction.to_soma import camera_object_poses, world_from_camera


def _transform(rotvec, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def test_alignment_recovers_world_trajectory_from_first_frame():
    world_camera = _transform([0.3, -1.2, 0.5], [1.0, 2.0, -0.5])
    world_object = np.stack([_transform([0.0, 0.0, 0.1 * t], [0.05 * t, 0.0, 3.0]) for t in range(5)])
    camera_object = np.linalg.inv(world_camera) @ world_object

    estimate = world_from_camera(world_object[0], camera_object[0])
    np.testing.assert_allclose(estimate, world_camera, atol=1e-12)
    np.testing.assert_allclose(estimate @ camera_object, world_object, atol=1e-12)


def test_camera_object_poses_compose_the_mesh_recentering():
    source_to_reconstruction = _transform([0.0, 0.2, 0.0], [0.1, -0.2, 0.05])
    pose_abs = np.stack([_transform([0.1, 0.0, 0.0], [0.0, 0.0, 2.0])] * 3)
    poses = camera_object_poses(pose_abs, source_to_reconstruction)
    np.testing.assert_allclose(poses[1], pose_abs[1] @ source_to_reconstruction)
