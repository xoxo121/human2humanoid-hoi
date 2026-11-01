"""Convert a CARI4D video reconstruction into the SOMA folder layout read by ``soma_to_g1.py``.

CARI4D reconstructs the human (MHR) and the object in the uncalibrated camera frame. One rigid
camera-to-world transform is fixed from the object's known first-frame pose,
``world_T_camera = world_T_object(0) @ inv(camera_T_object(0))``, and applied to every frame. Scale comes from
CARI4D's metric reconstruction; no later frame of any reference trajectory is used.

Usage: python -m hoi_imitation.reconstruction.to_soma <reconstruction dir> <dataset root> <episode> <output root>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from hoi_imitation.data.dataset import first_frame_object_pose, load_episodes

# MHR's native axes to OpenCV, and the OpenCV world to the SOMA world (Y up): both flip Y and Z.
FLIP_YZ = np.diag([1.0, -1.0, -1.0])


def camera_object_poses(pose_abs: np.ndarray, source_to_reconstruction: np.ndarray) -> np.ndarray:
    """Pose of the original object mesh in the camera frame, from CARI4D's pose of its re-centred mesh."""
    return np.asarray(pose_abs, dtype=np.float64) @ np.asarray(source_to_reconstruction, dtype=np.float64)


def world_from_camera(world_object_first: np.ndarray, camera_object_first: np.ndarray) -> np.ndarray:
    """Rigid transform that places the first reconstructed object pose at its known world pose."""
    return world_object_first @ np.linalg.inv(camera_object_first)


def decode_human(prediction: dict, mhr_assets: Path, device: str) -> tuple[np.ndarray, ...]:
    """Decode MHR into camera-frame vertices (T, V, 3), faces, shape (T, 45) and model parameters (T, 204)."""
    import roma
    import torch
    from lib_mhr.mhr_layer import MHRLayer
    from lib_mhr.rotations import rot6d_to_rotmat
    from sam_3d_body.models.modules.mhr_utils import compact_cont_to_model_params_body

    layer = MHRLayer.from_mhr_assets(mhr_assets_root=mhr_assets, device=device)
    head = layer.backend._ensure_head(device)

    def tensor(key: str) -> torch.Tensor:
        return torch.as_tensor(np.asarray(prediction[key]), dtype=torch.float32, device=device)

    vertices, shapes, parameters = [], [], []
    with torch.no_grad():
        for start in range(0, len(prediction["mhr_trans"]), 64):
            batch = slice(start, start + 64)
            translation = tensor("mhr_trans")[batch]
            rotation = roma.rotmat_to_euler("ZYX", rot6d_to_rotmat(tensor("mhr_global_rot6d")[batch]))
            batch_vertices, model_parameters = head.mhr_forward(
                global_trans=torch.zeros_like(translation),
                global_rot=rotation,
                body_pose_params=compact_cont_to_model_params_body(tensor("mhr_body_pose_cont")[batch]),
                hand_pose_params=tensor("mhr_hand")[batch],
                scale_params=tensor("mhr_scale")[batch],
                shape_params=tensor("mhr_shape")[batch],
                expr_params=tensor("mhr_face")[batch] if "mhr_face" in prediction else None,
                return_model_params=True,
            )
            vertices.append(batch_vertices.cpu().numpy() @ FLIP_YZ + translation.cpu().numpy()[:, None])
            shapes.append(tensor("mhr_shape")[batch].cpu().numpy())
            parameters.append(model_parameters.cpu().numpy())
    return np.concatenate(vertices), head.faces.cpu().numpy(), np.concatenate(shapes), np.concatenate(parameters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reconstruction", type=Path, help="CARI4D output directory of one video")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("episode", type=int)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--weights", type=Path, required=True, help="CARI4D weights directory")
    parser.add_argument("--modules", type=Path, required=True, help="upstream reconstruction/modules directory")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.path[:0] = [str(args.modules / "v2d_cari4d/lib/cari4d"), str(args.modules / "v2d_sam3d_body/lib")]

    import torch
    from v2d.sam3d_body.lib.export_soma import export_soma

    episode = load_episodes(args.dataset_root)[args.episode]
    name = args.reconstruction.name
    bundle = torch.load(args.reconstruction / "inference" / "refined.pth", map_location="cpu", weights_only=False)
    prediction = {key: value.cpu().numpy() if torch.is_tensor(value) else value for key, value in bundle["pr"].items()}
    export = json.loads((args.reconstruction / "export" / name / "wild_export.json").read_text())

    camera_object = camera_object_poses(prediction["pose_abs"], export["source_object_mesh_to_aligned_transform"])
    world_camera = world_from_camera(first_frame_object_pose(episode), camera_object[0])
    vertices, faces, shape, model_parameters = decode_human(prediction, args.weights / "sam3d_body", args.device)
    vertices = vertices @ world_camera[:3, :3].T + world_camera[:3, 3]

    folder = args.output_root / episode.sequence_id
    (folder / "reconstructed_mesh").mkdir(parents=True, exist_ok=True)
    work = folder / "mhr"
    work.mkdir(exist_ok=True)
    # export_soma's mesh path undoes a Y/Z flip, which is exactly the OpenCV-to-SOMA world change.
    torch.save(
        {"pred_vertices": torch.as_tensor(vertices, dtype=torch.float32), "faces": torch.as_tensor(faces)},
        work / "mesh.pt",
    )
    torch.save(
        {"shape_params": torch.as_tensor(shape), "mhr_model_params": torch.as_tensor(model_parameters)},
        work / "params.pt",
    )
    export_soma(
        params_path=work / "params.pt",
        mesh_path=work / "mesh.pt",
        output_path=folder / "soma_params.npz",
        device=args.device,
    )

    np.save(folder / "poses.npy", world_camera @ camera_object)
    np.save(work / "world_from_camera.npy", world_camera)
    shutil.copyfile(episode.mesh, folder / "reconstructed_mesh" / "output_aligned.glb")
    shutil.copyfile(episode.ground_plane, folder / "ground_plane.json")
    print(f"{name} -> {folder}: {len(camera_object)} frames")


if __name__ == "__main__":
    main()
