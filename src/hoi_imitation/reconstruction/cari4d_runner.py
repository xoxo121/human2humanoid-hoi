"""Run CARI4D inference from a host environment instead of its Docker image.

The released runner points at the container's source roots. This module redirects them to the checkout and
to the host-built FoundationPose extensions (directory in ``CARI4D_FOUNDATIONPOSE_BUILD``), then calls the
unchanged command-line entry point with the same arguments.

Usage: python -m hoi_imitation.reconstruction.cari4d_runner <run_inference arguments>
"""

import os
import sys
from pathlib import Path

import v2d.cari4d.lib.run_inference as run_inference


def main() -> None:
    modules = Path(run_inference.__file__).resolve().parents[2]
    run_inference.SAM3D_SOURCE_ROOT = modules / "v2d_sam3d_body" / "lib"
    extensions = os.environ["CARI4D_FOUNDATIONPOSE_BUILD"]
    foundation_pose = modules / "v2d_foundation_pose" / "lib" / "FoundationPose"
    os.environ["PYTHONPATH"] = os.pathsep.join([extensions, str(foundation_pose), os.environ.get("PYTHONPATH", "")])
    sys.argv[0] = "run_inference"
    run_inference.main()


if __name__ == "__main__":
    main()
