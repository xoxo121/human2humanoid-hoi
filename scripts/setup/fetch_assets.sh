#!/usr/bin/env bash
# Download the upstream assets this project uses: Git LFS files (robot meshes, controller, reconstruction
# models) and the SOMA body model. Run after install_retarget_env.sh.
set -euo pipefail
source "$(dirname "$0")/../common.sh"

git -C "$UPSTREAM" lfs install --local > /dev/null
git -C "$UPSTREAM" lfs pull --include "robotic_grounding/**,reconstruction/modules/v2d_cari4d/**,\
reconstruction/modules/v2d_sam3d_body/**,reconstruction/modules/v2d_foundation_pose/**"

# SOMA body model used by the retargeter (needs the retarget environment).
(cd "$ROBOTIC_GROUNDING" && "$RETARGET_ENV/bin/python" scripts/setup_soma_assets.py)
