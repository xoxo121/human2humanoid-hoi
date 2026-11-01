#!/usr/bin/env bash
# Create the SAM2 + Grounding DINO environment used to prompt and segment the actor and the object.
# Grounding DINO is built without its CUDA operator; it only runs on a few frames per video.
set -euo pipefail
source "$(dirname "$0")/../common.sh"

env=$SEGMENTATION_ENV
pip() { uv pip install --python "$env/bin/python" "$@"; }

uv venv --clear --python 3.11 "$env"
pip torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip setuptools wheel numpy==1.26.4 transformers==4.27.4 huggingface-hub pillow opencv-python omegaconf decord \
  imageio supervision addict yapf timm pycocotools
SAM2_BUILD_CUDA=0 pip --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
mkdir -p "$env/src"
[ -d "$env/src/GroundingDINO" ] || git clone -q --depth 1 https://github.com/IDEA-Research/GroundingDINO.git "$env/src/GroundingDINO"
FORCE_CUDA=0 pip --no-build-isolation --no-deps "$env/src/GroundingDINO"
pip -e "$RECONSTRUCTION_MODULES/v2d_common[io]" -e "$RECONSTRUCTION_MODULES/v2d_mv" \
  -e "$RECONSTRUCTION_MODULES/v2d_sam2/lib" -e "$RECONSTRUCTION_MODULES/v2d_grounding_dino/lib"
pip --no-deps -e "$REPO_ROOT"
"$env/bin/python" -c "import sam2, groundingdino, hoi_imitation"
echo "segmentation environment ready: $env"
