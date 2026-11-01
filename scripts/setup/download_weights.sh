#!/usr/bin/env bash
# Download the reconstruction model weights into $WEIGHTS_DIR. CARI4D's checkpoints are gated on Hugging Face:
# export HF_TOKEN for an account with access to nvidia/cari4d_commercial and facebook/sam-3d-body-dinov3.
set -euo pipefail
source "$(dirname "$0")/../common.sh"

: "${HF_TOKEN:?export HF_TOKEN (see the comment at the top of this script)}"
"$SEGMENTATION_ENV/bin/python" -m v2d.sam2.lib.download_weights --output_dir "$WEIGHTS_DIR/sam2"
"$SEGMENTATION_ENV/bin/python" -m v2d.grounding_dino.lib.download_weights --output_dir "$WEIGHTS_DIR/grounding_dino"
"$RECONSTRUCTION_ENV/bin/python" -m v2d.cari4d.lib.download_weights --output_dir "$WEIGHTS_DIR/cari4d"
# SOMA-X body model used to convert the reconstructed human; reconstruct_episode.sh reads it offline.
HF_HOME=$WEIGHTS_DIR/huggingface "$RECONSTRUCTION_ENV/bin/python" -c \
  "from huggingface_hub import snapshot_download; snapshot_download('nvidia/soma-x')"
