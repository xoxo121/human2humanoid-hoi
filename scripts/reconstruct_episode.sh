#!/usr/bin/env bash
# Reconstruct a dataset episode from its exocentric video and retarget it to G1 reference motion.
# Usage: reconstruct_episode.sh <episode index>
#
# Grounding DINO prompts -> SAM2 masks -> CARI4D -> SOMA folder in the world frame fixed by the object's
# first-frame pose -> G1 retargeting. Steps whose output exists are skipped.
# Output: $WORK_DIR/reconstruction/<episode>/, SOMA folders in $WORK_DIR/video_soma and motion in
# $WORK_DIR/video_motion.
set -euo pipefail
source "$(dirname "$0")/common.sh"

episode=$1
name=episode_$(printf %06d "$episode")
out=$WORK_DIR/reconstruction/$name
export HF_HOME=$WEIGHTS_DIR/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CARI4D_FOUNDATIONPOSE_BUILD=${CARI4D_FOUNDATIONPOSE_BUILD:-$RECONSTRUCTION_ENV/src/FoundationPose}
segmentation=$SEGMENTATION_ENV/bin/python
reconstruction=$RECONSTRUCTION_ENV/bin/python

read -r video frames object_name mesh sequence < <("$reconstruction" - "$DATA_ROOT" "$episode" <<'PY'
import sys
from hoi_imitation.data.dataset import load_episodes
e = load_episodes(sys.argv[1])[int(sys.argv[2])]
print(e.video, e.num_frames, e.object_name, e.mesh, e.sequence_id)
PY
)

# CARI4D's tools require the "<name>.0.color.mp4" file name and reject symbolic links.
mkdir -p "$out/input"
input=$out/input/$name.0.color.mp4
[ -f "$input" ] || ln "$video" "$input" 2> /dev/null || cp "$video" "$input"

[ -f "$out/prompts.json" ] || "$segmentation" -m hoi_imitation.reconstruction.prompts --video_path "$input" \
  --object_prompt "${object_name//_/ }" --model_dir "$WEIGHTS_DIR/grounding_dino" --output_path "$out/prompts.json"

if [ ! -f "$out/masks.h5" ]; then
  "$segmentation" -m v2d.sam2.lib.video_to_masks --video_path "$input" --prompts_path "$out/prompts.json" \
    --masks_dir "$out/masks" --weights_dir "$WEIGHTS_DIR/sam2"
  "$reconstruction" -m v2d.cari4d.lib.pack_masks --video_path "$input" --human_masks_path "$out/masks/0" \
    --object_masks_path "$out/masks/1" --output_path "$out/masks.h5"
fi

[ -f "$out/cari4d/$name/inference/refined.pth" ] || "$reconstruction" -m hoi_imitation.reconstruction.cari4d_runner \
  --video_path "$input" --mask_h5_path "$out/masks.h5" --object_mesh_path "$mesh" \
  --weights_path "$WEIGHTS_DIR/cari4d" --output_dir "$out/cari4d" --skip_weight_download --expected_frames "$frames"

[ -f "$WORK_DIR/video_soma/$sequence/soma_params.npz" ] || "$reconstruction" -m hoi_imitation.reconstruction.to_soma \
  "$out/cari4d/$name" "$DATA_ROOT" "$episode" "$WORK_DIR/video_soma" \
  --weights "$WEIGHTS_DIR/cari4d" --modules "$RECONSTRUCTION_MODULES"

"$(dirname "$0")/retarget.sh" "$WORK_DIR/video_motion" "$WORK_DIR/video_soma/$sequence"
stage=$WORK_DIR/video_motion/whole_body/reconstructed_stage
mkdir -p "$stage"
support=$DATA_ROOT/support_surface/${sequence}_support.usda
[ -f "$support" ] && cp "$support" "$stage/"
echo "$name: reference motion for $sequence in $WORK_DIR/video_motion"
