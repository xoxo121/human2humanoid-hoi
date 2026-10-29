#!/usr/bin/env bash
# Train a G1 + Dex3 tracking policy for one reference sequence with FlashCHORD's FlashSAC recipe.
# Usage: train_policy.sh <motion root> <sequence id> <run dir>
#
# The released 350M-step curriculum is compressed by STEP_SCALE (default 0.2, i.e. 70M steps); every stage
# keeps its relative position, so object assistance still reaches zero at 60% of the run.
# Environment:
#   RECIPE         released | pose | surface (default pose): object reward variant, see README
#   OVERRIDES      extra space-separated Hydra overrides
#   SEED           learner and reset seed (default 42)
#   WORLD_COUNTS   parallel world counts to try in order, e.g. "1024 512" for large objects (default: 2048)
#   STALL_MINUTES  stop a run whose log has not advanced for this long (default 15)
set -euo pipefail
source "$(dirname "$0")/common.sh"

motion_root=$(realpath "$1")
sequence=$2
run_dir=$(realpath -m "$3")
scale=${STEP_SCALE:-0.2}
seed=${SEED:-42}

case ${RECIPE:-pose} in
  released) recipe=() ;;
  pose) recipe=(objective.object_position_std=0.05 +objective.object_orientation_std=0.5) ;;
  surface) recipe=(+objective.object_surface_auc_softness=0.005) ;;
  *) echo "unknown RECIPE: $RECIPE" >&2; exit 2 ;;
esac
read -r -a extra <<< "${OVERRIDES:-}"

parquet=$motion_root/whole_body/soma/sequence_id=$sequence/robot_name=g1
[ -f "$parquet/data.parquet" ] || { echo "no retargeted motion for $sequence in $motion_root" >&2; exit 2; }
# Objects that start on the floor have no support geometry; FlashCHORD rejects an empty support stage.
support=false
grep -qsE "def (Cube|Cylinder) " "$motion_root/whole_body/reconstructed_stage/${sequence}_support.usda" && support=true

use_policy_env
read -r total interval thresholds < <(python - "$scale" <<'PY'
import sys
scale, worlds = float(sys.argv[1]), 2048
released = [35000320, 70000640, 104998912, 139999232, 174999552, 209999872, 288749568, 306249728, 349999104]
steps = [round(t * scale / worlds) * worlds for t in released]
print(steps[-1], steps[0], "[" + ",".join(map(str, steps)) + "]")
PY
)

export WANDB_MODE=offline WANDB_DIR=$run_dir
mkdir -p "$run_dir"
cd "$FLASH_CHORD"

train() {  # train <world count | default>
  local worlds=()
  [ "$1" != default ] && worlds=("scene.world_count=$1")
  python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac "task.parquet='$parquet'" \
    "reset.seed=$seed" "training.seed=$seed" logging.mode=offline logging.entity=null logging.upload_checkpoints=false \
    "training.total_environment_steps=$total" "training.checkpoint.save_interval_environment_steps=$interval" \
    "curriculum.thresholds=$thresholds" "scene.support=$support" "${worlds[@]}" "${recipe[@]}" "${extra[@]}" \
    "output_dir=$run_dir/train" > "$run_dir/train.log" 2>&1 &
  local trainer=$! stall=$((60 * ${STALL_MINUTES:-15}))
  # A scene that exhausts GPU memory can hang without stepping; stop runs whose log stops advancing.
  while kill -0 "$trainer" 2> /dev/null; do
    if (($(date +%s) - $(stat -c %Y "$run_dir/train.log") > stall)); then
      kill "$trainer" 2> /dev/null || true
      echo "stopped: no log output for ${STALL_MINUTES:-15} minutes" >> "$run_dir/train.log"
    fi
    sleep 30
  done
  wait "$trainer"
}

for count in ${WORLD_COUNTS:-default}; do
  if train "$count"; then
    echo "$sequence: $run_dir/train/policy_$total.safetensors"
    exit 0
  fi
  mv "$run_dir/train.log" "$run_dir/train_worlds_${count}_failed.log"
  [ -d "$run_dir/train" ] && mv "$run_dir/train" "$run_dir/train_worlds_${count}_failed"
done
echo "$sequence: training failed, see $run_dir" >&2
exit 1
