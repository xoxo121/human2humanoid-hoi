#!/usr/bin/env bash
# Continue a finished run to more environment steps in its final, assistance-free curriculum stage.
# Usage: finetune_policy.sh <run dir> <total environment steps> <output run dir>
# World count, curriculum thresholds and support setting are taken from the source run; the learning-rate
# schedule keeps the source run's horizon, so it continues at its final value. The replay buffer is refilled.
# OVERRIDES adds Hydra overrides, e.g. a different object reward.
set -euo pipefail
source "$(dirname "$0")/common.sh"

source_dir=$(realpath "$1")
target=$2
run_dir=$(realpath -m "$3")
state=$(ls "$source_dir"/train/state_*.safetensors | sort -V | tail -1)
read -r -a extra <<< "${OVERRIDES:-}"

use_policy_env
read -r parquet worlds thresholds horizon support < <(python - "$state" <<'PY'
import json, struct, sys
with open(sys.argv[1], "rb") as handle:
    metadata = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))["__metadata__"]
config = json.loads(metadata["user.resolved_config_json"])
training = config["training"]
horizon = training["optimization"].get("learning_rate_schedule_updates") or (
    training["total_environment_steps"] // training["world_count"] * training["updates_per_collection"])
thresholds = "[" + ",".join(str(t) for t in config["curriculum"]["thresholds"]) + "]"
print(config["task"]["parquet"], config["scene"]["world_count"], thresholds, int(horizon),
      str(config["scene"]["support"]).lower())
PY
)

export WANDB_MODE=offline WANDB_DIR=$run_dir
mkdir -p "$run_dir"
cd "$FLASH_CHORD"
python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac "task.parquet='$parquet'" \
  "reset.seed=${SEED:-42}" "training.seed=${SEED:-42}" logging.mode=offline logging.entity=null \
  logging.upload_checkpoints=false "scene.world_count=$worlds" "scene.support=$support" \
  "curriculum.thresholds=$thresholds" "training.total_environment_steps=$target" \
  "training.optimization.learning_rate_schedule_updates=$horizon" training.resume=true \
  "training.checkpoint_path=$state" "${extra[@]}" "output_dir=$run_dir/train" > "$run_dir/train.log" 2>&1
echo "finetuned from $(basename "$state"): $run_dir/train/policy_$target.safetensors"
