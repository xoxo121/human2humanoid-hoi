#!/usr/bin/env bash
# Evaluate a checkpoint over independent single-world rollouts with object assistance disabled.
# Usage: evaluate_policy.sh <checkpoint> <output dir> [rollouts, default 6]
# Each rollout writes run_<i>.json (metrics) and run_<i>.parquet (object trajectory); finished ones are kept.
set -uo pipefail
source "$(dirname "$0")/common.sh"

checkpoint=$(realpath "$1")
out=$(realpath -m "$2")
rollouts=${3:-6}
use_policy_env
mkdir -p "$out"
cd "$FLASH_CHORD"
for run in $(seq 1 "$rollouts"); do
  [ -f "$out/run_$run.parquet" ] && continue
  python scripts/evaluate_policy.py "evaluation.checkpoint=$checkpoint" evaluation.world_count=1 \
    evaluation.reset_mode=explicit "evaluation.metrics_output=$out/run_$run.json" \
    "evaluation.object_trajectories_output=$out/run_$run.parquet" "hydra.run.dir=$out/hydra/run_$run" \
    logging.mode=disabled > "$out/run_$run.log" 2>&1 || echo "rollout $run failed: $out/run_$run.log" >&2
done
