#!/usr/bin/env bash
# Start draining a task queue with worker.sbatch, unless a worker for it is already queued or running.
# Usage: submit_worker.sh <queue dir> [sbatch options...]
# The sbatch options (partition, GPUs, time) are stored with the queue and reused by successor workers.
set -euo pipefail

queue_dir=$(realpath -m "$1")
shift
mkdir -p "$queue_dir"/{pending,running,done,failed,logs}
[ $# -gt 0 ] && echo "$*" > "$queue_dir/sbatch_args"
[ -f "$queue_dir/sbatch_args" ] || { echo "first submission needs sbatch options" >&2; exit 2; }
if squeue -h -u "$USER" -n hoi-worker -o %Z | grep -qx "$queue_dir"; then
  echo "a worker is already queued or running for $queue_dir"
  exit 0
fi
read -r -a args < "$queue_dir/sbatch_args"
worker=$(dirname "$(realpath "$0")")/worker.sbatch
sbatch "${args[@]}" --chdir="$queue_dir" -o "$queue_dir/logs/worker_%j.out" "$worker" "$queue_dir" "$worker"
