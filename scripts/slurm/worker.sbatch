#!/usr/bin/env bash
#SBATCH --job-name=hoi-worker
# Run queued tasks inside one Slurm allocation, one lane per visible GPU. Submit with submit_worker.sh.
#
# Tasks are bash scripts in <queue>/pending, claimed in name order (enqueue.sh names start with a timestamp).
# While work remains the worker submits its own successor (afterany), so a queue drains without a login
# session. Tasks left behind by a dead worker are requeued. A lane stops claiming tasks when less than
# RESERVE_HOURS of the allocation remain.
set -uo pipefail

queue_dir=$1
worker_script=$2  # Slurm runs a spooled copy of this file, so successors need the original path
reserve_hours=${RESERVE_HOURS:-4}
mkdir -p "$queue_dir"/{pending,running,done,failed,logs}
log() { echo "$(date '+%F %T') [$SLURM_JOB_ID] $*" | tee -a "$queue_dir/worker.log"; }

for task in "$queue_dir"/running/*; do  # running/<task>.<job id>.<lane>
  [ -e "$task" ] || continue
  owner=$(basename "$task" | awk -F. '{print $(NF-1)}')
  if [ "$owner" != "$SLURM_JOB_ID" ] && ! squeue -h -j "$owner" -o %i 2> /dev/null | grep -q .; then
    mv "$task" "$queue_dir/pending/$(basename "$task" | sed -E 's/\.[0-9]+\.[0-9]+$//')"
    log "requeued $(basename "$task")"
  fi
done

submit_successor() {
  [ -n "$(ls "$queue_dir/pending")" ] || return 0
  squeue -h -u "$USER" -t PENDING -n hoi-worker -o %Z | grep -qx "$queue_dir" && return 0
  local args
  read -r -a args < "$queue_dir/sbatch_args"
  sbatch "${args[@]}" --dependency=afterany:"$SLURM_JOB_ID" --chdir="$queue_dir" \
    -o "$queue_dir/logs/worker_%j.out" "$worker_script" "$queue_dir" "$worker_script" > /dev/null && log "submitted successor"
}
submit_successor

remaining=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{
  if (NF == 4) print $1 * 86400 + $2 * 3600 + $3 * 60 + $4; else if (NF == 3) print $1 * 3600 + $2 * 60 + $3;
  else print $1 * 60 + $2 }')
deadline=$(($(date +%s) + remaining - reserve_hours * 3600))

run_lane() {
  local gpu=$1 task
  while (($(date +%s) < deadline)); do
    task=$(
      flock "$queue_dir/.lock" bash -c '
        for name in $(ls "$1/pending"); do
          mv "$1/pending/$name" "$1/running/$name.$2.$3" && echo "$name" && break
        done' _ "$queue_dir" "$SLURM_JOB_ID" "$gpu"
    )
    [ -n "$task" ] || return
    log "lane $gpu started $task"
    if CUDA_VISIBLE_DEVICES=$gpu bash "$queue_dir/running/$task.$SLURM_JOB_ID.$gpu" > "$queue_dir/logs/${task%.sh}.log" 2>&1; then
      mv "$queue_dir/running/$task.$SLURM_JOB_ID.$gpu" "$queue_dir/done/$task" && log "lane $gpu finished $task"
    else
      mv "$queue_dir/running/$task.$SLURM_JOB_ID.$gpu" "$queue_dir/failed/$task" && log "lane $gpu failed $task"
    fi
  done
}

lanes=$(nvidia-smi -L 2> /dev/null | wc -l)
for ((gpu = 0; gpu < (lanes > 0 ? lanes : 1); gpu++)); do run_lane "$gpu" & done
wait
submit_successor
log "exiting with $(ls "$queue_dir/pending" | wc -l) pending"
