#!/usr/bin/env bash
# Add a command to a task queue. Usage: enqueue.sh <queue dir> <task name> <command> [args...]
set -euo pipefail

queue_dir=$(realpath -m "$1")
name=$2
shift 2
mkdir -p "$queue_dir/pending"
task=$queue_dir/pending/$(date +%Y%m%d-%H%M%S-%N)_$name.sh
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf '%q ' "$@"
  echo
} > "$task"
echo "$task"
