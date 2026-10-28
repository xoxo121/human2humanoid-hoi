#!/usr/bin/env bash
# Retarget SOMA episode folders to G1 + Dex3 reference motion with upstream's soma_to_g1.py.
# Usage: retarget.sh <motion root> <soma folder>...
# Folders whose motion already exists are skipped. JOBS (default 4) sets the number of parallel episodes.
set -euo pipefail
source "$(dirname "$0")/common.sh"

motion_root=$(realpath -m "$1")
shift
retarget() {
  local folder=$1 sequence
  sequence=$(basename "$folder")
  if [ -f "$motion_root/whole_body/soma/sequence_id=$sequence/robot_name=g1/data.parquet" ]; then
    echo "skip $sequence"
    return
  fi
  (cd "$ROBOTIC_GROUNDING" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$RETARGET_ENV/bin/python" \
    scripts/retarget/soma_to_g1.py "$folder" --save --motion-root "$motion_root" > "$folder/retarget.log" 2>&1) \
    && echo "done $sequence" || { echo "failed $sequence: $folder/retarget.log"; return 1; }
}
export -f retarget
export motion_root ROBOTIC_GROUNDING RETARGET_ENV
printf '%s\n' "$@" | xargs -P "${JOBS:-4}" -I{} bash -c 'retarget "$(realpath "{}")"'
