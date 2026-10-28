#!/usr/bin/env bash
# Build G1 reference motion for dataset episodes: export to SOMA, retarget, and stage support surfaces.
# Usage: prepare_references.sh [episode index ...]   (default: every episode)
# Output: $WORK_DIR/soma/<sequence>/ and the motion root $WORK_DIR/motion. With REPAIR_CONTACTS=1, a copy
# with repaired hand contact labels is written to $WORK_DIR/motion_repaired.
set -euo pipefail
source "$(dirname "$0")/common.sh"

soma_root=$WORK_DIR/soma
motion_root=$WORK_DIR/motion
python=$RETARGET_ENV/bin/python

mapfile -t folders < <($python -m hoi_imitation.data.soma_export "$DATA_ROOT" "$soma_root" ${1:+--episodes "$@"} \
  | sed -n 's/^episode [0-9]* -> \([^ ]*\) .*/\1/p')
"$(dirname "$0")/retarget.sh" "$motion_root" "${folders[@]}"

stage=$motion_root/whole_body/reconstructed_stage
mkdir -p "$stage"
for folder in "${folders[@]}"; do
  support=$DATA_ROOT/support_surface/$(basename "$folder")_support.usda
  [ -f "$support" ] && cp "$support" "$stage/"
done

if [ "${REPAIR_CONTACTS:-0}" = 1 ]; then
  $python -m hoi_imitation.data.contact_labels "$motion_root" "$WORK_DIR/motion_repaired"
fi
