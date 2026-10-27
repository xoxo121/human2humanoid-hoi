#!/usr/bin/env bash
# Create the CPU environment used to retarget SOMA human motion to the G1 humanoid. Requires uv.
set -euo pipefail
source "$(dirname "$0")/../common.sh"

uv venv --clear --python 3.11 "$RETARGET_ENV"
uv pip install --python "$RETARGET_ENV/bin/python" --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
uv pip install --python "$RETARGET_ENV/bin/python" -r "$REPO_ROOT/requirements/retarget.txt"
uv pip install --python "$RETARGET_ENV/bin/python" --no-deps -e "$ROBOTIC_GROUNDING/source/robotic_grounding" -e "$REPO_ROOT"
"$RETARGET_ENV/bin/python" -c "import robotic_grounding.retarget.read_soma, pinocchio, hoi_imitation"
echo "retarget environment ready: $RETARGET_ENV"
