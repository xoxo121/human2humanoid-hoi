#!/usr/bin/env bash
# Create the policy environment used for training and evaluation. Requires uv and an NVIDIA driver with CUDA 12.
set -euo pipefail
source "$(dirname "$0")/../common.sh"

uv venv --clear --python 3.11 "$POLICY_ENV"
uv pip install --python "$POLICY_ENV/bin/python" -r "$REPO_ROOT/requirements/policy.txt"
uv pip install --python "$POLICY_ENV/bin/python" --no-deps -e "$FLASH_CHORD" -e "$REPO_ROOT"
"$POLICY_ENV/bin/python" -c "import flash_chord, hoi_imitation"
echo "policy environment ready: $POLICY_ENV"
