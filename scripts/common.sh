# Shared configuration for the scripts in this directory. Source it; do not execute it.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

resolve() { case $1 in /*) echo "$1" ;; *) echo "$REPO_ROOT/$1" ;; esac; }

DATA_ROOT=$(resolve "${DATA_ROOT:-data/hoi_dataset}")
WORK_DIR=$(resolve "${WORK_DIR:-outputs}")
ENV_DIR=$(resolve "${ENV_DIR:-.venvs}")
WEIGHTS_DIR=$(resolve "${WEIGHTS_DIR:-weights}")
POLICY_ENV=${POLICY_ENV:-$ENV_DIR/policy}
RETARGET_ENV=${RETARGET_ENV:-$ENV_DIR/retarget}
RECONSTRUCTION_ENV=${RECONSTRUCTION_ENV:-$ENV_DIR/reconstruction}
SEGMENTATION_ENV=${SEGMENTATION_ENV:-$ENV_DIR/segmentation}

UPSTREAM=$REPO_ROOT/third_party/video_to_data
FLASH_CHORD=$UPSTREAM/robotic_grounding/flash_chord
ROBOTIC_GROUNDING=$UPSTREAM/robotic_grounding
RECONSTRUCTION_MODULES=$UPSTREAM/reconstruction/modules

# Activate the policy environment, including the CUDA 12 runtime libraries shipped in its wheels.
use_policy_env() {
  export PATH=$POLICY_ENV/bin:$PATH XLA_PYTHON_CLIENT_PREALLOCATE=false
  local libs
  libs=$(ls -d "$POLICY_ENV"/lib/python3.*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
  export LD_LIBRARY_PATH=${libs}${LD_LIBRARY_PATH:-}
}
