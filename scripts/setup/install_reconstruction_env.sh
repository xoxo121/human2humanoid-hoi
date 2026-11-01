#!/usr/bin/env bash
# Create the CARI4D reconstruction environment and build its CUDA extensions (PyTorch3D, nvdiffrast, kaolin,
# FoundationPose). Requires uv, a CUDA 12.4 toolkit in CUDA_HOME, and a C++17 compiler.
# Usage: install_reconstruction_env.sh [download|build|all]   (default all)
# "download" needs network access; "build" does not, so it can run on an offline compute node.
# TORCH_CUDA_ARCH_LIST selects the GPU architectures to compile for (default "8.0;8.6;8.9;9.0").
set -euo pipefail
source "$(dirname "$0")/../common.sh"

phase=${1:-all}
env=$RECONSTRUCTION_ENV
sources=$env/src
pip() { uv pip install --python "$env/bin/python" "$@"; }

download() {
  uv venv --python 3.11 "$env"
  mkdir -p "$sources"
  pip torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
  pip setuptools wheel ninja cmake pybind11 cmeel-boost==1.83.0 cmeel-eigen
  pip -r "$REPO_ROOT/requirements/reconstruction.txt"
  pip --no-deps "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"
  pip "git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030"
  pip -c "$RECONSTRUCTION_MODULES/v2d_cari4d/lib/pip-constraints.txt" \
    -e "$RECONSTRUCTION_MODULES/v2d_common[io]" -e "$RECONSTRUCTION_MODULES/v2d_mv[math,vis]" \
    -e "$RECONSTRUCTION_MODULES/v2d_mesh/lib" -e "$RECONSTRUCTION_MODULES/v2d_foundation_pose/lib" \
    -e "$RECONSTRUCTION_MODULES/v2d_moge/lib" -e "$RECONSTRUCTION_MODULES/v2d_sam3d_body/lib" \
    -e "$RECONSTRUCTION_MODULES/v2d_cari4d/lib"
  pip --no-deps -e "$REPO_ROOT"
  [ -d "$sources/pytorch3d" ] || git clone -q --depth 1 --branch v0.7.9 https://github.com/facebookresearch/pytorch3d.git "$sources/pytorch3d"
  [ -d "$sources/nvdiffrast" ] || git clone -q --depth 1 https://github.com/NVlabs/nvdiffrast.git "$sources/nvdiffrast"
  [ -d "$sources/kaolin" ] || git clone -q --depth 1 --branch v0.18.0 --recursive https://github.com/NVIDIAGameWorks/kaolin.git "$sources/kaolin"
}

build() {
  : "${CUDA_HOME:?set CUDA_HOME to a CUDA 12.4 toolkit}"
  export PATH=$CUDA_HOME/bin:$PATH FORCE_CUDA=1 IGNORE_TORCH_VER=1 UV_OFFLINE=1
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0} MAX_JOBS=${MAX_JOBS:-8}
  local site prefix
  site=$("$env/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  prefix=$site/cmeel.prefix
  export CPATH=$prefix/include/eigen3:${CPATH:-}
  pip --no-build-isolation --no-deps "$sources/pytorch3d" "$sources/nvdiffrast" "$sources/kaolin"

  # FoundationPose's extensions are built from a copy so the upstream checkout stays clean.
  local foundation_pose=$sources/FoundationPose
  rm -rf "$foundation_pose" && mkdir -p "$foundation_pose/bundlesdf"
  cp -r "$RECONSTRUCTION_MODULES/v2d_foundation_pose/lib/FoundationPose/mycpp" "$foundation_pose/"
  cp -r "$RECONSTRUCTION_MODULES/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda" "$foundation_pose/bundlesdf/"
  (cd "$foundation_pose/mycpp" && mkdir -p build && cd build \
    && cmake .. -DPYTHON_EXECUTABLE="$env/bin/python" \
      -DCMAKE_PREFIX_PATH="$prefix;$("$env/bin/python" -m pybind11 --cmakedir)" \
    && make -j "$MAX_JOBS")
  (cd "$foundation_pose/bundlesdf/mycuda" && pip --no-build-isolation --no-deps -e .)
  "$env/bin/python" -c "import pytorch3d, nvdiffrast.torch, kaolin, moge, hoi_imitation"
  echo "reconstruction environment ready: $env (FoundationPose extensions in $foundation_pose)"
}

case $phase in
  download) download ;;
  build) build ;;
  all) download && build ;;
  *) echo "usage: $0 [download|build|all]" >&2; exit 2 ;;
esac
