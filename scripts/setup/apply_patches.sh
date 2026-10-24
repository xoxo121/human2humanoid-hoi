#!/usr/bin/env bash
# Apply patches/*.patch in order to the pinned upstream checkout. Safe to rerun: patches that no longer
# apply are treated as applied, which is then confirmed by reverse-checking the whole series.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
upstream=$repo/third_party/video_to_data
patches=("$repo"/patches/*.patch)

for patch in "${patches[@]}"; do
  if git -C "$upstream" apply --check "$patch" 2>/dev/null; then
    git -C "$upstream" apply "$patch"
    echo "applied $(basename "$patch")"
  fi
done

for ((i = ${#patches[@]} - 1; i >= 0; i--)); do
  git -C "$upstream" apply --reverse --check "${patches[i]}" 2>/dev/null \
    || { echo "upstream checkout does not match the patch series at $(basename "${patches[i]}")" >&2; exit 1; }
  git -C "$upstream" apply --reverse "${patches[i]}"
done
for patch in "${patches[@]}"; do git -C "$upstream" apply "$patch"; done
echo "patch series is applied"
