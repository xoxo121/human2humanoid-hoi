# human2humanoid-hoi

Whole-body control policies for a Unitree G1 humanoid with Dex3 hands that reproduce human-object
interactions: carrying, dragging, rotating and placing everyday objects. Reference motion comes either from
motion capture or from a single exocentric RGB video, and one tracking policy is trained per interaction
with reinforcement learning in GPU-parallel simulation.

```
 motion capture ──► SOMA export ─────────────────────┐
                                                     ├─► G1 retargeting ─► RL tracking policy ─► evaluation
 RGB video ──► prompts ─► SAM2 ─► CARI4D ─► world ───┘    (SOMA → G1+Dex3)   (FlashSAC + SONIC)
               (Grounding DINO)   (human+object 4D)  alignment
```

The simulation, the SONIC whole-body controller, the SOMA-to-G1 retargeter and the CARI4D reconstruction
model come from NVIDIA's [video_to_data](https://github.com/nvidia-isaac/video_to_data), pinned as a submodule.
This repository adds the pieces needed to go from raw episodes or videos to evaluated policies:

- **Object tracking rewards.** Two variants of FlashCHORD's object reward, applied as a patch:
  a *pose* reward that multiplies the position kernel by an orientation kernel `exp(-θ²/σ²)`, and a *surface*
  reward, a smooth surrogate of the ADD-AUC tracking metric. The surface reward averages sigmoids of
  `(k cm − d)` for `k = 1…9`, where `d` is the RMS distance between corresponding mesh vertices under the
  simulated and target poses. `d` is computed in closed form from the mesh's vertex mean and second moment,
  so it costs the same per world as the position term.
- **Data preparation.** Export of LeRobot HOI episodes to the retargeter's input layout with validation, and
  repair of false-negative hand contact labels (a palm-distance threshold drops contact mid-carry).
- **Training at a fraction of the published budget.** The 350M-step curriculum compressed to any budget with
  every stage at its original relative position, assistance-free finetuning from a finished run, a stall
  watchdog, and per-scene world-count fallbacks for large objects.
- **Evaluation and model selection.** Repeated single-world rollouts without object assistance, and a
  conservative rule that replaces a baseline checkpoint only on a clear gain.
- **Reconstruction without Docker.** A host build of the CARI4D pipeline, automatic prompting, and a
  camera-to-world alignment that turns a monocular reconstruction into retargetable motion.
- **Cluster execution.** A Slurm task queue whose workers resubmit themselves, so long experiment queues
  drain without an open login session.

## Repository layout

```
patches/                 changes to the pinned upstream (object rewards, CARI4D fix)
requirements/            locked environments: policy (CUDA 12), retarget (CPU), reconstruction
scripts/
  setup/                 environment installation, upstream assets, weights, patching
  prepare_references.sh  dataset episodes → retargeted G1 reference motion
  train_policy.sh        train one tracking policy
  finetune_policy.sh     continue a finished run without object assistance
  evaluate_policy.sh     independent evaluation rollouts
  reconstruct_episode.sh video → reconstruction → reference motion
  slurm/                 self-resubmitting task queue
src/hoi_imitation/
  data/                  dataset indexing, SOMA export, contact-label repair
  reconstruction/        prompting, CARI4D runner, world alignment and SOMA conversion
  evaluation/            rollout summaries and checkpoint selection
tests/
third_party/video_to_data
```

## Setup

Requirements: Linux, an NVIDIA GPU with a CUDA 12 driver, [uv](https://docs.astral.sh/uv/) and
[Git LFS](https://git-lfs.com/).

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/xoxo121/human2humanoid-hoi.git
cd human2humanoid-hoi
cp .env.example .env                       # data, output and environment locations
bash scripts/setup/apply_patches.sh
bash scripts/setup/install_retarget_env.sh
bash scripts/setup/install_policy_env.sh
bash scripts/setup/fetch_assets.sh         # robot and model assets (~2 GB)
```

The video pipeline needs two more environments and gated model weights:

```bash
bash scripts/setup/install_segmentation_env.sh
CUDA_HOME=/usr/local/cuda-12.4 bash scripts/setup/install_reconstruction_env.sh
HF_TOKEN=... bash scripts/setup/download_weights.sh
```

`install_reconstruction_env.sh download` and `... build` split the networked and the compile steps for
clusters whose compute nodes are offline.

### Data

The scripts expect a LeRobot dataset of human-object interaction episodes with SOMA-X human parameters,
object poses, object meshes, ground planes and exocentric videos, such as NVIDIA's multi-view HOI release:

```bash
huggingface-cli download nvidia/video_to_data_challenge --repo-type dataset \
  --include "track_2/tier_1_multiview_caption/*" --local-dir data/
```

Point `DATA_ROOT` in `.env` at the directory that contains `meta/`.

## Usage

```bash
bash scripts/prepare_references.sh 8 18         # episodes 8 and 18 → outputs/motion
bash scripts/train_policy.sh outputs/motion <sequence id> outputs/runs/<name>
bash scripts/evaluate_policy.sh outputs/runs/<name>/train/policy_70000640.safetensors outputs/eval/<name>
python -m hoi_imitation.evaluation.summary outputs/eval/*
```

`train_policy.sh` defaults to the pose reward and a 70M-step run (`STEP_SCALE=0.2`). Set `RECIPE=surface`
for the surface reward or `RECIPE=released` for the original object reward, and `WORLD_COUNTS="1024 512"` for
objects too large for 2048 parallel worlds on one GPU.

To compare variants, list them in a CSV (`episode,checkpoint,evaluation_dir`, baseline first per episode) and
let the selection rule pick one checkpoint per episode:

```bash
python -m hoi_imitation.evaluation.selection candidates.csv checkpoints.json
```

From video instead of motion capture:

```bash
bash scripts/reconstruct_episode.sh 18           # → outputs/video_motion
bash scripts/train_policy.sh outputs/video_motion <sequence id> outputs/runs/<name>
```

The reconstruction is placed in the world by the object's first-frame pose; everything after the first frame
comes from the video.

### Slurm

```bash
bash scripts/slurm/enqueue.sh outputs/queue train_ep18 bash scripts/train_policy.sh outputs/motion <sequence id> outputs/runs/ep18
bash scripts/slurm/submit_worker.sh outputs/queue -p <partition> --gres=gpu:2 -t 24:00:00
```

Each worker runs one task per GPU, requeues tasks orphaned by a dead worker, and submits its successor while
work remains.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The patched upstream objective has its own tests:
`cd third_party/video_to_data/robotic_grounding/flash_chord && pytest tests/objectives`.

## License

Apache License 2.0. The upstream code in `third_party/video_to_data` is NVIDIA's, under its own license.
