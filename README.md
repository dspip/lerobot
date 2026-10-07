# Fault injection

Faults, recovery datagen, and eval recording on a **pinned** [LeRobot](https://github.com/huggingface/lerobot) commit. This repo does not contain the LeRobot library. `uv sync` installs that pin for you.

| | |
|---|---|
| Package | `fault_system` (`src/fault_system`) |
| Eval | `lerobot-eval-faults` |
| Scene edits | `lerobot_env_libero_overlay` → `--env.type=libero_overlay` |
| Pin | `huggingface/lerobot` @ `e0d50211` |
| Python | 3.12+ (uv installs it; do not use the system interpreter) |
| License | Apache-2.0 |

The overlay distribution keeps the `lerobot_env_` prefix. Upstream plugin discovery only auto-imports names that start with `lerobot_env_`, `lerobot_policy_`, `lerobot_robot_`, `lerobot_camera_`, `lerobot_teleoperator_`, or `lerobot_strategy_`.

## Quick Start

Linux, NVIDIA GPU, from a fresh clone to a mid-air drop eval. Run every command from the repository root.

### Prerequisites

- **Git**
- **uv** — [install uv](https://docs.astral.sh/uv/getting-started/installation/). It creates `.venv` and installs Python. `uv run` is the environment; you do not activate a conda env.
- **NVIDIA driver** new enough for the CUDA 12.8 PyTorch wheel (driver 570.86 or newer). Check with `nvidia-smi`.
- **System packages** MuJoCo needs for headless rendering:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential cmake git \
  libegl1 libgl1 libglib2.0-0 ffmpeg
```

- **Internet** on the first run. `uv sync` downloads the pinned LeRobot. The first LIBERO episode downloads scene assets. A Pi0 / Pi0.5 checkpoint that names `google/paligemma-3b-pt-224` also needs that tokenizer: accept the model license on Hugging Face, then `uv run hf auth login`.

### 1. Clone

```bash
git clone https://github.com/dspip/lerobot.git
cd lerobot
```

### 2. Install

```bash
uv sync --locked
```

If that fails while building `egl-probe` with `Compatibility with CMake < 3.5 has been removed`:

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --locked
```

Check the environment:

```bash
uv run python -c "import torch, libero; print('CUDA', torch.cuda.is_available())"
```

`CUDA True` is what the eval needs. `False` means the driver or GPU is not visible to this process.

### 3. Environment variables

```bash
export MUJOCO_GL=egl
```

Keep that shell open. Headless machines render LIBERO through EGL. Without it, the first episode fails inside MuJoCo.

Do not set `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` until the checkpoint, tokenizer, and LIBERO assets have been downloaded once.

### 4. Checkpoint

Point `CHECKPOINT` at a local LeRobot folder that contains `config.json` next to `model.safetensors`:

```bash
export CHECKPOINT=/absolute/path/to/pretrained_model
```

This drop command assumes that file's image keys are LIBERO's defaults, `observation.images.image` and `observation.images.image2`. A Pi0 or Pi0.5 model trained on this repo's LIBERO datasets is that shape. A third view named `empty_camera_0` in `config.json` is a padded camera; leave it there. The policy fills a missing view. Do not add SmolVLA's camera rename for this checkpoint.

If `config.json` lists `observation.images.camera1` instead (the public `lerobot/smolvla_libero` weights), add the two flags in [Commands](#commands). If `"device"` is missing or not `cuda`, add `--policy.device=cuda`.

### 5. Drop-and-recovery eval

`libero_object` task `0` is alphabet soup (`alphabet_soup_1`) and basket `basket_1`. The gripper opens once, while the can is grasped, between steps 10 and 400, and 0.34–0.38 m from the basket. After it lands, **the same policy stays in control**. The flag for that is `--fault.post_drop_mode=immediate_policy`. It works for Pi0, Pi0.5, SmolVLA, or any other loaded policy. The IK planner does not start.

`--eval.batch_size=1` and `--eval.use_async_envs=false` are required. The drop wrapper has no per-env simulator handle inside an async vector env. `--fault.t_max=400` is required. The default window ends at step 30, which is before most grasps.

```bash
uv run lerobot-eval-faults \
  --policy.path="$CHECKPOINT" \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --output_dir=outputs/eval/pi0_drop_policy \
  --fault.enabled=true \
  --fault.type=midair_drop \
  --fault.t_max=400 \
  --fault.drop_xy_band_min=0.34 \
  --fault.drop_xy_band_max=0.38 \
  --fault.post_drop_mode=immediate_policy \
  --fault.object_name=alphabet_soup_1 \
  --fault.basket_name=basket_1
```

Pi0 and SmolVLA both keep an action chunk (often 50 steps). This eval does not clear that chunk when the can drops. Actions already queued are still executed after the landing, and the next chunk is the first one planned from the dropped scene.

### 6. Results

```text
outputs/eval/pi0_drop_policy/eval_info.json
outputs/eval/pi0_drop_policy/videos/
outputs/eval/pi0_drop_policy/fault_events.jsonl
```

`eval_info.json` → `overall.pc_success` is the LIBERO task success rate. `fault_events.jsonl` should contain one `"status": "triggered"` line per drop, with `"drop_trigger_reason": "xy_band"` and `"post_drop_mode": "immediate_policy"`. It should not contain `"status": "recovery_started"` — that line means the IK planner took over. An empty log means the can was never grasped inside the XY band before step 400.

## How a fault eval runs

`lerobot-eval-faults` parses its own config, builds the sim, then calls stock `eval_policy_all`. Stock `lerobot-eval` has no `--fault.*` and does not record through this wrapper.

```mermaid
flowchart LR
  CLI["lerobot-eval-faults"] --> ENV["make_env"]
  ENV --> HEAD["headless LIBERO"]
  HEAD --> WRAP["fault wrappers"]
  WRAP --> REC["recording wrapper"]
  REC --> EVAL["eval_policy_all"]
  EVAL --> DS["LeRobot dataset"]

  classDef ours fill:#0f6e56,stroke:#084c3e,color:#fff
  classDef pin fill:#1f4e79,stroke:#0b2545,color:#fff
  classDef out fill:#8a4b08,stroke:#5c3204,color:#fff
  class CLI,WRAP,REC ours
  class ENV,HEAD,EVAL pin
  class DS out
```

1. **Config.** `FaultEvalPipelineConfig` adds `fault` and `eval.recording_success_only`. Those fields stay off upstream `EvalPipelineConfig`.
2. **Sim.** LIBERO and overlay envs get `install_headless_libero_renderer()` before the first reset. Scene YAML is `libero_overlay`, applied inside `OverlayLiberoEnv` after the stock env is constructed.
3. **Faults.** `maybe_wrap_env_tree` picks a wrapper from `factory.py`: action (`hold`, `delay`, `jitter`), observation (dropout, blur, occlusion, latency), sim inject (slip, bump), or `midair_drop`. One type per run. Fault logs require `--env.max_parallel_tasks=1` (that is already the LIBERO default).
4. **Recording.** If `--eval.recording=true`, `EvalRecordingVecWrapper` stores the pre-step observation and the action, maps keys with `features_map`, and finishes the episode on done or max steps. `recording_success_only` drops failed buffers. Stock rollout is called with `recording_dir=None`, so it does not write a second dataset.
5. **Episode fields.** `success` is merged by wrapping `dataset.meta.save_episode` before the parquet flush. The dataset class itself is unchanged.

On a mid-air drop the wrapper sits on that same path:

1. The policy outputs an action. The simulator steps.
2. When the can is grasped, high enough, inside the time window, and inside the XY band, the wrapper opens the gripper. The can falls. During the fall the wrapper holds the arm and ignores the policy action.
3. After the can lands, `immediate_policy` hands control back to the policy. The IK planner is not started, and the seat-into-basket assist does not run.
4. The policy's later actions are what try to pick the can up again. The eval writes `eval_info.json`, a video, and `fault_events.jsonl`.

`immediate_ik` (the default if you omit `--fault.post_drop_mode`) is the other choice: after landing, the scripted planner takes over and `fault_events.jsonl` records `"status": "recovery_started"`. Use that only when you want to measure the planner, not the checkpoint.

## Layout

```text
src/fault_system/          faults, wrappers, eval CLI, recording, datagen
  action/  observation/  sim/  recovery/  datagen/  models/
  README.md              model ↔ fault wrapper ↔ LIBERO, with inputs and outputs
packages/lerobot_env_libero_overlay/   LIBERO scene plugin
tests/faults/              unit tests (mocked sim)
scripts/run_fault_smoke.sh one-episode SmolVLA smoke
```

Datagen (`src/fault_system/datagen/`) is a separate path: a JSON recipe drives layout, drop timing, and either Simple IK or SmolVLA, then `dataset_writer` saves episodes. It uses stock `LiberoEnv` and the same headless install. It does not go through `lerobot-eval-faults`.

## Commands

Smoke (one episode, cached `lerobot/smolvla_libero`, CUDA). The script sets `MUJOCO_GL` if you have not, and it sets `HF_HUB_OFFLINE=1`. Unset that pair if the Hub checkpoint is not cached yet.

```bash
MUJOCO_GL=egl bash scripts/run_fault_smoke.sh baseline
MUJOCO_GL=egl bash scripts/run_fault_smoke.sh injected
```

`baseline` writes `outputs/eval/fault_smoke_baseline/eval_info.json` and no `fault_events.jsonl`. `injected` is an action hold and writes `outputs/eval/fault_smoke_injected/fault_events.jsonl`. This script does not run a mid-air drop.

Record one episode into a dataset. Hub SmolVLA names its cameras `camera1` / `camera2` and expects a padded third camera. LIBERO's cameras are `agentview_image` and `robot0_eye_in_hand_image`.

```bash
uv run lerobot-eval-faults \
  --policy.path=lerobot/smolvla_libero \
  --policy.empty_cameras=1 \
  --env.type=libero --env.task=libero_object --env.task_ids='[0]' \
  --eval.n_episodes=1 --eval.batch_size=1 --eval.use_async_envs=false \
  --eval.recording=true \
  --policy.device=cuda --output_dir=outputs/eval/record_once \
  --env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
```

Mid-air drop, policy keeps control (the [Quick Start](#5-drop-and-recovery-eval) command). Verified with `lerobot/smolvla_libero` for one episode: the log had `"status": "triggered"`, `"drop_trigger_reason": "xy_band"`, `"post_drop_mode": "immediate_policy"`, and no `"recovery_started"`. The same run needs the SmolVLA camera flags above; a Pi0 checkpoint on `image` / `image2` does not. Task success on stock SmolVLA was 0/1 — that checkpoint was not trained to recover. A Pi0 run is still required to measure her checkpoint.

```bash
uv run lerobot-eval-faults \
  --policy.path="$CHECKPOINT" \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --output_dir=outputs/eval/pi0_drop_policy \
  --fault.enabled=true \
  --fault.type=midair_drop \
  --fault.t_max=400 \
  --fault.drop_xy_band_min=0.34 \
  --fault.drop_xy_band_max=0.38 \
  --fault.post_drop_mode=immediate_policy \
  --fault.object_name=alphabet_soup_1 \
  --fault.basket_name=basket_1
```

Other `libero_object` tasks use other body names (`cream_cheese_1` is task 1, and so on). `basket_1` is the basket on these scenes. Keep `task_ids` matched to `object_name`.

Overlay scenes, same SmolVLA camera flags:

```bash
uv run lerobot-eval-overlay \
  --policy.path=lerobot/smolvla_libero \
  --policy.empty_cameras=1 \
  --env.type=libero_overlay \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --env.overlay=examples/libero_overlays/add_red_cube.yaml \
  --eval.n_episodes=1 --eval.batch_size=1 --eval.use_async_envs=false \
  --policy.device=cuda \
  --env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
```

More overlay examples: `examples/libero_overlays/README.md`.

Unit tests: `uv run pytest tests/faults tests/envs/test_libero_overlays.py tests/scripts/test_eval_recording_features.py -q`
