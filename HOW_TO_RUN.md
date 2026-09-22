# Run a checkpoint in LIBERO with fault injection

Linux + NVIDIA GPU. Run every command from the repository root.

## Setup

```bash
uv sync --locked --extra libero --extra smolvla
export MUJOCO_GL=egl
```

CMake 4.x build error on `egl-probe`: prefix the same `uv sync` with `CMAKE_POLICY_VERSION_MINIMUM=3.5`.

```bash
uv run python -c "import torch, libero; print('CUDA', torch.cuda.is_available())"
```

## Checkpoint path

**Standard LeRobot** (`config.json` and `model.safetensors` in the same folder):

```bash
export CHECKPOINT=/absolute/path/to/pretrained_model
test -f "$CHECKPOINT/config.json" && test -f "$CHECKPOINT/model.safetensors"
```

**Extended two-pass failure-head** (`policy/`, `processors/`, `auxiliary/`): do not pass the top-level folder to `lerobot-eval`. See [Extended failure-head checkpoint](#extended-failure-head-checkpoint).

## Copy-paste commands

Change **only** `--policy.path` and, if needed, `--env.task` / `--env.task_ids`. Everything else below uses library defaults.

Device is selected automatically (CUDA when available). `fault.enabled` defaults to `false`. Camera keys default to `image` / `image2`. Control mode defaults to `relative`. Parallel tasks default to `1` (required for faults).

### Baseline (no fault)

```bash
uv run lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]"
```

`libero_object` task `0` is Alphabet Soup. Omit `--env.task_ids` to run the whole suite. Default episode count is **50**.

### Action hold

```bash
uv run lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --fault.enabled=true \
  --fault.type=action_hold
```

Defaults: hold starts at step **55**, lasts **8** steps, probability **1.0**. Event log: `<output_dir>/fault_events.jsonl`.

Other types (one per run): `action_delay`, `action_jitter`, `sensor_dropout`, `visual_occlusion`, `visual_blur`, `brightness_drop`, `obs_latency`.

### Mid-air drop + IK recovery

```bash
uv run lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --fault.enabled=true \
  --fault.type=midair_drop \
  --fault.t_max=400
```

`--fault.t_max=400` is the one non-default on purpose. The library default window is steps **10–30**, which is usually before the arm has grasped. Object `alphabet_soup_1`, basket `basket_1`, grasp required, `min_object_z=0.12`.

A successful inject logs `"event": "midair_drop"` in `fault_events.jsonl`. Task success is still LIBERO success in `eval_info.json`, not “the drop fired.”

## Optional flags

| What you want | Flag | Library default |
| --- | --- | --- |
| One smoke episode | `--eval.n_episodes=1` | `50` |
| Fixed output folder | `--output_dir=outputs/eval/my_run` | `outputs/eval/<timestamp>_<job>` |
| Reproducible seed | `--seed=1000` | `1000` |
| Force GPU / CPU | `--policy.device=cuda` | auto |
| Carry ~1 s before drop | `--fault.post_grasp_delay_steps=20` | `0` (drop as soon as lifted) |
| No seat-into-basket assist | `--fault.seat_assist_enabled=false` | `true` |
| Different object | `--fault.object_name=...` | `alphabet_soup_1` |
| Action-hold timing | `--fault.trigger_step=20 --fault.duration=8` | `55` / `8` |
| Delay / jitter | `--fault.delay_steps=3` / `--fault.noise_std=0.05` | `3` / `0.05` |

### Hub `lerobot/smolvla_libero` only

That Hub checkpoint uses `camera1` / `camera2` plus a padded camera. Project checkpoints that list `observation.images.image` in `config.json` do **not** need this:

```bash
  --policy.empty_cameras=1 \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
```

## Where results go

```text
<output_dir>/eval_info.json
<output_dir>/videos/
<output_dir>/fault_events.jsonl    # created when --fault.enabled=true
```

First run: `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE`. After assets are cached: `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Extended failure-head checkpoint

Layout:

```text
CHECKPOINT/policy/{config.json,model.safetensors}
CHECKPOINT/processors/
CHECKPOINT/auxiliary/weights.pt
```

Generic `lerobot-eval --policy.path` cannot load that top-level directory. The two-pass runner is not on `main`; you need a checkout that has:

```text
examples/faults/run_head_recovery_rollout.py
third_party/Gangelia_Project/smolvla_r_package/head_vlm_conditioning.py
```

and the XY60 extract at `/tmp/xy_band_mix_60ep_extract/dataset/data/chunk-000/` (used to replay nominal episode 12 before the scripted release).

```bash
export CHECKPOINT=/absolute/path/to/extended_checkpoint
export MUJOCO_GL=egl
uv run python examples/faults/run_head_recovery_rollout.py \
  --checkpoint "$CHECKPOINT"
```

Defaults: `--threshold 0.5`, `--steps 120`, output `reports/xy60_verify/head_recovery_rollout`. Recovery starts only when the auxiliary head’s drop probability crosses the threshold.

## Troubleshooting

- **`config.json` not found** — `--policy.path` must be the folder that contains it, not the parent of an extended checkpoint.
- **Empty `fault_events.jsonl` on mid-air drop** — the policy never grasped and lifted before `t_max`. Watch the video; widen `--fault.t_max` or confirm task `0`.
- **Camera / feature mismatch** — Hub SmolVLA-LIBERO needs the optional mapping above; this project’s fine-tunes usually do not.
- **EGL / CUDA** — `nvidia-smi`, keep `MUJOCO_GL=egl`, try `--eval.n_episodes=1`.
