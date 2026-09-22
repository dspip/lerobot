# Run a checkpoint in LIBERO with fault injection

Linux + NVIDIA GPU. Run every command from the repository root.

## Setup (once)

```bash
uv sync --locked --extra libero --extra smolvla
export MUJOCO_GL=egl
```

CMake 4.x build error on `egl-probe`: prefix the same `uv sync` with `CMAKE_POLICY_VERSION_MINIMUM=3.5`.

```bash
uv run python -c "import torch, libero; print('CUDA', torch.cuda.is_available())"
```

## Set the checkpoint

Hub policy (copy-paste as-is; this is what the mix recorder uses):

```bash
export CHECKPOINT=lerobot/smolvla_libero
```

A local **standard** LeRobot folder (`config.json` next to `model.safetensors`):

```bash
export CHECKPOINT=/absolute/path/to/pretrained_model
```

An **extended two-pass failure-head** tree (`policy/`, `processors/`, `auxiliary/`) cannot be passed to `lerobot-eval`. Use [Extended failure-head checkpoint](#extended-failure-head-checkpoint).

Keep this shell open so `$CHECKPOINT` and `MUJOCO_GL` stay set.

## Copy-paste eval

These flags are required for `lerobot/smolvla_libero`: LIBERO cameras are `image` / `image2`, that Hub policy expects `camera1` / `camera2` plus a padded camera. One episode, one env, so the first run is a smoke test (library defaults are 50 episodes and a large auto batch).

### Baseline (no fault)

```bash
uv run lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --policy.empty_cameras=1 \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
```

Keep the single quotes around the camera mapping. Task `0` is Alphabet Soup.

### Action hold

```bash
uv run lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --policy.empty_cameras=1 \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --fault.enabled=true \
  --fault.type=action_hold
```

Hold starts at step **55** for **8** steps. Other types (one per run): `action_delay`, `action_jitter`, `sensor_dropout`, `visual_occlusion`, `visual_blur`, `brightness_drop`, `obs_latency`.

### Mid-air drop + IK recovery

```bash
uv run lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --policy.empty_cameras=1 \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --fault.enabled=true \
  --fault.type=midair_drop \
  --fault.t_max=400
```

`--fault.t_max=400` is required so a late grasp can still drop (library window is 10–30). Object `alphabet_soup_1`, basket `basket_1`.

A real inject writes `"event": "midair_drop"` in `fault_events.jsonl`. Task success is LIBERO success in `eval_info.json`, not “the drop fired.”

If `config.json` lists `observation.images.image` (not `camera1`), remove `--policy.empty_cameras=1` and the `--env.camera_name_mapping=...` line only.

## Optional flags

| What you want | Flag | Otherwise |
| --- | --- | --- |
| More episodes | `--eval.n_episodes=10` | smoke uses `1` |
| Fixed output folder | `--output_dir=outputs/eval/my_run` | `outputs/eval/<timestamp>_<job>` |
| Reproducible seed | `--seed=1000` | `1000` |
| Carry ~1 s before drop | `--fault.post_grasp_delay_steps=20` | drop as soon as lifted (`0`) |
| No seat-into-basket assist | `--fault.seat_assist_enabled=false` | `true` |
| Action-hold timing | `--fault.trigger_step=20 --fault.duration=8` | `55` / `8` |
| Delay / jitter | `--fault.delay_steps=3` / `--fault.noise_std=0.05` | `3` / `0.05` |

## Where results go

```text
<output_dir>/eval_info.json
<output_dir>/videos/
<output_dir>/fault_events.jsonl    # when --fault.enabled=true
```

First Hub download: `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE`. After cache: `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Extended failure-head checkpoint

Layout:

```text
CHECKPOINT/policy/{config.json,model.safetensors}
CHECKPOINT/processors/
CHECKPOINT/auxiliary/weights.pt
```

Needs `examples/faults/run_head_recovery_rollout.py` and `third_party/Gangelia_Project/smolvla_r_package/head_vlm_conditioning.py` (not used by generic `lerobot-eval`), plus `/tmp/xy_band_mix_60ep_extract/dataset/data/chunk-000/` for the scripted episode-12 setup.

```bash
export CHECKPOINT=/absolute/path/to/extended_checkpoint
export MUJOCO_GL=egl
uv run python examples/faults/run_head_recovery_rollout.py \
  --checkpoint "$CHECKPOINT"
```

Defaults: `--threshold 0.5`, `--steps 120`, output `reports/xy60_verify/head_recovery_rollout`.

## Troubleshooting

- **Missing `camera1` / `camera2` / `camera3`** — you omitted the Hub camera flags. Put them back.
- **Missing `image` / `image2`** — this checkpoint is not Hub SmolVLA-LIBERO. Drop `empty_cameras` and `camera_name_mapping`.
- **Empty `fault_events.jsonl` on mid-air drop** — no grasp+lift before `t_max`. Watch the video.
- **EGL / CUDA** — `nvidia-smi`, keep `MUJOCO_GL=egl`.
