# Run a checkpoint in LIBERO with fault injection

Linux + NVIDIA GPU. Run every command from the repository root.

## Setup (once)

```bash
uv sync --locked
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

Run baseline first. If that episode never grasps, the drop runs will not inject either.

`scripts/run_fault_smoke.sh` covers baseline and action faults only. Mid-air drop is the commands below.

Every command below shares the same scene: `libero_object` task `0` (Alphabet Soup, `alphabet_soup_1`, basket `basket_1`), one episode. Keep the single quotes around the camera mapping. The Hub checkpoint wants `camera1` / `camera2` and a padded `camera3`; LIBERO’s two cameras are named `agentview_image` and `robot0_eye_in_hand_image`.

Leave these off. They are already the default, or they follow from `--eval.n_episodes=1`:

| You might add | Why it is already set |
| --- | --- |
| `--policy.device=cuda` | checkpoint `config.json` sets `cuda` |
| `--seed=1000` | eval default |
| `--env.control_mode=relative` | LIBERO default |
| `--env.max_parallel_tasks=1` | LIBERO default; faults reject a higher value |
| `--eval.batch_size=1` | one episode forces batch size 1 |
| `--eval.use_async_envs=false` | one env stays synchronous |
| `--fault.object_name` / `--fault.basket_name` | `alphabet_soup_1` / `basket_1` |
| `--fault.probability=1` / `--fault.seed=42` / `--fault.t_min=10` | fault defaults |
| `--fault.post_drop_dwell_steps=0` | fault default. A value above 0 waits, then starts IK even in `immediate_policy` |
| `--fault.post_drop_mode=immediate_ik` | this is the drop default: IK recovery after landing |

If `config.json` lists `observation.images.image` (not `camera1`), remove `--policy.empty_cameras=1` and the camera-mapping line only.

### Baseline (no fault)

```bash
uv run lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --policy.empty_cameras=1 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --eval.n_episodes=1 \
  --output_dir=outputs/eval/smolvla_baseline \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
```

Pass: `outputs/eval/smolvla_baseline/eval_info.json` exists, and that folder has no `fault_events.jsonl`. `overall.pc_success` is the LIBERO task result (0 or 100 for one episode).

### Action hold

```bash
uv run lerobot-eval-faults \
  --policy.path="$CHECKPOINT" \
  --policy.empty_cameras=1 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --eval.n_episodes=1 \
  --output_dir=outputs/eval/smolvla_hold \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --fault.enabled=true \
  --fault.type=action_hold
```

Hold starts at step **55** for **8** steps. Other types (one per run, same command, change `--fault.type`): `action_delay`, `action_jitter`, `sensor_dropout`, `visual_occlusion`, `visual_blur`, `brightness_drop`, `obs_latency`.

### Mid-carry drop, no recovery

The gripper opens when the can is **0.34–0.38 m** from the basket (the recipe’s `mid` band). After it lands, SmolVLA keeps control. IK does not start.

`--fault.t_max=400` is required. The default window ends at step 30, which is before most grasps.

```bash
uv run lerobot-eval-faults \
  --policy.path="$CHECKPOINT" \
  --policy.empty_cameras=1 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --eval.n_episodes=1 \
  --output_dir=outputs/eval/smolvla_drop_norecovery \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --fault.enabled=true \
  --fault.type=midair_drop \
  --fault.t_max=400 \
  --fault.drop_xy_band_min=0.34 \
  --fault.drop_xy_band_max=0.38 \
  --fault.post_drop_mode=immediate_policy
```

Pass: `fault_events.jsonl` has `"status": "triggered"`, `"drop_trigger_reason": "xy_band"`, `"post_drop_mode": "immediate_policy"`, and no `"status": "recovery_started"`.

### Mid-carry drop, with IK recovery

Same drop. Omit `post_drop_mode`: the default `immediate_ik` starts the planner after the can lands. Seat-into-basket assist stays on.

```bash
uv run lerobot-eval-faults \
  --policy.path="$CHECKPOINT" \
  --policy.empty_cameras=1 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --eval.n_episodes=1 \
  --output_dir=outputs/eval/smolvla_drop_recovery \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --fault.enabled=true \
  --fault.type=midair_drop \
  --fault.t_max=400 \
  --fault.drop_xy_band_min=0.34 \
  --fault.drop_xy_band_max=0.38
```

Pass: the log has both `"status": "triggered"` and `"status": "recovery_started"`. `pc_success` on this run can come from the IK planner. On the no-recovery run it can come only from SmolVLA. Compare the two videos, not just the two success rates.

### Manual keyboard drop + recovery

This runs the no-checkpoint manual demo. It replays nominal episode 12, then lets you press `d` to drop the object and `r` to start IK recovery. The demo needs the `xy_band_mix_60ep` archive because it reads the recorded actions and nominal-episode audit report.

Run this from the repository root. The first block finds the archive, including its current location in the desktop Trash, extracts it into the path expected by the script, and verifies both required inputs:

```bash
cd ~/Projects/lerobot

ARCHIVE="$HOME/.local/share/Trash/files/xy_band_mix_60ep.tar.gz"
if [ ! -f "$ARCHIVE" ]; then
  ARCHIVE="$(find "$HOME" -type f -name 'xy_band_mix_60ep.tar.gz' -print -quit)"
fi
if [ -z "$ARCHIVE" ]; then
  echo "Cannot find xy_band_mix_60ep.tar.gz under $HOME" >&2
  exit 1
fi

rm -rf /tmp/xy_band_mix_60ep_extract
mkdir -p /tmp/xy_band_mix_60ep_extract
tar -xzf "$ARCHIVE" -C /tmp/xy_band_mix_60ep_extract

test -f /tmp/xy_band_mix_60ep_extract/audit_report.json
test -d /tmp/xy_band_mix_60ep_extract/dataset/data/chunk-000
if [ ! -f /tmp/xy_band_mix_60ep_extract/audit_report.json ] || \
   [ ! -d /tmp/xy_band_mix_60ep_extract/dataset/data/chunk-000 ]; then
  echo "Dataset extraction is incomplete" >&2
  exit 1
fi
```

Start the UI:

```bash
uv run python examples/faults/run_manual_drop_recovery.py
```

Click the Tk window to focus it. Press `d` while the object is grasped, then `r` to recover. Press `q` or `Esc` to quit. The Robosuite private-macro warning is harmless for this demo.

## Optional flags

| What you want | Flag | Otherwise |
| --- | --- | --- |
| Drop as soon as the can is lifted, anywhere ≥ 0.18 m from the basket | omit both `--fault.drop_xy_band_*` flags | the commands above use the mid band 0.34–0.38 m |
| Other carry bands | lift `0.48–0.52`, early `0.42–0.46`, late `0.30–0.33` | mid `0.34–0.38` |
| More episodes | `--eval.n_episodes=10` | `1` |
| Whole `libero_object` suite | omit `--env.task_ids` | task `0` only. Also set `--fault.object_name` to that task’s object |
| Another suite | `--env.task=libero_spatial` (or `libero_goal`, `libero_10`) | `libero_object`. Default if you omit `--env.task` is `libero_10` |
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
- **Empty `fault_events.jsonl` on mid-air drop** — the can was never grasped inside the XY band before step 400. Watch the baseline video. Other bands are in the optional-flags table.
- **`recovery_started` on the no-recovery run** — `post_drop_mode` is not `immediate_policy`, or `post_drop_dwell_steps` is above 0.
- **No `recovery_started` on the recovery run** — look for `"status": "fall_aborted"`. The can never settled, so IK does not start.
- **EGL / CUDA** — `nvidia-smi`, keep `MUJOCO_GL=egl`.
