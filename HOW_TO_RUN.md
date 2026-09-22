# Run a checkpoint in LIBERO with fault injection

This guide runs a LeRobot policy checkpoint in the LIBERO simulator, first
without a fault and then with the in-tree fault-injection system. It is written
for Linux, an NVIDIA GPU, the `libero_object` suite, and a SmolVLA-compatible
checkpoint.

Run every command from the repository root.

## 1. Install the environment

Requirements:

- Python 3.12 or newer
- NVIDIA driver and a CUDA-capable PyTorch installation
- `uv`
- `ffmpeg` for rollout videos

```bash
cd /path/to/lerobot
uv sync --locked --extra libero --extra smolvla
```

If `egl-probe` fails to build with CMake 4.x, retry with:

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  uv sync --locked --extra libero --extra smolvla
```

Check the important runtime dependencies:

```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
uv run python -c "import libero; print('LIBERO import: OK')"
ffmpeg -version
```

For headless simulation:

```bash
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

On the first run, allow Hugging Face downloads so any base-model assets that
are not packaged in the checkpoint can be cached:

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
```

## 2. Identify the checkpoint format

There are two supported workflows. Do not pass an extended checkpoint's
top-level directory to `lerobot-eval`.

### A. Standard LeRobot checkpoint

The directory passed to `--policy.path` contains:

```text
CHECKPOINT/
├── config.json
└── model.safetensors
```

Set and validate the path:

```bash
export CHECKPOINT=/absolute/path/to/pretrained_model
test -f "$CHECKPOINT/config.json"
test -f "$CHECKPOINT/model.safetensors"
```

Continue with sections 3–7.

### B. Extended two-pass failure-head checkpoint

This format keeps the policy, processors, and auxiliary failure head in
separate directories:

```text
CHECKPOINT/
├── policy/
│   ├── config.json
│   └── model.safetensors
├── processors/
└── auxiliary/
    └── weights.pt
```

Set and validate the path:

```bash
export CHECKPOINT=/absolute/path/to/extended_checkpoint
test -f "$CHECKPOINT/policy/config.json"
test -f "$CHECKPOINT/policy/model.safetensors"
test -f "$CHECKPOINT/auxiliary/weights.pt"
test -f "$CHECKPOINT/processors/policy_preprocessor.json"
```

This top-level directory is not a standard `Policy.save_pretrained` directory.
For the full two-pass policy and failure head, use section 8. The generic
commands below are for standard checkpoints.

## 3. Run a no-fault baseline

Always run the baseline first. It proves that the checkpoint, processors,
camera mapping, task, and simulator work before a fault is introduced.

```bash
uv run lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --policy.device=cuda \
  --policy.empty_cameras=1 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --env.control_mode=relative \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --seed=1000 \
  --fault.enabled=false \
  --output_dir=outputs/eval/checkpoint_baseline \
  --job_name=checkpoint_baseline
```

Use the same task, seed, episode count, and checkpoint for the faulted run.
Only the `--fault.*`, output, and job-name arguments should change.

The camera mapping and `--policy.empty_cameras=1` match the repository's
SmolVLA LIBERO setup. For another policy type, use the camera and policy
options from that checkpoint's training configuration.

## 4. Run an action fault

This example repeats the previous action for eight simulator steps beginning at
step 20:

```bash
uv run lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --policy.device=cuda \
  --policy.empty_cameras=1 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --env.control_mode=relative \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --seed=1000 \
  --fault.enabled=true \
  --fault.type=action_hold \
  --fault.trigger_step=20 \
  --fault.duration=8 \
  --fault.probability=1.0 \
  --fault.seed=42 \
  --fault.log_path=fault_events.jsonl \
  --output_dir=outputs/eval/checkpoint_action_hold \
  --job_name=checkpoint_action_hold
```

Other simple fault types include:

- `action_delay` with `--fault.delay_steps=3`
- `action_jitter` with `--fault.noise_std=0.05`
- `sensor_dropout` with `--fault.trigger_step=20 --fault.duration=8`
- `visual_occlusion`, `visual_blur`, `brightness_drop`, and `obs_latency`

Use one fault type per run.

## 5. Run mid-air drop with IK recovery

This fault waits until `alphabet_soup_1` is grasped and lifted, carries it for
20 control steps, applies a drop impulse, and switches control to the in-tree IK
recovery planner.

```bash
uv run lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --policy.device=cuda \
  --policy.empty_cameras=1 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --env.control_mode=relative \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --seed=1000 \
  --fault.enabled=true \
  --fault.type=midair_drop \
  --fault.object_name=alphabet_soup_1 \
  --fault.basket_name=basket_1 \
  --fault.t_min=0 \
  --fault.t_max=400 \
  --fault.require_grasp=true \
  --fault.min_object_z=0.12 \
  --fault.post_grasp_delay_steps=20 \
  --fault.probability=1.0 \
  --fault.seed=42 \
  --fault.seat_assist_enabled=false \
  --fault.log_path=fault_events.jsonl \
  --output_dir=outputs/eval/checkpoint_midair_drop \
  --job_name=checkpoint_midair_drop
```

Important:

- `--env.max_parallel_tasks=1` is required when faults are enabled.
- The object and basket names are specific to `libero_object` task 0.
- `seat_assist_enabled=false` keeps recovery unaided; set it to `true` only
  for a diagnostic demo.
- If the policy never grasps and lifts the soup before step 400, the drop
  correctly does not fire.

## 6. Inspect the result

For each standard evaluation, inspect:

```bash
cat outputs/eval/checkpoint_midair_drop/eval_info.json
ls outputs/eval/checkpoint_midair_drop/videos
cat outputs/eval/checkpoint_midair_drop/fault_events.jsonl
```

The log path is relative to `--output_dir`, so the command above writes:

```text
outputs/eval/checkpoint_midair_drop/fault_events.jsonl
```

A mid-air drop run should contain an event with `"event": "midair_drop"`.
The terminal and `eval_info.json` report task success. A detected drop or a
successful regrasp alone is not task success; the LIBERO task must finish.

For a useful comparison, run several seeds and report baseline and faulted
success separately. Start with one episode to verify the setup before running a
larger batch.

## 7. Reproduce a run offline

After all model assets have been cached locally:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Keep these values fixed between compared runs:

- checkpoint
- `--seed`
- `--fault.seed`
- task and task id
- episode count
- all fault parameters

Unset offline mode if an error says a model or tokenizer is missing:

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
```

## 8. Run an extended two-pass failure-head checkpoint

The full extended checkpoint needs the specialized two-pass loader and rollout
script. These experimental companion files are not part of `main`; obtain the
feature checkout or an archive that contains:

```text
examples/faults/run_head_recovery_rollout.py
third_party/Gangelia_Project/smolvla_r_package/head_vlm_conditioning.py
third_party/Gangelia_Project/smolvla_r_package/train_phase1.py
```

The current specialized rollout also uses nominal episode 12 to prepare a
repeatable grasp before releasing the object. Extract the supplied XY60 dataset
so this exact path exists:

```text
/tmp/xy_band_mix_60ep_extract/dataset/data/chunk-000/
```

For an archive whose top-level directory is `dataset/`:

```bash
rm -rf /tmp/xy_band_mix_60ep_extract
mkdir -p /tmp/xy_band_mix_60ep_extract
tar -xzf /path/to/xy_band_mix_60ep.tar.gz \
  -C /tmp/xy_band_mix_60ep_extract
```

Preflight:

```bash
test -f examples/faults/run_head_recovery_rollout.py
test -f third_party/Gangelia_Project/smolvla_r_package/head_vlm_conditioning.py
test -f /tmp/xy_band_mix_60ep_extract/dataset/data/chunk-000/file-000.parquet
test -f "$CHECKPOINT/auxiliary/weights.pt"
```

Run:

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_head_recovery_rollout.py \
  --checkpoint "$CHECKPOINT" \
  --threshold 0.5 \
  --steps 120 \
  --output outputs/eval/failure_head_recovery
```

Inspect:

```bash
cat outputs/eval/failure_head_recovery/summary.json
cat outputs/eval/failure_head_recovery/fault_events.jsonl
ls outputs/eval/failure_head_recovery/rollout.mp4
```

In this experiment, the scripted release prepares the failure but does not
start recovery. Recovery starts only when the auxiliary head's drop
probability crosses `--threshold`.

This loader reconstructs the two-pass architecture from the checkpoint format.
Confirm it matches the code used to train the checkpoint before treating the
result as a model benchmark.

## Troubleshooting

### `config.json` not found

`--policy.path` is one directory too high or too low. For a standard checkpoint,
pass the directory containing both `config.json` and `model.safetensors`. Do not
pass the top of an extended checkpoint to generic `lerobot-eval`.

### CUDA is unavailable or the process runs out of memory

Confirm `torch.cuda.is_available()` is `True`, close other GPU jobs, keep
`--eval.batch_size=1`, and test one episode first.

### EGL or MuJoCo cannot create a context

Confirm the NVIDIA driver is visible with `nvidia-smi` and keep
`MUJOCO_GL=egl`. On a desktop with a working display, `MUJOCO_GL=glfw` can be
used for an interactive window.

### The fault log is empty

- Confirm `--fault.enabled=true`.
- For `midair_drop`, confirm the policy actually grasps and lifts
  `alphabet_soup_1` before `--fault.t_max`.
- Confirm the selected task uses the configured object and basket names.
- Check that `--fault.probability=1.0` during setup testing.

### The checkpoint works without faults but fails immediately with a fault

First compare the exact same seed and task. Then reduce the fault severity:
shorten `action_hold`, lower `noise_std`, or delay the mid-air drop longer.
Do not change the checkpoint, task, and fault parameters at the same time.
