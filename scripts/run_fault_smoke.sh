#!/usr/bin/env bash
# A/B smoke using stock lerobot-eval + in-tree fault injection.
# Requires: uv sync, LeRobot + LIBERO extras, CUDA for SmolVLA.
# Does NOT cover midair_drop — use examples/faults/run_full_drop_recovery_pipeline.py.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-baseline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

LEROBOT_EVAL=(uv run lerobot-eval)

COMMON=(
  --policy.path=lerobot/smolvla_libero
  --env.type=libero
  --env.task=libero_object
  --env.task_ids="[0]"
  --env.control_mode=relative
  --env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
  --policy.empty_cameras=1
  --eval.batch_size=1
  --eval.n_episodes=1
  --eval.use_async_envs=false
  --env.max_parallel_tasks=1
  --seed=1000
  --policy.device=cuda
)

case "$MODE" in
  baseline)
    "${LEROBOT_EVAL[@]}" \
      --fault.enabled=false \
      "${COMMON[@]}" \
      --output_dir=outputs/eval/fault_smoke_baseline \
      --job_name=fault_smoke_baseline
    ;;
  injected)
    "${LEROBOT_EVAL[@]}" \
      --fault.enabled=true \
      --fault.type=action_hold \
      --fault.trigger_step=20 \
      --fault.duration=8 \
      --fault.probability=1.0 \
      --fault.seed=42 \
      --fault.log_path=outputs/eval/fault_smoke_injected/fault_events.jsonl \
      "${COMMON[@]}" \
      --output_dir=outputs/eval/fault_smoke_injected \
      --job_name=fault_smoke_injected
    ;;
  delay)
    "${LEROBOT_EVAL[@]}" \
      --fault.enabled=true \
      --fault.type=action_delay \
      --fault.delay_steps=3 \
      --fault.log_path=outputs/eval/fault_smoke_delay/fault_events.jsonl \
      "${COMMON[@]}" \
      --output_dir=outputs/eval/fault_smoke_delay \
      --job_name=fault_smoke_delay
    ;;
  jitter)
    "${LEROBOT_EVAL[@]}" \
      --fault.enabled=true \
      --fault.type=action_jitter \
      --fault.noise_std=0.35 \
      --fault.seed=42 \
      --fault.log_path=outputs/eval/fault_smoke_jitter/fault_events.jsonl \
      "${COMMON[@]}" \
      --output_dir=outputs/eval/fault_smoke_jitter \
      --job_name=fault_smoke_jitter
    ;;
  sensor)
    "${LEROBOT_EVAL[@]}" \
      --fault.enabled=true \
      --fault.type=sensor_dropout \
      --fault.trigger_step=20 \
      --fault.duration=40 \
      --fault.log_path=outputs/eval/fault_smoke_sensor/fault_events.jsonl \
      --fault.diag_dir=outputs/eval/fault_smoke_sensor/diag \
      "${COMMON[@]}" \
      --output_dir=outputs/eval/fault_smoke_sensor \
      --job_name=fault_smoke_sensor
    ;;
  *)
    echo "Usage: $0 {baseline|injected|delay|jitter|sensor}" >&2
    echo "Note: midair_drop is not a smoke mode. Use examples/faults/ pipeline scripts." >&2
    exit 1
    ;;
esac
