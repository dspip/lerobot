# Fault injection

Faults, recovery datagen, and eval recording on a **pinned** [LeRobot](https://github.com/huggingface/lerobot) commit. This repo does not contain the LeRobot library.

| | |
|---|---|
| Package | `fault_system` (`src/fault_system`) |
| Eval | `lerobot-eval-faults` |
| Scene edits | `lerobot_env_libero_overlay` → `--env.type=libero_overlay` |
| Pin | `huggingface/lerobot` @ `e0d50211` |

The overlay distribution keeps the `lerobot_env_` prefix. Upstream plugin discovery only auto-imports names that start with `lerobot_env_`, `lerobot_policy_`, `lerobot_robot_`, `lerobot_camera_`, `lerobot_teleoperator_`, or `lerobot_strategy_`.

```bash
uv sync --locked
```

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
3. **Faults.** `maybe_wrap_env_tree` picks a wrapper from `factory.py`: action (`hold`, `delay`, `jitter`), observation (dropout, blur, occlusion, latency), sim inject (slip, bump), or recovery (`midair_drop` + IK planner). One type per run. Fault logs require `--env.max_parallel_tasks=1`.
4. **Recording.** If `--eval.recording=true`, `EvalRecordingVecWrapper` stores the pre-step observation and the action, maps keys with `features_map`, and finishes the episode on done or max steps. `recording_success_only` drops failed buffers. Stock rollout is called with `recording_dir=None`, so it does not write a second dataset.
5. **Episode fields.** `success` is merged by wrapping `dataset.meta.save_episode` before the parquet flush. The dataset class itself is unchanged.

## Layout

```text
src/fault_system/          faults, wrappers, eval CLI, recording, datagen
  action/  observation/  sim/  recovery/  datagen/
packages/lerobot_env_libero_overlay/   LIBERO scene plugin
tests/faults/              unit tests (mocked sim)
scripts/run_fault_smoke.sh one-episode SmolVLA smoke
```

Datagen (`src/fault_system/datagen/`) is a separate path: a JSON recipe drives layout, drop timing, and either Simple IK or SmolVLA, then `dataset_writer` saves episodes. It uses stock `LiberoEnv` and the same headless install. It does not go through `lerobot-eval-faults`.

## Commands

Smoke (one episode, cached `lerobot/smolvla_libero`, CUDA, offline hub):

```bash
MUJOCO_GL=egl bash scripts/run_fault_smoke.sh baseline
MUJOCO_GL=egl bash scripts/run_fault_smoke.sh injected
```

Record that eval into a dataset:

```bash
uv run lerobot-eval-faults \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero --env.task=libero_object --env.task_ids='[0]' \
  --eval.n_episodes=1 --eval.batch_size=1 --eval.recording=true \
  --policy.device=cuda --output_dir=outputs/eval/record_once
```

Overlay scenes: `lerobot-eval-overlay --env.type=libero_overlay --env.overlay=<yaml>`.

Unit tests: `uv run pytest tests/faults tests/envs/test_libero_overlays.py tests/scripts/test_eval_recording_features.py -q`
