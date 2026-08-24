---
name: test-fault
description: >-
  Use when testing, smoke-running, or validating in-tree fault injectors under
  `src/lerobot/faults/`, including midair_drop recovery, action_hold, or LIBERO eval.
---

# Test fault injection

All paths are in-tree under `/home/aviya/Projects/lerobot`. Do **not** use the
`lerobot_faults` sidecar package.

## Unit tests (no simulator, no GPU)

```bash
cd /home/aviya/Projects/lerobot
uv run pytest tests/faults -q
```

Pass = pytest exit 0. These do not prove the GPU pipeline.

Drop-middle / recovery subset:

```bash
uv run pytest tests/faults/test_midair_drop_fault.py \
  tests/faults/test_side_grasp_and_drop_gates.py \
  tests/faults/test_planner.py \
  tests/faults/test_libero_sim_helpers.py -q
```

## Drop-middle dry-run (no LIBERO)

```bash
cd /home/aviya/Projects/lerobot
uv run python examples/faults/run_drop_recovery_demo.py --dry-run
```

Expect a synthetic dataset under `outputs/demo_drop_recovery_dry/` (or `--output-dir`).

## Drop-middle full pipeline (GPU + LIBERO)

```bash
cd /home/aviya/Projects/lerobot
export MUJOCO_GL=egl
uv run python examples/faults/run_full_drop_recovery_pipeline.py \
  --output-dir outputs/full_pipeline_demo \
  --policy-path lerobot/smolvla_libero \
  --device cuda
```

**Pass only if all of:**

| Artifact | Expect |
|----------|--------|
| `outputs/full_pipeline_demo/fault_events.jsonl` | exists; an event with `"event": "midair_drop"` |
| `outputs/full_pipeline_demo/pipeline_log.json` | `"success": true` — grasp + drop + object in view, not merely fault fired |
| `outputs/full_pipeline_demo/videos/` | at least one rollout video |

Missing JSONL = wrapper did not attach. `"success": false` means behavioral checks failed.

## LIBERO A/B smoke (baseline + action faults)

Smoke covers baseline, hold, delay, jitter, sensor — **not** `midair_drop`. See
`run-libero-smoke` skill.

```bash
cd /home/aviya/Projects/lerobot
export MUJOCO_GL=egl
bash scripts/run_fault_smoke.sh baseline
bash scripts/run_fault_smoke.sh injected   # action_hold
```

| Mode | Must exist |
|------|------------|
| `baseline` | `outputs/eval/fault_smoke_baseline/eval_info.json` and **no** `fault_events.jsonl` |
| `injected` | `outputs/eval/fault_smoke_injected/fault_events.jsonl` with hold events |

## Eval CLI smoke (any fault type)

```bash
uv run lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --eval.n_episodes=1 \
  --env.max_parallel_tasks=1 \
  --fault.enabled=true \
  --fault.type=action_hold \
  --fault.trigger_step=20 \
  --fault.duration=8 \
  --output_dir=outputs/eval/fault_hold
```

Requires `--env.max_parallel_tasks=1` when faults are enabled.

## Troubleshooting

- `ImportError: lerobot` → run from repo root with `uv run`.
- Headless GL → `export MUJOCO_GL=egl`.
- Offline Hub → smoke sets `HF_HUB_OFFLINE=1`; unset if checkpoints must download.
