---
name: run-libero-smoke
description: >-
  Use when running in-tree LIBERO A/B smoke evals (baseline, action_hold, delay,
  jitter, sensor) via `scripts/run_fault_smoke.sh` and stock `lerobot-eval`.
  For midair_drop / drop-middle recovery, use the test-fault skill instead.
---

# Run LIBERO fault smoke evals

Smoke covers hold/delay/jitter/sensor only. **Drop-middle (`midair_drop`) is not a
smoke mode** — follow `test-fault` and `examples/faults/run_full_drop_recovery_pipeline.py`.

## Prerequisites

- Repo root: `/home/aviya/Projects/lerobot`
- `uv sync --locked` (or `--extra test`)
- CUDA GPU recommended for SmolVLA
- `MUJOCO_GL=egl` on headless machines

## Commands

```bash
cd /home/aviya/Projects/lerobot
export MUJOCO_GL=egl

bash scripts/run_fault_smoke.sh baseline
bash scripts/run_fault_smoke.sh injected   # action_hold
bash scripts/run_fault_smoke.sh delay
bash scripts/run_fault_smoke.sh jitter
bash scripts/run_fault_smoke.sh sensor
```

Uses stock **`lerobot-eval`** with nested **`--fault.*`** flags (draccus). There is
no `lerobot-faults` CLI in this repo.

## Validation

| Mode | Must exist | Expect |
|------|------------|--------|
| `baseline` | `outputs/eval/fault_smoke_baseline/eval_info.json` | Often success; **no** `fault_events.jsonl` |
| `injected` | `outputs/eval/fault_smoke_injected/fault_events.jsonl` | Hold events around steps 20–27 |
| `delay` | `outputs/eval/fault_smoke_delay/fault_events.jsonl` | `action_delay` after warm-up |
| `jitter` | `outputs/eval/fault_smoke_jitter/fault_events.jsonl` | `proposed_action` ≠ jittered action |
| `sensor` | JSONL + `outputs/eval/fault_smoke_sensor/diag/*.png` | Black diag PNGs during blackout |

If `fault_events.jsonl` is missing on an injected run, `maybe_wrap_env_tree` did not wrap the env.

## Troubleshooting

- `lerobot-eval: command not found` → use `uv run lerobot-eval` (script does this).
- Sensor video looks fine but policy is blind → open **diag PNGs** under `--fault.diag_dir`.
