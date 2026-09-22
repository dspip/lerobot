# Manual drop and recovery (live)

Use this to test midair drop and IK recovery on the real LIBERO sim with your keyboard—no prediction head, no checkpoint.

## Run it

From the repo root:

```bash
cd ~/Projects/lerobot
uv run python examples/faults/run_manual_drop_recovery.py
```

Click the tkinter window once so it has focus. Keys **d**, **r**, **q**, and **Esc** (and the two buttons) only work when that window is focused.

## What you should see

The arm plays a **full recorded pickup** from nominal dataset episode **12** by default (each action held for two sim steps). When that recording finishes, the same episode **starts again from the beginning** so the arm never sits frozen waiting for you. Press **Drop (d)** while the can is in the hand, at whatever moment you choose—the terminal should print `Drop: is_object_grasped=True` when the grasp check passes. Then press **Recover (r)** and watch the arm go back for the can. The window stays up until recovery finishes (planner done, then a short still hold), or until you press **q**, **Esc**, or close the window.

If you press **Recover** before **Drop**, the script prints `Press Drop while the can is in the hand first.` and does not start the planner.

Optional:

- `--episode N` — another **nominal** episode id from the audit report (must be `type: nominal`).
- `--output reports/xy60_verify/manual_drop_recovery` — where artifacts are written.

Requires `/tmp/xy_band_mix_60ep_extract/audit_report.json` and the parquet under `/tmp/xy_band_mix_60ep_extract/dataset/data/chunk-000`.

## Artifacts

Written under `reports/xy60_verify/manual_drop_recovery/` by default:

| File | Purpose |
| --- | --- |
| `rollout.mp4` | Agent-view recording with phase overlay |
| `summary.json` | Run metadata |
| `fault_events.jsonl` | Structured fault/recovery events |

A good manual test in `summary.json`:

- `grasped_before_drop`: `true`
- `drop_pressed_at` and `recovery_pressed_at`: both set (step numbers)
- `quit_reason`: `recovery_done` (finished recovery) or `user_quit` (you closed early on purpose)

If `quit_reason` is `step_cap`, the 4000-step safety limit was hit—usually a bug or recovery stuck; it should not happen on a normal run.
