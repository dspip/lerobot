# HANDOFF — Frame-level failure annotations (LeRobot Parquet)

Single source of truth for this workstream.

## 1. Project goal & current context

**Goal:** LIBERO / MuJoCo episodes recorded as LeRobot datasets (`data/**/*.parquet` + `videos/` + `meta/`) must carry **per-frame failure labels** from live simulator ground truth, not from a sidecar JSONL join.

| Column | Meaning |
| ------ | ------- |
| `is_failure` | Physical failure this frame (object was held mid-air, now ungrasped, not in the basket). |
| `failure_onset` | Rising edge of `is_failure`. |
| `failure_type` | Category id (`0` none, `1` midair drop, `2` slip, …). Sticky after injector fire or spontaneous drop. |
| `injection_active` | Injector glitched **this** frame (independent of physics). |
| `phase` | `0` nominal, `1` injection, `2` post-fault, `3` recovery. Injection pulse wins over recovery on the glitch frame. |

SmolVLA camera/state/action keys are unchanged. Extra columns are ignored by the policy preprocessor.

## 2. Architecture

**Recorders**

1. Eval: `src/lerobot/scripts/lerobot_eval.py` `rollout()` when `--eval.recording=true`. Features include the five label columns (zeros if no wrapper).
2. Drop-recovery: `FaultRecoveryDatasetLogger` (`src/lerobot/faults/recovery/dataset_logger.py`) used by `examples/faults/run_full_drop_recovery_pipeline.py`.
3. Hardware `lerobot-record`: still out of scope (faults attach only in `eval_main`).

**Injection:** `maybe_wrap_env_tree` after `make_env`. Wrappers annotate **after** `env.step` and **before** `notify_dones`, merging arrays into Gym `info`.

**Physics:** `src/lerobot/faults/annotation.py` + `sim/libero.py` (`is_object_held_midair`, `is_object_grasped`, `is_object_in_basket`).

## 3. What shipped (this branch)

Branch: `feature/failure-annotation-parquet`

| File | Role |
| ---- | ---- |
| `src/lerobot/faults/annotation.py` | Latch + schema + Gym info helpers |
| `src/lerobot/faults/wrappers.py` | All three wrappers stamp `info` |
| `src/lerobot/faults/recovery/dataset_logger.py` | Writes label columns |
| `src/lerobot/scripts/lerobot_eval.py` | Eval recording includes labels |
| `src/lerobot/faults/action/hold.py`, `observation/burst.py`, `observation/sensor_dropout.py`, `sim/object_slip.py`, `sim/eef_bump.py` | `just_injected` pulse so duration=1 is labeled |
| `tests/faults/test_failure_annotation.py` | Latch / wrapper tests (no GPU) |
| `examples/faults/verify_failure_parquet.py` | Opens real Parquet and asserts |

## 4. Test results (2026-09-16)

- `uv run pytest tests/faults -q` → **207 passed**
- Dry-run dataset: `/tmp/lerobot_fail_ann_dry` (schema + nonzero flags)
- **Real LIBERO + SmolVLA + midair_drop** (CUDA, `MUJOCO_GL=egl`):
  - Command: `uv run python examples/faults/run_full_drop_recovery_pipeline.py --output-dir outputs/failure_annotation_verify --policy-path lerobot/smolvla_libero --device cuda`
  - `pipeline_log.json` `"success": true`, drop at sim step 63
  - Parquet `outputs/failure_annotation_verify/dataset`: **105 frames**, **14 `is_failure`**, **1 `failure_onset`**, **1 `injection_active`**, `failure_type=1`, injection frame **phase=1**, later failure frames **phase=3** (recovery planner running while the can is on the table)

## 5. Risks / follow-ups

- Eval recording still stores **pre-step** images with **post-step** labels (same convention as `next.reward`). The drop-recovery logger uses post-step obs, so labels match the cameras there.
- `midair_drop` starts recovery on the same env step as the drop, so you rarely see `phase=2` on that fault type; `object_slip` / `eef_bump` will.
- Jetson / RealSense depth and `lerobot-record` hardware faults are not in this branch.
- Dataset root must be fresh (`LeRobotDataset.create` refuses an existing directory unless `append=True`).
