# Drop datagen (LIBERO-Object)

Typed JSON recipes, experiment-matrix runner, and path-based drop triggers.

**Current training pair:** `libero_object_success.json` and `libero_object_drop.json` (see `examples/faults/recipes/README.md`). They write `outputs/datasets/libero_object_v1/success` and `…/drop`.

Do not treat `libero_object_drop_train.json` / `libero_object_drop_heldout.json` as the current recording plan.

## Simulation and recording rates

| Layer | Rate | Notes |
| --- | --- | --- |
| MuJoCo physics | 500 Hz | `DEFAULT_MUJOCO_MODEL_TIMESTEP = 0.002` s in `fault_system.recovery.fps` |
| Control / env step | 20 Hz | Recipe `control_hz` |
| Physics substeps per control tick | 25 | One control period (0.05 s) ÷ model timestep |

Each run writes **one** LeRobot dataset at `dataset/` (`recording.dataset_fps`, default 10). Control stays 20 Hz. At 10 fps, stride is 2 (even ticks). `--fps 20` stores every control tick in that same `dataset/` folder. `run_drop_datagen.py` does not write a sibling `dataset_20hz/` (that path is leftover from older dual-view recipes; `master_fps` is ignored by the CLI).

Recording uses **uniform stride only**. There are **no forced injection frames**. At 10 fps, boolean pulses (`drop_release`, `failure_onset`) are OR-accumulated across skipped ticks and applied on the next logged frame.

## Action–observation pairing

POST `env.step()` contract (`POST_STEP_LOGGING_CONTRACT` in `frame_logging.py`):

- **Observation** is the dict returned **after** `env.step` (\(o_{t+1}\)).
- **Action** is the command that was just executed (\(a_t\)).

## Drop mechanism

- Drops fire on the SimpleIK carry path during `lift` / `to_container` when the paired plan requests a drop.
- On release the gripper opens and the object falls. The arm keeps the nominal carry command while falling.
- Logging stride does not change physics. A no-drop matrix row is a different episode, not a different simulator.

## On-disk layout

Under `recording.output_dir`:

- `dataset/` — the recorded LeRobot dataset (default 10 fps) + `meta/failure_segments.parquet`
- `run_manifest.json` — kept and rejected attempts, seeds, object names
- Per-variant diagnostics under `{controller}/{post_drop_mode}/episode_XXXX/` (no-drop rows use `{mode}_no_drop`)

Rerun the same `run_drop_datagen.py` command to resume. Completed variant keys (kept or rejected) are skipped. A kill discards only the in-progress episode.

## Frame columns

`DATAGEN_FRAME_LABEL_FEATURES` plus existing failure columns. Policy inputs stay images + 8-D state + 7-D action + `task`.

| Column | Meaning |
| --- | --- |
| `tick_index` | Control tick for this row |
| `drop_release` | The frame the gripper released (pulse) |
| `drop_event` | Object in free fall / injection window (unpadded) |
| `drop_window` | Short drop label: two **10 fps** frames before release, the fall, two 10 fps frames after landing (4 control ticks of pad at 20 Hz control). Same physical window if you recorded with `--fps 20`. `False` if the episode never released. Backfilled at commit. |
| `attempt_index` | `0` from the start through landing; `1` after landing until episode end. No-drop episodes stay `0`. This is the “first try / after the drop” split. There is no separate `subepisode` column. |
| `loss_mask` | Whether this frame is used as an imitation target. See below. |
| `is_failure` | Physical latch: was held mid-air, then not grasped, not in the basket, not releasing over the basket. **Not** the drop label. It can stay true for seconds after a drop because of dwell. |

`is_failure` may appear downstream as `object_lost`. The parquet name is `is_failure`.

### `loss_mask`

- **1** on a normal carry, including the 2 frames before release (still a normal hold).
- **0** from release through the post-drop wait (fall + dwell) so the policy does not learn the freeze.
- **1** again when IK recovery is active (`recovery_active`).
- **`immediate_smolvla`:** **0** from release through the **rest of the episode**, even if IK later starts.
- **No-drop episodes:** **1** on every frame.

`loss_mask_from_fault` zeroes injection and dwell. `MidAirDropFault.loss_mask_for_env` adds the SmolVLA rule.

### Sidecar `meta/failure_segments.parquet`

One or two rows per episode (`attempt_index` 0 / 1). Stores release and landing ticks/frames, plus **`object_name` and `task_id` of that episode** (not `object_names[0]` of the recipe). Written on every commit so a resume does not lose rows.

### Model inputs vs labels

- **Inputs:** two RGB images + 8-D `observation.state` + language `task`.
- **Labels / filters:** `loss_mask`, `drop_window`, `drop_event`, `attempt_index`, `is_failure`, sidecar. Do not feed privileged columns as policy inputs unless the trainer is built for that.

## Current recipes

| Recipe | Seed | Planned attempts | Matrix |
| --- | --- | --- | --- |
| `libero_object_success.json` | 21000 | 400 no-drop | one row |
| `libero_object_drop.json` | 22000 | 400 | five equal rows (4 drop modes + 1 no-drop) |

Eight recorded objects, round-robin by logical index. Held-out names `tomato_sauce_1` and `orange_juice_1` are excluded from layout sampling and hidden in the sim. They are **not** recorded as a second training dataset.

Public SmolVLA LIBERO checkpoints were likely trained on all ten official tasks. Held-out here is a **recording** split, not a claim the base VLA never saw those objects.

Missed grasp / stalled carry → episode not saved. A success cell of “50 per object” can finish under 50. A drop cell of “10 per mode per object” can finish under 10.

## Experiment matrix (drop recipe)

Each **logical episode index** runs **five** rows with the same layout, motion profile, and drop draw. All rows use `simple_ik`.

| `post_drop_mode` | `drop` |
| --- | --- |
| `immediate_ik` | true |
| `continue_then_ik` | true |
| `reset_then_ik` | true |
| `immediate_smolvla` | true |
| `immediate_ik` | false |

The success recipe is only the no-drop row.

## Drop timing (10 Hz)

Fall after release is typically **2–4 frames** at 10 fps (`drop_event`). `drop_window` adds two stored frames before release and two after landing. Use `drop_window` for a short detector target. Do not invent a 16-frame window that is not in the parquet.

## Trainer notes

- Train on **`dataset/`**. Default recording is 10 fps. Use `--fps 20` only if you want that one dataset at control rate.
- Filter imitation loss with **`loss_mask`**. Do not use `is_failure` as that filter.
- For a short “this is the drop” label, use **`drop_window`**, not `is_failure` and not the whole dwell.
- `attempt_index == 0` is approach + carry (+ fall if any). `attempt_index == 1` is after landing (recovery).
- Language comes from the dataset `task` / `task_index` (per-object LIBERO instruction). Use `instruction_mode="dataset_task"` if the loader has that switch.

## Sharding and audit

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/libero_object_drop.json \
  --logical-start 0 --logical-end 16
```

- `--logical-start` / `--logical-end`: half-open logical range.
- Merge: `examples/faults/merge_drop_datagen_shards.py`
- Audit: `examples/faults/audit_drop_dataset.py <run_dir> --heldout-objects tomato_sauce_1,orange_juice_1`  
  Audit only flags held-out names used as the **pick target**. Pixel exclusion is hide-at-record-time, not that check.

After merge, `logical_episode_index % 100 < test_percent` (default 15) is `test_in_distribution`.

## Known timing deviations

Gripper settle runs extra MuJoCo substeps inside a control tick without extra logged frames (regrasp close ~40 substeps vs 25; release open ~30 vs 25). About 0.04 s of physics on those ticks is not extra dataset frames.

## Recording-quality notes

- Grasp uses finger-link geoms, not pad-only groups.
- LIBERO wrapper does not `reset()` inside `step`; datagen stops on `done`.
- `nominal_grasp_missed` and `nominal_carry_stalled` are rejected (not saved).

## Module map

| Area | Modules |
| --- | --- |
| Recipes / matrix | `recipe.py`, `recipe_identity.py`, `runner.py`, `paired_context.py` |
| Recording | `recording_views.py`, `frame_logging.py`, `dataset_writer.py` |
| Segments / card / audit | `failure_segments.py`, `dataset_card.py`, `audit.py`, `merge_shards.py`, `shard_range.py` |
| Controllers | `controllers/simple_ik.py`, `controllers/smolvla.py`, `smolvla_pipeline.py` |
| Sim / wrap | `../sim/libero.py`, `../wrappers.py`, `../annotation.py` |

CLI: `examples/faults/run_drop_datagen.py`, `merge_drop_datagen_shards.py`, `audit_drop_dataset.py`.
