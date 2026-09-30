# Drop datagen (LIBERO-Object)

Unified drop datagen: typed JSON recipes, experiment-matrix runner, controller adapters, and path-based drop triggers. This document focuses on the **multi-object LIBERO-Object drop** datasets (`libero_object_drop_train`, `libero_object_drop_heldout`); see `examples/faults/recipes/README.md` for recipe entry points.

## Simulation and recording rates

| Layer | Rate | Notes |
| --- | --- | --- |
| MuJoCo physics | 500 Hz | `DEFAULT_MUJOCO_MODEL_TIMESTEP = 0.002` s in `lerobot.faults.recovery.fps` |
| Control / env step | 20 Hz | Recipe `control_hz`; matches `recording.master_fps` when the master view is enabled |
| Physics substeps per control tick | 25 | One control period (0.05 s) ÷ model timestep |

Each run writes **two LeRobot views** under the recipe `recording.output_dir`:

| View directory | FPS | Stride | Role |
| --- | --- | --- | --- |
| `dataset_20hz/` | 20 | 1 | Master view: one frame per control tick |
| `dataset/` | 10 | 2 | Policy view: logs ticks where `tick % stride == 0` |

Recording uses **uniform stride only** (`should_log_view_tick` / `should_log_sim_step`); there are **no forced injection frames**. For strided views, boolean **pulses** in `PULSE_ANNOTATION_KEYS` (`drop_release`, `failure_onset`) are **OR-accumulated** across skipped ticks and applied on the next logged frame (`offer_tick_to_views` in `recording_views.py`).

For a given episode, if the master view logs `N` frames at 20 Hz, the 10 Hz view logs **⌈N / 2⌉** frames (stride 2, ticks 0, 2, 4, …).

## Action–observation pairing

Datagen follows the **POST `env.step()` contract** (`POST_STEP_LOGGING_CONTRACT` in `frame_logging.py`):

- **Observation** on each row is the dict returned **after** `env.step` (state at \(o_{t+1}\)).
- **Action** is `env.last_executed_action` when set, otherwise the action tensor passed into `env.step` (the command executed for that transition, \(a_t\)).

So each stored transition pairs **next observation with the action that was just executed**—standard “observe result of action” alignment for imitation learning.

## Drop mechanism

- Drops fire on the SimpleIK carry path during `lift` / `to_container` when the paired plan requests a drop (`path_drop` + shared `drop_u`).
- On release, the fault opens the gripper and sets **`falling`**; the arm **keeps executing** the nominal carry command (scaled release branch in `_nominal_action`) while the object falls.
- **Recording does not alter physics**: logging stride and the no-drop matrix row only change which ticks are written. `force_drop_injection` is ignored in `should_log_sim_step`. Audit checks that drop vs no-drop episodes share **identical `observation.state` prefixes before `drop_release`** (`_audit_pairs` in `audit.py`).

## On-disk layout

Under `recording.output_dir`:

- `dataset/` — 10 Hz LeRobot dataset + `meta/failure_segments.parquet` for that view
- `dataset_20hz/` — 20 Hz sibling (`master_dataset_directory`)
- `run_manifest.json` — kept/discarded episodes, seeds, matrix metadata
- Per-variant diagnostics under `{controller}/{post_drop_mode}/episode_XXXX/` (no-drop rows use `{mode}_no_drop`)

## Frame columns (policy view)

New datagen labels (`DATAGEN_FRAME_LABEL_FEATURES`) are added alongside existing failure columns; **dtypes/shapes of prior columns are unchanged**.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `tick_index` | int64 (1,) | Control tick index for this row |
| `drop_release` | bool (1,) | Rising edge of `triggered`: release pulse |
| `drop_event` | bool (1,) | `triggered` and (`falling` or `drop_injection_step`): object in free fall / injection window |
| `attempt_index` | int64 (1,) | `0` through landing (`drop_event` true); `1` after landing until episode end (no-drop stays `0`) |

Existing privileged / label columns (unchanged contract) include `is_failure`, `failure_onset`, `injection_active`, `phase`, `failure_type`, `ever_held_midair`, plus `loss_mask`, images, 8-D state, 7-D action, and `task`. **There are no pad-force columns.**

### Failure semantics

- **`is_failure`** (`annotation.py`): physical latch—object was held mid-air, then is not grasped, not in the basket, and not releasing over the basket. Recovery regrasp / basket placement clears subsequent “ungrasp” from counting as failure.
- **Training alias**: downstream Action Expert code may refer to this as **`object_lost`**; the dataset column name is **`is_failure`**.
- **Detector target**: “did a drop occur during this sub-task?” — successful recovery is **not** treated as failure for that objective.
- **`injection_active`**: injector window (includes the fall while `drop_injection_step` / fall logic is active)—separate from the physical latch so models do not learn the injection clock alone.
- **No per-episode verdict broadcast** onto frame rows; attempt-level timing lives in the sidecar.

### Sidecar `meta/failure_segments.parquet`

One or two rows per **(episode, attempt)** (`failure_segments.py`), with release/landing tick and frame indices, times at `control_hz`, and `object_z_at_release`. Built from per-tick snapshots during logging, not from a global episode verdict.

### Model inputs vs labels

- **Inputs**: two RGB images (`observation.images.image`, `observation.images.image2`) + **8-D** `observation.state`.
- **Privileged / training labels**: failure and drop columns above, `loss_mask`, and sidecar segments—kept separate from policy inputs.

## Experiment matrix (five rows per logical episode)

Each **logical episode index** runs **five** matrix rows with the **same** layout, motion profile, and drop draw (`paired_context.build_paired_episode_plan`). Rows differ by post-drop behavior:

| `post_drop_mode` | `drop` | Behavior (all rows use `controller: simple_ik` in train/heldout recipes) |
| --- | --- | --- |
| `immediate_ik` | true | Drop then IK recovery |
| `continue_then_ik` | true | Dwell / continue carry, then IK |
| `reset_then_ik` | true | SmolVLA dwell segment, policy reset, then IK |
| `immediate_ik` | false | **No drop** control baseline (`…/immediate_ik_no_drop/` artifacts) |
| `immediate_smolvla` | true | Drop then SmolVLA policy actions until done or timeout |

Drop and no-drop **pairs share identical observations before release** (audit + shared paired plan).

## Drop timing (empirical, 10 Hz view)

From finalized train merges (audit / dataset card histograms):

| Stat | Release → landing | Release height (`object_z`) |
| --- | --- | --- |
| p50 | 0.30 s | — |
| p90 | 0.32 s | — |
| max | 0.35 s | 0.12–0.23 m |

At 10 Hz, fall duration is typically **2–4 frames** after the release pulse.

**Training window (not stored in parquet)**: e.g. **k = 3** post-release frames at 10 Hz, total window **W ≈ 1.6 s (16 frames)** for temporal drop detectors.

## Object splits

| Split | Pick targets (`object_names`) | Recording seed | Planned dataset episodes (`recording.episodes`) |
| --- | --- | --- | --- |
| Train | `alphabet_soup_1`, `cream_cheese_1`, `salad_dressing_1`, `bbq_sauce_1`, `ketchup_1`, `butter_1`, `milk_1`, `chocolate_pudding_1` | `20000` | 800 (160 logical × 5 rows) |
| Held-out objects | `tomato_sauce_1`, `orange_juice_1` (evaluation only, recorded at evaluation time) | `30000` | 100 (20 logical × 5 rows) |

Targets rotate by **logical episode index** (`select_episode_object_round_robin`). Official LIBERO task ids come from `libero_object_tasks.official_task_id` per target, not from a single static scene.

- The training dataset is recorded from the train recipe only. Held-out objects are never recorded for training and never trained on; `libero_object_drop_heldout.json` exists so the unseen-object evaluation set can be recorded with the same pipeline after training.
- Held-out SKUs are removed from the scene at record time. They are not pick targets and they are not left in frame as distractors.
- **Leakage caveat**: public SmolVLA LIBERO checkpoints were likely trained on **all ten** LIBERO-Object tasks; held-out objects are a **recording** split, not a guarantee the base VLA never saw them.

### In-distribution test split

After merge, `merge_drop_datagen_shards.build_splits` assigns episodes with **`logical_episode_index % 100 < test_percent`** (default **15**) to **`test_in_distribution`**; all five matrix rows for that logical index share the split.

## Reproducibility and sharding

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/libero_object_drop_train.json \
  --logical-start 0 --logical-end 32
```

- **`--logical-start` / `--logical-end`**: half-open range of logical episodes (`shard_range.validate_logical_shard_range`).
- **Merge**: `examples/faults/merge_drop_datagen_shards.py --shards … --output … --test-percent 15`
- **Audit**: `examples/faults/audit_drop_dataset.py <run_dir> --heldout-objects tomato_sauce_1,orange_juice_1`

Operational notes (single GPU shard, order-of-magnitude):

- **~5.6 GB RAM** (SmolVLA + env + video encoders), **~1.6 GB GPU** for rows that load the policy
- On a **23 GB** machine, run shards **sequentially** or under a memory cap
- **~2.5 min** per logical episode (five matrix rows)

## Trainer notes

- **`injection_active`** is true across the fall/injection window; combine with **`loss_mask`** for Phase-1-style filtering.
- In the five-row smoke run, roughly **~45%** of frames had **`loss_mask = 0`** (post-drop dwell, fall ticks, and post-drop SmolVLA segment while `triggered && !recovery_active && !drop_injection_step`).
- **`loss_mask` derivation**: `loss_mask_from_fault` — zero on `drop_injection_step` or post-drop dwell; nominal no-drop episodes log **1.0** throughout.
- **Alpha-S `Phase1Dataset`** (`third_party/Eran's_code/Gangelia_AlphaS/alpha_s_xy60/phase1_data.py`, private, not vendored here) loads either view (`dataset/` or `dataset_20hz/`) unmodified; use **`instruction_mode="dataset_task"`** so `task` strings match the per-object LIBERO instructions. `loss_mask` already zeroes the fall ticks.

## Known timing deviations (documented, not changed)

Gripper settle runs **extra MuJoCo substeps inside a control tick** without additional logged frames:

| Event | Extra substeps | Typical tick total |
| --- | --- | --- |
| Recovery regrasp close (`force_close_gripper`) | **15** (`max(gripper_settle_steps, 15)`) | **40** vs normal 25 |
| Recovery release open (`force_open_gripper`) | **`gripper_settle_steps`** (default **5**) | **30** vs normal 25 |

~**0.04 s** of physics per affected tick is **not** represented as extra dataset frames.

## Fixes relevant to multi-object recording

- **Grasp detection** uses **finger link geoms** (`left_finger` / `right_finger`), not pad-only groups, so tall bottles pinched above the pads count as grasped (`is_object_grasped` in `sim/libero.py`).
- **LIBERO wrapper** no longer **`reset()` inside `step`** (`step_without_success_reset`); post-step annotation and labels see the terminal scene.
- **Vector env** uses **`AutoresetMode.NEXT_STEP`**; datagen loop **stops on `done`** so post-success frames are not recorded after autoreset would fire.
- **SimpleIK carry stall**: if the EEF moves **< 1 cm** for **40** control ticks during carry, outcome **`nominal_carry_stalled`** → **`evaluate_datagen_keep` rejects** the episode (far-edge singularities).
- **Pre-drop rejects**: **`nominal_grasp_missed`**, **`nominal_carry_stalled`** (and other keep rules in `dataset_writer.evaluate_datagen_keep`).

## Module map

| Area | Modules |
| --- | --- |
| Recipes / matrix | `recipe.py`, `recipe_identity.py`, `runner.py`, `paired_context.py` |
| Recording | `recording_views.py`, `frame_logging.py`, `dataset_writer.py` |
| Segments / card / audit | `failure_segments.py`, `dataset_card.py`, `audit.py`, `merge_shards.py`, `shard_range.py` |
| Controllers | `controllers/simple_ik.py`, `controllers/smolvla.py`, `smolvla_pipeline.py` |
| Sim / wrap | `../sim/libero.py`, `../wrappers.py`, `../annotation.py` |

CLI wrappers: `examples/faults/run_drop_datagen.py`, `merge_drop_datagen_shards.py`, `audit_drop_dataset.py`.
