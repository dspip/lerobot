# Fault datagen JSON recipes

Shared JSON recipes for unified drop datagen live here. Full pipeline semantics (rates, columns, splits, sharding) are in [`src/lerobot/faults/datagen/README.md`](../../src/lerobot/faults/datagen/README.md).

## How episode counts work

`recording.episodes` is the planned **dataset total** (sum of episodes across all matrix rows). Each `experiment_matrix` row has a `weight` that sets the mix; equal weights split the total evenly.

Example: `recording.episodes: 100` with five rows at `weight: 1` → 20 episodes per row → **20 logical episodes** (each logical index runs all five rows once). Override the total at runtime with `--episodes` (same allocation rules); e.g. `--episodes 5` yields one episode per row for a smoke run.

## Recording command

Record these two recipes. Do not record `libero_object_drop_train.json` or `libero_object_drop_heldout.json`.

`libero_object_success.json` is 400 no-drop episodes (50 per recorded object). `libero_object_drop.json` is 400 episodes split evenly across four drop-recovery modes and one no-drop row (10 of each per object). Both record the same eight objects and remove `tomato_sauce_1` and `orange_juice_1` from the scene. Stored datasets stay at 10 fps. Rerunning the same command continues after the last committed episode.

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/libero_object_success.json

uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/libero_object_drop.json
```

Optional overrides: `--base-seed`, `--output`, `--episodes`, `--device`, `--logical-start` / `--logical-end` (half-open logical range for sharding).

---

## `libero_object_drop_train.json`

**Purpose:** Main **train-target** multi-object drop dataset (eight LIBERO-Object SKUs).

| Field | Value |
| --- | --- |
| `name` | `libero_object_drop_train` |
| `task` | `libero_object` |
| `task_id` | `0` (validator anchor when multiple official task ids appear in `object_names`) |
| `control_hz` | 20 |
| `q` | 1.0 (matrix `drop: true/false` overrides per row via paired plan) |
| `recording.base_seed` | **20000** |
| `recording.output_dir` | `outputs/datasets/libero_object_drop_v1/train` |
| `recording.dataset_fps` | 10 |
| `recording.master_fps` | 20 (writes sibling `dataset_20hz/`) |
| `recording.episodes` | **800** → **160** logical episodes × **5** matrix rows |

**Pick targets (`object_names`):**  
`alphabet_soup_1`, `cream_cheese_1`, `salad_dressing_1`, `bbq_sauce_1`, `ketchup_1`, `butter_1`, `milk_1`, `chocolate_pudding_1`.

**Not in this list (held-out, never recorded for training):** `tomato_sauce_1`, `orange_juice_1`.

**Experiment matrix (five rows, equal weight):**

| `post_drop_mode` | `drop` | Role |
| --- | --- | --- |
| `immediate_ik` | true | Drop + IK recovery |
| `continue_then_ik` | true | Drop + continue then IK |
| `reset_then_ik` | true | Drop + dwell/reset + IK |
| `immediate_ik` | false | **No-drop** paired control (`immediate_ik_no_drop` artifact dir) |
| `immediate_smolvla` | true | Drop + SmolVLA post-drop policy |

Shared blocks: `placement`, `simple_ik.path_drop` (drops only in `lift` / `to_container`), `smolvla` bands, `post_drop.dwell_steps: 80`.

**After sharding, merge and audit:**

```bash
uv run python examples/faults/merge_drop_datagen_shards.py \
  --shards outputs/.../shard0 outputs/.../shard1 \
  --output outputs/datasets/libero_object_drop_v1/train_merged \
  --test-percent 15

uv run python examples/faults/audit_drop_dataset.py \
  outputs/datasets/libero_object_drop_v1/train_merged \
  --heldout-objects tomato_sauce_1,orange_juice_1
```

---

## `libero_object_drop_heldout.json`

**Purpose:** Evaluation only. Not part of dataset generation: run it after training to record the unseen-object test set with the same five-row matrix and physics as train (separate output, never merged with train).

| Field | Value |
| --- | --- |
| `name` | `libero_object_drop_heldout` |
| `task_id` | **5** (`tomato_sauce_1` anchor per recipe validator) |
| `recording.base_seed` | **30000** |
| `recording.output_dir` | `outputs/datasets/libero_object_drop_v1/heldout_objects` |
| `recording.episodes` | **100** → **20** logical × **5** rows |

**Pick targets:** `tomato_sauce_1`, `orange_juice_1`.

Use the same `run_drop_datagen.py` / sharding flags as train. Audit train merges with `--heldout-objects` so held-out SKUs never appear as **targets** in the train corpus (they may still appear as clutter in train scenes).

---

## `libero_object_grasp_pilot.json`

**Purpose:** Small **grasp / layout pilot** — all ten LIBERO-Object instances listed, **no drop**, single matrix row.

| Field | Value |
| --- | --- |
| `name` | `libero_object_grasp_pilot` |
| `task_id` | `0` |
| `recording.base_seed` | **13000** |
| `recording.output_dir` | `outputs/probes/libero_object_grasp_pilot` |
| `recording.dataset_fps` | 10 |
| `recording.master_fps` | *(omitted — only `dataset/` at 10 Hz unless you add `"master_fps": 20`)* |
| `recording.episodes` | 100 |

**Matrix:** one row — `simple_ik`, `immediate_ik`, **`drop: false`**.

Use this to validate placement randomization and grasp detection across every object name before long drop runs. Not a substitute for the train/heldout drop recipes.

---

## Legacy / other recipes

- **`can_drop_datagen.json`** — single-target CAN drop POC: five episode kinds (SimpleIK carry; three IK recoveries, one no-drop, one SmolVLA-after-drop). Type 3 (`reset_then_ik`) is SimpleIK → drop → SmolVLA dwell → IK place.
- **`alphabet_soup_ik_random.json`** — SimpleIK only, no drop, 100 alphabet-soup episodes with layout + trajectory randomization. Writes `outputs/alphabet_soup_success_ik_random`. Do not point this at `outputs/IK_no_noise`.

Example smoke run on the CAN recipe:

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/can_drop_datagen.json \
  --episodes 5
```
