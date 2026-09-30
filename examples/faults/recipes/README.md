# Fault datagen JSON recipes

Shared JSON recipes for drop datagen. Rates, frame columns, and `loss_mask` rules are in [`src/lerobot/faults/datagen/README.md`](../../src/lerobot/faults/datagen/README.md).

## What to record

Use these two recipes. They write **two datasets**, not one mixed run.

| Recipe | Role | Planned attempts | Output |
| --- | --- | --- | --- |
| `libero_object_success.json` | No-drop pick-and-place only | 400 (50 per object) | `outputs/datasets/libero_object_v1/success` |
| `libero_object_drop.json` | Four drop-recovery modes + a no-drop slice | 400 (10 of each mode per object) | `outputs/datasets/libero_object_v1/drop` |

Do **not** record `libero_object_drop_train.json` or `libero_object_drop_heldout.json`. Those files stay in the tree so older tests load, but they are the previous 800-episode / held-out-recording design.

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/libero_object_success.json

uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/libero_object_drop.json
```

Rerun the same command to continue after Ctrl+C. Finished attempts (kept or rejected) are skipped. The episode that was in progress is recorded again.

Optional overrides: `--base-seed`, `--output`, `--episodes`, `--fps` (`10` or `20`), `--device`, `--logical-start` / `--logical-end`. Changing `--fps` after a run has started requires a new `output_dir`.

`recording.episodes` is the **planned** attempt count. Missed grasps and stalled carries are not saved, so the on-disk episode count can be lower.

## Shared recording contract

Both recipes:

- Control at 20 Hz. One dataset: `dataset/` at **10 fps** by default. Pass `--fps 20` to store that one dataset at 20 fps instead. There is no second `dataset_20hz/` copy.
- Eight pick targets, round-robin: `alphabet_soup_1`, `cream_cheese_1`, `salad_dressing_1`, `bbq_sauce_1`, `ketchup_1`, `butter_1`, `milk_1`, `chocolate_pudding_1`.
- Held out of this recording: `tomato_sauce_1`, `orange_juice_1`. They are not pick targets. At record time they are hidden and parked off the table so they do not stay in the frames. Official LIBERO tasks 5 and 9 stay on disk for later eval; they are not recorded here.
- Sidecar `dataset/meta/failure_segments.parquet` stores the episode's real `object_name` and `task_id`.

### `libero_object_success.json`

| Field | Value |
| --- | --- |
| `recording.base_seed` | 21000 |
| `recording.episodes` | 400 |
| Matrix | one row: `simple_ik`, `immediate_ik`, `drop: false` |

Recorded success set (2026-09-30): **385 kept / 15 rejected**. Per-object kept: soup 50, ketchup 50, milk 50, chocolate pudding 50, butter 49, cream cheese 46, BBQ sauce 46, salad dressing 44.

### `libero_object_drop.json`

| Field | Value |
| --- | --- |
| `recording.base_seed` | 22000 |
| `recording.episodes` | 400 → 80 logical episodes × 5 equal-weight rows |

| `post_drop_mode` | `drop` | Per object (planned) |
| --- | --- | --- |
| `immediate_ik` | true | 10 |
| `continue_then_ik` | true | 10 |
| `reset_then_ik` | true | 10 |
| `immediate_smolvla` | true | 10 |
| `immediate_ik` | false | 10 |

This set is **not recorded yet**.

## How episode counts work

`recording.episodes` is the planned total across all matrix rows. Equal weights split the total evenly.

Example: `episodes: 400` with five rows at `weight: 1` → 80 episodes per row → **80 logical episodes** (each logical index runs all five rows once). `--episodes 5` on the drop recipe is one episode per row (smoke).

---

## Other recipes (not the v1 training pair)

### `libero_object_grasp_pilot.json`

Small grasp / layout pilot: all ten LIBERO-Object names, no drop, one matrix row. Output `outputs/probes/libero_object_grasp_pilot`. Not a substitute for the two v1 recipes.

### Legacy

- `libero_object_drop_train.json` — old 800-episode mix at `outputs/datasets/libero_object_drop_v1/train`. Do not record.
- `libero_object_drop_heldout.json` — old recipe that **records** tomato sauce and orange juice. Do not record. Held-out objects stay out of the v1 training pixels.
- `can_drop_datagen.json` — single-object CAN drop mix (alphabet soup).
- `alphabet_soup_ik_random.json` — no-drop soup only.

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/can_drop_datagen.json \
  --episodes 5
```
