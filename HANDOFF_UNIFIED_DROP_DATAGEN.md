# Handoff — record paired drop-recovery datasets

This is the only recorder for the current POC. Do not use `run_can_simpleik_datagen`, `run_xy_band_checkpoint`, or `run_full_drop_recovery_pipeline`.

**Branch:** `feat/unified-drop-datagen`

**Goal:** two comparable datasets, same object / same init+drop pairing / three post-drop modes:

1. SimpleIK always (carry + recovery).
2. SmolVLA until drop, then IK recovery (and dwell / reset-then-IK on SmolVLA variants).

`reset_then_ik` is **invalid** for SimpleIK carry. The recipe already excludes it.

---

## 0. Machine

- CUDA GPU
- `MUJOCO_GL=egl` (headless)
- `uv sync --locked --extra test` already done on this machine
- Policy weights: `lerobot/smolvla_libero` (Hugging Face cache)

---

## 1. Tests to run first (no GPU required)

These lock pairing, recipe, masks, and keep/reject. Run them before recording.

```bash
cd /path/to/lerobot
uv run pytest tests/faults -q --tb=short
```

Expected: pass. If they fail, **do not record**.

Optional extra (no sim):

```bash
uv run pytest tests/faults/test_datagen_runner.py tests/faults/test_datagen_recipe.py tests/faults/test_datagen_final_review_contracts.py -q
```

There is **no required live MuJoCo pytest** for this recorder. Live validation is the first `--episodes 1` run below.

Dry-run the old demo (does not write the paired matrix):

```bash
uv run python examples/faults/run_drop_recovery_demo.py --dry-run
```

---

## 2. Recipe

`examples/faults/recipes/can_drop_datagen.json`

Checked-in matrix is **1 episode per variant** (5 variants) so a first pass is cheap. Raise `--episodes` for the real sets.

| controller | post_drop_mode |
|---|---|
| simple_ik | immediate_ik |
| simple_ik | continue_then_ik |
| smolvla | immediate_ik |
| smolvla | continue_then_ik |
| smolvla | reset_then_ik |

Outputs land under `outputs/can_drop_datagen/<controller>/<post_drop_mode>/` unless you override `--output`.

---

## 3. Commands

### Smoke (one episode per variant)

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/can_drop_datagen.json \
  --device cuda
```

### Real POC datasets (same pairing, N episodes per variant)

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/can_drop_datagen.json \
  --episodes 20 \
  --output outputs/can_drop_datagen_poc \
  --device cuda
```

`--episodes` overrides **every** matrix row. `--base-seed` overrides `recording.base_seed` (default 9000). Keep one seed if you want comparable reruns.

CPU SmolVLA is possible (`--device cpu`) but slow; do not use it for the training set.

---

## 4. What to check after a run

Per variant directory:

- `data/**/*.parquet` + `videos/` + `meta/`
- `episode_metadata.jsonl` — one row per attempt (`keep`, `reject_reason`, `episode_seed`, `layout_seed`, `drop_seed`, `controller_seed`, `post_drop_mode`)
- `manifest.json` — accepted episodes only

Paired check: same `logical_episode_index` across controllers must share `layout_seed` / `drop_seed` / `episode_seed`. `controller_seed` / `variant_seed` may differ.

Frame check (first kept episode):

- `loss_mask == 0` on drop + dwell frames
- `loss_mask == 1` on recovery IK
- `ever_held_midair` present

If `keep` is false for a row, the episode was discarded (never written as a training episode). That is expected for failed grasp / reject reasons.

---

## 5. Do not

- Edit `src/lerobot/faults/datagen/smolvla_pipeline.py` “just to record”
- Point training at `outputs/failure_annotation_verify/`
- Mix SimpleIK and SmolVLA into one LeRobot repo without the variant split (the runner already partitions)
- Use `reset_then_ik` on SimpleIK
