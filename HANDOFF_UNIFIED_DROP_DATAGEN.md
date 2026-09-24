# Handoff — record paired drop-recovery datasets

This is the only recorder for the current POC. Do not use `run_can_simpleik_datagen`, `run_xy_band_checkpoint`, or `run_full_drop_recovery_pipeline`.

**Branch:** `feat/unified-drop-datagen`

**Goal:** one LeRobot dataset, five episode kinds, same object / same init+drop pairing:

1. SimpleIK → drop → immediate IK
2. SimpleIK → drop → dwell → IK
3. SimpleIK → drop → SmolVLA dwell → IK place (`reset_then_ik`)
4. SimpleIK, no drop
5. SimpleIK → drop → SmolVLA only (`immediate_smolvla`, planned fail)

Failed IK recoveries (1–3) are discarded. Type 5 is kept.

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
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
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

| controller | post_drop_mode | drop |
|---|---|---|
| simple_ik | immediate_ik | true |
| simple_ik | continue_then_ik | true |
| simple_ik | reset_then_ik | true |
| simple_ik | immediate_ik | false |
| simple_ik | immediate_smolvla | true |

Outputs land under `outputs/can_drop_datagen/dataset/` unless you override `--output`.

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

Single dataset directory `dataset/`:

- `data/**/*.parquet` + `videos/` + `meta/`
- `meta/episodes` has per-episode `success`
- `run_manifest.json` — one row per attempt (`keep`, `success`, seeds, `post_drop_mode`)

Paired check: same `logical_episode_index` shares `layout_seed` / `drop_seed` / `episode_seed` / `init_state_id`.

Canonical `use_stock_layout` is false: all five SimpleIK rows share one randomized layout.

Frame check:

- types 1–3: `loss_mask == 0` on drop + dwell; `1` on IK recovery
- type 5: `loss_mask == 0` from drop to end
- `ever_held_midair` present

If `keep` is false, the episode is not in the training dataset. Failed IK recoveries (types 1–3) are discarded. Type 5 failed runs are kept.

---

## 5. Do not

- Edit `src/lerobot/faults/datagen/smolvla_pipeline.py` “just to record”
- Point training at `outputs/failure_annotation_verify/`
- Treat `reset_then_ik` as a MuJoCo scene reset; it is SimpleIK → drop → SmolVLA dwell → IK
