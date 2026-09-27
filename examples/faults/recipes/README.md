Shared JSON recipes for fault datagen live here.

- **`can_drop_datagen.json`** — one dataset, five episode kinds (SimpleIK carry; three IK recoveries, one no-drop, one SmolVLA-after-drop fail). Type 3 (`reset_then_ik`) is SimpleIK → drop → SmolVLA dwell → IK place.
- **`alphabet_soup_ik_random.json`** — SimpleIK only, no drop, 100 alphabet-soup episodes with layout + trajectory randomization. Writes `outputs/alphabet_soup_success_ik_random`. Do not point this at `outputs/IK_no_noise`.

`recording.episodes` is the planned **dataset total** (sum of episodes across all matrix rows). Each `experiment_matrix` row has a `weight` that sets the mix; equal weights split the total evenly. Example: `recording.episodes: 100` with five rows at `weight: 1` → 20 episodes per row. Override the total at runtime with `--episodes` (same allocation rules); e.g. `--episodes 5` on that recipe yields 1 episode per row for a quick smoke run.

Record datasets with:

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/can_drop_datagen.json \
  --episodes 5
```
