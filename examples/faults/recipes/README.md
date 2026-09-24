Shared JSON recipes for fault datagen live here.

- **`can_drop_datagen.json`** — one dataset, five episode kinds (SimpleIK carry; three IK recoveries, one no-drop, one SmolVLA-after-drop fail). Type 3 (`reset_then_ik`) is SimpleIK → drop → SmolVLA dwell → IK place.

Record datasets with:

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/can_drop_datagen.json
```
