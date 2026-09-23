Shared JSON recipes for fault datagen live here.

- **`can_drop_datagen.json`** — sole drop-datagen recipe (experiment matrix, placement, SimpleIK + SmolVLA settings, recording).

Record datasets with:

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/can_drop_datagen.json
```
