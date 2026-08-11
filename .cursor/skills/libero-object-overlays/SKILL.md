---
name: libero-object-overlays
description: >-
  Non-destructively replace or add objects in LeRobot LIBERO scenes via YAML
  overlays (stock assets/BDDL untouched). Use when customizing LIBERO objects,
  swapping tomato_sauce for a cube, adding distractors, or --env.overlay.
---

# LIBERO object overlays

## Copyright (required)

New files created for this work (and any new Gangelia fork code) must use:

```text
# Copyright 2026 Gangelia. All rights reserved.
```

Do **not** use `The HuggingFace Inc. team` on new files. Leave existing HuggingFace
headers unchanged on files you only modify.

## Rules

- Never edit installed LIBERO `assets/` or `bddl_files/`.
- Default eval stays identical when `--env.overlay` is omitted.
- Prefer `examples/libero_overlays/` as the template.

## Modes

| Mode | YAML | Effect | Init states |
|------|------|--------|-------------|
| default | none | Stock BDDL | unchanged |
| replace | `mode: replace` + `replacements` | Swap category tokens (e.g. `tomato_sauce` → `red_cube`) | kept unless `keep_init_states: false` |
| add | `mode: add` + `add_objects` | Inject object + region + `(On ...)` | disabled unless `keep_init_states: true` |

## Add a custom object

1. Create MuJoCo XML with `object` body + `bottom_site` / `top_site` / `horizontal_radius_site` (see `examples/libero_overlays/assets/red_cube/red_cube.xml`).
2. Register with `@register_object` CamelCase class → snake_case category (`RedCube` → `red_cube`). Do not redefine stock classes.
3. Point `objects_module` in the overlay YAML at that Python file (path relative to the YAML).
4. Put the Gangelia copyright header on every new `.py` file.

## Overlay YAML sketch

```yaml
mode: replace   # or add
suite: libero_object
task_id: 5
objects_module: ./objects/red_cube.py
replacements:
  tomato_sauce: red_cube
language: Pick the red cube and place it in the basket
```

```yaml
mode: add
suite: libero_object
task_id: 5
objects_module: ./objects/red_cube.py
add_objects:
  - category: red_cube
    target: floor
    ranges: [0.05, 0.10, 0.10, 0.15]  # xmin ymin xmax ymax
```

## Commands

```bash
# Unchanged default
lerobot-eval --policy.path=lerobot/smolvla_libero \
  --env.type=libero --env.task=libero_object --env.task_ids="[5]" \
  --eval.batch_size=1 --eval.n_episodes=1

# Replace
lerobot-eval --policy.path=lerobot/smolvla_libero \
  --env.type=libero --env.task=libero_object --env.task_ids="[5]" \
  --env.overlay=examples/libero_overlays/replace_tomato_with_red_cube.yaml \
  --eval.batch_size=1 --eval.n_episodes=1

# Add
lerobot-eval --policy.path=lerobot/smolvla_libero \
  --env.type=libero --env.task=libero_object --env.task_ids="[5]" \
  --env.overlay=examples/libero_overlays/add_red_cube.yaml \
  --eval.batch_size=1 --eval.n_episodes=1
```

## Tests

```bash
uv run pytest tests/envs/test_libero_overlays.py -q
```

## Implementation map

- Config load: `lerobot.envs.libero_overlays.config`
- BDDL patch: `lerobot.envs.libero_overlays.bddl`
- Apply + temp BDDL: `lerobot.envs.libero_overlays.apply`
- Env hook: `LiberoEnv(overlay=...)` / `--env.overlay`
