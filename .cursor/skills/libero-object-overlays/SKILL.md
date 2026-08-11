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
- Prefer `examples/libero_overlays/` as the template (full how-to-run:
  `examples/libero_overlays/README.md`).

## Modes

| Mode | YAML | Effect | Init states |
|------|------|--------|-------------|
| default | none | Stock BDDL | unchanged |
| replace | `mode: replace` + `replacements` | Swap category tokens (e.g. `tomato_sauce` → `red_cube`) | kept unless `keep_init_states: false` |
| add | `mode: add` + `add_objects` | Inject object + region + `(On ...)` | disabled unless `keep_init_states: true` |

## Find the BDDL category name (replace keys)

`replacements` maps **categories**, not instance ids and not English words.

BDDL:

```text
(:objects
  tomato_sauce_1 - tomato_sauce    # instance - category
  basket_1 - basket
)
```

- Use the token **after** `-` as the YAML key (`tomato_sauce`).
- Replace rewrites the category **and** instance prefix
  (`tomato_sauce_1` → `red_cube_1`).
- Wrong keys fail at apply time (`assert_categories_present`).

Resolve the stock file and list categories:

```bash
uv sync --locked --extra libero   # once

uv run python - <<'PY'
from pathlib import Path
import re
from libero.libero import benchmark, get_libero_path

suite_name, task_id = "libero_object", 5
suite = benchmark.get_benchmark_dict()[suite_name]()
task = suite.get_task(task_id)
bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
print("language:", task.language)
print("bddl:    ", bddl)
block = bddl.read_text()
objs = block[block.find("(:objects") : block.find("(:obj_of_interest")]
for m in re.finditer(r"(\w+)\s+-\s+(\w+)", objs):
    print(f"  instance={m.group(1):20s} category={m.group(2)}")
PY
```

Then in YAML:

```yaml
replacements:
  tomato_sauce: red_cube   # category → custom category
```

## Add custom object(s)

1. Create MuJoCo XML with `object` body + `bottom_site` / `top_site` / `horizontal_radius_site` (see `examples/libero_overlays/assets/`).
2. Register with `@register_object` CamelCase class → snake_case category (`RedCube` → `red_cube`). Do not redefine stock classes.
3. Point `objects_module` at a file **or package** that imports every custom category you list (see `examples/libero_overlays/objects/`).
4. Put the Gangelia copyright header on every new `.py` file.
5. To add **several** objects, append more entries under `add_objects` (see below).

## Overlay YAML sketch

```yaml
mode: replace
suite: libero_object
task_id: 5
objects_module: ./objects/red_cube.py
replacements:
  tomato_sauce: red_cube
language: Pick the red cube and place it in the basket
```

```yaml
# Add one or many objects — `add_objects` is a list.
mode: add
suite: libero_object
task_id: 5
objects_module: ./objects          # package that registers red_cube + blue_cube
add_objects:
  - category: red_cube
    target: floor
    ranges: [0.05, 0.10, 0.10, 0.15]   # xmin ymin xmax ymax

  - category: blue_cube
    target: floor
    ranges: [-0.18, 0.05, -0.13, 0.10]

  # Same category twice → unique instance + region_name required
  - category: red_cube
    instance: red_cube_2
    region_name: red_cube_2_init_region
    target: floor
    ranges: [0.12, -0.12, 0.17, -0.07]
```

## How to run (real commands)

From repo root. Needs `uv sync --locked --extra libero --extra smolvla`.
If `egl-probe` fails to build under CMake 4, prefix with
`CMAKE_POLICY_VERSION_MINIMUM=3.5`.

`smolvla_libero` expects `camera1`/`camera2` + one padded slot, so every command
needs `--env.camera_name_mapping` and `--policy.empty_cameras=1` (same as
`.github/workflows/benchmark_tests.yml`). Omitting them raises
`Feature mismatch between dataset/environment and policy config`.

```bash
CAM='--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'

# Unchanged default (baseline)
uv run lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[5]" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  "$CAM" \
  --policy.empty_cameras=1

# Replace tomato_sauce → red_cube
uv run lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[5]" \
  --env.overlay=examples/libero_overlays/replace_tomato_with_red_cube.yaml \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  "$CAM" \
  --policy.empty_cameras=1

# Add distractors (red_cube + blue_cube + second red_cube)
uv run lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[5]" \
  --env.overlay=examples/libero_overlays/add_red_cube.yaml \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  "$CAM" \
  --policy.empty_cameras=1
```

Other policies map keys differently — e.g. `pi0fast-libero` uses
`--rename_map={"observation.images.image": "observation.images.base_0_rgb", "observation.images.image2": "observation.images.left_wrist_0_rgb"}`.

Dry-run patch only (print temp BDDL, never touch stock files):

```bash
uv run python - <<'PY'
from pathlib import Path
from libero.libero import benchmark, get_libero_path
from lerobot.envs.libero_overlays import apply_overlay

suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(5)
base = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
applied = apply_overlay(
    base,
    "examples/libero_overlays/replace_tomato_with_red_cube.yaml",
    suite_name="libero_object",
    task_id=5,
)
print(applied.bddl_path)
print(applied.bddl_path.read_text())
PY
```

## Tests

```bash
CUDA_VISIBLE_DEVICES= uv run pytest tests/envs/test_libero_overlays.py -q
```

## Implementation map

- Config load: `lerobot.envs.libero_overlays.config`
- BDDL patch: `lerobot.envs.libero_overlays.bddl`
- Apply + temp BDDL: `lerobot.envs.libero_overlays.apply`
- Env hook: `LiberoEnv(overlay=...)` / `--env.overlay`
- How-to-run + BDDL discovery: `examples/libero_overlays/README.md`
