---
name: libero-object-overlays
description: >-
  Non-destructively replace or add objects in LeRobot LIBERO scenes via YAML
  overlays without editing stock assets or BDDL. Use when customizing LIBERO
  objects, swapping tomato_sauce, adding distractors, writing overlay YAML,
  registering @register_object classes, or passing --env.overlay.
---

# LIBERO object overlays

Copy `examples/libero_overlays/`, edit YAML / XML / a `@register_object` class,
then pass `--env.overlay=<yaml>`. Stock LIBERO `assets/` and `bddl_files/` are
never modified. Patched BDDL is written under `$TMPDIR/lerobot_libero_overlays/`.

Eval commands, camera flags, and CMake notes: `examples/libero_overlays/README.md`.

## Hard rules

- Never edit installed LIBERO `assets/` or `bddl_files/`.
- Omit `--env.overlay` → default eval is unchanged.
- New `.py` files: `Copyright 2026 Gangelia. All rights reserved.` (not HuggingFace).
- One YAML = one mode. `replace` cannot set `add_objects`; `add` cannot set `replacements`.
- `objects_module` paths are resolved **relative to the YAML file**.
- Keep `hard_reset=True` when init states are disabled (`mode: add` default).

## Choose a mode

| Goal | Mode | Init states |
| --- | --- | --- |
| Swap a stock object (same count) | `replace` | kept unless `keep_init_states: false` |
| Keep stock objects, inject extras | `add` | disabled unless `keep_init_states: true` |

## Implement

Copy this checklist and work it in order:

```
- [ ] Copied examples/libero_overlays/ (do not invent a new layout)
- [ ] Chose replace XOR add
- [ ] Custom XML + @register_object (if the category is not already registered)
- [ ] Overlay YAML next to objects_module
- [ ] Dry-run apply_overlay; stock BDDL still unchanged
- [ ] CUDA_VISIBLE_DEVICES= uv run pytest tests/envs/test_libero_overlays.py -q
```

### 1. Custom object (skip if reusing `red_cube` / `blue_cube`)

Copy `examples/libero_overlays/assets/red_cube/` and `objects/red_cube.py`.

**XML** (`assets/<name>/<name>.xml`) must include:

- inner `<body name="object">` with visual (`group="1"`) + collision (`group="0"`) geoms
- `bottom_site`, `top_site`, `horizontal_radius_site` (LIBERO placement)

**Python** — CamelCase class name becomes the snake_case category (`RedCube` → `red_cube`):

```python
@register_object
class RedCube(MujocoXMLObject):
    def __init__(self, name: str = "red_cube", obj_name: str = "red_cube"):
        ...
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
```

- Do **not** redefine stock classes (`TomatoSauce`, …). Register a new class.
- Point `objects_module` at a **file** (`./objects/red_cube.py`) or a **package**
  (`./objects`) whose `__init__.py` imports every custom category you list.

### 2. Replace — YAML keys are BDDL **categories**

In `(:objects tomato_sauce_1 - tomato_sauce)`, the key is the token **after** `-`
(`tomato_sauce`), not the instance id and not English. Replace rewrites both
(`tomato_sauce_1` → `red_cube_1`). Wrong keys fail at apply (`assert_categories_present`).

```bash
uv run python - <<'PY'
from pathlib import Path
import re
from libero.libero import benchmark, get_libero_path
suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(5)
bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
print(bddl)
block = bddl.read_text()
objs = block[block.find("(:objects") : block.find("(:obj_of_interest")]
for m in re.finditer(r"(\w+)\s+-\s+(\w+)", objs):
    print(f"  instance={m.group(1):20s} category={m.group(2)}")
PY
```

```yaml
mode: replace
suite: libero_object
task_id: 5
objects_module: ./objects/red_cube.py
replacements:
  tomato_sauce: red_cube
language: Pick the red cube and place it in the basket
```

Canonical file: `examples/libero_overlays/replace_tomato_with_red_cube.yaml`.

### 3. Add — `add_objects` is a list

Each entry injects `:objects`, a `:regions` box, and `(On <instance> <target>_<region_name>)`.

```yaml
mode: add
suite: libero_object
task_id: 5
objects_module: ./objects
add_objects:
  - category: red_cube
    target: floor
    ranges: [0.05, 0.10, 0.10, 0.15] # xmin ymin xmax ymax

  - category: blue_cube
    target: floor
    ranges: [-0.18, 0.05, -0.13, 0.10]

  # Same category twice → unique instance + region_name
  - category: red_cube
    instance: red_cube_2
    region_name: red_cube_2_init_region
    target: floor
    ranges: [0.12, -0.12, 0.17, -0.07]
```

Canonical file: `examples/libero_overlays/add_red_cube.yaml`.

Optional per entry: `yaw_rotation: [min, max]`. Default instance is `{category}_1`;
default region is `{category}_init_region`. `target` is a fixture already in the
BDDL (`floor`, `kitchen_table`, …).

### 4. Dry-run (stock file must stay untouched)

```bash
uv run python - <<'PY'
from pathlib import Path
from libero.libero import benchmark, get_libero_path
from lerobot.envs.libero_overlays import apply_overlay
suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(5)
base = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
applied = apply_overlay(base, "examples/libero_overlays/replace_tomato_with_red_cube.yaml",
                        suite_name="libero_object", task_id=5)
print(applied.bddl_path)
print(applied.bddl_path.read_text())
PY
```

Then `lerobot-eval ... --env.overlay=<yaml>`. Full commands: `examples/libero_overlays/README.md`.

## YAML fields

| Field | Required | Notes |
| --- | --- | --- |
| `mode` | yes | `replace` or `add` |
| `replacements` | replace | `{stock_category: custom_category}` |
| `add_objects` | add | list of maps (`category` + `ranges` required) |
| `objects_module` | if custom objects | `.py` file or package dir, relative to YAML |
| `language` | no | overrides `(:language ...)` |
| `suite` / `task_id` | no | if set, must match the active env |
| `keep_init_states` | no | `null` → keep on replace, disable on add |

## Common mistakes

| Symptom | Cause |
| --- | --- |
| `BDDL does not contain categories to replace` | YAML key is an instance id or English, not the category |
| `Feature mismatch ... policy config` | missing `--env.camera_name_mapping` / `--policy.empty_cameras=1` for smolvla_libero |
| Object missing in sim | forgot `objects_module`, or package `__init__.py` does not import the class |
| `hard_reset=False requires init_states=True` | `mode: add` disabled init states; leave hard reset on |
| Duplicate instance / region | second object of the same category without unique `instance` + `region_name` |
| Mix `replacements` and `add_objects` | illegal; use two YAML files or pick one mode |
| Edited site-packages LIBERO | overlays must write temp BDDL only |

## Code map

- Config: `src/lerobot/envs/libero_overlays/config.py`
- BDDL patch: `src/lerobot/envs/libero_overlays/bddl.py`
- Apply + temp file: `src/lerobot/envs/libero_overlays/apply.py`
- Object import: `src/lerobot/envs/libero_overlays/objects.py`
- Env hook: `LiberoEnv(overlay=...)` / `--env.overlay`
- Tests: `tests/envs/test_libero_overlays.py`
