# LIBERO object overlays — how to run

Non-destructive scene edits for LeRobot + LIBERO. Stock LIBERO `assets/` and
`bddl_files/` are **never** modified; overlays write a patched BDDL under
`$TMPDIR/lerobot_libero_overlays/`.

| File | Purpose |
|------|---------|
| `replace_tomato_with_red_cube.yaml` | Swap `tomato_sauce` → `red_cube` on task 5 |
| `add_red_cube.yaml` | Add several distractors (`red_cube`, `blue_cube`, second `red_cube`) |
| `assets/red_cube/`, `assets/blue_cube/` | MuJoCo XML for the cubes |
| `objects/` | Package registering `RedCube` + `BlueCube` via `@register_object` |

Also see `.cursor/skills/libero-object-overlays/SKILL.md` and
`docs/source/libero.mdx`.

## Prerequisites

From the repo root (Linux):

```bash
uv sync --locked --extra libero
# optional: policy deps for the smolvla eval below
uv sync --locked --extra libero --extra smolvla
```

## Find the BDDL category name (what to put in `replacements`)

Replace mode maps **BDDL categories**, not instance ids and not English phrases.

In BDDL:

```text
(:objects
  tomato_sauce_1 - tomato_sauce    # instance - category
  basket_1 - basket
)
```

- Left of `-` is the **instance** (`tomato_sauce_1`).
- Right of `-` is the **category** (`tomato_sauce`) — use this as the YAML key.
- Replace rewrites both: `tomato_sauce` → `red_cube` and `tomato_sauce_1` → `red_cube_1`.

### Resolve the stock BDDL path for a task

```bash
uv run python - <<'PY'
from pathlib import Path
from libero.libero import benchmark, get_libero_path

suite_name = "libero_object"
task_id = 5

suite = benchmark.get_benchmark_dict()[suite_name]()
task = suite.get_task(task_id)
bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
print("language:", task.language)
print("bddl:    ", bddl)
print()
text = bddl.read_text()
start = text.find("(:objects")
print(text[start : text.find(")", start + 1) + 200])  # objects block (approx)
PY
```

Example output for `libero_object` task 5 includes `tomato_sauce_1 - tomato_sauce`,
so the replace key is `tomato_sauce`:

```yaml
replacements:
  tomato_sauce: red_cube
```

If the key is wrong, apply fails with a clear error that the category is missing
from the BDDL.

### Inspect objects without opening the full file

```bash
uv run python - <<'PY'
from pathlib import Path
from libero.libero import benchmark, get_libero_path
import re

suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(5)
bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
text = bddl.read_text()
# instance - category pairs
for m in re.finditer(r"(\w+)\s+-\s+(\w+)", text[text.find("(:objects") : text.find("(:obj_of_interest")]):
    print(f"instance={m.group(1):20s}  category={m.group(2)}")
PY
```

## Real eval commands

Run from the **repo root**. Paths below are relative to that root.

### 1) Stock scene (no overlay — baseline)

```bash
uv run lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[5]" \
  --eval.batch_size=1 \
  --eval.n_episodes=1
```

### 2) Replace tomato sauce with a red cube

```bash
uv run lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[5]" \
  --env.overlay=examples/libero_overlays/replace_tomato_with_red_cube.yaml \
  --eval.batch_size=1 \
  --eval.n_episodes=1
```

### 3) Add several distractors (keep stock objects)

```bash
uv run lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[5]" \
  --env.overlay=examples/libero_overlays/add_red_cube.yaml \
  --eval.batch_size=1 \
  --eval.n_episodes=1
```

`mode: add` disables fixed init states by default (MuJoCo free-joint layout
changes). Override with `keep_init_states: true` in the YAML only if you know
the state vector still matches.

## Quick sanity checks (no full eval)

Patch BDDL only and print the temp path:

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
print("patched BDDL:", applied.bddl_path)
print(applied.bddl_path.read_text())
PY
```

Confirm the stock file was not touched:

```bash
# stock path still contains tomato_sauce; patched file contains red_cube
grep -E 'tomato_sauce|red_cube' "$(uv run python -c "
from pathlib import Path
from libero.libero import benchmark, get_libero_path
s=benchmark.get_benchmark_dict()['libero_object']()
t=s.get_task(5)
print(Path(get_libero_path('bddl_files'))/t.problem_folder/t.bddl_file)
")"
```

## Unit tests

```bash
CUDA_VISIBLE_DEVICES= uv run pytest tests/envs/test_libero_overlays.py -q
```
