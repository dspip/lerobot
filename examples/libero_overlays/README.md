# LIBERO object overlays (examples)

Non-destructive scene edits for LeRobot + LIBERO:

- `replace_tomato_with_red_cube.yaml` — swap `tomato_sauce` → `red_cube` on task 5
- `add_red_cube.yaml` — keep the stock scene and add a red cube distractor
- `assets/red_cube/` — MuJoCo XML for the cube
- `objects/red_cube.py` — registers `RedCube` via LIBERO's `@register_object`

Stock LIBERO install files are never modified. See
`.cursor/skills/libero-object-overlays/SKILL.md` for the full workflow.
