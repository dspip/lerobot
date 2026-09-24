# IK Episode Previews and Controller-Specific Layouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save SmolVLA-style diagnostic artifacts for every SimpleIK attempt and run SmolVLA on stock LIBERO poses while SimpleIK retains randomized poses.

**Architecture:** Extract the existing preview encoder and proof-shot renderer into a controller-neutral module, then inject preview capture into the SimpleIK loop through a callback. Add an explicit `smolvla.use_stock_layout` recipe flag; the runner resolves one randomized IK layout and one zero-offset/zero-yaw SmolVLA layout per logical episode and records the actual layout used.

**Tech Stack:** Python 3.12, NumPy, Pillow, imageio/LeRobot video encoder, MuJoCo/LIBERO, pytest.

## Global Constraints

- SimpleIK episode directories contain `videos/full_pipeline.mp4`, `videos/full_pipeline.gif`, `final_state_multicam.png`, and `fault_events.jsonl`.
- Preview artifacts are saved for kept and rejected attempts.
- SmolVLA stock layout means XY displacement `0.0` and yaw displacement `0°`.
- Both SimpleIK modes share the randomized layout; all SmolVLA modes share the stock layout.
- The LeRobot dataset keep/discard policy is unchanged.
- Raw per-camera MP4s are not added to episode directories.

---

### Task 1: Shared episode preview component

**Files:**
- Create: `src/lerobot/faults/datagen/episode_preview.py`
- Modify: `src/lerobot/faults/datagen/smolvla_pipeline.py`
- Create: `tests/faults/test_episode_preview.py`

**Interfaces:**
- Produces: `EpisodePreview(output_dir: Path, control_fps: int)`.
- Produces: `capture(env: Any, label: str, color: tuple[int, int, int]) -> None`.
- Produces: `finalize(rs_env: Any) -> dict[str, str | None]`.
- Preserves SmolVLA artifact names and rendering behavior.

- [ ] **Step 1: Write failing unit tests**

Test a fake render environment and patched encoders:

```python
def test_episode_preview_captures_frames_and_writes_expected_artifacts(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(preview, "_write_mp4", lambda path, frames, fps: written.append((path.name, len(frames), fps)))
    monkeypatch.setattr(preview, "_write_gif", lambda path, frames, fps: written.append((path.name, len(frames), fps)))
    monkeypatch.setattr(preview, "_final_proof_shot", lambda rs, path: str(path))
    recorder = preview.EpisodePreview(tmp_path, control_fps=20)
    recorder.capture(_FakeRenderEnv(), "PHASE: SIMPLE IK", (30, 90, 200))
    artifacts = recorder.finalize(object())
    assert written == [("full_pipeline.gif", 1, 8), ("full_pipeline.mp4", 1, 20)]
    assert artifacts["video_mp4"].endswith("videos/full_pipeline.mp4")
    assert artifacts["video_gif"].endswith("videos/full_pipeline.gif")
    assert artifacts["final_state_multicam"].endswith("final_state_multicam.png")
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run pytest tests/faults/test_episode_preview.py -q
```

Expected: collection fails because `lerobot.faults.datagen.episode_preview` does not exist.

- [ ] **Step 3: Implement the shared preview module**

Move `_overlay_banner`, `_write_gif`, `_write_mp4`, `_render_looking_into_basket`, and `_final_proof_shot` from `smolvla_pipeline.py`. Add:

```python
class EpisodePreview:
    def __init__(self, output_dir: Path, control_fps: int) -> None:
        self.output_dir = Path(output_dir)
        self.control_fps = int(control_fps)
        self.frames: list[np.ndarray] = []

    def capture(self, env: Any, label: str, color: tuple[int, int, int]) -> None:
        raw = env.call("render") if hasattr(env, "call") else [env.envs[0].render()]
        self.frames.append(_overlay_banner(np.asarray(raw[0]), label, color))

    def finalize(self, rs_env: Any) -> dict[str, str | None]:
        if not self.frames:
            raise RuntimeError("cannot finalize an episode preview without frames")
        videos_dir = self.output_dir / "videos"
        gif_path = videos_dir / "full_pipeline.gif"
        mp4_path = videos_dir / "full_pipeline.mp4"
        gif_stride = max(1, len(self.frames) // 100)
        _write_gif(gif_path, self.frames[::gif_stride], fps=8)
        _write_mp4(mp4_path, self.frames, fps=self.control_fps)
        proof = _final_proof_shot(rs_env, self.output_dir / "final_state_multicam.png")
        return {"video_mp4": str(mp4_path), "video_gif": str(gif_path), "final_state_multicam": proof}
```

Update SmolVLA imports to consume shared helpers without changing its output paths.

- [ ] **Step 4: Run focused and SmolVLA tests**

Run:

```bash
uv run pytest tests/faults/test_episode_preview.py tests/faults/test_smolvla_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/faults/datagen/episode_preview.py src/lerobot/faults/datagen/smolvla_pipeline.py tests/faults/test_episode_preview.py
git commit -m "refactor: share episode preview rendering"
```

---

### Task 2: Save SimpleIK preview artifacts

**Files:**
- Modify: `src/lerobot/faults/datagen/controllers/simple_ik.py`
- Modify: `tests/faults/test_simple_ik_episode_logging.py`
- Modify: `tests/faults/test_datagen_final_review_contracts.py`

**Interfaces:**
- `run_simple_ik_episode_loop(..., on_post_step: Callable[[int, str], None] | None = None)`.
- The adapter owns `EpisodePreview` lifetime and finalizes it before `env.close()`.
- `EpisodeResult.details` includes returned artifact paths.

- [ ] **Step 1: Write failing callback and adapter-finalization tests**

Add a loop test:

```python
captured = []
facts = run_simple_ik_episode_loop(
    ...,
    on_post_step=lambda step, phase: captured.append((step, phase)),
)
assert captured
assert captured[0][0] == 0
```

Add an adapter test with a fake `EpisodePreview`:

```python
preview = MagicMock()
preview.finalize.return_value = {
    "video_mp4": ".../full_pipeline.mp4",
    "video_gif": ".../full_pipeline.gif",
    "final_state_multicam": ".../final_state_multicam.png",
}
assert result.details["video_mp4"].endswith("full_pipeline.mp4")
preview.finalize.assert_called_once_with(rs_env)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/faults/test_simple_ik_episode_logging.py tests/faults/test_datagen_final_review_contracts.py -q
```

Expected: FAIL because the loop rejects `on_post_step` and the adapter does not create/finalize a preview.

- [ ] **Step 3: Implement callback capture and finalization**

After each `env.step(action)` call:

```python
if on_post_step is not None:
    on_post_step(step, planner.phase_name)
```

In `SimpleIKControllerAdapter.run_episode`, instantiate the preview after reset, capture an initial frame, pass a callback that renders `PHASE: SIMPLE IK <phase> step=<step>`, finalize before environment close, and merge artifact paths into `EpisodeResult.details`. Structure result creation so all ordinary success/failure returns from the loop are finalized. Exceptions still close the environment and propagate.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/faults/test_simple_ik_episode_logging.py tests/faults/test_datagen_final_review_contracts.py tests/faults/test_episode_preview.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/faults/datagen/controllers/simple_ik.py tests/faults/test_simple_ik_episode_logging.py tests/faults/test_datagen_final_review_contracts.py
git commit -m "feat: save SimpleIK episode previews"
```

---

### Task 3: Controller-specific stock SmolVLA layout

**Files:**
- Modify: `src/lerobot/faults/datagen/recipe.py`
- Modify: `src/lerobot/faults/datagen/runner.py`
- Modify: `examples/faults/recipes/can_drop_datagen.json`
- Modify: `tests/faults/test_datagen_recipe.py`
- Modify: `tests/faults/test_datagen_runner.py`
- Modify: `HANDOFF_UNIFIED_DROP_DATAGEN.md`

**Interfaces:**
- `SmolVLARecipe.use_stock_layout: bool`.
- JSON field: `smolvla.use_stock_layout`.
- The existing layout-provider signature remains unchanged.

- [ ] **Step 1: Write failing recipe and runner tests**

Recipe assertion:

```python
recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
assert recipe.smolvla.use_stock_layout is True
```

Runner provider and assertions:

```python
def layout_provider(ctx):
    p = ctx.recipe.placement
    return {"kind": "stock" if p.xy_range_m == 0 and p.yaw_range_deg == (0.0, 0.0) else "random"}

assert [request_layout_by_controller[c] for c in simple_modes] == [{"kind": "random"}] * 2
assert [request_layout_by_controller[c] for c in smolvla_modes] == [{"kind": "stock"}] * 3
assert layout_provider_calls == 2
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/faults/test_datagen_recipe.py tests/faults/test_datagen_runner.py -q
```

Expected: FAIL because `use_stock_layout` is unknown and only one layout is resolved.

- [ ] **Step 3: Parse the explicit recipe flag**

Add `use_stock_layout` to `_SMOLVLA_KEYS` and `SmolVLARecipe`, require a JSON boolean in the unified loader, and add:

```json
"use_stock_layout": true
```

to `can_drop_datagen.json`.

- [ ] **Step 4: Resolve layouts per controller**

Use `dataclasses.replace`:

```python
randomized_layout = layout_fn(layout_ctx)
stock_recipe = replace(
    recipe,
    placement=replace(recipe.placement, xy_range_m=0.0, yaw_range_deg=(0.0, 0.0)),
)
stock_layout = layout_fn(replace(layout_ctx, recipe=stock_recipe))
layout_by_controller = {
    DatagenController.SIMPLE_IK: randomized_layout,
    DatagenController.SMOLVLA: stock_layout if recipe.smolvla.use_stock_layout else randomized_layout,
}
```

Set `EpisodeRequest.shared_layout = layout_by_controller[manifest.controller]`. Preserve `EpisodeResult.from_run(..., layout=request.shared_layout)` so manifests contain the applied pose.

- [ ] **Step 5: Update handoff contract**

Document that equal logical episode indices share seeds and init state, while layout pairing is within controller families only when `use_stock_layout` is enabled.

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run pytest tests/faults/test_datagen_recipe.py tests/faults/test_datagen_runner.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/lerobot/faults/datagen/recipe.py src/lerobot/faults/datagen/runner.py examples/faults/recipes/can_drop_datagen.json tests/faults/test_datagen_recipe.py tests/faults/test_datagen_runner.py HANDOFF_UNIFIED_DROP_DATAGEN.md
git commit -m "feat: use stock layouts for SmolVLA"
```

---

### Task 4: Full verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run the fault test suite**

```bash
uv run pytest tests/faults -q --tb=short
```

Expected: PASS.

- [ ] **Step 2: Check lint diagnostics and diff**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the pre-existing untracked watcher and intentional changes/commits remain.

- [ ] **Step 3: Run one live SimpleIK-only preview smoke**

Use a temporary recipe containing the two SimpleIK matrix rows and a fresh output directory. Confirm both episode directories contain the four expected artifacts and both dataset variants remain valid.

- [ ] **Step 4: Run one live stock-layout SmolVLA smoke when GPU time is available**

Use a temporary recipe containing one SmolVLA row. Confirm the manifest’s applied object poses match the selected stock LIBERO init state and no XY/yaw perturbation is introduced.
