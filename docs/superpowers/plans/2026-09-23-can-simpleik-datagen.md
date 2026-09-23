# Can-only SimpleIK datagen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate LIBERO-Object task-0 (can) episodes with randomized clutter poses, SimpleIK pick/place, optional one mid-air drop with `P(any drop)=q` uniform over eligible lift/transport steps, SimpleIK recovery, and a Tk viewer that shows camera plus the full event log — without writing a training dataset.

**Architecture:** Pure functions own recipe, layout, eligible-set, and drop sampling. A two-pass executor records a nominal SimpleIK trace (phase + can XY per step), builds `E`, then resets, reapplies the same layout, and replays with at most one inject at the sampled index. Tk UI subscribes to a fan-out event log (memory + JSONL + stdout).

**Tech Stack:** Python 3.12, numpy, pytest, Tk/PIL (viewer only), existing `SimpleIKRecoveryPlanner` + `MidAirDropFault` in `lerobot/src/lerobot/faults/`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-23-can-simpleik-datagen-design.md`
- Pick target is only `alphabet_soup_1`; basket `basket_1`; suite `libero_object` task 0
- No GraspGen, no cuRobo, no SmolVLA, no parquet recording, no seat-assist
- SimpleIK noise identity: `waypoint_noise_m=0`, `recovery_action_noise_std=0`, `arm_posture_noise_deg=0`, `speed_multiplier=1.0`
- Drop math: if `n=|E|>0`, `P(no drop)=1-q` else `i~Uniform(E)`; never independent per-step Bernoulli; at most one drop
- Keep-out: `max(min_drop_distance_from_basket_m, HARD_BASKET_KEEPOUT_M)` with `HARD_BASKET_KEEPOUT_M=0.22`
- Disable automatic `MidAirDropFault._should_trigger` (`probability=0`); inject only via `trigger_manual_drop` then `request_recovery`
- New Python files: `Copyright 2026 Gangelia. All rights reserved.` plus the Apache-2.0 header used by other fault files
- Do not commit unless the human explicitly asks; skip commit steps if not asked
- Run tests from `lerobot/` with `uv run pytest ...`

## File structure

| Path | Role |
| --- | --- |
| `src/lerobot/faults/datagen/recipe.py` | Load/validate JSON recipe |
| `src/lerobot/faults/datagen/drop_timing.py` | Phase aliases, eligible indices, drop sample |
| `src/lerobot/faults/datagen/layout.py` | Rejection-sample XY/yaw for all movable objects |
| `src/lerobot/faults/datagen/events.py` | Fan-out event log |
| `src/lerobot/faults/datagen/runtime.py` | LIBERO object discovery and quaternion helpers |
| `src/lerobot/faults/datagen/__init__.py` | Public exports |
| `examples/faults/recipes/can_simpleik_datagen.json` | Default recipe |
| `examples/faults/can_simpleik_viewer.py` | Tk camera + log pane (no LIBERO) |
| `examples/faults/run_can_simpleik_datagen.py` | LIBERO two-pass loop + viewer |
| `tests/faults/test_datagen_recipe.py` | Recipe tests |
| `tests/faults/test_datagen_drop_timing.py` | Drop/eligible tests |
| `tests/faults/test_datagen_layout.py` | Layout tests |
| `tests/faults/test_datagen_events.py` | Event log tests |

Do not modify `run_full_drop_recovery_pipeline.py` in this slice.

---

### Task 1: Recipe loader

**Files:**
- Create: `src/lerobot/faults/datagen/__init__.py`
- Create: `src/lerobot/faults/datagen/recipe.py`
- Create: `examples/faults/recipes/can_simpleik_datagen.json`
- Test: `tests/faults/test_datagen_recipe.py`

**Interfaces:**
- Consumes: JSON file on disk
- Produces: `class RecipeError(ValueError)`, `load_recipe(path: Path) -> DatagenRecipe`

`DatagenRecipe` fields (all required after load; nested dataclasses):

```python
@dataclass(frozen=True)
class PlacementRecipe:
    xy_range_m: float
    min_basket_clearance_m: float
    min_pairwise_clearance_m: float
    yaw_range_deg: tuple[float, float]
    max_attempts: int

@dataclass(frozen=True)
class DropRecipe:
    eligible_phases: tuple[str, ...]
    min_drop_distance_from_basket_m: float

@dataclass(frozen=True)
class SimpleIKRecipe:
    waypoint_noise_m: float
    recovery_action_noise_std: float
    arm_posture_noise_deg: float
    speed_multiplier: float

@dataclass(frozen=True)
class DatagenRecipe:
    q: float
    object_name: str
    basket_name: str
    placement: PlacementRecipe
    drop: DropRecipe
    simple_ik: SimpleIKRecipe
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/faults/test_datagen_recipe.py
from pathlib import Path

import pytest

from lerobot.faults.datagen.recipe import RecipeError, load_recipe


def test_load_recipe_reads_q_and_object(tmp_path: Path):
    p = tmp_path / "r.json"
    p.write_text(
        '{"q": 0.5, "object_name": "alphabet_soup_1", "basket_name": "basket_1",'
        '"placement": {"xy_range_m": 0.12, "min_basket_clearance_m": 0.35,'
        '"min_pairwise_clearance_m": 0.08, "yaw_range_deg": [-180, 180], "max_attempts": 200},'
        '"drop": {"eligible_phases": ["lift", "to_container"], "min_drop_distance_from_basket_m": 0.30},'
        '"simple_ik": {"waypoint_noise_m": 0.0, "recovery_action_noise_std": 0.0,'
        '"arm_posture_noise_deg": 0.0, "speed_multiplier": 1.0}}'
    )
    r = load_recipe(p)
    assert r.q == 0.5
    assert r.object_name == "alphabet_soup_1"
    assert r.drop.eligible_phases == ("lift", "to_container")


def test_load_recipe_missing_file():
    with pytest.raises(RecipeError, match="not found"):
        load_recipe(Path("/no/such/recipe.json"))


@pytest.mark.parametrize("q", [-0.01, 1.01])
def test_load_recipe_rejects_q_out_of_range(tmp_path: Path, q: float):
    p = tmp_path / "r.json"
    p.write_text(
        '{"q": %s, "object_name": "alphabet_soup_1", "basket_name": "basket_1",'
        '"placement": {"xy_range_m": 0.12, "min_basket_clearance_m": 0.35,'
        '"min_pairwise_clearance_m": 0.08, "yaw_range_deg": [-180, 180], "max_attempts": 1},'
        '"drop": {"eligible_phases": ["lift"], "min_drop_distance_from_basket_m": 0.3},'
        '"simple_ik": {"waypoint_noise_m": 0.0, "recovery_action_noise_std": 0.0,'
        '"arm_posture_noise_deg": 0.0, "speed_multiplier": 1.0}}' % q
    )
    with pytest.raises(RecipeError, match="q"):
        load_recipe(p)


def test_load_recipe_requires_object_name(tmp_path: Path):
    p = tmp_path / "r.json"
    p.write_text('{"q": 0.0, "basket_name": "basket_1"}')
    with pytest.raises(RecipeError, match="object_name"):
        load_recipe(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faults/test_datagen_recipe.py -v`

Expected: FAIL with `ModuleNotFoundError` or `cannot import name load_recipe`

- [ ] **Step 3: Write minimal implementation**

`recipe.py`: `json.loads`, require keys, validate `0<=q<=1`, `object_name` non-empty, `speed_multiplier>0`, `xy_range_m>=0`, `max_attempts>=1`, `len(yaw_range_deg)==2` and `hi>=lo`, `eligible_phases` non-empty. Raise `RecipeError` (not raw `KeyError`) for missing file (`FileNotFoundError`) and validation.

Default JSON at `examples/faults/recipes/can_simpleik_datagen.json` matching the spec example.

`__init__.py` exports `load_recipe`, `DatagenRecipe`, `RecipeError`.

- [ ] **Step 4: Run tests and make sure they pass**

Run: `uv run pytest tests/faults/test_datagen_recipe.py -v`

Expected: PASS

- [ ] **Step 5: Commit only if the human asked**

---

### Task 2: Eligible set and drop sampler

**Files:**
- Create: `src/lerobot/faults/datagen/drop_timing.py`
- Test: `tests/faults/test_datagen_drop_timing.py`

**Interfaces:**
- Consumes: `HARD_BASKET_KEEPOUT_M` from `lerobot.faults.recovery.midair_drop`
- Produces:
  - `canonicalize_phase(name: str) -> str` (`to_container` → `to_basket_hover`; other names unchanged)
  - `keepout_m(min_drop_distance_from_basket_m: float) -> float` = `max(min_drop, HARD_BASKET_KEEPOUT_M)`
  - `TraceFrame` dataclass: `step: int`, `phase: str`, `object_xy: np.ndarray` shape (2,), `basket_xy: np.ndarray` shape (2,), `grasped: bool`
  - `eligible_indices(frames: list[TraceFrame], eligible_phases: tuple[str, ...], min_drop_distance_from_basket_m: float) -> list[int]`
  - `DropDecision` dataclass: `drop: bool`, `step: int | None`, `reason: str` (`no_eligible_frames` | `skipped_q` | `injected`)
  - `sample_drop(q: float, eligible_steps: list[int], rng: np.random.Generator) -> DropDecision`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from lerobot.faults.datagen.drop_timing import (
    DropDecision,
    TraceFrame,
    eligible_indices,
    sample_drop,
)


def _frame(step, phase, obj_xy, basket_xy=(0.0, 0.0)):
    return TraceFrame(
        step=step,
        phase=phase,
        object_xy=np.array(obj_xy, dtype=np.float64),
        basket_xy=np.array(basket_xy, dtype=np.float64),
    )


def test_eligible_excludes_keepout_and_non_carry_phases():
    frames = [
        _frame(0, "descend_grasp", (0.5, 0.0)),
        _frame(1, "lift", (0.5, 0.0)),
        _frame(2, "to_basket_hover", (0.10, 0.0)),  # 0.10 m < 0.30 keep-out
        _frame(3, "to_container", (0.40, 0.0)),
        _frame(4, "open_place", (0.40, 0.0)),
    ]
    e = eligible_indices(frames, ("lift", "to_container"), 0.30)
    assert e == [1, 3]


def test_sample_drop_q0_never_drops():
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = sample_drop(0.0, [10, 20, 30], rng)
        assert d.drop is False
        assert d.step is None
        assert d.reason == "skipped_q"


def test_sample_drop_q1_always_in_e():
    rng = np.random.default_rng(1)
    e = [10, 20, 30]
    for _ in range(50):
        d = sample_drop(1.0, e, rng)
        assert d.drop is True
        assert d.step in e
        assert d.reason == "injected"


def test_sample_drop_empty_e():
    d = sample_drop(1.0, [], np.random.default_rng(0))
    assert d == DropDecision(drop=False, step=None, reason="no_eligible_frames")


def test_sample_drop_empirical_rate_and_uniform():
    rng = np.random.default_rng(42)
    e = [1, 2, 3, 4]
    n = 4000
    drops = 0
    counts = {s: 0 for s in e}
    for _ in range(n):
        d = sample_drop(0.5, e, rng)
        if d.drop:
            drops += 1
            counts[d.step] += 1
    rate = drops / n
    assert 0.47 < rate < 0.53
    for s in e:
        assert counts[s] > 0
    assert max(counts.values()) / min(counts.values()) < 1.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faults/test_datagen_drop_timing.py -v`

Expected: FAIL import

- [ ] **Step 3: Write minimal implementation**

`sample_drop`: if `eligible_steps` empty → `no_eligible_frames` (ignore `q`). Else if `rng.random() >= q` → `skipped_q`. Else `step = int(eligible_steps[int(rng.integers(0, len(eligible_steps)))])`.

`eligible_indices`: for each frame, `phase = canonicalize_phase(frame.phase)`; allowed = `{canonicalize_phase(p) for p in eligible_phases}`; include `frame.step` if `frame.grasped`, `phase in allowed`, and `hypot(object_xy-basket_xy) >= keepout_m(...)`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/faults/test_datagen_drop_timing.py -v`

Expected: PASS

- [ ] **Step 5: Commit only if asked**

---

### Task 3: Layout sampler (pure)

**Files:**
- Create: `src/lerobot/faults/datagen/layout.py`
- Test: `tests/faults/test_datagen_layout.py`

**Interfaces:**
- Consumes: `PlacementRecipe`
- Produces:
  - `ObjectPose2d` dataclass: `name: str`, `xy: np.ndarray` (2,), `yaw_rad: float`
  - `sample_layout(*, objects: list[ObjectPose2d], basket_xy: np.ndarray, table_xy_lim: float, rng, placement: PlacementRecipe) -> list[ObjectPose2d] | None`

Rules: for each object independently sample `dx,dy ~ Unif[-xy_range, xy_range]` added to **reset** `xy`; `yaw ~ Unif[deg2rad(lo), deg2rad(hi)]`. Reject candidate if `|xy|` any coord > `table_xy_lim`, or distance to `basket_xy` < `min_basket_clearance_m`, or distance to any already-accepted object < `min_pairwise_clearance_m`. Retry whole layout up to `max_attempts`. Return `None` on failure (do not return overlapping poses).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from lerobot.faults.datagen.layout import ObjectPose2d, sample_layout
from lerobot.faults.datagen.recipe import PlacementRecipe


def _place(**kwargs):
    base = dict(
        xy_range_m=0.12,
        min_basket_clearance_m=0.35,
        min_pairwise_clearance_m=0.08,
        yaw_range_deg=(-180.0, 180.0),
        max_attempts=200,
    )
    base.update(kwargs)
    return PlacementRecipe(**base)


def test_sample_layout_keeps_away_from_basket():
    objs = [ObjectPose2d("can", np.array([0.4, 0.0]), 0.0)]
    basket = np.array([0.0, 0.0])
    rng = np.random.default_rng(0)
    out = sample_layout(
        objects=objs, basket_xy=basket, table_xy_lim=0.7, rng=rng, placement=_place()
    )
    assert out is not None
    dist = float(np.linalg.norm(out[0].xy - basket))
    assert dist >= 0.35


def test_sample_layout_rejects_overlap():
    objs = [
        ObjectPose2d("a", np.array([0.5, 0.0]), 0.0),
        ObjectPose2d("b", np.array([0.5, 0.0]), 0.0),
    ]
    out = sample_layout(
        objects=objs,
        basket_xy=np.array([-1.0, -1.0]),
        table_xy_lim=0.7,
        rng=np.random.default_rng(1),
        placement=_place(xy_range_m=0.0, min_pairwise_clearance_m=0.08, max_attempts=5),
    )
    assert out is None


def test_sample_layout_failed_after_max_attempts():
    objs = [ObjectPose2d("can", np.array([0.0, 0.0]), 0.0)]
    out = sample_layout(
        objects=objs,
        basket_xy=np.array([0.0, 0.0]),
        table_xy_lim=0.7,
        rng=np.random.default_rng(0),
        placement=_place(xy_range_m=0.01, min_basket_clearance_m=0.35, max_attempts=8),
    )
    assert out is None
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/faults/test_datagen_layout.py -v`

- [ ] **Step 3: Implement `sample_layout`**

Inner loop: copy reset poses; place objects in list order; on pairwise/basket/table fail, restart full layout (increment attempt).

- [ ] **Step 4: Tests pass**

Run: `uv run pytest tests/faults/test_datagen_layout.py -v`

- [ ] **Step 5: Commit only if asked**

---

### Task 4: Event log fan-out

**Files:**
- Create: `src/lerobot/faults/datagen/events.py`
- Test: `tests/faults/test_datagen_events.py`

**Interfaces:**
- Produces:
  - `LogEvent` dataclass: `ts: float`, `step: int | None`, `kind: str`, `message: str`, `payload: dict`
  - `class DatagenEventLog`:
    - `__init__(self, jsonl_path: Path | None = None, *, stdout: bool = True)`
    - `add_listener(self, fn: Callable[[LogEvent], None]) -> None`
    - `emit(self, kind: str, message: str, *, step: int | None = None, **payload) -> LogEvent`
    - `close(self) -> None`

`emit` appends JSONL (`kind`, `message`, `step`, `ts`, `payload`), prints `f"[{kind}] {message}"` if stdout, calls listeners. Use `time.time()` for `ts`.

Minimum `kind` strings used later: `recipe`, `layout_ok`, `layout_failed`, `plan_ok`, `plan_failed`, `eligible`, `drop`, `impulse`, `recovery`, `drop_landed_in_basket`, `recovery_failed`, `episode_end`

- [ ] **Step 1: Failing tests**

```python
from pathlib import Path

from lerobot.faults.datagen.events import DatagenEventLog


def test_emit_writes_jsonl_and_listeners(tmp_path: Path, capsys):
    path = tmp_path / "e.jsonl"
    seen = []
    log = DatagenEventLog(path, stdout=True)
    log.add_listener(seen.append)
    ev = log.emit("drop", "injected", step=12, q=0.5)
    log.close()
    assert ev.kind == "drop"
    assert ev.step == 12
    assert seen[0].message == "injected"
    text = path.read_text()
    assert "injected" in text
    assert "drop" in capsys.readouterr().out
```

- [ ] **Step 2: Run fail** `uv run pytest tests/faults/test_datagen_events.py -v`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Pass**
- [ ] **Step 5: Commit only if asked**

---

### Task 5: Tk viewer (camera + log pane)

**Files:**
- Create: `examples/faults/can_simpleik_viewer.py`
- Test: `tests/faults/test_datagen_events.py` (add HUD string helper test in `events.py` or viewer module)

**Interfaces:**
- Consumes: `DatagenEventLog.add_listener`, RGB `uint8` HxWx3
- Produces: `class DatagenViewer` in `can_simpleik_viewer.py`:
  - `quit_requested: bool`
  - `pause: bool`
  - `next_episode: bool` (edge-triggered; runner clears)
  - `force_q: float | None` (`None`, `0.0`, or `1.0`)
  - `pump()`, `show(frame_rgb, hud: str)`, `append_log(event: LogEvent)`, `clear_log()`, `destroy()`
  - Classmethod `try_create() -> DatagenViewer | None` — return `None` if Tk/ImageTk import fails

Layout: `tk.Frame` left = image `Label`; right = `tk.Text` state=disabled, wrap=word, yscroll. Controls row: Next, Pause, q=0, q=1, Quit. Bind `q`/`Escape` quit.

Log pane: `append_log` inserts `f"{event.ts:.1f} step={event.step} [{event.kind}] {event.message}\n"`. Auto-scroll to end unless the user scroll position is not at the bottom (`yview()[1] < 0.999`).

HUD is **not** the log. `show` draws overlay on the numpy frame (reuse `_overlay` pattern from `run_manual_drop_recovery.py`: black banner + `hud` text) then displays.

- [ ] **Step 1: Write a unit test for auto-scroll helper (no Tk)**

Put in `src/lerobot/faults/datagen/events.py`:

```python
def should_autoscroll(yview_hi: float) -> bool:
    return float(yview_hi) >= 0.999
```

Test:

```python
from lerobot.faults.datagen.events import should_autoscroll

def test_should_autoscroll():
    assert should_autoscroll(1.0) is True
    assert should_autoscroll(0.5) is False
```

- [ ] **Step 2: Fail then implement `should_autoscroll`**
- [ ] **Step 3: Implement `DatagenViewer`** using the test for scroll policy. If Tk import fails, `try_create` returns None.
- [ ] **Step 4: `uv run pytest tests/faults/test_datagen_events.py -v` PASS**
- [ ] **Step 5: Commit only if asked**

---

### Task 6: Two-pass runner (LIBERO)

**Files:**
- Create: `examples/faults/run_can_simpleik_datagen.py`
- Modify: `src/lerobot/faults/datagen/__init__.py` exports

**Interfaces:**
- Consumes: `load_recipe`, `sample_layout`, `eligible_indices`, `sample_drop`, `DatagenEventLog`, `DatagenViewer.try_create`, `SimpleIKRecoveryPlanner`, `DropRecoveryEnvWrapper`, `MidAirDropFault.trigger_manual_drop`, `request_recovery`, `offset_object_xy_on_table` (and a small yaw setter if free-joint `q` has 7 dims — set `q[0], q[1]` plus optional yaw via quaternion on `q[3:7]`; if yaw write is unsafe, skip yaw and log `yaw_skipped`).
- Produces: CLI `main(argv) -> int`

**Two-pass (required for exact `n`):** SimpleIK is closed-loop so a spline length is not `n`.

1. `reset(seed)`
2. Read all `rs_env.objects` keys except `basket_name` and the robot; snapshot reset XY; `sample_layout`; apply offsets; log `layout_ok` or skip with `layout_failed`
3. **Pass A:** `SimpleIKRecoveryPlanner` with identity noise; `plan(...)` from live EEF/can/basket; step until `planner.done` or cap (e.g. 800). Each step append `TraceFrame(step, planner.phase_name, can_xy, basket_xy)`. Gripper/actions from `planner.next_action`. Do **not** drop. If plan never starts (`plan` raises / empty waypoints) log `plan_failed` and skip.
4. `E = eligible_indices(...)`; log `eligible` with `n=len(E)`
5. `decision = sample_drop(q_eff, E, rng)` where `q_eff = viewer.force_q if set else recipe.q`; log `drop`
6. **Pass B:** `reset` same init_state; re-apply **stored** layout poses (not a new sample); new planner (identity noise); if `decision.drop`, on `step == decision.step` call `trigger_manual_drop` then `request_recovery(consume_first_action=True)` and thereafter take recovery actions from the wrapper/fault. Else run planner to place. `seat_assist_enabled=False`. `probability=0` so `_should_trigger` never fires.
7. If can XY in basket after drop before regrasp, `emit("drop_landed_in_basket", ...)`
8. Viewer: each physics step `pump`, `show(frame, hud)`, log listener already appending. Pause: spin `pump` until unpaused.

CLI:

```
uv run python examples/faults/run_can_simpleik_datagen.py \
  --recipe examples/faults/recipes/can_simpleik_datagen.json \
  --seed 0 \
  --episodes 1 \
  --output outputs/can_simpleik_datagen
```

`--recipe` required. `--headless` skips Tk. JSONL at `{output}/events.jsonl`.

Fault config for wrapper:

```python
FaultInjectionConfig(
    enabled=True,
    type="midair_drop",
    probability=0.0,
    t_min=0,
    t_max=10_000,
    object_name=recipe.object_name,
    basket_name=recipe.basket_name,
    seat_assist_enabled=False,
    waypoint_noise_m=0.0,
    recovery_action_noise_std=0.0,
    arm_posture_noise_deg=0.0,
    speed_multiplier_min=1.0,
    speed_multiplier_max=1.0,
    post_grasp_delay_steps=0,
    drop_xy_band_min=None,
    drop_xy_band_max=None,
    seed=seed,
    log_path=output / "fault_events.jsonl",
)
```

List objects: `list(getattr(rs_env, "objects", {}) or [])` excluding `basket_name`. If `objects` missing, only offset `recipe.object_name`.

Apply pose: `dx, dy = new_xy - reset_xy`; `offset_object_xy_on_table(rs, name, dx, dy, basket_name=..., min_basket_xy=0.0)` because keep-out already enforced in `sample_layout` (passing 0.0 avoids double-reject after a legal sample). If offset returns False, log and skip episode.

- [ ] **Step 1: Smoke-test without LIBERO** — add `tests/faults/test_datagen_runner_helpers.py`:

```python
from lerobot.faults.datagen.drop_timing import TraceFrame, eligible_indices, sample_drop
import numpy as np

def test_two_pass_uses_pass_a_trace_not_live_resample():
    frames = [
        TraceFrame(0, "lift", np.array([0.5, 0.0]), np.array([0.0, 0.0])),
        TraceFrame(1, "to_basket_hover", np.array([0.4, 0.0]), np.array([0.0, 0.0])),
    ]
    e = eligible_indices(frames, ("lift", "to_container"), 0.30)
    d = sample_drop(1.0, e, np.random.default_rng(0))
    assert d.step in e
```

This locks the math the runner must call. Do not import LIBERO in this test.

- [ ] **Step 2: Fail/pass that helper test**
- [ ] **Step 3: Implement `run_can_simpleik_datagen.py`** as specified. Env setup copy from `run_manual_drop_recovery.py`: `MUJOCO_GL=egl`, `LiberoEnv(task="libero_object", task_ids=[0], ...)`, `make_env`, unwrap, `DropRecoveryEnvWrapper`.
- [ ] **Step 4: Run CPU tests**

Run: `uv run pytest tests/faults/test_datagen_recipe.py tests/faults/test_datagen_drop_timing.py tests/faults/test_datagen_layout.py tests/faults/test_datagen_events.py tests/faults/test_datagen_runner_helpers.py tests/faults/test_midair_drop_fault.py -q`

Expected: all PASS. Existing mid-air tests unchanged.

- [ ] **Step 5: Live check (manual, display)**

Run: `uv run python examples/faults/run_can_simpleik_datagen.py --recipe examples/faults/recipes/can_simpleik_datagen.json --seed 0 --episodes 3 --output outputs/can_simpleik_datagen`

Confirm: clutter moved, only can grasped, log pane shows layout/eligible/drop lines, at most one drop, no drop when can–basket XY < 0.30 m, HUD ≠ log pane.

- [ ] **Step 6: Commit only if asked**

---

## Spec coverage

| Spec item | Task |
| --- | --- |
| JSON recipe, no silent `q` | 1 |
| Uniform drop / `q` / empty `E` | 2 |
| Keep-out + phases lift/to_container | 2 |
| All-object layout, can-only pick | 3 + 6 |
| Event log + JSONL + stdout | 4 |
| Tk camera + scrollable logs | 5 |
| SimpleIK both passes, no extra noise, no auto trigger | 6 |
| No dataset recording | 6 (not calling `FaultRecoveryDatasetLogger`) |
| Error skip layout/plan | 6 |

## Type names (do not drift)

`DatagenRecipe`, `PlacementRecipe`, `DropRecipe`, `SimpleIKRecipe`, `RecipeError`, `load_recipe`, `TraceFrame`, `eligible_indices`, `DropDecision`, `sample_drop`, `ObjectPose2d`, `sample_layout`, `LogEvent`, `DatagenEventLog`, `should_autoscroll`, `DatagenViewer`.
