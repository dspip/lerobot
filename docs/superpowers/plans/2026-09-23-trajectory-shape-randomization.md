# Coherent Trajectory Shape and Speed Randomization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-step execution noise with seeded, coherent via-point path variation and one speed multiplier per episode for nominal pickup/place and post-drop recovery.

**Architecture:** A focused motion-profile module samples deterministic nominal/recovery offsets, posture biases, and shared episode speed. `SimpleIKRecoveryPlanner` consumes those values, inserts safe pickup/transport via-points, and exposes the exact carry polyline it executes. The path-drop sampler clips that polyline against the basket keep-out and samples a single drop point uniformly over eligible arc length.

**Tech Stack:** Python 3.12, NumPy, pytest, LIBERO/robosuite, Tk/Pillow viewer.

## Global Constraints

- Preserve exact grasp, lift, basket-hover, and release endpoints.
- Apply path-shape variation to both nominal motion and post-drop recovery.
- Use `pickup_via_offset_m=0.03` and `transport_via_offset_m=0.06` defaults.
- Sample one speed per episode uniformly from `[0.8, 1.2]`; nominal and recovery share it.
- Keep per-step `recovery_action_noise_std=0` in this generator.
- Keep the one-shot arm posture range at ±3 degrees when randomization is enabled.
- When randomization is disabled, effective offsets and posture are zero and speed is `1.0`.
- Preserve one-pass generation, exact run-level `P(drop)=q`, at most one drop, and the 0.20 m keep-out.
- Motion-profile, layout, and drop RNG streams must be independent.
- Do not commit, push, merge, or discard changes; this branch remains uncommitted unless the user explicitly changes that instruction.

---

### Task 1: Recipe schema and deterministic episode motion profiles

**Files:**
- Create: `src/lerobot/faults/datagen/motion_profile.py`
- Create: `tests/faults/test_datagen_motion_profile.py`
- Modify: `src/lerobot/faults/datagen/recipe.py`
- Modify: `tests/faults/test_datagen_recipe.py`
- Modify: `examples/faults/recipes/can_simpleik_datagen.json`

**Interfaces:**
- Produces:
  - `SimpleIKRecipe.trajectory_randomization_enabled: bool`
  - `SimpleIKRecipe.pickup_via_offset_m: float`
  - `SimpleIKRecipe.transport_via_offset_m: float`
  - `SimpleIKRecipe.arm_posture_noise_deg: float`
  - `SimpleIKRecipe.speed_multiplier_range: tuple[float, float]`
  - `MotionLegProfile(pickup_offset_xy_m, transport_offset_m, posture_bias_rad)`
  - `EpisodeMotionProfile(speed_multiplier, nominal, recovery)`
  - `sample_episode_motion_profile(recipe: SimpleIKRecipe, episode_seed: int) -> EpisodeMotionProfile`

- [ ] **Step 1: Replace the old recipe tests with the new explicit schema**

Add tests that load:

```python
"simple_ik": {
    "trajectory_randomization_enabled": True,
    "pickup_via_offset_m": 0.03,
    "transport_via_offset_m": 0.06,
    "arm_posture_noise_deg": 3.0,
    "speed_multiplier_range": [0.8, 1.2],
}
```

Assert disabled mode remains legal with non-zero configured magnitudes, and add parameterized rejection tests for:

```python
[
    ("pickup_via_offset_m", -0.01),
    ("transport_via_offset_m", -0.01),
    ("arm_posture_noise_deg", -1.0),
]
```

Add separate malformed speed-range cases `[]`, `[0.8]`, `[1.2, 0.8]`, and `[0.0, 1.2]`.

- [ ] **Step 2: Run the recipe tests and confirm red**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/faults/test_datagen_recipe.py -q
```

Expected: failures because the dataclass and loader still use `noise_enabled`, scalar `speed_multiplier`, and execution-noise fields.

- [ ] **Step 3: Implement the recipe dataclass and strict validation**

Use:

```python
@dataclass(frozen=True)
class SimpleIKRecipe:
    trajectory_randomization_enabled: bool
    pickup_via_offset_m: float
    transport_via_offset_m: float
    arm_posture_noise_deg: float
    speed_multiplier_range: tuple[float, float]
```

Require every field. Validate non-negative offsets/posture, exactly two positive speed values, and `maximum >= minimum`. Remove the generator recipe's `waypoint_noise_m` and `recovery_action_noise_std` keys rather than silently accepting obsolete behavior.

- [ ] **Step 4: Write failing motion-profile tests**

Cover:

```python
def test_disabled_profile_is_identity()
def test_profile_is_reproducible_from_episode_seed()
def test_different_seeds_change_the_profile()
def test_offsets_stay_inside_configured_bounds()
def test_speed_stays_inside_configured_range()
def test_nominal_and_recovery_have_distinct_offsets()
def test_speed_is_shared_by_nominal_and_recovery()
def test_profile_sampling_does_not_consume_caller_rng()
```

For pickup offsets, assert Euclidean norm is at most `pickup_via_offset_m`. For transport, assert signed magnitude is at most `transport_via_offset_m`. For posture, assert every component lies within ±`deg2rad(arm_posture_noise_deg)`.

- [ ] **Step 5: Run the new profile tests and confirm red**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/faults/test_datagen_motion_profile.py -q
```

Expected: collection failure because `motion_profile.py` does not exist.

- [ ] **Step 6: Implement profile sampling with a dedicated seeded stream**

Use immutable, JSON-friendly tuples:

```python
@dataclass(frozen=True)
class MotionLegProfile:
    pickup_offset_xy_m: tuple[float, float]
    transport_offset_m: float
    posture_bias_rad: tuple[float, float, float]

@dataclass(frozen=True)
class EpisodeMotionProfile:
    speed_multiplier: float
    nominal: MotionLegProfile
    recovery: MotionLegProfile
```

Create the RNG internally from a stable stream:

```python
rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 0x4D4F544E]))
```

Sample pickup points uniformly by area inside a disk:

```python
radius = max_radius * np.sqrt(rng.random())
angle = rng.uniform(-np.pi, np.pi)
offset = (radius * np.cos(angle), radius * np.sin(angle))
```

Sample signed transport offsets uniformly from `[-maximum, maximum]`, posture components uniformly from the configured angular range, and speed uniformly from the configured range. Return exact identity values when disabled without consuming random draws.

- [ ] **Step 7: Update the JSON recipe**

Replace `simple_ik` with:

```json
"simple_ik": {
  "trajectory_randomization_enabled": false,
  "pickup_via_offset_m": 0.03,
  "transport_via_offset_m": 0.06,
  "arm_posture_noise_deg": 3.0,
  "speed_multiplier_range": [0.8, 1.2]
}
```

- [ ] **Step 8: Run Task 1 tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  tests/faults/test_datagen_recipe.py \
  tests/faults/test_datagen_motion_profile.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Review checkpoint**

Inspect `git diff` for Task 1 only. Confirm there are no edits outside the five listed files and do not commit.

---

### Task 2: Shared trajectory geometry and planner via-points

**Files:**
- Create: `src/lerobot/faults/recovery/trajectory.py`
- Create: `tests/faults/test_trajectory_geometry.py`
- Modify: `src/lerobot/faults/recovery/planner.py`
- Modify: `tests/faults/test_planner.py`

**Interfaces:**
- Consumes: `MotionLegProfile` values as primitive constructor arguments.
- Produces:
  - `PathSegment(name: str, start_xyz: tuple[float, float, float], end_xyz: tuple[float, float, float])`
  - `CarryPath(segments: tuple[PathSegment, ...])`
  - `build_pickup_via(...) -> np.ndarray`
  - `build_carry_path(...) -> CarryPath`
  - `SimpleIKRecoveryPlanner.carry_path: CarryPath | None`

- [ ] **Step 1: Write failing trajectory-geometry tests**

Test these exact behaviors:

```python
def test_zero_pickup_offset_preserves_direct_approach()
def test_pickup_via_uses_safe_hover_height()
def test_transport_via_is_perpendicular_to_direct_route()
def test_positive_and_negative_offsets_bend_opposite_directions()
def test_carry_path_preserves_lift_and_basket_endpoints()
def test_via_resolution_rejects_workspace_violation_instead_of_clamping()
def test_via_resolution_rejects_basket_keepout_violation()
def test_via_resolution_shrinks_magnitude_without_flipping_direction()
def test_no_legal_via_returns_straight_path_after_32_candidates()
```

Represent all public dataclass coordinates as tuples; convert to arrays only inside geometry functions.

- [ ] **Step 2: Run geometry tests and confirm red**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/faults/test_trajectory_geometry.py -q
```

Expected: collection failure because `trajectory.py` does not exist.

- [ ] **Step 3: Implement focused geometry primitives**

Define:

```python
@dataclass(frozen=True)
class PathSegment:
    name: str
    start_xyz: tuple[float, float, float]
    end_xyz: tuple[float, float, float]

    @property
    def length(self) -> float: ...

@dataclass(frozen=True)
class CarryPath:
    segments: tuple[PathSegment, ...]
```

Use phase names `"lift"`, `"to_basket_via"`, and `"to_basket_hover"`. `build_carry_path` returns lift plus two transport segments when a non-zero legal transport offset exists; otherwise it returns lift plus the direct basket segment. Resolve an illegal requested offset by testing 32 deterministic scales from full magnitude toward zero without changing its sign/direction. Return the requested offset, resolved offset, and `fallback: bool` alongside the geometry.

- [ ] **Step 4: Add failing planner tests**

Add:

```python
def test_pickup_via_is_inserted_before_approach_hover()
def test_transport_via_is_inserted_between_lift_and_basket()
def test_grasp_and_place_endpoints_match_identity_planner()
def test_carry_path_is_refreshed_from_live_object_at_lift_entry()
def test_carry_path_matches_the_waypoints_the_planner_executes()
def test_zero_offsets_preserve_existing_phase_sequence()
def test_speed_multiplier_changes_actions_but_not_waypoint_geometry()
```

In the live-object test, move `object_pos` before entering lift and assert `carry_path.segments[0].start_xyz` reflects the moved pose.

- [ ] **Step 5: Run planner tests and confirm red**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  tests/faults/test_planner.py \
  tests/faults/test_trajectory_geometry.py -q
```

Expected: new planner tests fail because constructor arguments and `carry_path` are absent.

- [ ] **Step 6: Wire via-points into `SimpleIKRecoveryPlanner`**

Add constructor inputs:

```python
pickup_via_offset_xy_m: tuple[float, float] = (0.0, 0.0)
transport_via_offset_m: float = 0.0
trajectory_seed: int | None = None
```

Keep `waypoint_noise_m` only for backward compatibility with other callers; this generator passes zero. Insert `"pickup_via"` before `"approach_hover"` when its offset is non-zero. On entering `"lift"`, rebuild lift and transport targets from the live object pose, update `"to_basket_via"`/`"to_basket_hover"` waypoints, and set `self._carry_path` to that exact geometry.

Extend object-centric arrival logic to `"to_basket_via"`:

```python
if wp.name in ("to_basket_via", "to_basket_hover", "open_place"):
    target_object_xy = ...
    return np.linalg.norm(object_pos[:2] - target_object_xy), self.basket_xy_tol
```

Do not allow stuck-advance from either transport phase.

- [ ] **Step 7: Run Task 2 tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  tests/faults/test_planner.py \
  tests/faults/test_trajectory_geometry.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Review checkpoint**

Confirm the planner's existing direct path is byte-for-byte equivalent in phase endpoints when offsets are zero. Do not commit.

---

### Task 3: Uniform drop sampling over a bent carry polyline

**Files:**
- Modify: `src/lerobot/faults/datagen/path_drop.py`
- Modify: `tests/faults/test_datagen_path_drop.py`

**Interfaces:**
- Consumes: `CarryPath` and current planner phase/object position.
- Produces:
  - `EligibleArcPiece(segment_name: str, t0: float, t1: float, length_m: float)`
  - `EligiblePath(pieces: tuple[EligibleArcPiece, ...], total_m: float)`
  - `PathTrigger(segment_name: str, target_t: float, segment_order: int)`
  - `eligible_path(path: CarryPath, basket_xy, keepout_m) -> EligiblePath`
  - `sample_path_drop(q, path, rng) -> tuple[DropDecision, PathTrigger | None]`

- [ ] **Step 1: Replace straight-path tests with polyline tests**

Cover:

```python
def test_vertical_lift_outside_keepout_is_fully_eligible()
def test_segment_entering_keepout_is_clipped_at_circle_intersection()
def test_segment_outside_to_outside_crossing_disk_produces_two_pieces()
def test_via_path_total_is_sum_of_clipped_piece_lengths()
def test_sampling_frequency_is_proportional_to_piece_length()
def test_q_zero_never_drops()
def test_q_one_drops_when_any_piece_exists()
def test_empty_eligible_path_skips()
def test_trigger_uses_projection_progress_on_active_segment()
def test_trigger_fires_when_a_step_crosses_target_progress()
def test_trigger_does_not_fire_on_a_different_segment()
```

- [ ] **Step 2: Run path-drop tests and confirm red**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/faults/test_datagen_path_drop.py -q
```

Expected: failures because the implementation still assumes one vertical and one monotone straight segment.

- [ ] **Step 3: Implement line-circle clipping**

For each segment parameterized by `p(t)=p0+t*(p1-p0)`, solve:

```python
a = dot(dxy, dxy)
b = 2 * dot(p0xy - basket_xy, dxy)
c = dot(p0xy - basket_xy, p0xy - basket_xy) - keepout_m**2
```

Split `[0, 1]` at real roots, test each interval midpoint, and retain intervals whose XY distance is at least the keep-out. Vertical segments have `a == 0` and are either fully eligible or fully excluded.

- [ ] **Step 4: Implement weighted arc sampling and projection triggers**

Choose an arc distance uniformly from `[0, total_m]`, locate its piece by cumulative length, and convert to the segment's target `t`. Compute runtime progress using clamped projection:

```python
t = np.clip(
    np.dot(position - start, end - start) / np.dot(end - start, end - start),
    0.0,
    1.0,
)
```

Fire only on the selected segment when `t >= target_t`; if execution has already advanced to a later carry segment, fire immediately at its first frame as the nearest remaining eligible point.

- [ ] **Step 5: Run Task 3 tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/faults/test_datagen_path_drop.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Review checkpoint**

Use a 10,000-draw deterministic statistical test and confirm observed per-piece frequencies are within ±3 percentage points of expected arc-length shares. Do not commit.

---

### Task 4: Explicit recovery motion-profile plumbing

**Files:**
- Modify: `src/lerobot/faults/recovery/midair_drop.py`
- Modify: `src/lerobot/faults/config.py`
- Modify: `tests/faults/test_midair_drop_fault.py`
- Modify: `tests/faults/test_side_grasp_and_drop_gates.py`

**Interfaces:**
- Consumes: recovery leg from `EpisodeMotionProfile`.
- Produces:

```python
MidAirDropFault.set_recovery_motion_profile(
    env_idx: int,
    *,
    speed_multiplier: float,
    pickup_offset_xy_m: tuple[float, float],
    transport_offset_m: float,
    posture_bias_rad: tuple[float, float, float],
) -> None
```

- [ ] **Step 1: Write failing recovery-profile tests**

Test:

```python
def test_explicit_recovery_profile_reaches_planner()
def test_explicit_speed_overrides_config_random_speed()
def test_recovery_action_noise_remains_zero()
def test_profile_is_isolated_per_vector_env()
def test_invalid_env_index_is_rejected()
def test_non_positive_speed_is_rejected()
def test_reset_clears_previous_episode_override()
```

Spy on `SimpleIKRecoveryPlanner` construction and assert exact primitive values rather than only checking that recovery starts.

- [ ] **Step 2: Run recovery tests and confirm red**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  tests/faults/test_midair_drop_fault.py \
  tests/faults/test_side_grasp_and_drop_gates.py -q
```

Expected: failures because the setter and per-env override state do not exist.

- [ ] **Step 3: Add per-env recovery override state and setter**

Store the explicit values on `_EnvDropState`. Clear them in episode reset. In `_start_recovery_planner`, use explicit values when present; otherwise preserve existing generic behavior for callers outside this generator.

Pass:

```python
SimpleIKRecoveryPlanner(
    speed_multiplier=override.speed_multiplier,
    waypoint_noise_m=0.0,
    arm_posture_noise_rad=np.asarray(override.posture_bias_rad),
    pickup_via_offset_xy_m=override.pickup_offset_xy_m,
    transport_via_offset_m=override.transport_offset_m,
    trajectory_seed=episode_seed,
)
```

Do not apply `_apply_recovery_action_noise` in this generator because its config value is fixed at zero.

- [ ] **Step 4: Add config fields used by generic recovery callers**

Add zero-default planner geometry fields:

```python
pickup_via_offset_m: float = 0.0
transport_via_offset_m: float = 0.0
```

Validate both are non-negative. Existing callers remain direct-path by default.

- [ ] **Step 5: Run Task 4 tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  tests/faults/test_midair_drop_fault.py \
  tests/faults/test_side_grasp_and_drop_gates.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Review checkpoint**

Confirm no existing fault caller receives non-zero via-points or changed speed behavior unless it calls the explicit setter. Do not commit.

---

### Task 5: Single-pass runner, event log, and viewer integration

**Files:**
- Modify: `examples/faults/run_can_simpleik_datagen.py`
- Modify: `examples/faults/can_simpleik_viewer.py`
- Modify: `tests/faults/test_datagen_runner_helpers.py`
- Modify: `tests/faults/test_datagen_events.py`

**Interfaces:**
- Consumes: `EpisodeMotionProfile`, planner `carry_path`, polyline `PathTrigger`, recovery-profile setter.
- Produces event kinds:
  - `motion_profile`
  - `trajectory_randomization_fallback`
  - existing `drop`, `impulse`, `recovery`, and `episode_end` with added path fields.

- [ ] **Step 1: Write failing runner-helper tests**

Test:

```python
def test_runner_uses_separate_layout_drop_and_motion_rng_streams()
def test_nominal_planner_receives_nominal_leg_and_episode_speed()
def test_recovery_fault_receives_recovery_leg_and_same_episode_speed()
def test_drop_is_not_sampled_until_planner_exposes_carry_path()
def test_drop_trigger_uses_actual_planner_segment_progress()
def test_consumed_drop_trigger_cannot_fire_twice()
def test_execution_noise_is_always_zero()
def test_disabled_randomization_builds_identity_profile()
```

- [ ] **Step 2: Write failing event/HUD tests**

Assert `motion_profile` contains:

```python
{
    "speed_multiplier": ...,
    "nominal": {
        "pickup_offset_xy_m": [...],
        "transport_offset_m": ...,
        "posture_bias_rad": [...],
    },
    "recovery": {...},
}
```

Assert drop events include `segment_name`, `target_t`, actual XYZ, and basket distance. Assert HUD text contains `speed=`, active segment, and target progress.

- [ ] **Step 3: Run runner/event tests and confirm red**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  tests/faults/test_datagen_runner_helpers.py \
  tests/faults/test_datagen_events.py -q
```

Expected: new tests fail because runner still builds planner noise ad hoc and the HUD exposes the old threshold model.

- [ ] **Step 4: Integrate independent RNG streams and profile logging**

At each episode:

```python
layout_rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 0x4C41594F]))
drop_rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 0x44524F50]))
profile = sample_episode_motion_profile(recipe.simple_ik, episode_seed)
```

Use `layout_rng` only for layout and `drop_rng` only for the q coin/arc sample. Emit the requested profile before executing the planner. Emit resolved nominal via-points after nominal planner construction and resolved recovery via-points when recovery starts.

- [ ] **Step 5: Wire nominal and recovery profiles**

Construct nominal planner with nominal via offsets, posture bias, and shared speed. Before execution, call `fault.set_recovery_motion_profile(...)` with recovery values and the same speed. Build `FaultInjectionConfig` with `recovery_action_noise_std=0.0`.

- [ ] **Step 6: Replace straight trigger handling**

On the first lift frame where `planner.carry_path` is non-null:

```python
path = eligible_path(
    planner.carry_path,
    basket_xy=basket[:2],
    keepout_m=keepout,
)
decision, trigger = sample_path_drop(q, path, drop_rng)
```

Each carry step calls the projection-based trigger with current object XYZ and active phase. Consume it immediately after firing. Keep runtime grasp and keep-out checks before `trigger_manual_drop`.

- [ ] **Step 7: Update viewer HUD and event payloads**

Display:

```text
episode=3 step=81 phase=to_basket_via speed=0.87
q=0.500 eligible=0.54m drop=to_basket_hover@0.62
can-basket=0.31m grasped=True
```

The scrollable log remains the full structured event stream.

- [ ] **Step 8: Run Task 5 tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  tests/faults/test_datagen_runner_helpers.py \
  tests/faults/test_datagen_events.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Review checkpoint**

Confirm the runner still performs exactly one environment execution per episode and has no trace/reset/replay pass. Do not commit.

---

### Task 6: Regression, live LIBERO comparison, and documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-09-23-can-simpleik-datagen-design.md`
- Modify: `docs/superpowers/specs/2026-09-23-trajectory-shape-randomization-design.md` only if measured behavior differs from the approved design.

**Interfaces:**
- Consumes all prior tasks.
- Produces verification evidence and user-facing run instructions.

- [ ] **Step 1: Run focused and broad automated suites**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/faults tests/envs -q
```

Expected: all tests pass; only known CUDA/robosuite warnings and existing skips remain.

- [ ] **Step 2: Run static checks**

Run:

```bash
.venv/bin/python -m compileall -q \
  src/lerobot/faults \
  examples/faults \
  tests/faults
git diff --check
```

Expected: exit code 0 and no output from `git diff --check`.

- [ ] **Step 3: Run a disabled-randomization LIBERO baseline**

Create `/tmp/can_simpleik_identity.json` from the recipe with:

```json
"trajectory_randomization_enabled": false
```

Run:

```bash
MUJOCO_GL=egl PYTHONPATH=examples/faults:src timeout 900 \
  .venv/bin/python examples/faults/run_can_simpleik_datagen.py \
  --recipe /tmp/can_simpleik_identity.json \
  --seed 1000 --episodes 20 \
  --output /tmp/can_identity --headless
```

Record completed episodes, recovery failures, drop phase counts, and elapsed time from `events.jsonl`.

- [ ] **Step 4: Run the enabled-randomization comparison**

Create `/tmp/can_simpleik_randomized.json` with:

```json
"trajectory_randomization_enabled": true
```

Run the same 20 seeds:

```bash
MUJOCO_GL=egl PYTHONPATH=examples/faults:src timeout 900 \
  .venv/bin/python examples/faults/run_can_simpleik_datagen.py \
  --recipe /tmp/can_simpleik_randomized.json \
  --seed 1000 --episodes 20 \
  --output /tmp/can_randomized --headless
```

Record the same metrics plus sampled speed range, positive/negative transport offsets, and fallback count.

- [ ] **Step 5: Evaluate the safety criterion**

Compare completion and recovery success. If enabled randomization loses more than 5 percentage points versus baseline, reduce defaults in this order and rerun:

1. transport offset `0.06 -> 0.04`;
2. pickup offset `0.03 -> 0.02`;
3. speed range `[0.8, 1.2] -> [0.9, 1.1]`.

Do not change multiple ranges in one rerun; the evidence must show which reduction restored reliability.

- [ ] **Step 6: Run one live-viewer episode**

Run:

```bash
DISPLAY=:1 MUJOCO_GL=egl PYTHONPATH=examples/faults:src timeout 600 \
  .venv/bin/python examples/faults/run_can_simpleik_datagen.py \
  --recipe examples/faults/recipes/can_simpleik_datagen.json \
  --seed 3 --episodes 1 \
  --output /tmp/can_randomized_ui
```

Confirm the 720 px camera panel visibly shows a bent path, HUD speed/segment data, and full log entries.

- [ ] **Step 7: Update the original generator design**

Replace statements that force identity SimpleIK with references to the new profile. Document that diversity comes from bounded via-points and per-episode speed, not execution noise, and that drop sampling follows the planner's bent polyline.

- [ ] **Step 8: Final verification**

Rerun:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/faults tests/envs -q
.venv/bin/python -m compileall -q src/lerobot/faults examples/faults tests/faults
git diff --check
```

Expected: tests pass and both static commands exit 0.

- [ ] **Step 9: Final uncommitted-state checkpoint**

Run:

```bash
git status --short --branch
```

Report the branch, changed files, baseline/randomized success metrics, drop distribution, actual speed range, and fallback count. Leave all work uncommitted.
