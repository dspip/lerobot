# Branch changelog: `feat/can-simpleik-datagen` vs `main`

Date: 2026-09-23  
Compared to: `main` at `2b70e2bc` (`merge: integrate manual recovery improvements`)  
This file is the branch-vs-`main` inventory. The design specs in this folder lag the recipe in two places: they still say `alphabet_soup_1` and `speed_multiplier_range: [0.8, 1.2]`. The recipe on the branch (commit `1821617d`) uses `milk_1` and `[0.9, 1.1]`.

**Not in this branch:** LeRobot dataset recording, picking objects other than the recipe `object_name`, GraspGen, per-step action noise, a max-reach layout constraint.

## For a Cursor agent merging this branch

Read this section before editing shared files. Do not reconstruct the work from chat transcripts. Base SHA is `main` `2b70e2bc`. Take **all additive paths** from this branch. Only merge-edit the **eight shared files** below. Defaults on new planner/config knobs are `0` / off so evaluation callers on `main` keep old motion unless they opt in.

### Merge policy

1. Keep both sides' unique files. This branch's unique tree is listed under "Additive paths (no conflict expected)".
2. On a conflict in a shared file, **keep both behaviors** unless they literally assign the same identifier. Do not drop `disable_gui_renderer`, `CARRY_PHASES`, `to_basket_via`, `pickup_via`, `waypoint_blend_radius_m`, or `PathTrigger` `held_midair` waiting.
3. Do not "simplify" blending by applying the full radius to every waypoint. Grasp/release phases must stay exact. Cap blend at `_BLEND_LEG_FRACTION * leg_length` (`0.4`).
4. Do not restore a two-pass (dry-run then replay) drop sampler. Drop is one arc-length sample on the carry path in the same episode.
5. Recipe JSON on this branch is `object_name: milk_1`. Specs in `docs/superpowers/specs/` still say `alphabet_soup_1`. Prefer the JSON + `recipe.py` as source of truth for schema.
6. After merge, run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/faults tests/envs/test_libero_gui_renderer.py -q` from `lerobot/`.

### Additive paths (take whole files; should not conflict)

```
src/lerobot/faults/datagen/           # new package
src/lerobot/faults/recovery/trajectory.py
examples/faults/run_can_simpleik_datagen.py
examples/faults/can_simpleik_viewer.py
examples/faults/recipes/can_simpleik_datagen.json
tests/faults/test_datagen_*.py
tests/faults/test_trajectory_geometry.py
tests/envs/test_libero_gui_renderer.py
docs/superpowers/
```

If the other branch also added `src/lerobot/faults/datagen/`, merge module-by-module; this package is the recipe loader, layout sampler, motion profile, path drop, events, viewer helpers.

### Shared files (conflict likely) — keep these symbols

| File | This branch added | Do not lose |
| --- | --- | --- |
| `src/lerobot/envs/libero.py` | `disable_gui_renderer(env)` + call in `LiberoEnv` ctor after env create, before `reset` | Headless reset livelock fix. Other branch may also construct `OffScreenRenderEnv`; call this helper there too. |
| `src/lerobot/faults/recovery/libero_hook.py` | import `disable_gui_renderer`; call after `OffScreenRenderEnv(**kwargs)` | Same. |
| `src/lerobot/faults/config.py` | `FaultInjectionConfig` fields `pickup_via_offset_m=0.0`, `transport_via_offset_m=0.0`, `waypoint_blend_radius_m=0.0` + `__post_init__` `>= 0` checks | Insert next to existing `arm_posture_noise_deg`. Dataclass field **order** matters if anyone constructs positionally. |
| `src/lerobot/faults/recovery/planner.py` | See API block below | Highest conflict risk. |
| `src/lerobot/faults/recovery/midair_drop.py` | Import `CARRY_PHASES`; pass via offsets + `waypoint_blend_radius_m` into `SimpleIKRecoveryPlanner(...)`; `carrying = phase in CARRY_PHASES` | Old check `phase in ("lift", "to_basket_hover")` **misses** `to_basket_via` and opens the gripper mid-carry. |
| `tests/faults/test_planner.py` | Keep both test sets | New tests import `BLENDABLE_PHASES`. |
| `tests/faults/test_midair_drop_fault.py` | Extra cases for new planner kwargs / carry phases | Keep both. |

### `SimpleIKRecoveryPlanner` API this branch expects

New **keyword-only** constructor args (all default so old call sites compile):

```python
pickup_via_offset_xy_m: tuple[float, float] = (0.0, 0.0)
transport_via_offset_m: float = 0.0
basket_keepout_m: float = 0.20
waypoint_blend_radius_m: float = 0.0  # 0 = stop on every waypoint (main behavior)
```

New module constants:

```python
CARRY_PHASES = frozenset({"lift", "to_basket_via", "to_basket_hover"})
BLENDABLE_PHASES = frozenset({"retract", "pickup_via", "lift", "to_basket_via"})
```

New / used attributes and phase names:

- `planner.phase_name` — string; extra values vs `main`: `"pickup_via"`, `"to_basket_via"`.
- `planner.carry_path` — `CarryPath | None` from `lerobot.faults.recovery.trajectory`; set when entering `lift`.
- Waypoint names in order when offsets are non-zero: `retract`, `pickup_via`, `approach_hover`, `descend_grasp`, `close_grasp`, `lift`, `to_basket_via`, `to_basket_hover`, `open_place`, `retract_done`.
- Any `if phase == ...` or `phase in (...)` on the other branch that lists only `lift` / `to_basket_hover` must include `to_basket_via` for carry, and must allow `pickup_via` as a pre-grasp transit phase.

Geometry helpers live in `lerobot.faults.recovery.trajectory`: `build_pickup_via`, `build_carry_path`, `CarryPath`, `ResolvedPickupVia`. Prefer importing those rather than inlining keep-out math.

### Path drop contract (if the other branch touches drop timing)

- `PathTrigger.fires(..., held_midair: bool = True)` returns `False` when not held (wait), **not** a permanent cancel.
- Sample once on keep-out-clipped carry arc length; inject on the same run.
- Object pose logged around `trigger_manual_drop` must be `.copy()`'d first; the impulse mutates the live array.

### Datagen runner wiring (other branch should not need to rewrite this)

`examples/faults/run_can_simpleik_datagen.py` constructs the nominal planner with recipe blend/offsets and `FaultInjectionConfig(..., waypoint_blend_radius_m=recipe.simple_ik.waypoint_blend_radius_m)` so recovery blending matches. If the other branch has a different runner, copy that kwargs pattern rather than forking planner internals.

### Intentional non-goals (do not "complete" during merge)

- Picking clutter objects (recipe stays one `object_name`).
- LeRobot parquet/video dataset writer.
- Layout max-reach filter (far XY can stall at `max_steps_per_waypoint=100` and still count as success).
- Deleting `drop_timing.py` two-pass leftovers still exported from `datagen/__init__.py`.

## Commits


| SHA        | Message                                                                                                                                                        |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `6f0f2a8d` | `feat(faults): add recipe-driven can SimpleIK datagen with path drops`                                                                                         |
| `1821617d` | `updated json` — recipe `object_name` switched to `milk_1`; speed range and blend radius already at `[0.9, 1.1]` / `0.02` from local edits in the first commit |


34 files, +5640 / −25.

## What existed on `main`

On `main`, mid-air drop work is evaluation-oriented:

- Nominal pickup is SmolVLA (or similar policy), recovery is `SimpleIKRecoveryPlanner`.
- Layout is almost fixed (roughly ±4 cm on the pick object).
- Drop timing is delay / XY-band / a one-shot `probability`, not `P(any drop)=q` uniform on the carry path.
- The planner stops on every waypoint (`hold_steps` plus proportional creep into the target).
- There is no JSON recipe, no datagen runner, no live datagen viewer.

This branch keeps the same LIBERO-Object task 0 env, `MidAirDropFault` impulse, and SimpleIK recovery, and replaces the dataset-generation loop.

## How to run

From `lerobot/`:

```bash
MUJOCO_GL=egl PYTHONPATH=examples/faults:src .venv/bin/python \
  examples/faults/run_can_simpleik_datagen.py \
  --recipe examples/faults/recipes/can_simpleik_datagen.json \
  --seed 0 --episodes 10 \
  --output outputs/can_simpleik_datagen
```

`--headless` skips the Tk viewer. `--recipe` is required; a bad JSON aborts before env reset.

## Recipe (`examples/faults/recipes/can_simpleik_datagen.json`)

Loaded by `src/lerobot/faults/datagen/recipe.py`. Every field is required (no silent defaults once the file is loaded).


| Field                                                           | Role                                                                                                                                                                          |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q`                                                             | `P(drop)` for the episode if the eligible carry arc is non-empty. `0` never drops, `1` always drops. At most one drop.                                                        |
| `object_name`                                                   | Grasp target. Branch recipe: `milk_1`. Specs still say `alphabet_soup_1`. Other objects stay clutter.                                                                         |
| `basket_name`                                                   | Place target (`basket_1`).                                                                                                                                                    |
| `placement.xy_range_m`                                          | Square half-width around each object's **reset** XY (0.12 m, vs ~0.04 m on `main`).                                                                                           |
| `placement.min_basket_clearance_m`                              | Pick-target vs basket keep-out (0.35 m).                                                                                                                                      |
| `placement.distractor_basket_clearance_m`                       | Smaller keep-out for clutter (0.15 m). Applying the full 0.35 m to every object makes layout sampling almost always fail because stock clutter already sits inside that disk. |
| `placement.min_pairwise_clearance_m`                            | Object–object XY (0.08 m).                                                                                                                                                    |
| `placement.yaw_range_deg`                                       | `[-180, 180]`.                                                                                                                                                                |
| `drop.eligible_phases`                                          | Lift and path-to-container.                                                                                                                                                   |
| `drop.min_drop_distance_from_basket_m` / `hard_keepout_floor_m` | Forbidden drop disk around the basket (0.20 m). Per-recipe so `mix_audit` keep-out on old datasets does not change.                                                           |
| `simple_ik.trajectory_randomization_enabled`                    | If false: offsets/posture are zero, speed is 1.0. Blend radius is independent of this switch.                                                                                 |
| `simple_ik.pickup_via_offset_m` / `transport_via_offset_m`      | Bounds on coherent via-point offsets (0.03 / 0.06 m).                                                                                                                         |
| `simple_ik.arm_posture_noise_deg`                               | One-shot wrist bias (±3°).                                                                                                                                                    |
| `simple_ik.speed_multiplier_range`                              | One speed per episode, shared by nominal and recovery. Recipe: `[0.9, 1.1]`.                                                                                                  |
| `simple_ik.waypoint_blend_radius_m`                             | Pass-through radius for free-space via points. Recipe: `0.02`. `0` restores stop-at-every-waypoint.                                                                           |


## New package: `src/lerobot/faults/datagen/`


| Module              | What it does                                                                                                                                                                                                                                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `recipe.py`         | Strict JSON load + validation.                                                                                                                                                                                                                                                                                           |
| `layout.py`         | Rejection-sample XY (and yaw) for **every** movable object; table AABB, basket clearance, pairwise clearance. **No max-reach check.**                                                                                                                                                                                    |
| `motion_profile.py` | Independent RNG stream (`SeedSequence([seed, 0x4D4F544E])`). Samples one speed, independent nominal vs recovery pickup/transport offsets and posture. Layout and drop use other streams so they do not steal draws from each other.                                                                                      |
| `path_drop.py`      | After lift, clip the carry polyline against the basket keep-out, sample one arc-length `s`*, fire `PathTrigger` when the object projection reaches it. If the can is not yet `held_midair`, the trigger **waits** (does not cancel). That was a silent `q=1` miss when the sampled lift point arrived before z ≥ 0.12 m. |
| `runtime.py`        | Carry-phase gripper clamp / slower translation (`stabilize_carry_action`, `hold_gripper_closed`).                                                                                                                                                                                                                        |
| `events.py`         | JSONL + stdout + viewer log (`DatagenEventLog`).                                                                                                                                                                                                                                                                         |
| `display.py`        | Overlay / panel helpers for the viewer.                                                                                                                                                                                                                                                                                  |
| `drop_timing.py`    | Leftover two-pass helpers (`TraceFrame`, `eligible_indices`, `sample_drop`) still exported from `__init__.py`. The runner does **not** use a dry-run pass.                                                                                                                                                               |


## Planner and recovery (`src/lerobot/faults/recovery/`)

### `planner.py` (`SimpleIKRecoveryPlanner`)

- **Pickup via** (`pickup_via`) before hover/descend when the offset is non-zero.
- **Transport via** (`to_basket_via`) on the keep-out-legal carry polyline (`CarryPath` in `trajectory.py`).
- Carry path is rebuilt from the **live** object pose when `lift` starts, so drop sampling matches the executed lift, not the planned-at-reset pose.
- `CARRY_PHASES = {lift, to_basket_via, to_basket_hover}` — recovery used to miss `to_basket_via` and unclamped the gripper mid-carry.
- **Waypoint blending** (`waypoint_blend_radius_m`): on `BLENDABLE_PHASES = {retract, pickup_via, lift, to_basket_via}` dwell is 0 and arrival is a blend radius early so the arm does not stop on via points. Grasp/release phases (`approach_hover`, `descend_grasp`, `close_grasp`, `to_basket_hover`, `open_place`, `retract_done`) still settle exactly.
- Effective blend radius is capped at 40% of that leg's length. A full 0.04 m on the ~5 cm retract erased the retract and cost recoveries.
- Constructor default `waypoint_blend_radius_m=0` so other callers keep old motion unless they opt in.

### `trajectory.py` (new)

Pickup-via and carry-path geometry: workspace clamp, basket keep-out clip, signed perpendicular transport offset.

### `midair_drop.py`

Recovery planner construction takes the same via offsets, speed, posture, blend radius, and basket keep-out as the recipe/config. Carry check uses `CARRY_PHASES`.

### `config.py`

New fields, default 0 (old behavior): `pickup_via_offset_m`, `transport_via_offset_m`, `waypoint_blend_radius_m`. Reject negatives.

## Runner and viewer

`examples/faults/run_can_simpleik_datagen.py`:

1. Load recipe, make LIBERO env, wrap with `DropRecoveryEnvWrapper`.
2. Per episode: sample layout → sample motion profile → plan SimpleIK once → execute.
3. At lift: log carry path; if `q` fires, arm `PathTrigger`.
4. Drop uses a **copy** of object XYZ before `trigger_manual_drop` (live numpy alias previously logged the post-impulse pose).
5. Recovery is the same SimpleIK planner with the recovery-leg profile.
6. Events: `layout_ok`, `motion_profile`, `plan_ok`, `trajectory_phase`, `drop`, `impulse`, `recovery`, `episode_end` / `recovery_failed`.

`examples/faults/can_simpleik_viewer.py`: Tk window, larger camera panel, scrollable event log (same stack as the manual drop demo).

## LIBERO headless reset (`src/lerobot/envs/libero.py`)

`disable_gui_renderer()` turns off the on-screen OpenCV viewer that `OffScreenRenderEnv` forces on. Headless `cv2` builds raise in `destroyAllWindows()`; LIBERO's reset retry then spins forever. Hooked from env construction and `libero_hook.py`. Observations still come from the offscreen renderer.

## Tests (new or extended)

- Recipe / layout / motion-profile / path-drop / events / display / runner helpers.
- Planner: via-points, carry-path refresh, blending (no transit dwell, command floor, grasp/release still settle, geometry unchanged, negative radius rejected, blend capped by leg length).
- Trajectory geometry, GUI renderer disable, mid-air drop fault updates.

`tests/faults` for this work is green. Unrelated `tests/robots` / `tests/teleoperators` / `tests/annotations` failures on this machine predate the branch.

## Behaviour that is on the branch but easy to miss

**Single pass.** Path is computed at lift; the drop is injected on that same execution. There is no record-then-replay episode.

**Independent RNG streams** from the episode seed: layout `0x4C41594F`, drop `0x44524F50`, motion `0x4D4F544E`.

**Out-of-reach placements.** Layout does not reject far XY. Seed 0, later episodes with the can ~0.44 m from gripper home, the Panda plateaus ~2 cm off the can. Each waypoint then burns `max_steps_per_waypoint=100` (phases land on 33 / 133 / 233 / 333). The episode can still log as success. That is placement reaching the arm, not blending.

**Marginal recoveries.** Seed 0 episode 0 is chaotic: a ~1 cm change in drop height flips a clean recovery vs a regrasp loop to the 800-step cap. Combined measured rate on earlier soup batches was ~19/20, matching the identity/no-blend baseline.

## File map vs `main`

**Added**

- `src/lerobot/faults/datagen/`*
- `src/lerobot/faults/recovery/trajectory.py`
- `examples/faults/run_can_simpleik_datagen.py`
- `examples/faults/can_simpleik_viewer.py`
- `examples/faults/recipes/can_simpleik_datagen.json`
- `tests/faults/test_datagen_*.py`, `tests/faults/test_trajectory_geometry.py`, `tests/envs/test_libero_gui_renderer.py`
- `docs/superpowers/plans/*`, `docs/superpowers/specs/*` (this changelog)

**Modified**

- `src/lerobot/faults/recovery/planner.py`, `midair_drop.py`, `libero_hook.py`
- `src/lerobot/faults/config.py`
- `src/lerobot/envs/libero.py`
- `tests/faults/test_planner.py`, `tests/faults/test_midair_drop_fault.py`

