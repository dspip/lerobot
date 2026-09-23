# Can-only SimpleIK datagen (layout + drop timing)

Date: 2026-09-23  
Status: approved  
Scope: generation and live viewing. **No LeRobot dataset recording in this slice.**

## Problem

The current VLA drop pipeline (`examples/faults/run_full_drop_recovery_pipeline.py`) is not usable as a diverse generator:

- Pickup is SmolVLA; recovery is SimpleIK. This slice uses SimpleIK for **pickup, transport, and recovery**.
- Only the can is the pick target (`alphabet_soup_1`, LIBERO-Object task 0). Other objects stay in the scene as clutter and are **not** grasp targets.
- Start poses are almost fixed (XY60 used `(0,0)`; otherwise ±4 cm on the can only).
- Drop timing is delay-based or XY-band, plus a one-shot `probability` draw. That is not `P(any drop)=q` with at most one drop, uniform over eligible frames.

## Non-goals

- GraspGen / cuRobo.
- Per-step action noise; coherent via-point, posture, and episode-speed
  randomization are specified separately.
- Picking any object other than the can.
- Parquet/video dataset commit, XY-band quotas, keep-gates, seat-assist for training.
- Per-step independent Bernoulli drops (rejected: multiple drops, `1-(1-p)^n` rate, early-step bias).

## Locked decisions

| Topic | Decision |
| --- | --- |
| Target | `alphabet_soup_1` only; task text stays soup → basket |
| Clutter | Rejection-sample start XY (and yaw when the free joint allows) for **every** movable object; basket stays fixed |
| Basket keep-out | `min_basket_clearance_m` applies to the pick target only. Distractors use the smaller `distractor_basket_clearance_m`, because the stock scene already seats `cream_cheese_1` (0.237 m) and `tomato_sauce_1` (0.271 m) inside the target keep-out — applying it to everything drops per-attempt layout success to 0.08% |
| Motion | SimpleIK for pick, lift, transport, place, and post-drop recovery. Shape diversity comes from bounded pickup/transport via-points plus a one-shot posture bias; nominal and recovery share one sampled episode speed. Per-step action noise stays disabled. |
| Config | JSON recipe required (`--recipe`). No silent default for `q` once loaded |
| Drop | Run-level `q`. If drop: one point uniform over the eligible **arc length** of the computed carry path. `P(no drop)=1-q` when that arc is non-empty |
| Eligible arc | Phases lift **or** path-to-container, can actually grasped, and object–basket XY ≥ keep-out |
| Recording | Not in this slice |
| Visual check | Live Tk window (camera + **scrollable log panel**, same stack as `run_manual_drop_recovery.py`) |

## Architecture

Keep the LeRobot/LIBERO env + `MidAirDropFault` impulse + `SimpleIKRecoveryPlanner`. Replace the SmolVLA nominal loop with plan-then-execute.

```
JSON recipe
    → reset libero_object task 0
    → sample layouts for all movable objects
    → execute nominal closed-loop SimpleIK (single pass)
    → at lift entry the carry path is known, so measure its eligible arc:
        clip each lift/bent-transport segment against the basket keep-out
        eligible_len = sum of the remaining polyline arc lengths
    → with prob 1-q run on nominally
      else: s* ~ Uniform(0, lift_len + transport_len); inject when the carry
            passes s*; SimpleIK recover
    → live UI streams agentview + overlay + the same event log as JSONL/stdout
```

### Why one pass, not two

An earlier revision sampled a step index uniformly over the executed trace,
which needed a dry run to learn how many eligible steps there would be. The
planner builds its waypoints analytically, so the carry *geometry* is known at
lift entry without executing. The immutable carry path contains the vertical
lift and either one direct transport segment or two segments through a safe
via-point. Sampling uniformly along its keep-out-clipped arc gives exactly one
drop and `P(drop)=q` in a single pass. The distribution is uniform in distance
travelled rather than in timesteps.

Both trigger quantities are monotone within their phase (can height during
lift, can-basket distance during transport), so the sampled point cannot be
stepped over. A lift target that closed-loop execution overshoots fires at the
first transport frame, the nearest eligible point remaining.

CLI: `--recipe path.json` is required. Missing or invalid recipe aborts before reset.

### Recipe schema

```json
{
  "q": 0.5,
  "object_name": "alphabet_soup_1",
  "basket_name": "basket_1",
  "placement": {
    "xy_range_m": 0.12,
    "min_basket_clearance_m": 0.35,
    "distractor_basket_clearance_m": 0.15,
    "min_pairwise_clearance_m": 0.08,
    "yaw_range_deg": [-180, 180],
    "max_attempts": 200
  },
  "drop": {
    "eligible_phases": ["lift", "to_container"],
    "min_drop_distance_from_basket_m": 0.20,
    "hard_keepout_floor_m": 0.20
  },
  "simple_ik": {
    "trajectory_randomization_enabled": true,
    "pickup_via_offset_m": 0.03,
    "transport_via_offset_m": 0.06,
    "arm_posture_noise_deg": 3.0,
    "speed_multiplier_range": [0.8, 1.2],
    "waypoint_blend_radius_m": 0.04
  }
}
```

- `q` must be in `[0, 1]`. `0` never drops; `1` always drops if `E` is non-empty.
- `waypoint_blend_radius_m` is the pass-through radius for free-space transit legs, so the arm rounds those corners instead of stopping on them. `0` restores stop-at-every-waypoint motion. See the trajectory-shape design doc for the per-leg cap.
- `xy_range_m` is a square half-width around each object’s **reset** XY (larger than today’s ±0.04 m). Implementation may clamp samples to the real table AABB if draws fall off the table.
- `min_drop_distance_from_basket_m` is the forbidden drop disk. `hard_keepout_floor_m` is the per-recipe floor beneath it. It is per-recipe rather than the module-level `HARD_BASKET_KEEPOUT_M` (0.22 m) because `mix_audit` uses that constant to judge already-recorded datasets, so moving it would retroactively change their verdicts.
- Measured LIBERO geometry: basket XY footprint 0.151 x 0.276 m (half-extent 0.138 m), can 0.062 x 0.082 m (half-extent 0.041 m). The can clears the basket only beyond ~0.18 m, so 0.20 m is the floor in use: geometric clearance plus a small margin for the impulse's lateral drift.

Default recipe file: `examples/faults/recipes/can_simpleik_datagen.json`.

## Components

| Unit | Responsibility | Depends on |
| --- | --- | --- |
| Recipe loader | Parse/validate JSON | stdlib |
| Layout sampler | Offset every movable object; reject basket/pairwise/off-table | `lerobot.faults.sim.libero` |
| Nominal SimpleIK plan | Seeded pickup/transport via-points, exact grasp/place endpoints, and one episode speed | `SimpleIKRecoveryPlanner` |
| Eligible-path builder | Clip the planner's carry polyline against the basket keep-out | `CarryPath` + basket pose |
| Drop sampler | Skip or sample one point uniformly over eligible arc length | `q`, eligible path, RNG |
| Executor | Drive OSC/SimpleIK actions; call existing `_trigger_drop` / impulse at `i` | `MidAirDropFault` |
| Live viewer | Tk window: camera, HUD overlay, **append-only log pane** mirroring generator events | executor + event log |

Do not teach `run_pipeline` to write datasets in this slice. A thin runner (`examples/faults/run_can_simpleik_datagen.py`) owns the loop and UI.

## Drop sampling (exact)

Let `A` be the total length of all lift/transport polyline pieces outside the
basket keep-out. If `A=0`, execute nominal placement and log
`no_eligible_path`. Otherwise draw the run-level `q` coin once. On success,
sample `s ~ Uniform(0,A)`, map it to a segment and segment progress, and inject
when closed-loop execution reaches or passes that point.

Consequences: one environment execution, at most one drop, and
`P(any drop)=q` whenever `A>0`. Runtime held-midair and keep-out gates remain
authoritative. Grasp approach and in-basket release are never eligible.

## Placement

After `env.reset`:

1. Enumerate movable objects (LIBERO object bodies excluding the basket).
2. For each object, sample `dx, dy ~ Unif[-xy_range_m, xy_range_m]` from its reset XY, optional yaw in `yaw_range_deg`.
3. Reject if: too close to basket, too close to another already-placed object, or clearly off-table.
4. Up to `max_attempts` full-layout retries. Failure → skip episode, log `layout_failed`. Do not force overlapping poses.

The can is placed with the same sampler as clutter; only the can is later grasped.

## Error handling

| Case | Behavior |
| --- | --- |
| Bad recipe | Abort; no env |
| Layout exhausted | Skip episode; continue if looping |
| IK plan fails | Skip episode; log `plan_failed`; no SmolVLA fallback |
| Empty `E` | Nominal execute; not an error |
| Drop lands in basket | Log `drop_landed_in_basket`; `seat_assist_enabled=false` |
| Recovery fails | Log; end episode; no keep-gate |

## Live UI (required verification)

Pytest cannot show whether layouts look legal or whether the drop fires on lift vs transport. Ship a live viewer:

- Stack: Tk + PIL, same as `examples/faults/run_manual_drop_recovery.py` (no new UI framework).
- Layout: camera on the left (or top); **log pane on the right (or bottom)**. Overlay text on the camera is not a substitute for the log pane.
- Playback: **this generator**, not XY60 parquet replay.
- Overlay (HUD): seed, `q`, eligible arc length, episode speed, drop segment/progress (or skip reason), current phase, can–basket XY, and grasped flag.
- **Log pane:** every generator event that would go to stdout or `*.jsonl` is appended here in order, one line (or wrapped JSON) per event, with wall-clock or sim step prefix. Minimum events: recipe loaded, layout ok/fail, sampled motion profile, plan ok/fail, eligible path, drop skip/inject (`q`, segment, progress, keep-out), nominal/recovery phase changes, impulse with the copied pre-drop pose, recovery start, `drop_landed_in_basket`, recovery fail, and episode end. User can scroll; newest lines stay visible (auto-scroll unless they scrolled up). Optional Clear between episodes. The same stream is still written to a JSONL file next to the run so a session is inspectable after closing the window.
- Controls: Next episode, Pause, optional force `q=0` / `q=1` for the next run (does not rewrite the JSON file).
- Camera: LIBERO `agentview` (add wrist if cheap). Display rate may subsample sim steps; physics still runs at control freq.

Headless machines without Tk skip the window and print the **full event log** (not only HUD fields) to stdout. Unit tests never require the UI.

## Tests (CPU, no GPU)

- Recipe: valid load; reject missing file, `q` out of range, missing `object_name`.
- Drop sampler with fixture polylines: `q=0` never drops; `q=1` samples one eligible segment/progress; many seeds follow arc-length proportions; never two drops.
- Keep-out: line-circle clipping excludes every polyline interval inside `min_drop_distance_from_basket_m`.
- Empty eligible path → skip, no exception.
- Placement: reject overlap / in-basket; `layout_failed` after max tries (mocked poses).
- Existing `tests/faults/` for impulse + SimpleIK recovery still pass.

Live check (manual, LIBERO + display): `uv run python examples/faults/run_can_simpleik_datagen.py --recipe examples/faults/recipes/can_simpleik_datagen.json` and confirm randomized clutter, can-only grasp, at most one drop, no drop inside the keep-out disk.

## Environment gotcha: LIBERO's forced on-screen renderer

`OffScreenRenderEnv` hardcodes `has_renderer=True`, so robosuite builds an OpenCV GUI
window and calls `cv2.destroyAllWindows()` on every hard reset. Headless `cv2` builds
raise there, and LIBERO's `reset()` retry loop uses `finally: continue`, which swallows
the exception and retries forever — a silent 100% CPU spin that looks like a slow load.

`lerobot.envs.libero.disable_gui_renderer` turns the unused on-screen renderer off before
the first reset. Both construction sites (`LiberoEnv._ensure_env` and the control-freq
hook) call it. Offscreen rendering, which supplies observations, is unaffected.

## Risks

- SimpleIK nominal pickup is a different action distribution than SmolVLA; fine for later recording, not a drop-in XY60 replacement.
- `xy_range_m=0.12` may still clip the table; clamp to table AABB in implementation.
- Planner phase names must be mapped explicitly to `eligible_phases` or `E` will be empty and `q` will appear broken.
- Existing `config.probability` / XY-band / `post_grasp_delay_steps` must not run in parallel with the new sampler (would double-trigger). The runner sets delay/band off and uses the planned index instead.
