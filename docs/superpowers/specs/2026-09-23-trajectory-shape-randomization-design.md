# Coherent SimpleIK trajectory-shape and speed randomization

Date: 2026-09-23  
Status: approved  
Scope: nominal can pickup/place and post-drop SimpleIK recovery in
`run_can_simpleik_datagen.py`.

## Problem

Per-step action noise produces controller jitter, not tele-operation-like path
diversity. The generator needs coherent episode-level differences in trajectory
shape and speed while preserving precise grasp/place endpoints and the
single-pass, path-based drop guarantees.

## Goals

- Vary the shape of both nominal pickup/place and post-drop recovery paths.
- Preserve exact grasp and basket endpoints.
- Keep the default variation conservative enough for a high success rate.
- Sample one speed per episode and reuse it for nominal motion and recovery.
- Keep `P(drop)=q`, at most one drop, and no drop inside the configured keep-out.
- Reproduce the complete episode motion profile from the episode seed.
- Disable all trajectory and speed variation with one recipe flag.

## Non-goals

- Per-step execution/action noise.
- GraspGen, cuRobo, or learned trajectory generation.
- Random grasp endpoints.
- Varying lift or place height in this slice.
- Recording a LeRobot dataset.

## Recipe

Replace the current mixed noise configuration with:

```json
{
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

When `trajectory_randomization_enabled=false`, the effective values are:

- pickup and transport offsets: `0`;
- posture bias: `0`;
- speed multiplier: `1.0`.

`waypoint_blend_radius_m` is independent of that switch: it shapes execution
rather than the sampled trajectory.

## Waypoint Blending

Without blending the arm visibly stops on every waypoint, for two reasons: each
transit waypoint carries a dwell of zero-command steps, and the commanded delta
is proportional to the remaining distance, so the arm creeps into the waypoint
before retargeting. Both are removed for free-space transit legs only
(`retract`, `pickup_via`, `lift`, `to_basket_via`): the dwell becomes zero and
arrival is declared a blend radius early, so the corner is rounded instead of
stopped.

The effective radius is capped at 40% of the leg length. A radius comparable to
the leg would erase that leg's geometry — measured in LIBERO, applying the full
0.04 m to the 0.05 m retract leg cost a recovery (the regrasp looped to the
800-step cap), while the capped radius does not.

`approach_hover`, `descend_grasp`, `close_grasp`, `to_basket_hover`,
`open_place`, and `retract_done` keep exact arrival and full settling time, so
grasp and release precision is unchanged.

Validation:

- offsets and posture magnitude must be non-negative;
- speed range must contain exactly two positive numbers;
- speed maximum must be greater than or equal to speed minimum.

The old `recovery_action_noise_std` is removed from this recipe. The fault
implementation may retain its generic option for other callers, but this
generator always passes zero.

## Episode motion profile

At episode start, sample an immutable `EpisodeMotionProfile` from an RNG derived
from the episode seed:

- `speed_multiplier ~ Uniform(0.8, 1.2)`;
- nominal pickup via offset, bounded by `pickup_via_offset_m`;
- nominal transport via offset, bounded by `transport_via_offset_m`;
- recovery pickup and transport offsets sampled independently from the same
  episode RNG;
- nominal and recovery posture biases sampled independently within ±3 degrees.

Nominal and recovery use the same sampled speed. They use separate shape and
posture samples so the recovery does not mechanically replay the nominal route.
Layout and drop RNG streams remain independent from motion-profile sampling so
adding layout retries cannot change an episode's speed or trajectory shape.

The requested sampled profile is written to `events.jsonl` before execution.
Each planner then logs its resolved via-points after applying the safety rules.

## Path geometry

### Pickup

Keep the precise object-hover and grasp waypoints unchanged. Insert a safe
pickup via-point before object hover:

- height is the safe hover height, never the grasp height;
- lateral displacement is bounded by `pickup_via_offset_m`;
- illegal offsets are progressively shrunk; positions are never boundary-clamped.

This bends the approach without moving the grasp endpoint.

### Transport

Keep lift and basket-hover endpoints unchanged. Insert a transport via-point:

- base position is the midpoint of the keep-out-eligible transport stretch,
  from the actual lift position to the point where the direct route enters the
  basket keep-out;
- displacement is perpendicular to the direct XY route;
- signed displacement is sampled within `±transport_via_offset_m`;
- height is the existing safe carry height;
- illegal offsets are progressively shrunk; positions are never boundary-clamped.

This creates smooth left/right route variation while preserving lift, basket,
and release endpoints.

### Recovery

The recovery planner uses the same construction with the recovery-specific
offsets from the episode profile. It starts from the actual post-drop object
pose and therefore computes its via-points when recovery begins.

## Drop sampling on a bent path

The existing straight-path assumption is invalid once a transport via-point is
present. The planner and drop sampler share one immutable carry-path
description:

1. vertical lift;
2. lift point to transport via-point;
3. transport via-point to basket hover.

At lift entry:

1. construct the carry polyline from the actual can and basket poses;
2. clip each segment against the 0.20 m basket keep-out disk;
3. sum eligible arc lengths;
4. draw the run-level `q` coin;
5. on drop, sample one point uniformly over the eligible arc.

Runtime firing uses projection onto the active segment, not action count or
can-basket distance. The trigger fires when projected segment progress passes
the sampled progress. A step that crosses the target still fires on that step.
The trigger is consumed after firing, so there can be at most one drop.

The planner and sampler consume the same carry-path description. They must not
construct duplicate via-points independently.

## Safety and fallback

Random via-points must:

- stay inside the planner workspace;
- stay outside the configured basket keep-out;
- keep pickup approach at safe hover height;
- keep transport at safe carry height;
- preserve exact grasp, lift, basket-hover, and release endpoints.

Resolving a requested offset rejects a point rather than clamping it when
either workspace or basket clearance would be violated, because clamping would
bias points toward the boundary. It tries at most 32 progressively smaller
magnitudes along the same sampled direction. This preserves the requested
left/right route while finding the largest legal bend. If even the straight
segment is illegal, use the straight nominal segment and emit
`trajectory_randomization_fallback`. A fallback is not an episode failure.

The existing runtime grasp and keep-out gates remain authoritative. If either
gate fails at the sampled point, skip the drop and log the reason.
The grasp gate uses the stricter held-midair check (contact, minimum height,
and EEF proximity), not robosuite contact alone.

## Viewer and logs

The live viewer and `events.jsonl` show:

- sampled episode speed;
- requested and resolved nominal pickup and transport offsets;
- requested and resolved recovery pickup and transport offsets;
- posture biases;
- selected drop segment and target progress;
- actual drop phase, position, and basket distance;
- fallback events.

The HUD shows speed, active path phase, and the sampled drop segment/progress.

## Verification

### Unit tests

- Disabled randomization yields zero offsets, speed `1.0`, and deterministic
  nominal paths.
- Enabled profiles stay within configured bounds and reproduce from seed.
- Nominal and recovery receive separate offsets but share episode speed.
- Grasp and basket endpoints remain unchanged.
- Bent-path drop sampling is uniform over eligible arc length.
- No sampled drop point enters the 0.20 m keep-out.
- Segment triggers fire once and cannot be skipped by a large controller step.
- Invalid via-points use the logged straight-path fallback.
- Recipe validation rejects negative offsets, malformed speed ranges, and
  non-positive speed.

### Real LIBERO checks

Run disabled and enabled batches with the same seed range:

- compare completion and recovery-success rates;
- confirm both left and right bends occur;
- confirm sampled speed spans `[0.8, 1.2]`;
- inspect nominal and recovery paths in the enlarged viewer;
- confirm lift and transport drops still occur;
- report measured success-rate change.

The feature is not called safe merely because unit tests pass. If the enabled
batch materially reduces completion or recovery success, reduce offsets before
recording data.

Measured on seeds 0–19 with `q=1`: identity and enabled profiles both completed
19/20 episodes. The enabled run injected exactly 20/20 drops (eight lift, seven
first transport segment, five final transport segment). The absolute 95%
recovery rate is not a production dataset acceptance target; larger
pre-recording batches are still required.
Observed sampled speeds covered 0.811–1.154, both transport-offset signs were
used, lift and both bent-transport phases received drop targets, and the
enabled batch produced no nominal trajectory fallback events.
