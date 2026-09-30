# IK episode previews and controller-specific layouts

## Goal

The unified drop-datagen runner will:

1. Save the same diagnostic preview artifacts for every SimpleIK attempt that the
   SmolVLA pipeline saves.
2. Keep the current randomized object layout for SimpleIK while running SmolVLA
   on the stock LIBERO layout.

This deliberately removes layout pairing between SimpleIK and SmolVLA. Pairing
remains within each controller family: both SimpleIK modes share one randomized
layout, and all three SmolVLA modes share one stock layout for a logical episode.
Episode, drop, and controller seeds retain their current behavior.

## SimpleIK preview artifacts

Every SimpleIK `episode_XXXX/` directory will contain:

- `videos/full_pipeline.mp4`
- `videos/full_pipeline.gif`
- `final_state_multicam.png`
- the existing `fault_events.jsonl`

The MP4 and GIF use the same external environment render and top phase banner as
SmolVLA. A frame is captured initially and after every simulation step, including
attempts later rejected by the dataset writer. The banner identifies the SimpleIK
phase and simulation step. The MP4 uses the simulation control frequency. The GIF
is downsampled to at most roughly 100 frames and encoded at 8 FPS, matching the
current SmolVLA diagnostic behavior.

The final proof image retains the current SmolVLA multi-camera format: a basket
top view followed by available `agentview`, `frontview`, `birdview`, and
`sideview` renders.

Preview output is diagnostic and independent of the LeRobot dataset. Dataset
camera streams continue to be written under each variant's `dataset/videos/`
tree, subject to the existing keep/discard policy.

## Shared preview component

Rendering and encoding utilities currently private to `smolvla_pipeline.py` will
move into a focused datagen preview module. That module owns:

- phase-banner rendering;
- environment frame capture with last-frame fallback;
- MP4 and GIF encoding;
- final multi-camera proof rendering;
- final artifact path creation.

Both controller paths use this component. SmolVLA artifact names and behavior
remain unchanged. SimpleIK supplies phase and step labels through an optional
post-step callback in its episode loop. The adapter finalizes previews before
closing the MuJoCo environment, including early-return and failed-outcome paths.
Encoding failures raise a contextual recording error instead of silently claiming
that artifacts were written.

## Controller-specific layouts

The recipe will explicitly request stock SmolVLA placement. Stock means:

- XY displacement is zero;
- yaw displacement is zero;
- the selected LIBERO init state still varies by episode seed.

The runner resolves two layouts per logical episode:

1. `simple_ik_layout`: sampled with the top-level randomized `placement` recipe;
2. `smolvla_layout`: sampled from the same init state with zero XY and zero yaw.

Each matrix row receives the layout for its controller. Both modes of a
controller receive the same layout object. The actual applied layout remains
recorded in each result and manifest details so the output is auditable. The
layout seed remains recorded, but consumers must not interpret equal layout
seeds as equal cross-controller poses when the stock override is enabled.

The handoff documentation will be updated to remove the cross-controller layout
pairing claim and describe the new within-controller pairing contract.

## Tests

Tests will be written before production changes and will cover:

- the SimpleIK loop invoking preview capture initially and after simulation
  steps with phase/step labels;
- SimpleIK finalizing MP4, GIF, and final proof paths before environment close;
- preview artifacts being generated for unsuccessful/rejected attempts;
- the runner passing one randomized layout to both SimpleIK modes and one stock
  layout to all SmolVLA modes;
- stock SmolVLA placement preserving reset positions while applying no XY or yaw
  perturbation;
- result metadata containing the actual controller-specific layout;
- existing SmolVLA preview behavior remaining unchanged.

## Non-goals

- Keeping rejected attempts in the LeRobot training dataset.
- Fixing SmolVLA policy performance.
- Adding per-camera raw MP4 files to episode directories.
- Restoring cross-controller layout comparability while the stock SmolVLA
  override is enabled.
