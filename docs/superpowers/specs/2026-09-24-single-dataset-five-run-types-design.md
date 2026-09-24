# Single-dataset drop datagen (five run types)

Date: 2026-09-24  
Status: draft — type 3 is SimpleIK → drop → SmolVLA dwell → IK place  
Scope: recording layout and recipe matrix. Frame annotation schema stays the existing Failure Annotated LeRobot spec. No training-loop changes.

## Problem

The current recorder writes **five LeRobot datasets** (`<out>/<controller>/<post_drop_mode>/dataset/`) from a frozen experiment matrix that is the wrong five pairs (SmolVLA-carry variants, `simple_ik × reset_then_ik` forbidden, no no-drop row). It also splits repos and cannot keep a planned fail (type 5) while discarding failed IK recoveries (types 1–3).

Needed: **one** LeRobot dataset fit for VLA finetuning / drop detection, with five **kinds of episode**, not five repos.

## Non-goals

- New frame-level columns (`controller`, `post_drop_mode`, `run_class`, `episode_success` on parquet frames).
- Changing `is_failure` / `failure_onset` / `failure_type` / `injection_active` / `phase` semantics (already implemented in `lerobot.faults.annotation`).
- GraspGen / cuRobo.
- Per-step action noise.
- Resume into a non-empty output dir.
- Implementing a second recording job this slice. Replay with another carry controller / noise flag is a **recipe copy** later; this spec only locks the seed and drop-replay contract so that job is possible.

## Locked decisions

| Topic | Decision |
| --- | --- |
| Dataset count | One LeRobot v3 repo per recording job |
| Episode | One run of **one** type. Never five types in one episode |
| Pairing | For logical index `N`, the five types are five **separate** episodes sharing layout + drop point (type 4 skips the impulse) |
| Frame schema | Existing spec fields only |
| Mission success | Episode-level bool `success` in `meta/episodes`, not a frame column |
| Failed 1–3 | **Discard.** Do not commit. Type 5 is the only planned fail in the dataset |
| Type 3 | JSON key stays `reset_then_ik`. Sequence is **SimpleIK → drop → SmolVLA dwell (`post_drop.dwell_steps`) → IK regrasp/place**. SmolVLA is `policy.reset()` at dwell start. Not a sim reset. |
| Type 5 | Planned: after drop, SmolVLA continues, no IK. Commit even if SmolVLA luckily succeeds |
| Pre-drop grasp fail | Discard (no drop to annotate; not a training episode) |

## On-disk layout

```
<recording.output_dir>/
  dataset/                 # the only LeRobot dataset
    data/
    videos/
    meta/                  # info.json, stats, tasks, episodes (includes success)
  run_manifest.json        # diagnostic; not required for training
  <controller>/<mode>/episode_XXXX/   # existing preview videos / jsonl (unchanged role)
```

`variant_dataset_directory` / `variant_repo_id` become a single root: `<output_dir>/dataset`, repo id `<recipe.name>`.

One `FaultRecoveryDatasetLogger` for the whole matrix. Episodes append in runner order.

## Frame annotations (unchanged)

Every committed frame already carries:

| Field | Meaning |
| --- | --- |
| `is_failure` | Physical failure latch (dropped / ungrasped outside basket after mid-air hold) |
| `failure_onset` | True only on the first failure frame |
| `failure_type` | `0` none, `1` midair_drop (this job only injects 1) |
| `injection_active` | True only while the injector mutates sim state |
| `phase` | `0` nominal, `1` injection, `2` post-fault/unsettled, `3` recovery |

`loss_mask` stays a training helper (not in the Failure Annotated spec):

| Episode kind | `loss_mask` |
| --- | --- |
| Type 4 (no drop) | `1` all frames (only committed if place succeeds) |
| Types 1–3 | Only committed if recovery succeeds: `0` on injection + dwell (type 3 dwell is SmolVLA); `1` on recovery IK and pre-drop SimpleIK |
| Type 5 | `0` from the drop frame through episode end; pre-drop nominal stays `1` |

## Episode metadata

On `save_episode`, write at least:

- `success`: bool — can seated in basket at episode end (existing `EpisodeResult.success`)
- standard LeRobot fields (`episode_index`, `length`, `tasks`, …)

Do **not** put controller or post-drop mode in parquet frames. They may remain on `run_manifest.json` for debugging.

## Five matrix rows (this recipe)

`examples/faults/recipes/can_drop_datagen.json` `experiment_matrix`:

```json
[
  {"controller": "simple_ik", "post_drop_mode": "immediate_ik", "drop": true, "episodes": 1},
  {"controller": "simple_ik", "post_drop_mode": "continue_then_ik", "drop": true, "episodes": 1},
  {"controller": "simple_ik", "post_drop_mode": "reset_then_ik", "drop": true, "episodes": 1},
  {"controller": "simple_ik", "post_drop_mode": "immediate_ik", "drop": false, "episodes": 1},
  {"controller": "simple_ik", "post_drop_mode": "immediate_smolvla", "drop": true, "episodes": 1}
]
```

`--episodes` still overrides every row’s count.

| Row | Carry | After drop | Drop? |
| --- | --- | --- | --- |
| 1 | SimpleIK | IK immediately | yes |
| 2 | SimpleIK | dwell (pass-through / no SmolVLA), then IK | yes |
| 3 | SimpleIK | SmolVLA for `dwell_steps` (fresh `policy.reset()`), then IK regrasp/place | yes |
| 4 | SimpleIK | n/a | no |
| 5 | SimpleIK | SmolVLA until timeout / done; **no** second IK | yes |

Row 4 still lists a `post_drop_mode` because the current schema requires the key; it is unused when `drop` is false.

All five rows use the **randomized** SimpleIK layout (not SmolVLA stock layout). Pairing would be a lie if type 5 used stock poses.

Top-level recipe `q` is ignored for drop/no-drop. The row’s `drop` flag is the coin: `true` → always inject if eligible path exists; `false` → never inject. If eligible path is empty on a `drop: true` row, skip/discard that attempt and retry the same logical index (same seeds) is **not** required this slice: log `no_eligible_path` and do not commit.

## Recipe / runner changes (required)

JSON-only is not enough. Locked validation today forbids this matrix.

1. **Matrix rows** may include `drop: bool`. Equal `episodes` across rows still required.
2. **Stop requiring** the old five `(controller, post_drop_mode)` pairs. Require uniqueness of `(controller, post_drop_mode, drop)` instead. Do not require a fixed set of pairs.
3. **Allow** `simple_ik × reset_then_ik`. Behavior is the type-3 hybrid above, not a no-op reset on SimpleIK.
4. **Add** `PostDropMode.IMMEDIATE_SMOLVLA = "immediate_smolvla"`: after impulse, do not start IK; run SmolVLA until done. `recovery_active` stays false so `phase` is 1 then 2, not 3. Type 3 uses existing `reset_then_ik` dwell then IK (`phase` 2 then 3).
5. **Keep policy:**  
   - Type 4: commit only if place succeeds.  
   - Types 1–3: commit only if drop happened **and** recovery succeeds (`success=true`). Failed IK recovery → discard.  
   - Type 5: commit if drop happened (planned fail). `success` is whatever the sim ended with.  
   - Always discard: layout fail, plan fail, never grasped, no frames, `no_eligible_path`.
6. **Single writer** as above.
7. **Seeds:** see [Replay contract](#replay-contract). Logical `N` shares `layout_seed` / `drop_seed` / `episode_seed` across the five rows. Type 4 uses the same drop seed but does not fire. Persist `drop_trigger` / `trigger_pose` on `run_manifest.json` (already present) so a later job can replay the **same world drop**, not only the same RNG.

## Error handling

| Case | Behavior |
| --- | --- |
| Bad recipe | Abort before env |
| Layout / IK plan fail | Skip that episode attempt; no commit |
| Empty eligible path on `drop: true` | No commit; log `no_eligible_path` |
| Types 1–3 recovery fails | Discard (not in the dataset). Not relabeled as type 5 |
| Types 1–3 drop lands in basket | Keep only if `success=true` at episode end; else discard |
| Type 5 SmolVLA places the can | Rare; still commit. `success=true` if in basket. Mask still 0 from drop to end |
| Output dir not empty | Refuse (existing `StaleRunOutputError`) |

## Tests (CPU)

- Recipe: load the new five-row matrix; reject unknown keys; reject duplicate `(controller, mode, drop)`; accept `simple_ik × reset_then_ik`; accept `immediate_smolvla`.
- Writer: one logger / one `dataset/` path for all rows.
- Keep: types 1–3 commit iff `success=true`; failed IK recovery discards; type 5 commits after a drop; grasp-fail discards.
- Mask: drop→end is 0 only for committed type 5.
- Pairing: same logical index → same `layout_seed` and `drop_seed` on all five rows.
- `success` present on `save_episode` episode metadata (unit-test the logger call), absent from `FAILURE_ANNOTATION_FEATURES`.

## Risks

- Unfreezing the matrix will break tests that assert exactly the old five pairs. Update those tests; do not keep a dual schema.
- `reset_then_ik` with SimpleIK carry now means a post-drop SmolVLA dwell, then IK. GPU is required for types 3 and 5.
- `immediate_smolvla` never starts IK; type 3 always does after dwell.
- Mixing successful 1–4 with planned type 5 fails in one dataset is intended. Failed 1–3 must not appear.
- Same `drop_seed` + a **different carry path** (new controller or trajectory noise) is a **different drop location**. Replay of stored `trigger_pose` is required for identical physics.

## Replay contract

A later dataset with another **nominal controller** and/or **trajectory noise on/off** must be a **new recording** (`output_dir` empty) from a **copied recipe**, not an in-place edit of this dataset.

**Copy unchanged**

- `recording.base_seed`
- `placement` (and `object_names`, `basket_name`, task ids)
- `simple_ik.path_drop` keep-out / eligible phases
- `--episodes` / matrix episode counts
- Logical episode indices `0 .. N-1`

**Change**

- Matrix row `controller` (carry)
- `simple_ik.trajectory_randomization_enabled` (or a future per-row override)
- `recording.output_dir` (required; stale dir is refused)

**What stays the same automatically today**

`layout_seed` and `drop_seed` are derived only from `base_seed + logical_episode_index`. They must **not** include controller, post-drop mode, or the noise flag. Object XY/yaw for scene `N` then matches this job.

**What does *not* stay the same**

Drop is sampled on the **carry polyline**. Changing SimpleIK vs SmolVLA, or turning via-point/posture noise on, changes that polyline. The same `drop_seed` then picks a different point.

**How to get the same drop later**

1. This job writes `drop_trigger` / `trigger_pose` per committed drop episode in `run_manifest.json`.
2. A later runner (not this slice) takes that manifest and **replays the stored pose/step** instead of sampling from `drop_seed`.
3. Until that replay path exists, a recook with a new controller or noise flag is “same table layout, new drop timing.”

`controller_seed` may depend on controller × mode (it already does). That is allowed: arm motion can differ; layout must not.
