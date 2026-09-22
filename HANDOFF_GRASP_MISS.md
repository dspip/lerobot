# HANDOFF — `grasp_miss` failure type (dataset + recovery)

Single source of truth for the **next implementation agent**. Read this before editing. The drop workstream remains [`HANDOFF.md`](./HANDOFF.md); do not change drop latch semantics.

**Owner decision (Aviya, 2026-09-20):** expand the failure-recovery POC with a **new failure type** (not more alphabet-soup drops, not stacked objects). Target object is **Alphabet Soup**. Videos that define the two outcomes live at:

- `/home/aviya/Downloads/eval_episode_0_failure.mp4` — transient miss, VLA recovers on a second pick
- `/home/aviya/Downloads/eval_episode_0_failure_initial_pos.mp4` — soup shifted at **reset**; VLA retries forever at the remembered pose

Do **not** re-derive those stories from extracted frames. An earlier pass misread clutter/perspective as stacking and as a carry-drop. **User ground truth overrides frames.**

---

## 1. Goal

Add in-tree `fault.type=grasp_miss` so we can:

1. **Label** failed soup acquires from simulator GT (including when the EEF is far from the can).
2. **Record** mixed datasets: existing `midair_drop` + `grasp_miss` (transient retries + persistent loops with expert recovery).
3. **Intervene** only after a **persistent** miss: interrupt the VLA, then escalate to the existing IK planner (GraspGen-class regrasp to the **current** soup pose).

Success for this implementation slice (ask before scaling CUDA recording):

- Unit tests in `tests/faults/` with **no GPU/LIBERO**.
- Annotator: a never-held table miss **can** set `is_failure` (today it cannot).
- Drop tests still pass unchanged.
- One dry-run / mock path logs `event: grasp_miss` in JSONL.

CUDA mix recording and Phase 2 head expansion are **follow-on** unless the user asks in the same session.

---

## 2. Investigation summary

### 2.1 Existing drop pipeline (do not break)

| Piece | Path | Behavior |
| --- | --- | --- |
| Latch | `src/lerobot/faults/annotation.py` | `is_failure` = `_ever_held` and not grasped and not in basket. **A grasp miss never latches.** |
| Injector | `src/lerobot/faults/recovery/midair_drop.py` | Fires after a **good mid-air grasp**, then impulse + `SimpleIKRecoveryPlanner`. |
| Wrapper | `DropRecoveryEnvWrapper` in `wrappers.py` | `type='midair_drop'` only. Suppresses success-reset while `recovery_active`. |
| Pulse | `injector_injection_active` | `drop_injection_step` / `just_injected` / `remaining`. |
| Dataset columns | `FAILURE_ANNOTATION_FEATURES` | `is_failure`, `failure_onset`, `failure_type`, `injection_active`, `phase`. **No new columns required for v1.** |
| `loss_mask` | `recovery/loss_mask.py` | `0` only on the inject frame for drops. |
| Phase 1 training | `third_party/Gangelia_Project/smolvla_r_package/phase1_data.py` | Valid iff `loss_mask==1` and not `injection_active`. `sample_type=recovery` iff `phase==3`. |
| Phase 2 head | `phase2_data.py`, `phase2_model.py` | Classes `("none", "slip_or_drop")`. Only `failure_type==1` with `is_failure=True` is a positive. **Any other type is masked invalid.** |

Panda convention in this repo: **`action[6] = +1` close, `-1` open**. Finger close is slow (`speed=0.01`); do **not** treat `action[6] >= 0.5` as proof the gripper is empty-closed. Use **measured finger qpos** (see `force_close_gripper` / gripper joint helpers in `sim/libero.py`).

IDs already used: `0` none, `1` midair_drop, `2`–`11` other injectors. **Assign `FAILURE_TYPE_GRASP_MISS = 12`.** Never reuse `1`.

Teacher-forcing (prior product decision): sim GT may teacher-force the **action adapter** at train time; **test metric is closed-loop success with no GT injection**. Previous-step **predicted** failure may condition the LLM; current GT must not.

### 2.2 The two eval videos (user GT)

| File | What happens | What it is |
| --- | --- | --- |
| `eval_episode_0_failure.mp4` | First pick does not get a solid grasp on **soup**; second try succeeds. No stacking. | **Transient miss** — VLA recovers. Leave policy in control. |
| `eval_episode_0_failure_initial_pos.mp4` | Soup **translated at frame 0 (reset)**. Policy goes to the expected pose, closes empty, loops. Never lifts. | **Persistent miss / memory trap**. Will not self-correct. Escalate. |

Both: target = Alphabet Soup. Offset in the second video is **init perturbation**, not a mid-grasp nudge.

### 2.3 Literature (2025–2026) — recommended vs our first draft

Looked up: ProbeAct ([arXiv:2606.09740](https://arxiv.org/abs/2606.09740), LIBERO-plus), FLARE ([2608.26645](https://arxiv.org/abs/2608.26645)), FailSafe ([2510.01642](https://arxiv.org/abs/2510.01642)), VLA-Corrector, RoboBRIDGE, ProgressVLA ([2603.27670](https://arxiv.org/abs/2603.27670)).

| Keep | Change from the first in-repo sketch |
| --- | --- |
| Retry vs Reset (two regimes, one physical “soup not held”) | Do **not** name this ProbeAct’s “soft/hard empty grasp” (those mean gripper kinematics). Use **transient / persistent**. |
| Do not imitate the stuck loop | Detect with **gripper width + EEF vs object motion**, not “close command for N steps” |
| Do not require EEF-near-soup to label a miss | Far offset → track **soup pose**, not arm proximity |
| Expert recovery for persistent miss | **Ladder**, not IK on the first timeout: (1) VLA retry (2) same-XY miss → interrupt chunk / avoid that approach (3) re-perceive + IK/GraspGen |
| FailSafe: failure paired with a **working** recovery action | Phase 2 must see `loss_mask=0` stall frames for the **classifier**; Phase 1 must not imitate them |

ProbeAct empty-grasp GT (use this for recording):

1. **Empty close:** gripper width at mechanical minimum (nothing between fingers).
2. **Lift without acquire:** `Δz_eef > τ_lift` while `Δz_soup ≈ 0`.
3. **Memory trap:** miss **again** near the same XY → do not keep commanding that pose.

FailSafe injects **translation** failures (the init-offset video) and only keeps pairs whose recovery action passes a sanity check.

---

## 3. Recommended design (implement this)

### 3.1 Physical predicates (LIBERO GT)

Reuse `is_object_grasped`, `is_object_held_midair`, `is_object_in_basket`, `get_object_pose`, `get_eef_pose` from `sim/libero.py`. Add small helpers if needed (gripper opening width, soup XY at reset).

```text
held     = is_object_held_midair(soup)          # grasped + z>=0.12 + eef near soup
table    = soup_z < 0.08
empty_close = gripper_width <= epsilon_closed and not grasped
lift_wo  = (eef_z rose by > tau_lift) and (soup_z change <= tau_noise) and table
stalled  = t >= t_pick_min and table and ||soup_xy - soup_xy0|| < 0.02 and not held

miss_evidence = empty_close or lift_wo or stalled
```

**No** `||eef - soup|| < 12 cm` requirement.

`injection_active`:

- **True** only on reset-offset step(s) if we spawned a shifted soup.
- Native miss (`failure.mp4`) → `injection_active=false` the whole episode.

### 3.2 Latch (`FailureAnnotator`)

Branch on `injector_type_id == 12` (or a `physical_kind="grasp_miss"` flag). **Do not** use `_ever_held` for this type.

```text
onset: first frame miss_evidence is true (and not already cleared)
is_failure: onset_seen and not held and not in_basket and not cleared
clear: first held_midair OR in_basket
_cleared_after_failure: same as drop — basket release / later ungrasp is not a second onset
```

If type is drop (`1`), keep today’s `_ever_held` logic bit-for-bit.

Do **not** default unknown physical failures to `FAILURE_TYPE_MIDAIR_DROP`.

### 3.3 Control ladder (runtime)

Config defaults (tunable; see unresolved questions):

| Rung | When | Action |
| --- | --- | --- |
| 0 Nominal | not `is_failure` | VLA |
| 1 Transient | first miss, `n_misses_at_xy < 2`, within `native_retry_timeout` | **VLA stays in control**. `recovery_active=false`. `phase=2`. |
| 2 Repeat XY | second miss with `||eef_xy - last_miss_xy|| < r_trap` | Interrupt remaining action chunk if the eval loop allows it; optional small retract. Still VLA unless timeout. Log `memory_trap`. |
| 3 Persistent | timeout **or** still not `held` after rung 2 | `recovery_active=true`. `SimpleIKRecoveryPlanner` from **current soup pose** (same as drop regrasp). `phase=3`. |
| Abort | wall-clock cap, still not held, planner also failed | `episode_outcome=failure`. Stop. Do not loop forever. |

Do **not** start the planner on the first failed close.

### 3.4 `loss_mask` and training

| Frames | `loss_mask` | Phase 1 actions | Phase 2 failure label |
| --- | --- | --- | --- |
| Before miss | 1 | yes | `none` |
| Failed close / VLA loop | **0** | **no** | `grasp_miss` if `is_failure` |
| Native successful retry | 1 | yes | `none` after lift (`is_failure` false; type may stay 12) |
| Inject offset frame | 0 | no | as physics |
| Planner recovery | 1 | yes | `none` once `held`; `phase=3` while planner runs |
| Aborted loop (no planner) | 0 after onset | no | `grasp_miss` |

**Phase 2 follow-on (do not silently skip):** `FAILURE_CLASSES = ("none", "slip_or_drop", "grasp_miss")`, map `{1: 1, 12: 2}`, bump `LABEL_SCHEMA_VERSION` to `gangelia.current_failure.v2`. Old 2-class checkpoints will not load. **Also:** auxiliary labels must be readable on `loss_mask=0` frames or the head never sees persistent misses. Today `Phase2Dataset.samples` only includes `_valid` action starts.

### 3.5 Init-offset recording

Apply XY (and optional small yaw) **in `reset`**, store `soup_xy0` **after** the offset. Log one JSONL event `grasp_miss_init_offset`. Do not nudge again at grasp time.

### 3.6 What not to build in this slice

- Stacked-object / GraspGen unstack (different skill).
- ProbeAct CBF QP (optional later; rung 2 can be log + retract first).
- Progress-head GT (`frame_index/T` is forbidden; progress still disabled in current Phase 2).
- New Parquet columns for v1 (optional `persist_miss` later).
- Editing stock LIBERO assets; use overlay / runtime pose write only.

---

## 4. Implementation plan (ordered)

Follow [`.cursor/skills/add-fault/SKILL.md`](./.cursor/skills/add-fault/SKILL.md). New Python: `Copyright 2026 Gangelia. All rights reserved.` + Apache-2.0. Do **not** rewrite HuggingFace headers on files you only edit.

### Slice A — labels + config + tests (no GPU)

1. `annotation.py`
   - `FAILURE_TYPE_GRASP_MISS = 12`
   - `FAILURE_TYPE_FROM_CONFIG["grasp_miss"] = 12`
   - Latch branch; tests in `tests/faults/test_failure_annotation.py`:
     - never-held empty close / lift-without-object → `is_failure`
     - native lift clears; no second onset on place-open
     - drop tests unchanged
2. `config.py`
   - `"grasp_miss"` in `_SUPPORTED_TYPES` and `_RECOVERY_TYPES`
   - Fields (defaults off / conservative): see §6
   - `validate()`
3. Unit tests only: `uv run pytest tests/faults/test_failure_annotation.py -q`

### Slice B — injector + wrapper

4. New `src/lerobot/faults/recovery/grasp_miss.py`
   - Mirror `MidAirDropFault` API: `reset`, `notify_dones`, `on_step`, `after_physics_step` if needed, `_states`
   - State: `soup_xy0`, `miss_count`, `last_miss_xy`, `t_onset`, `recovery_active`, `just_injected` / pulse, `planner`
   - JSONL: `event=grasp_miss`, `status=onset|memory_trap|planner|cleared`
   - On rung 3: reuse `SimpleIKRecoveryPlanner.plan(...)` (already table-pick capable)
5. `factory.py`: `_RECOVERY_TYPES |= {grasp_miss}`; `RecoveryFaultInjector = MidAirDropFault | GraspMissFault`; `make_recovery_fault` (keep `make_midair_drop_fault` as wrapper that only builds drop, or branch inside)
6. `wrappers.py`: `maybe_wrap_env` — `type in {midair_drop, grasp_miss}` → recovery wrapper. Generalize `DropRecoveryEnvWrapper` so it does not hard-require `midair_drop` (keep class name if you want less churn; document it). Same no-autoreset hook.
7. `recovery/__init__.py`, `faults/__init__.py` exports
8. `tests/faults/test_grasp_miss_fault.py` — mocks, no MuJoCo:
   - empty close triggers onset, not planner
   - second miss near same XY logs trap
   - after timeout, `recovery_active`
   - close in free space with soup already `held` does not fire
   - stall with far EEF still onsets
9. `docs/source/fault_injection.mdx` + skill table in `add-fault/SKILL.md`

### Slice C — recording recipe (code only until user greenlights CUDA)

10. `recording_recipe.py`: `training_grasp_miss_kwargs()` (init offset range, timeouts, `seat_assist` N/A)
11. Extend mix planner (`mix_recording.py`) with `kind: grasp_miss` **only if** user wants mix in the same PR; otherwise a dedicated example script `examples/faults/run_grasp_miss_demo.py` mirroring the drop demo (`--dry-run` required).
12. Audit: one `failure_onset`, type `12` on miss frames, `is_failure` false after first `held_midair`, `loss_mask=0` on loop.

### Slice D — training (separate PR unless asked)

13. Phase 2 schema v2 + tests in `third_party/Gangelia_Project/smolvla_r_package/tests/test_phase2_data.py`
14. Do not train until a small CUDA mix is audited (same bar as drop).

**Verify:** `uv run pytest tests/faults -q`

---

## 5. File map

| File | Change |
| --- | --- |
| `src/lerobot/faults/annotation.py` | Type 12 + miss latch |
| `src/lerobot/faults/config.py` | Type + fields |
| `src/lerobot/faults/recovery/grasp_miss.py` | **New** injector |
| `src/lerobot/faults/factory.py` | Register |
| `src/lerobot/faults/wrappers.py` | Dispatch + annotator pulse |
| `src/lerobot/faults/__init__.py` | Exports |
| `src/lerobot/faults/recovery/__init__.py` | Exports |
| `src/lerobot/faults/sim/libero.py` | Gripper-width / optional `offset_object_xy_on_table` reuse if already present |
| `tests/faults/test_failure_annotation.py` | Miss cases |
| `tests/faults/test_grasp_miss_fault.py` | **New** |
| `docs/source/fault_injection.mdx` | Table row |
| `.cursor/skills/add-fault/SKILL.md` | Recovery types list |
| `examples/faults/run_grasp_miss_demo.py` | Optional, with `--dry-run` |
| `third_party/Gangelia_Project/smolvla_r_package/phase2_data.py` | **Not in slice A/B** |

`offset_object_xy_on_table` may already exist in `sim/libero.py` from drop mix work — reuse, do not copy.

---

## 6. Suggested config fields

`FaultInjectionConfig` (all unused unless `type=grasp_miss`):

| Field | Suggested default | Role |
| --- | --- | --- |
| `object_name` | `alphabet_soup_1` | Same as drop |
| `grasp_miss_epsilon_closed` | measure from open/close qpos in tests; start `0.15 * init_qpos` like snap-close | Empty-close |
| `grasp_miss_lift_dz_m` | `0.04` | Lift-without-acquire |
| `grasp_miss_obj_dz_noise_m` | `0.01` | Soup didn’t move |
| `grasp_miss_t_pick_min` | `40` steps @ 20 Hz = 2 s | Stall / never-approach |
| `grasp_miss_native_retry_timeout` | `50` steps | Rung 1 window — **ask user** |
| `grasp_miss_trap_xy_m` | `0.05` | Same-place repeat |
| `grasp_miss_init_offset_xy_m` | `0.04` | Reset shift magnitude — **ask user** |
| `grasp_miss_enable_planner` | `true` | Rung 3 |
| `grasp_miss_episode_cap_steps` | `400` | Abort watchdog |

`enabled=False` by default.

CLI: `lerobot-eval --fault.enabled=true --fault.type=grasp_miss ...` (draccus). No sidecar CLI.

---

## 7. Eval / product metrics

- **Primary:** task success **without** oracle GT in the policy path.
- Report separately: transient (VLA-only) success vs persistent (planner-assisted) success.
- Oracle soup-pose → planner is diagnostic, not the deploy claim.

---

## 8. Unresolved questions — **ask Aviya before assuming**

Implementation can start on slices A–B with the suggested defaults, but **do not CUDA-record or lock timeouts** until these are answered:

1. **Rung 3 expert:** in-tree `SimpleIKRecoveryPlanner` only, or also call GraspGen (`~/Projects/general_grasp/GraspGen`) for regrasp? Handoff default: **in-tree IK first** (same as drop).
2. **Rung 2:** implement action-chunk flush in `lerobot_eval` / SmolVLA queue, or only log `memory_trap` + optional retract in the fault wrapper? Flushing chunks touches eval, not only `faults/`. Handoff default: **wrapper retract + log**; no eval-loop patch unless asked.
3. **Init offset:** magnitude, XY only vs yaw, one object vs random? Videos are “a few centimeters.” Default `0.04 m` if unanswered.
4. **Native retry timeout:** how long to wait for a `failure.mp4`-style second pick before planner? Default 2.5 s (50 steps @ 20 Hz).
5. **Phase 2 in the same PR?** Handoff default: **no** — labels + injector first. Schema v2 breaks 2-class checkpoints.
6. **Mix with the 60-ep drop set** vs a new repo-id? Default: **new dataset root**, do not rewrite `xy_band_mix_60ep`.
7. **Should `target_moved` be a separate `failure_type`** for init-offset vs near-can miss? Default: **one class `grasp_miss` (12)**; distinguish via `injection_active` (offset) vs not (native miss).
8. **Abort vs planner** on persistent miss during the first CUDA smoke: always planner, or record some aborted loops for the classifier only?
9. **Task:** `libero_object` task_id `0` (soup) only, or more objects later? Default: **task 0 only**.

If a question blocks a default above, **stop and ask** rather than inventing a third behavior.

---

## 9. Agent constraints

- Ask mode is off for the implementer; still **do not commit** unless the user asks.
- Do not start the next recording milestone until the user approves (they gated mix/scale for drops before).
- Delegate large coding to Composer if that is the house rule; keep this handoff as the brief.
- `uv run pytest …` not raw pytest.
- `--env.max_parallel_tasks=1` when faults are enabled.

## 10. Chat / extra context

Product discussion: [grasp miss design](7ae088de-f7e1-43ba-8cb0-626102c10188) (this thread). Drop dataset: [`HANDOFF.md`](./HANDOFF.md), `third_party/Gangelia_Project/SmolVLA_Recovery_Codex_Plan.md`.
