# HANDOFF — Post-drop dwell (teammate request + our fixes)

> **For agentic workers:** implement this file as the spec. Use
> `.cursor/skills/add-fault/SKILL.md` for fault-package conventions. Use TDD:
> several tests already exist and currently **fail** because production code
> was never updated. Finish those tests first; then add the mode/recipe work.
> Do **not** re-run SmolVLA GPU diagnostics unless a new claim needs traces.

**Repo:** `/home/aviya/Projects/lerobot`
**Working tree:** implement in the current tree (often `main`). Do not switch
to `feat/head-triggered-recovery` unless the user says so.
**Date:** 2026-09-23
**Status (2026-09-23):** Unified drop datagen is implemented on branch
`feat/unified-drop-datagen`. Use `examples/faults/recipes/can_drop_datagen.json`
and `examples/faults/run_drop_datagen.py` (five controller×mode variants,
paired seeds, `run_manifest.json`). Sections below that reference deleted
files (`can_simpleik_datagen.json`, 50/50-only mix, `matrix_results.json`,
`DatagenEventLog`, viewer modules) are **historical** unless marked current.

This file supersedes:

- `/home/aviya/Downloads/HEAD_DATA_PLAN_post_drop_dwell_2026-09-22.md`
  (scripted Low/High hover, wander, missed-grasp mix, 2–20 s uniform dwell,
  70/30 recovered split, delete `_cleared_after_failure`) — **rejected**.
- The earlier “do not implement reset at all” paragraph in this handoff —
  **replaced**. Reset is an **optional tagged mode**, not the default, not
  scripted hover.

---

## 0. What to tell the teammate (one screen)

Their goal stands: the failure head never saw “can on table, arm elsewhere,”
so live recall collapsed. **Fix: wait after the drop, then start the existing
IK planner from live poses.**

Our fixes vs their original plan:

| They asked | We ship | Why |
| ---------- | ------- | --- |
| Scripted hover / wander / missed grasp | **No.** Passive SmolVLA (or optional reset) during dwell | Measured: stock SmolVLA **leaves**; hover is not this policy |
| 2–20 s uniform dwell | **Fixed 80 env steps (4 s at 20 Hz)** first | Extra seconds after ~1 s are mostly duplicates |
| Always-on retry / hover | Three **tagged** modes in JSON; default is continue | Reset retries exist only if you clear the action queue |
| Delete `_cleared_after_failure` | **Do not** | Labels a successful place as a second drop |
| `basket_dist < 0.15` success | **Do not** | That oracle does not exist; keep cylinder checks |
| Shared recipe JSON | **Yes (current):** `examples/faults/recipes/can_drop_datagen.json` | Unified matrix: SimpleIK + SmolVLA × post-drop modes |

Default recording mix (current recipe): **all five** approved controller×mode
pairs with equal episode counts; `post_drop.dwell_steps` ≥ 1 (80 in the checked-in
recipe). `immediate_ik` is included for SimpleIK/SmolVLA where supported, not
weighted to zero.

---

## 1. Problem

The failure head was trained on `xy_band_mix_60ep`. There a drop is followed
**immediately** by scripted IK, so `is_failure` lasts ~1–2 s. In live SmolVLA
eval the can sits on the table while the queued carry chunk walks the arm
away. That scene is missing from training → live recall ~0.06 vs offline ~1.0.

The annotator rule does not need to change for a single drop. The recorder
must **wait**, mask dwell actions out of the **action** loss, then start IK.

---

## 2. Measured facts (do not re-litigate without new traces)

Policy: `lerobot/smolvla_libero`. Task: LIBERO object 0. Control **20 Hz**.
SmolVLA `n_action_steps=50` ≈ 2.5 s queued carry. Stock eval
(`src/lerobot/scripts/lerobot_eval.py` `rollout()`) calls `policy.reset()`
**once per episode**, not on drop.

Diagnostic: `examples/faults/diagnose_post_drop_vla.py` (injector monkeypatched
so drop does not start IK).

### 2.1 Mid-band drop, **no** `policy.reset()` — this is stock eval

Trigger: XY band **[0.34, 0.38] m**, `post_grasp_delay_steps=0`, `t_min=0`.

| Run | Seeds that dropped | Regrasp | Closest EEF–can XY after t>1 s | Grasp *attempt* (closed gripper and XY&lt;10 cm after t&gt;1 s) |
| --- | --- | --- | --- | --- |
| 20 s traces `outputs/post_drop_vla_diagnostic_mid/` | 9300, 9301, 9302 | **0** | ~0.21 m | **0** (only the first ~0.4 s still over the can) |
| 8 s traces `outputs/post_drop_8s_noreset/` | 7 of 12 seeds (9500, 9501, 9502, 9505, 9507, 9508, 9510) | **0** | **0.206 m** min | **0** |

After drop: t=0 EEF 2–5 cm from can, gripper closed. By **t≈1 s** EEF **>20 cm**
away and stays 22–36 cm for the rest of the window. Wrist camera rarely sees
the can (~3–14% of frames after 1 s). This is **departure**, not hover, not a
failed regrasp.

Delay-based drop (carry 20–60 steps, no XY band) **never fired** in the
diagnostic: the can reaches the 0.22 m keepout during the wait. Do not use
delay-based drop for this mix. Match `xy_band_mix_60ep`: XY bands + delay 0.

### 2.2 Same drop **plus** `policy.reset()` on the drop step

Traces: `outputs/post_drop_vla_diagnostic_mid_reset/`.

Reset **does** produce hover / close / retry (seed 9401 regrasped at step 137
≈ 6.9 s). Dwell pose is **not** the continue pose:

| Mode | median EEF–can XY | wrist sees can |
| ---- | ----------------- | -------------- |
| continue (no reset) | 0.23–0.31 m | 3–14% |
| reset on drop | 0.067–0.084 m | 84–100% |

Reset dwell overlaps the **pre-first-grasp approach pose** (hard negatives for
the head). That is why it must be **tagged and optional**, not the default.

### 2.3 Why the teammate saw reset-like motion

`examples/faults/run_head_recovery_rollout.py` calls `policy.reset()` inside
`_query()` on **every chunk refresh**. That harness never drains a leftover
carry chunk. Stock `lerobot_eval.py` does not do this. Do not “fix” eval to
match the head harness. Record both distributions if they want both, tagged.

---

## 3. Architecture (what to build)

Three post-drop **modes**. Dwell length is env **control** steps (20 Hz), not
dataset frames (10 FPS).

| `post_drop_mode` | On drop | During wait | When wait ends |
| ---------------- | ------- | ----------- | -------------- |
| `immediate_ik` | Drop + start IK **this step** (today) | n/a | n/a |
| `continue_then_ik` | Drop only. **Do not** `policy.reset()`. Execute incoming VLA actions | `triggered=True`, `recovery_active=False` | If still ungrasped and not in basket → `_start_recovery_planner` from **live** poses |
| `reset_then_ik` | Drop only. Set a one-shot flag so the **recorder** calls `policy.reset()` (clears SmolVLA action queue). Fault class has **no** policy object | Same as continue | Same IK start / skip rules |

**Library defaults** (keep existing unit tests green without extra kwargs):

- `post_drop_dwell_steps: int = 0`
- `post_drop_mode: str = "immediate_ik"`

`dwell_steps == 0` ⇒ always `immediate_ik` behavior (ignore mode except log it).
`reset_then_ik` with `dwell_steps == 0` ⇒ `ValueError`.

**Training defaults** (`training_midair_drop_kwargs` / recipe):

- `post_drop_dwell_steps = 80` (`DEFAULT_POST_DROP_DWELL_STEPS`)
- `post_drop_mode = "continue_then_ik"`

### 3.1 Timeline (`continue_then_ik` / `reset_then_ik`)

1. VLA pick + carry.
2. Drop when object–basket XY in the configured band (checkpoint mix: mid
   **[0.34, 0.38]** among other `XY_BANDS`; `post_grasp_delay_steps=0`).
3. Dwell N env steps of VLA (continue) or reset-then-VLA (reset mode).
4. Timeout: start existing IK if object ungrasped and not in basket.
5. If VLA regrasps **or** object is in basket before timeout → **do not**
   start IK; keep executing VLA.

Match the **already written** tests in `tests/faults/test_midair_drop_fault.py`
for the dwell counter (do not invent a different off-by-one):

- `post_drop_dwell_steps=2`: drop step + **two** further `on_step` calls still
  pass the policy action and do **not** start the planner; the **fourth**
  `on_step` starts IK (unless grasped / in basket).
- `post_drop_dwell_steps=0`: first `on_step` that drops also starts IK (today).

### 3.2 Labels / masks

Do **not** edit `src/lerobot/faults/annotation.py` unless a test proves a label
is wrong.

| Frames | `is_failure` | `loss_mask` |
| ------ | ------------ | ----------- |
| Drop injection | True (physical) | 0 (already) |
| Dwell (can free, not in basket) | True (existing annotator) | **0 (new)** |
| IK after dwell | False once grasped / in basket | **1** (action policy learns recovery) |
| VLA regrasp during dwell | False once grasped | 0 on all dwell frames |

`loss_mask_for_env` today is 0 **only** on `drop_injection_step`. Dwell must
also return 0.

`loss_mask_from_fault(..., post_drop_dwell_step: bool = False)`: if
`drop_injection_step or post_drop_dwell_step` → 0.0 else 1.0. Keep using
`drop_injection_step` for the physical drop frame even if recovery is already
active (`immediate_ik`).

`choose_phase`: with recovery delayed, `PHASE_POST_FAULT` (2) appears when
`is_failure` and not recovering. Desired. Do not change `choose_phase`.

### 3.3 Policy reset (reset mode only)

`MidAirDropFault` must **not** import or call the policy.

On the drop step in `reset_then_ik`:

- `state.policy_reset_requested = True` (one-shot).
- Log `status="triggered"` with `"post_drop_mode": "reset_then_ik"`.

`examples/faults/run_full_drop_recovery_pipeline.py` (the only place that
owns `policy`): after `env.step`, if
`getattr(env.fault._states[0], "policy_reset_requested", False)`:

```python
policy.reset()
env.fault._states[0].policy_reset_requested = False
```

Do this **once** on the drop step, not every chunk (that is the head-eval
bug), and **not** after a 1 s VLA wait. See §12.1: physics settle already
ran inside the drop; a control-step wait before reset walks the arm away.
`DropRecoveryEnvWrapper` may expose `consume_policy_reset(env_idx) -> bool`
instead of the pipeline poking `_states`; prefer that helper.

Eval (`lerobot_eval.py`) stays unchanged: one reset per episode.

### 3.4 Keep gate / annotator / basket — out of this slice

Keep `should_keep_checkpoint_drop` (regrasp + in basket, no seat assist).
Do **not** start saving timed-out table drops until videos look right.

Do **not** change:

- `_cleared_after_failure`
- `is_object_in_basket` (annotator `z_max=0.20`, keep-check `z_max=0.14`)
- `HARD_BASKET_KEEPOUT_M=0.22` / recipe min 0.30 m
- `xy_band_checkpoint.py` keep rules unless wiring a new kwarg through
  `run_pipeline` / `fault_overrides`

---

## 4. Files

| Path | Job |
| ---- | --- |
| `src/lerobot/faults/config.py` | Fields + `validate()` |
| `src/lerobot/faults/recovery/midair_drop.py` | Split drop vs planner; dwell counter; mask; log mode/dwell; `consume_policy_reset` |
| `src/lerobot/faults/recovery/loss_mask.py` | `post_drop_dwell_step` |
| `src/lerobot/faults/recovery/recording_recipe.py` | Training defaults + optional recipe load |
| `src/lerobot/faults/recovery/__init__.py` | Export new constants if added |
| `src/lerobot/faults/wrappers.py` | Optional `consume_policy_reset`; no policy import |
| `examples/faults/run_full_drop_recovery_pipeline.py` | Pass dwell/mode; reset once; summary `fault_config` keys |
| `examples/faults/run_xy_band_checkpoint.py` | Load recipe / pass kwargs (do not hardcode hover) |
| `examples/faults/recipes/can_simpleik_datagen.json` | **Create.** Shared with teammate |
| `examples/faults/recipes/README.md` | One paragraph: what belongs here |
| `tests/faults/test_midair_drop_fault.py` | **Already extended** (failing). Make them pass; add mode tests |
| `tests/faults/test_recording_recipe.py` | **Already expects** `DEFAULT_POST_DROP_DWELL_STEPS` |
| `tests/faults/test_libero_sim_helpers.py` | **Already expects** `post_drop_dwell_step=True` → 0.0 |
| `tests/faults/test_datagen_recipe.py` | **Create.** JSON load, weights, validation |

Do **not** edit `annotation.py` or invent world-frame hover scripts.

New files: `Copyright 2026 Gangelia. All rights reserved.` + Apache-2.0.
Do not rewrite HuggingFace headers on files you only touch.

---

## 5. Config contract

`FaultInjectionConfig` (midair_drop):

```python
POST_DROP_MODES = ("immediate_ik", "continue_then_ik", "reset_then_ik")

post_drop_dwell_steps: int = 0
post_drop_mode: str = "immediate_ik"
```

Validate (midair_drop branch of `validate()`):

- `post_drop_dwell_steps >= 0` (tests already match `ValueError` / `"post_drop_dwell_steps"`).
- `post_drop_mode` in `POST_DROP_MODES`.
- if `post_drop_mode == "reset_then_ik"` and `post_drop_dwell_steps == 0`: error.

`training_midair_drop_kwargs(...)` extra args (defaults shown):

```python
DEFAULT_POST_DROP_DWELL_STEPS = 80  # 4 s at 20 Hz

def training_midair_drop_kwargs(
    ...,
    post_drop_dwell_steps: int | None = None,
    post_drop_mode: str = "continue_then_ik",
) -> dict[str, Any]:
    ...
    "post_drop_dwell_steps": (
        DEFAULT_POST_DROP_DWELL_STEPS if post_drop_dwell_steps is None
        else int(post_drop_dwell_steps)
    ),
    "post_drop_mode": str(post_drop_mode),
```

`run_pipeline(...)` should accept `post_drop_dwell_steps` / `post_drop_mode`
and put them into `training_midair_drop_kwargs` (or `fault_overrides`). Log
them in `summary["fault_config"]`. Episode metadata must include the mode so
reset episodes can be dropped later without re-recording.

---

## 6. Shared recipe JSON (teammate merge)

Create `examples/faults/recipes/can_simpleik_datagen.json`. **Additive**
`post_drop` object only in this PR. Leave placeholders the teammate already
uses on their branch **if you know the keys**; otherwise do not invent
`initial_pose` / `object_set` schemas — they own those keys. Document in the
README: merge by adding sibling keys, do not rename `post_drop`.

```json
{
  "name": "can_simpleik_datagen",
  "task": "libero_object",
  "task_id": 0,
  "object_name": "alphabet_soup_1",
  "control_hz": 20,
  "post_drop": {
    "dwell_steps": 80,
    "mode_weights": {
      "continue_then_ik": 0.5,
      "reset_then_ik": 0.5,
      "immediate_ik": 0.0
    }
  }
}
```

Loader (pure, no GPU) e.g. `load_datagen_recipe(path) -> dict` +
`sample_post_drop_mode(rng, weights) -> str`:

- Weights must be ≥ 0 and sum > 0.
- Unknown mode keys → `ValueError`.
- Default file above samples `continue_then_ik` or `reset_then_ik` (50/50).

Checkpoint recorder: for each drop episode, sample mode from recipe weights,
pass dwell_steps + mode into `run_pipeline`. Write sampled mode into the
attempt log / `info.json`.

If the teammate JSON appears on merge with extra keys, loader must
`json.load` and only *require* `post_drop.dwell_steps` + `post_drop.mode_weights`.

---

## 7. Injector implementation notes

Hook today (`MidAirDropFault._trigger_drop`): drop physics then immediately
`_start_recovery_planner`. Split:

- Always `_drop_object` + `state.triggered = True` + log `triggered`.
- If `post_drop_dwell_steps == 0`: start planner and return recovery action
  (existing tests: `test_zero_dwell_starts_recovery_on_drop_step`,
  `test_drop_injection_step_loss_mask`).
- Else: **return the proposed VLA action** (not a planner action). Set
  `state.dwell_steps_completed = 0` (or equivalent). If mode is
  `reset_then_ik`, set `policy_reset_requested`.
- Pattern already in `trigger_manual_drop` (drop without planner).

`on_step` when `triggered` and not `recovery_active`:

- Increment dwell counter.
- If grasped or in basket: never start IK; pass VLA action.
- Elif dwell elapsed per §3.1 tests: `request_recovery` /
  `_start_recovery_planner` from current poses; return recovery action.
- Else: pass VLA action.

`loss_mask_for_env`: 0.0 if `drop_injection_step` **or**
(`triggered` and not `recovery_active`). After IK starts, 1.0 except the
injection frame in `immediate_ik` (injection and recovery same step → 0).

Log on trigger and on recovery start: `post_drop_dwell_steps`,
`post_drop_mode`.

`_EnvDropState` additions: dwell counter, `policy_reset_requested: bool = False`.
`reset()` / `notify_dones()` must clear them (new `notify_dones` test already
clears recovery; extend if needed).

Do not call `policy.reset()` from the injector.

---

## 8. Tests (no GPU / no MuJoCo)

**Already in tree, currently failing — make these pass first:**

- `tests/faults/test_libero_sim_helpers.py::test_loss_mask_from_fault_state`
- `tests/faults/test_recording_recipe.py::test_training_kwargs_disable_seat_assist_and_require_carry`
- `tests/faults/test_recording_recipe.py::test_training_kwargs_post_drop_dwell_override`
- `tests/faults/test_midair_drop_fault.py::test_invalid_config_errors` (`post_drop_dwell_steps: -1`)
- `test_zero_dwell_starts_recovery_on_drop_step`
- `test_post_drop_dwell_passes_policy_and_delays_planner`
- `test_dwell_loss_mask_zero_until_ik`
- `test_regrasp_during_dwell_skips_ik`
- `test_object_in_basket_after_dwell_skips_ik`

**Add (still mocked):**

- `reset_then_ik` drop sets `policy_reset_requested`; `consume_policy_reset`
  returns True once then False. Injector still does not start IK until dwell
  elapses.
- `immediate_ik` is unchanged when `dwell_steps=0`.
- Config: `reset_then_ik` + `dwell_steps=0` raises.
- Recipe load + `sample_post_drop_mode` on the default JSON (50/50 continue/reset
  per §12.6); with 100% continue; with 100% reset; invalid weights.
- Existing midair_drop tests with no new kwargs still pass (`dwell_steps=0`).

Run:

```bash
uv run pytest tests/faults/test_midair_drop_fault.py tests/faults/test_recording_recipe.py tests/faults/test_libero_sim_helpers.py tests/faults/test_datagen_recipe.py -q
uv run pytest tests/faults -q
```

Do not require CUDA for the implementation PR.

---

## 9. Explicitly rejected (still)

| Idea | Why |
| ---- | --- |
| Scripted hover / wander / missed-grasp percentages | Wrong distribution for stock SmolVLA |
| Default `policy.reset()` on every drop | Not what `lerobot_eval` does; head-precision risk (§2.2) |
| Delay-based drop for this dwell mix | Never fires (§2) |
| Delete `_cleared_after_failure` | False second failure on place-release |
| New `basket_dist < 0.15` success | Oracle split; cylinder already exists |
| 2–20 s uniform dwell, 70/30 recovered, 2000-frame quota | Unmeasured; 4 s recovered batch first |
| Second-drop labelling / latch deletion | Needs a real multi-interval machine later |
| Patching `lerobot_eval.py` for dwell | Eval integration remains `maybe_wrap_env_tree` only |
| Calling `policy.reset()` every chunk like `run_head_recovery_rollout.py` | Not eval; would manufacture retries always |

Compatible with teammate varied starts + generalized IK **if** dwell is
policy-driven (continue or reset) and IK still starts from current poses.

---

## 10. After code lands (stop and ask the user)

**Closed in §12.6:** default datagen recipe is **50/50** `continue_then_ik` /
`reset_then_ik`; recorded frames export **`ever_held_midair`** (physical
`is_failure` formula unchanged).

Remaining human gates:

1. One CUDA mid-band episode with **continue** dwell; parquet: failure window
   ≈ wait, `loss_mask=0` on dwell, IK `loss_mask=1`, video is departure.
2. Sampled wait vs fixed 80; keep non-recovered drops; unify basket `z_max`.
3. Before scaling a full XY-band mix: measure head precision on
   **pre-first-grasp** frames in reset episodes (§2.2). Tag
   `post_drop_mode` lets you drop those episodes without re-recording.

Related: `HANDOFF.md` (drop labels), `HANDOFF_XY60_HEAD_RECOVERY.md` (head
eval). Do not fold `grasp_miss` into this work (`HANDOFF_GRASP_MISS.md`).

Diagnostics already in tree (keep; not the product):

- `src/lerobot/faults/recovery/post_drop_trace.py`
- `tests/faults/test_post_drop_trace.py`
- `examples/faults/diagnose_post_drop_vla.py`

Re-run only if needed:

```bash
MUJOCO_GL=egl uv run python examples/faults/diagnose_post_drop_vla.py \
  --output-dir outputs/post_drop_vla_diagnostic_mid \
  --seeds 9300,9301,9302 \
  --post-grasp-delay-steps 0 --t-min 0 \
  --drop-xy-band-min 0.34 --drop-xy-band-max 0.38
```

`--reset-policy-on-drop` is the contrast experiment, not the training default.

Gripper open in the diagnostic: last action dim **&lt; 0** (LIBERO/Panda).
Wrist visibility is an approximate projection.

---

## 11. Suggested implement order

1. Config fields + `loss_mask_from_fault` so helper tests pass.
2. `training_midair_drop_kwargs` defaults so recipe tests pass.
3. `_trigger_drop` / `on_step` dwell so midair_drop tests pass.
4. `consume_policy_reset` + reset-mode tests.
5. Recipe JSON + loader tests.
6. Wire `run_pipeline` + checkpoint + summary metadata.
7. `uv run pytest tests/faults -q`.

Do not commit unless the user asks.

---

## 12. User decisions (2026-09-23, this session)

Supervisor records decisions here. Do not re-decide them in code.

1. **Implement this handoff**, not the teammate Downloads plan. The Downloads file stays rejected (§9).
2. **Reset is a real option and must be tested**, not only sketched. If later traces are bad, drop those episodes by tag. *(Superseded for recipe weights by §12.6.)*
3. **Default dataset stays `continue_then_ik` at weight 1.0**, dwell 80. *(Superseded by §12.6 — 50/50 continue/reset.)*
4. **Reset timeline the user asked for**, after the check below:

   `drop → (control wait only if the can is still airborne) → policy.reset() once → VLA for the dwell → existing IK from live poses`

### 12.1 Check: is a ~1 s wait before reset needed?

**No extra control-step wait.** [certain]

- `midair_drop` already opens the gripper and runs `physics_settle = max(settle_steps, 80)` MuJoCo substeps **before** the next env action (`src/lerobot/faults/sim/libero.py`). Training uses `settle_steps=100`. The can is already falling/landed inside the drop call. The arm has not taken another VLA step yet.
- Measured continue traces (§2.1): at the first control frame after the drop the EEF is **2–5 cm** from the can. By **t ≈ 1 s** it is **>20 cm** away, because the queued carry chunk is still executing.
- A 1 s **VLA** wait before `policy.reset()` would spend that second walking the arm away, then reset from the far pose. That is the continue distribution, not a retry near the can.
- Therefore `reset_then_ik` calls `policy.reset()` **once on the drop step** (after physics settle, before further VLA steps), then runs VLA for `post_drop_dwell_steps` (80), then starts IK if still ungrasped and not in basket. No `pre_reset_wait_steps` field unless a new trace shows the can still airborne at the first control frame.

### 12.2 Teammate plan — response to every ask

Source: `/home/aviya/Downloads/HEAD_DATA_PLAN_post_drop_dwell_2026-09-22.md`. Their goal (record “can on table, arm not immediately regrasping”) is accepted. Their generator is not.

| Their ask | Response | Where it is handled |
| --- | --- | --- |
| Failure head never saw long post-drop rest; live recall ~0.06 | **Accept the gap.** Record a real wait after the drop before IK. | §1, dwell in `midair_drop.py` |
| Label rule unchanged (`held earlier AND not grasped AND not in basket`) | **Accept.** Formula unchanged. Recorded schema also exports `ever_held_midair` from `_ever_held` (§12.6). *(Superseded: “do not edit `annotation.py`” — schema-only export landed in §12.6.)* | §3.2, §12.6 |
| Scripted low hover ~40% | **Reject.** Stock SmolVLA leaves; hover is not this policy (§2.1). | §9 |
| Scripted high hover ~25% | **Reject.** Same. | §9 |
| Scripted wander ~20% | **Reject.** Continue-mode VLA already leaves the wrist view (3–14% wrist-visible after 1 s). Do not script waypoints. | §2.1, §9 |
| Scripted missed grasp ~15% | **Reject.** Continue traces had **0** grasp attempts after 1 s. Do not fake offsets. | §2.1, §9 |
| Chain 1–3 behaviours, vary speed | **Reject.** Motion during dwell is the policy. | §9 |
| Dwell 2–20 s uniform | **Reject for this dataset.** Fixed **80 env steps (4 s at 20 Hz)**. Extra seconds after ~1 s were mostly duplicates on continue. | §0, recipe JSON |
| Keep recovered ~70% | **Not this slice.** Keep gate stays regrasp + in basket. Do not start saving `no_regrasp` until videos look right. | §3.4 |
| Keep non-recovered ~30%, `loss_mask=0` after drop | **Mask yes, keep no.** Dwell and the drop frame get `loss_mask=0`. Episodes that never regrasp are still rejected by the existing keep gate. | §3.2, §3.4 |
| Second drop ~10%, delete `_cleared_after_failure` | **Reject.** Labels a successful place-release as a second failure. Needs a real multi-interval annotator later. | §9 |
| Drop bands as today | **Accept.** XY bands, `post_grasp_delay_steps=0`. Delay-based drop never fired (§2.1). | checkpoint wiring |
| Force ≥20% tipped landings | **Reject for this slice.** Do not change the impulse hook to force tips. | §3.4 |
| Nudges that move the can 2–50 cm | **Reject.** That was the missed-grasp script. | §9 |
| Wrist-out-of-view and occlusion frames | **Emerge from continue VLA**, not from scripted wander. | §2.1 |
| Do not reduce pre-grasp / carry / in-basket negatives | **Accept.** No nominal-episode deletion. Reset dwell overlaps pre-grasp (§2.2). **Default recipe is 50/50 continue/reset (§12.6)**; tag `post_drop_mode` to filter reset episodes if head precision is poor. *(Superseded: reset weight 0 in default recipe.)* | §2.2, recipe, §12.6 |
| ~2000 dwell frames, ~20–25 episodes, +10 nominal, 20% holdout | **Reject the quota.** Ship the recorder + recipe first. One continue episode and one reset episode are the proof runs, not a 2000-frame mix. | §10 |
| Keep `no_regrasp` / `not_success` when they contain dwell | **Reject for this slice.** | §3.4 |
| Log `dwell_behaviours` | **Reject** (no scripted behaviours). **Do log** `post_drop_mode` and `post_drop_dwell_steps` on the episode summary so reset episodes can be dropped later. | §5 |
| Acceptance: median failure window ≥ 8 s, min ≥ 2 s | **Reject the 8 s bar.** Target window is the 4 s dwell (80 control steps; dataset frames at 10 FPS are about half of that). | §0 |
| ≥500 frames EEF >15 cm lateral | **Expected from continue**, not a scripted quota. | §2.1 |
| ≥500 frames EEF 2–10 cm above can, gripper open | **That is hover.** Continue does not do it. Reset does (§2.2); **50% of sampled episodes use reset (§12.6)**. | §2.2, §12.6 |
| ≥300 tipped-can failure frames | **Out of slice.** | §3.4 |
| Non-recovered episodes present with `loss_mask=0` | **Mask on dwell is in slice. Keeping non-recovered episodes is not.** | §3.2, §3.4 |
| Retrain head; recall ≥ 0.8 in every time bucket | **Out of slice.** Data recorder only. | — |
| Single-frame head cannot separate upright landing from pre-grasp without history | **Accept as a warning.** Reset dwell can look like pre-first-grasp (§2.2). Mix is **50/50 by §12.6** — use `ever_held_midair` history downstream and measure before scaling. *(Superseded: “reset is not the default”.)* | §2.2, §12.6 |
| Code pointers: dwell between trigger and planner | **Accept the insertion point.** | §7 |
| Code pointers: edit `annotation.py` | **Partial (§12.6).** Reject changing the physical `is_failure` rule; **accept** exporting `ever_held_midair` in the annotation schema only. *(Superseded: blanket reject.)* | §12.6 |
| Code pointers: loosen `should_keep_checkpoint_drop` | **Reject this slice.** | §3.4 |
| Code pointers: `loss_mask.py` | **Accept**, for dwell frames, not for a new non-recovered keep policy. | §3.2 |
| Code pointers: summary fields in the pipeline | **Accept** for mode + dwell steps. | §5 |

### 12.3 What “fixed dataset” means here

The next XY-band recording uses `examples/faults/recipes/can_simpleik_datagen.json`: dwell 80, **50/50** `continue_then_ik` / `reset_then_ik` (§12.6). Episode metadata stores the sampled mode. Recorded frames export **`ever_held_midair`** alongside `is_failure` for failure-head history input.

### 12.4 Implementation log

Implemented on branch `feat/fix-dataset-dwell` by worker [post-drop dwell](a7ea610c-751e-4443-aa7e-0cd8342e5a55). Supervisor reviewed the injector and re-ran tests. Not committed.

Code:

- `FaultInjectionConfig.post_drop_dwell_steps` default `0`, `post_drop_mode` default `immediate_ik`. `reset_then_ik` with dwell `0` raises.
- Training kwargs default dwell `80`, mode `continue_then_ik`.
- Drop with dwell `> 0` returns the VLA action and does not start IK. After `dwell` further `on_step` calls, IK starts from live poses unless grasped or in basket.
- `reset_then_ik` sets `policy_reset_requested` on the drop step. `DropRecoveryEnvWrapper.consume_policy_reset` is one-shot. The pipeline calls `policy.reset()` once after that `env.step` and prints `[pipeline] policy.reset() once after drop (reset_then_ik)`.
- `loss_mask` is `0` on the drop frame and on dwell (`triggered` and not `recovery_active`).
- Recipe `examples/faults/recipes/can_simpleik_datagen.json`: dwell 80; mode weights updated to **50/50** continue/reset in §12.6. Checkpoint samples the mode and passes it into `run_pipeline`.
- `suppress_grasp_skip` on `_EnvDropState`: the grasp that is still reported on the first dwell frames is not treated as a regrasp. It clears the first time the object reads ungrasped. Both sim proofs started IK at step 161, so this latch did not swallow recovery.
- **§12.6 follow-up:** `annotation.py` adds `ever_held_midair` to `FAILURE_ANNOTATION_FEATURES` and frame export; `_ever_held` / `_cleared_after_failure` / physical `is_failure` logic unchanged.

Not done, on purpose (initial dwell slice):

- No edit to keep gate or `lerobot_eval.py`.
- No scripted hover / wander / missed grasp.
- No full XY-band mix. Two proof episodes only (below). A full mix is the checkpoint script and will take hours.

*(Superseded: “no edit to `annotation.py`” — schema export only in §12.6.)*

### 12.5 Finish summary

**Teammate plan:** every ask is answered in §12.2. The gap they named is real. The generator they specified is not what shipped. The fixed recording recipe is passive VLA for 80 control steps, then the existing IK planner. Reset is implemented and tagged; **recipe mix is 50/50 continue/reset as of §12.6** (proof runs below used explicit modes).

**Unit tests the supervisor re-ran:** 13 passed (`reset_then_ik` handshake, zero-dwell error, dwell delay, recipe). Worker reported `tests/faults` 289 passed before the one-line reset log.

**Sim proofs** (LIBERO object 0, seed 9300, XY band `[0.34, 0.38]`, `post_grasp_delay_steps=0`, `soup_xy_offset=(0, 0)`, dwell 80). The Tkinter manual demo (`3cb6be11`) was not used: it replays a recorded nominal episode and has no SmolVLA queue to reset. These runs are the live MuJoCo recorder.

| | `reset_then_ik` | `continue_then_ik` |
| --- | --- | --- |
| Output | `outputs/post_drop_reset_proof` | `outputs/post_drop_continue_proof` |
| Drop step | 80 | 80 |
| `policy.reset()` | once, on the drop step | not called |
| IK start step | 161 (80 dwell steps later) | 161 |
| Regrasp | step 216, then basket success | step 234, then basket success |
| Control steps labeled drop (dwell) | 81 (drop + 80) | 81 |
| Dataset `loss_mask=0` | 41 frames | 41 frames |
| Dataset `is_failure=1` | 53 frames | (same annotator; failure lasts until regrasp, mask stays 0 only on dwell) |

41 masked frames is the 10 FPS recording of ~81 control steps (stride 2). Extra `is_failure` frames after that are IK-before-regrasp with `loss_mask=1`, which is what §3.2 asks for.

**First reset attempt failed** (`triggered_at=null`) because `run_pipeline` applies a random ±4 cm soup offset when `soup_xy_offset` is omitted. The checkpoint already passes `(0, 0)`. Proofs used `(0, 0)`. Do not treat that miss as a dwell bug.

**1 s wait before reset:** still no. Physics settle is inside the drop. Reset ran on step 80, VLA ran until step 161, then IK. The arm was not given a second of queued carry before the queue clear.

**Human gate (post §12.6):** reset is **in the default mix at 50%** with continue
(proof runs below used explicit modes per episode). Before a large XY-band
recording, check head precision on pre-first-grasp frames in reset dwell (§2.2);
filter by `post_drop_mode` if needed. Seed 9300 success in both modes does not
prove head precision on reset dwell alone. *(Superseded: reset weight 0 / raise
later.)*

**Mode is now visible on the recorded video.** The drop-phase banner used to read `PHASE: DROP (midair_drop)` for every frame of the wait, so an 80-step dwell looked exactly like a 1-frame `immediate_ik` drop and the three modes were indistinguishable on screen. The banner now reads `PHASE: DROP immediate_ik`, `PHASE: DWELL continue_then_ik 48/80`, or `PHASE: DWELL reset_then_ik 48/80 (queue cleared)`. `examples/faults/compare_post_drop_modes.py` stitches several episode videos into one labeled side-by-side mp4/gif.

Three-mode comparison recorded at seed 9300, mid band, `soup_xy_offset=(0, 0)`, all placed the can:

| Mode | Drop step | IK starts | Regrasp | Dataset `loss_mask=0` |
| --- | --- | --- | --- | --- |
| `immediate_ik` | 80 | 80 (same step) | 142 | 1 frame |
| `continue_then_ik` | 80 | 161 | 234 | 41 frames |
| `reset_then_ik` | 80 | 161 | 216 | 41 frames |

Artifacts: `outputs/modes_immediate`, `outputs/modes_continue`, `outputs/modes_reset`, comparison in `outputs/modes_compare/post_drop_modes.mp4`. Pass `copy_demo_gif=False` when recording proofs; the default copies the episode gif over `docs/assets/demo/full_pipeline_drop_recovery.gif`.

**Regression fix (manual demo):** Post-dwell `on_step` treated any `triggered` episode with default `post_drop_dwell_steps=0` like an automatic drop and started IK on the next step, breaking `examples/faults/run_manual_drop_recovery.py`. Added `_EnvDropState.awaiting_manual_recovery`: `trigger_manual_drop` sets it; `on_step` passes proposed actions through until `request_recovery` clears it and starts the planner. Manual Drop waits indefinitely; manual Recover (R) starts IK immediately. No policy reset and no SmolVLA in that script.

### 12.6 User decision (2026-09-23) — recipe mix + `ever_held_midair`

**Supersedes** §0 “100% continue”, §12 items 2–3 (reset weight 0 / continue-only default), §10 “raise reset weight”, §12.2 table rows that said weight 0 / no `annotation.py` edit, §12.4 “no edit to `annotation.py`”, §12.5 “weight 0 until you say otherwise”, and the old §12.3–12.4 recipe weight lines.

- **`reset_then_ik` is required** in the recording mix (not weight 0).
- Shared recipe: **`continue_then_ik` 0.5**, **`reset_then_ik` 0.5**, **`immediate_ik` 0.0**, dwell **80**.
- Reset dwell labeling unchanged: `is_failure=true`, `PHASE_POST_FAULT` until IK begins, then `PHASE_RECOVERY`.
- Export **`ever_held_midair`** (bool, shape `(1,)`) from the annotator’s existing `_ever_held` latch in recorded dataset/info features (oracle/sim history for downstream failure-head training). Latch semantics and **`_cleared_after_failure`** unchanged — recovery place release is still not a second failure.
- No in-repo failure-head trainer or proprioceptive estimator in this slice.
