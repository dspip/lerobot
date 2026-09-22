# HANDOFF — XY60 dataset proof and head-triggered recovery

Single source of truth for the **next agent** on this task. Read this whole file before doing anything.

**Owner:** Aviya. **Supervisor rule:** if an investigation finds a bug or an ambiguity, stop. Explain it in simple words, give a recommended fix with pros and cons and the checks you already ran, and wait for Aviya's approval. Do not "just fix it."

**Out of scope:** `grasp_miss`, soup initial-position variation, more dataset recording, retraining, pushing, and any commit on `main`.

**Branches (not merged, not pushed):**

| Repo | Branch | Commit | What |
| --- | --- | --- | --- |
| `/home/aviya/Projects/lerobot` | `feat/head-triggered-recovery` | `14313b83` | `request_recovery`, tests, rollout script, audit, videos. `main` does not contain this. |
| `third_party/Gangelia_Project` | `feat/head-vlm-loader` | `e80efe8` | Reconstructed two-pass loader. Eran's `eran_ai/initial-pipeline` does not contain this. |

Drop latch semantics stay as they are. See [`HANDOFF.md`](./HANDOFF.md). Grasp-miss work stays in [`HANDOFF_GRASP_MISS.md`](./HANDOFF_GRASP_MISS.md) and must not start until mid-air drop recovery is proven reliable.

---

## 0. What we already agreed

| Track | Status | What |
| --- | --- | --- |
| 1. Prove the XY60 dataset | **DONE, waiting on Aviya** | Report: `reports/xy60_verify/DATASET_AUDIT.md`. Inventory, labels, masks, and videos match the training report. Picture/action before-vs-after is **not proven**. Dataset was not edited. |
| 2. Recovery activator | **DONE** | `MidAirDropFault.request_recovery` starts the IK planner without an impulse. 270 fault tests passed. Drop trigger path was not changed. |
| 3. Vary soup start pose | **WAIT** | Only after Aviya says the dataset is sound enough. |
| 4. Head-triggered sim | **ONE SCENE ONLY** | Clean 40 steps: max P(drop)=0.006, recovery never started. After a scripted release: P(drop)=0.993 and `request_recovery` ran. Not a success-rate test. |

Eran trained SmolVLA. Aviya records data and owns the in-tree fault system. Eran is not the person to implement the recovery hook.

---

## 1. Plain-language goal

Eran trained several SmolVLA variants on a 60-episode mix of normal pick-and-place demos and drop/recovery demos. Task success was inconsistent (best clean start 7/10, best post-drop 6/10). The failure **prediction head** detected "drop right now" very well on validation frames (F1 about 0.96) but using that prediction to condition actions did not clearly raise task success.

Aviya wants two answers, separately:

1. **Is the dataset itself correct?** Especially: is each recorded action paired with the camera/state from the moment that action was meant for?
2. **Can we test the head the way the fault system already works?** If the head says "failure," start the same recovery planner the drop injector starts. The head is the switch. The drop physics is not required to flip that switch.

---

## 2. Facts already checked (do not re-guess these)

### Dataset location

The extracted folder was not in the repo. The archive is:

`/home/aviya/Projects/lerobot/xy_band_mix_60ep.tar.gz` (44 MB, 2026-09-17)

Top-level members:

- `dataset/` — LeRobot layout: `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, `meta/episodes/chunk-000/file-000..003.parquet`, `data/chunk-000/file-000..003.parquet`, videos for `observation.images.image` and `observation.images.image2`, plus an `images/` tree
- `kept_registry.json`, `checkpoint_log.json`, `audit_report.json`

Eran's Mac path in old reports (`/Users/rnbnrzy/Desktop/Gangelia_Project/xy_band_mix_60ep/dataset`) is **not** this machine.

A cancelled audit already extracted the archive to `/tmp/xy_band_mix_60ep_extract/dataset`. Reuse that if the files are intact. If anything looks partial, extract the tarball again over a fresh directory. Do not treat that extract as an audit result.

Existing reader (do not modify it): `third_party/Gangelia_Project/smolvla_r_package/audit_xy60.py`. It expects `ROOT / "xy_band_mix_60ep/dataset"` where `ROOT` is `third_party/Gangelia_Project`.

There is also a smaller unrelated archive: `failure_mix_small_dataset.tar`. Do not mix it into this audit.

### Known timing caveat (already written by the training side)

`third_party/Gangelia_Project/smolvla_r_package/SIMULATION.md` says the dataset **cannot by itself prove** whether a row's observation was captured before or after its action, and that `recording_stride: 2` is recording cadence, not proof that each action was held for both controller steps. Track 1 must try to prove alignment with frames and state, and must say "not proven" when the files do not contain the evidence.

### Checkpoints (downloaded, weights present)

Drive unzip left an extra folder layer and one name has a space. Weights are present:

| Role | Directory |
| --- | --- |
| Calibrated, Oracle → VLM, 3000 updates. Reported 7/10 clean, 4/10 post-drop. | `checkpoints/calibrated_oracle_to_vlm-total3000-20260922T091204Z-1-001/calibrated_oracle_to_vlm-total3000/` (`policy/model.safetensors`, `vlm_conditioning/adapter.pt`) |
| No calibration, Oracle → VLM, 2000 updates. Reported 6/10 and 6/10. | `checkpoints/no_calibration_oracle_to_vlm-total2000-20260922T091212Z-1-001/no_calibration_oracle_to_vlm-total2000/` |
| No calibration, predicted head → VLM, 3000 updates. This is the head checkpoint. Reported 5/10 clean, 4/10 post-drop. | `checkpoints/no_calibration_predicted_head_to_vlm-total3000-20260922T091225Z-1-001/no_calibration_predicted_head_to_vlm-total3000/` (`policy/model.safetensors`, `auxiliary/weights.pt`) |
| Calibration only, 1000 updates. Reported 6/10 clean, 2/10 post-drop. | `checkpoints/post_calibration-total 1000-20260922T091227Z-1-001/post_calibration-total 1000/SmolVLA_calibrated_checkpoint/SmolVLA_calibrated_1000/checkpoint/` |

Pinned training revision from Eran's report: `71a11efe77f55e61f3ab2ce45b40da8cf626afa9`. Do not move the repo to that commit.

### How recovery starts today

`MidAirDropFault.on_step` (`src/lerobot/faults/recovery/midair_drop.py`): if `recovery_active`, it replays `SimpleIKRecoveryPlanner`. Otherwise, if `_should_trigger`, `_trigger_drop` applies a physics impulse **and** sets `recovery_active = True` and builds the planner. There is no public "please start recovery now" call that skips the impulse.

`DropRecoveryEnvWrapper` only wraps `fault.type=midair_drop`.

---

## 3. Track 1 — dataset proof (investigation only)

**Worker:** thorough read-only audit. Extract the tarball under `/tmp/xy_band_mix_60ep_extract`. Do not write into the tarball or into `src/`.

**Report file (create this, do not edit this handoff):** `reports/xy60_verify/DATASET_AUDIT.md`

Pictures are allowed and wanted. Save frames under `reports/xy60_verify/frames/`. A picture without the episode index, frame index, and what it is supposed to show does not count.

### Checks (all of them, with counts)

1. **Inventory.** Episode count, frame count, nominal vs disturbance split. Eran's report says 60 episodes (20 nominal, 40 disturbance), 5,953 observations, 48 train / 12 val, and 40 observations whose action targets are invalid. Confirm or contradict each number from the files.
2. **Schema.** List every column in `data/chunk-000/*.parquet`. Confirm the failure columns the recorder is supposed to write (`is_failure`, `failure_onset`, `failure_type`, `injection_active`, `phase`, `loss_mask` — see `src/lerobot/faults/recovery/dataset_logger.py`). Say which are missing.
3. **Label sanity.**
   - Nominal episodes: `is_failure` should stay false.
   - Disturbance episodes: there should be a real failure interval, not a single flicker, unless the file shows otherwise.
   - `failure_type` for drops should be the mid-air drop id (1), not 0 and not 12.
   - `loss_mask == 0` only where the project says the action target is invalid (inject frame). Valid recovery actions after the drop must keep `loss_mask == 1`.
   - Count masked frames. Eran said 40 invalid action targets. Match or explain the difference.
4. **Observation / action timing.** This is the important one.
   - Pick at least 3 nominal episodes and 3 drop episodes.
   - For several frames around grasp, drop, and regrasp, dump state (EEF xyz, gripper) and the action in that row.
   - Extract the matching camera frame.
   - Say whether the action is a plausible next command for that picture (example: gripper above the can and still open → action should move down or close, not yank away).
   - If you cannot tell whether the image is before or after the action, write **not proven** and show the two neighboring frames so Aviya can see why.
5. **Duplicates / empty episodes.** Reuse logic from `audit_xy60.py` if it runs. If it does not run, say the exact error. Do not rewrite that script.
6. **Video vs table length.** For the sampled episodes, frame count in parquet vs decodable video frames.

### Do not

- Do not "correct" labels, drop rows, or re-encode video.
- Do not declare the dataset good or bad in a single adjective. End with a table: claim, evidence, verdict (`holds` / `fails` / `not proven`).
- Do not start Track 3.

---

## 4. Track 2 — recovery can be requested (implementation)

**Worker:** Composer. Code changes are allowed only for this track.

**Intent Aviya approved:** the same recovery behavior the drop injector uses should be startable by another activator. For the later test, that activator is the prediction head. This slice does **not** load Eran's checkpoint and does **not** run LIBERO.

### Build this

A small public request on the existing mid-air drop fault, something a caller can use like:

```text
request_recovery(env, env_idx, reason="head") -> first recovery action
```

Behavior:

- Uses the current end-effector pose and the **current** soup pose.
- Builds `SimpleIKRecoveryPlanner` the same way `_trigger_drop` does after the impulse.
- Sets `recovery_active = True` so later `on_step` calls keep replaying the planner.
- Does **not** apply an impulse, does **not** open the gripper, does **not** set `drop_injection_step`.
- Logs a JSONL event that the reason was an external request (`head` or whatever string was passed), distinct from `xy_band` / `delay_elapsed`.
- If recovery is already active, do not start a second planner.
- If the fault is disabled, do nothing.

Keep `_trigger_drop` as the physics path. Share the planner-setup code so the two paths cannot drift. Do not rename `DropRecoveryEnvWrapper`.

### Tests

`tests/faults/` with mocks. No GPU, no LIBERO, no MuJoCo.

- Request starts planner and `recovery_active` without calling the impulse helper.
- A following step returns the next planner action, not the policy action.
- Drop-trigger tests that already exist still pass.
- Request while already recovering does not reset the planner.

Run: `uv run pytest tests/faults/test_midair_drop_fault.py tests/faults -q --maxfail=10`
(narrow the second path if the full folder is too slow; do not skip the new tests).

### Docs / skill

- One short paragraph in `docs/source/fault_injection.mdx`: recovery can be requested without a new impulse; the drop injector is still the only thing that applies the impulse.
- If you add a real caller-facing flag or type, update `.cursor/skills/add-fault/SKILL.md`. If you only add a method and no new fault type, say so in `reports/xy60_verify/RECOVERY_REQUEST.md` and do not invent a skill section.

New Python files: `Copyright 2026 Gangelia. All rights reserved.` plus the Apache-2.0 header used by other new fault files. Do not rewrite HuggingFace headers.

### Stop and ask Aviya if

- The only clean design is a new fault `type=` or a change to drop latch rules.
- Wiring the head would require editing `lerobot_eval.py` a second time (the only hook is `maybe_wrap_env_tree`).
- Tests show the existing drop path changed.

Write what you did in `reports/xy60_verify/RECOVERY_REQUEST.md`.

### Explicitly not in this slice

- Loading `auxiliary/weights.pt`.
- A closed-loop sim where the head calls `request_recovery`. That is the **next** slice, after Aviya approves Track 2.
- `grasp_miss`.

---

## 5. Later slice (do not start)

After Aviya accepts Track 2:

1. Run the **predicted head** checkpoint (`no_calibration_predicted_head_to_vlm-total3000`) in sim.
2. On each query, if the head's current-failure probability crosses the threshold Eran used, call `request_recovery`. Do not also fire the drop impulse in that test.
3. Compare task success against the same checkpoint **without** the recovery request (Eran's reported predicted-head numbers: at 3000 updates, 5/10 clean and 4/10 post-drop).
4. Separately, optional sanity: `calibrated_oracle_to_vlm-total3000` still does **not** get a fake head. It is the oracle-conditioned policy, not the head test.

The point of that test: the head's offline F1 does not tell us if recovery helps. This test does.

---

## 6. How to talk about Eran's table (so nobody re-explains it wrong)

- **Clean-start:** task finished, robot began in the normal pose. No drop during the scored rollout.
- **Post-drop:** task finished, robot began in a prepared after-drop pose. Detecting the drop or grasping again is not success.
- **Oracle:** the simulator's true "dropped or not" flag is fed into the model. A real robot does not have this.
- **Prediction head:** a classifier on the current camera+state. Eran measured it at F1 0.9627 on 1,199 validation frames (116 true positives, 9 false negatives, 0 false positives). That is detection quality, not task success.
- **HF control 3/10 and 0/10** is the original LIBERO SmolVLA with zero extra updates, but with **this project's** normalization fitted on 16 nominal episodes, not a untouched Hugging Face eval. Eran thinks those weights look weak. That claim stays open until we reproduce it. Do not "fix" the base model in this task.

---

## 7. Progress log

| When | What |
| --- | --- |
| 2026-09-22 | Handoff written. Dataset confirmed only as `xy_band_mix_60ep.tar.gz`. Four checkpoint trees contain `model.safetensors`. Track 3 and checkpoint rollouts not started. |
| 2026-09-22 | Track 2 landed. `request_recovery(env, env_idx, reason="head")` in `src/lerobot/faults/recovery/midair_drop.py`. Shared planner setup with the drop path. No impulse, no `drop_injection_step`. Write-up: `reports/xy60_verify/RECOVERY_REQUEST.md`. `uv run pytest tests/faults -q` → 270 passed. No design fork. Nothing calls `request_recovery` from a policy yet. |
| 2026-09-22 | Aviya stopped the parallel workers. Dataset audit produced no report and no frames. Extract left at `/tmp/xy_band_mix_60ep_extract`. The next agent is the manager, not another pair of workers. |
| 2026-09-22 | Track 1 finished by the manager. `reports/xy60_verify/DATASET_AUDIT.md` plus 64 labeled frames under `reports/xy60_verify/frames/`. Counts hold (60 episodes, 5,953 rows, 20/40 split, 48/12 via `audit_xy60.py`, 40 masked inject frames). All 8 videos decode to those lengths. Observation/action lag is not proven (both lags match arm motion). No dataset edit. Track 3 and rollouts still waiting. |
| 2026-09-22 | Aviya approved the predicted-head sim (head calls `request_recovery`, no drop impulse, video). Stopped before any rollout. Checkpoint format `gangelia-phase2-head-to-vlm-v1` does not load in the in-tree `Phase2Policy` (adapter outputs 960, two VLM passes). `head_vlm_conditioning.py` and `train_head_vlm_no_calibration.py` are not on this machine. |
| 2026-09-22 | Continued with a reconstructed two-pass loader (`smolvla_r_package/head_vlm_conditioning.py`): unconditioned VLM pass, head, 960-d residual on the state token, second VLM pass. Clean 40-step rollout: max P(drop)=0.006, recovery never started (`reports/xy60_verify/head_recovery_rollout/`). After replaying nominal episode 12 and a physics release that does not call the drop trigger: can was grasped, head P(drop)=0.993 on the first scored step, `request_recovery` logged `drop_trigger_reason=head`. Video: `reports/xy60_verify/head_recovery_after_release/rollout.mp4`. This is not a 10-seed success rate. |
| 2026-09-22 | Aviya rejected a commit on `main`. That commit was moved, unpushed, to `feat/head-triggered-recovery` (`14313b83`). The loader commit was moved off `eran_ai/initial-pipeline` onto `feat/head-vlm-loader` (`e80efe8`). Do not commit to `main` or to Eran's branch. |
| 2026-09-22 | Manual drop/recovery live demo: `examples/faults/run_manual_drop_recovery.py` (tkinter, no head). Replay ep 12, then Aviya presses Drop or Recover. Write-up: `reports/xy60_verify/MANUAL_DROP_RECOVERY.md`. Not run interactively in CI; `py_compile` only. |
| 2026-09-22 | Manual drop/recovery demo fix: full nominal replay (random ep from audit), no pre-drop freeze; `after_physics_step` during recovery; 4000-step cap and `recovery_done` exit. Aviya hit `step_cap` at 472 on the old script. Docs: `MANUAL_DROP_RECOVERY.md`. |
| 2026-09-22 | Manual demo replay fix: default ep 12 only, loop same episode with `init_state_id` restored on reset (no random ep chain). Recover-before-drop message. |

The next agent owns this file and updates this table as work finishes. This handoff edit itself may be uncommitted on `feat/head-triggered-recovery`. Do not commit it unless Aviya asks.

## 7b. Manager role (Aviya, 2026-09-22)

You are one manager in this chat. Do not open background workers. Do not commit. Do not push. Do not touch `main`.

Discuss the next step with Aviya before writing more code. If a check finds a bug, stop and explain it in simple words with pros, cons, and the checks you ran.

---

## 8. Copy-paste prompt for a fresh agent

```text
You are talking with Aviya about an unfinished experiment. Do not write code, do not commit, do not push, and do not touch main until Aviya asks. Read HANDOFF_XY60_HEAD_RECOVERY.md first. Also skim reports/xy60_verify/DATASET_AUDIT.md and reports/xy60_verify/head_recovery_after_release/summary.json.

Repo: /home/aviya/Projects/lerobot
Checked-out branch: feat/head-triggered-recovery (commit 14313b83). main does not have this work. It was never pushed.
Training repo: third_party/Gangelia_Project on feat/head-vlm-loader (commit e80efe8). Eran's branch eran_ai/initial-pipeline does not have the loader. That commit was never pushed.

What is already true:
- XY60 dataset audit holds on counts, labels, masks, and video length. Whether each picture was taken before or after its action is not proven. Report: reports/xy60_verify/DATASET_AUDIT.md. Frames: reports/xy60_verify/frames/.
- MidAirDropFault.request_recovery starts the existing IK planner without a physics drop. Tests passed. The automatic drop path was not changed.
- A reconstructed two-pass head (third_party/Gangelia_Project/smolvla_r_package/head_vlm_conditioning.py) loads the predicted-head checkpoint. Eran's real head_vlm_conditioning.py is still missing, so this loader is not proven identical to his.
- Clean 40-step rollout: max P(drop)=0.006, recovery never started. Video: reports/xy60_verify/head_recovery_rollout/rollout.mp4.
- After replaying nominal episode 12 and a scripted release that does not start recovery: the can was grasped, then the head scored P(drop)=0.993 and request_recovery logged drop_trigger_reason=head. Video: reports/xy60_verify/head_recovery_after_release/rollout.mp4. Task success was not scored. One scene only.

Checkpoint used: checkpoints/no_calibration_predicted_head_to_vlm-total3000-20260922T091225Z-1-001/no_calibration_predicted_head_to_vlm-total3000 (gitignored local download).

Out of scope until Aviya says otherwise: grasp_miss, moving the soup's start position, retraining, committing to main, pushing.

Aviya wants to discuss the next step. Explain in simple words. If you recommend something, give pros and cons and wait.
```
