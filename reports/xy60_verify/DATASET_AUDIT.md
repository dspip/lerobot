# XY60 dataset audit

Read-only check of `xy_band_mix_60ep.tar.gz`, extracted at `/tmp/xy_band_mix_60ep_extract`. The dataset files were not edited. No training code was changed.

The extract matches the archive: 60 episodes, 5,953 rows in parquet, and 5,953 decodable frames in each camera. `dataset/images/` has no files. The pictures are in the mp4s.

`audit_xy60.py` was run as-is against this extract (`--output /tmp/xy60_audit_fresh`). It finished with status `rows_passed_media_pending` because that script does not decode video unless a `media_verification.json` is already present. Video counts below are from a separate full decode of all 8 mp4s.

## Inventory

| Claim from the training report | Count in the files | Verdict |
| --- | --- | --- |
| 60 episodes | 60 (`meta/info.json` and episode parquet) | holds |
| 20 nominal, 40 disturbance | 20 `type=nominal`, 40 `type=drop` in `audit_report.json` (10 lift, 10 early, 10 mid, 10 late) | holds |
| 5,953 observations | 5,953 data rows; `total_frames` is 5,953 | holds |
| 48 train / 12 val | Reproduced by `audit_xy60.py` with seed 42. Train episodes and val episodes match that script's stratified quotas (8/2 per drop band, 16/4 nominal). | holds as the trainer's split |
| 40 invalid action targets | 40 rows with `loss_mask == 0`, and those are the same 40 rows with `injection_active`. Eligible rows: 5,913. | holds |

`meta/info.json` `splits.train` is `"0:60"`. That field lists every episode as one LeRobot train split. The 48/12 cut is not stored inside the dataset. It is rebuilt from the episode groups in `audit_xy60.py`. A trainer that ignores that script and uses `info.json` would train on all 60 episodes.

Shortest episode: 62 frames. Longest is under the per-episode lengths in `audit_report.json`. No episode has length 0. Episode lengths match both `meta/episodes` and `audit_report.json`.

Task text (one task): `pick up the alphabet soup and place it in the basket`.

State layout used below: `observation.state` is end-effector xyz (3), axis-angle (3), two finger joints (2). Action is xyz delta (3), rotation delta (3), gripper (1). Gripper command is about `-1` when open and about `+1` when closed. Finger aperture is `finger0 - finger1` (about 0.080 open, about 0.060 while holding the can). The two finger joints have opposite signs, so their average stays near 0 even when the hand is open.

## Schema

Columns in `data/chunk-000/*.parquet`:

`observation.state`, `action`, `loss_mask`, `is_failure`, `failure_onset`, `failure_type`, `injection_active`, `phase`, `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`.

Every column the recorder is supposed to write is present: `is_failure`, `failure_onset`, `failure_type`, `injection_active`, `phase`, `loss_mask`. None of those are missing.

`phase` values that actually occur: 0 (2,937 frames), 1 (40 frames), 3 (2,976 frames). Phase 2 (post-fault without recovery) never appears. That matches a drop that starts recovery on the same step: the inject frame is phase 1, and the following frames are phase 3.

## Label sanity

Nominal episodes (20): `is_failure` is false on every frame, `injection_active` is false, `failure_type` is 0, `loss_mask` is 1.

Disturbance episodes (40):

- Each episode has one failure run. Lengths are 10 to 20 frames (median 17). No one-frame flicker. No episode has two failure runs.
- Each episode has one `failure_onset` frame, and it is the first frame of that run.
- On every `is_failure` frame, `failure_type` is 1. No frame in the dataset has type 12. The only types present are 0 (2,937) and 1 (3,016).
- Type stays 1 after the failure flag turns off. Those later frames are the recovery segment (phase 3). Type 0 is only the pre-drop and nominal frames.

`loss_mask == 0` on exactly the 40 inject frames, and on no other frame. Every phase-3 recovery frame has `loss_mask == 1`. The inject frame is excluded from training targets. The regrasp and the lift after it are kept.

## What the pictures show

64 labeled frames are in `reports/xy60_verify/frames/`. Each filename and the text on the picture has the episode, the frame index, and the event. Both cameras are saved (agent view and wrist).

Sampled episodes: nominal 11, 50, 59. Drops: 0 (lift), 4 (mid), 6 (late). All six are marked `final_success` in the collector log.

### Nominal grasp (episode 11)

The hand comes down with the gripper command open, then the command flips to close, then the hand goes up with the can.

| Frame | What you see in the numbers | Gripper command | End-effector z (m) | Finger aperture |
| --- | --- | --- | --- | --- |
| 20 | Still descending, hand open | -1.02 | 0.123 | open |
| 24 | At the can, still commanded open and down | -1.02 | 0.052 | 0.080 |
| 25 | Close command appears. Fingers have barely moved. | +1.01 | 0.046 | 0.076 |
| 26 | Still at the can, command stays closed. Downward command has stopped. | +1.00 | 0.047 | 0.067 |
| 30 | Can is off the table. Command stays closed and upward. | +1.00 | 0.111 | 0.061 |

Pictures: `ep11_f0024_before_close_agentview.png`, `ep11_f0025_close_command_agentview.png`, `ep11_f0026_after_close_agentview.png`, `ep11_f0030_lift_closed_agentview.png`, plus the wrist twins. Episodes 50 and 59 show the same close flip (frames 26 and 23).

The action on these frames is a plausible pick: down and open, then close, then up and closed. The pictures at frames 24–26 do not show a large finger motion. The lift is obvious by frame 30.

### Drop and regrasp (episode 0)

| Frame | Event | Gripper command | z (m) | Aperture | failure | mask | phase |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 31 | Holding the can, still moving up | +0.99 | 0.123 | 0.062 | no | 1 | 0 |
| 32 | Inject. Hand opens. Wrist view shows the can top under an open hand. | -1.00 | 0.181 | 0.080 | yes, onset, type 1 | 0 | 1 |
| 33 | Hand still open. Arm action is almost zero. | -1.00 | 0.186 | 0.080 | yes | 1 | 3 |
| 36 | Hand open, commanded straight down (`dz = -1`) | -1.00 | 0.159 | 0.080 | yes | 1 | 3 |
| 41 | Low, over the dropped can, still open | -1.00 | 0.074 | 0.080 | yes | 1 | 3 |
| 42 | Close command. Aperture shrinks. Failure flag turns off. | +1.00 | 0.075 | 0.063 | no | 1 | 3 |
| 55 | Can is up again, gripper still closed, `dz` about +0.64 | +1.00 | 0.136 | 0.063 | no | 1 | 3 |

Pictures: `ep00_f0031_pre_drop_holding_*.png`, `ep00_f0032_inject_drop_*.png`, `ep00_f0033_after_inject_*.png`, `ep00_f0036_recovery_descend_agentview.png`, `ep00_f0041_before_regrasp_agentview.png`, `ep00_f0042_regrasp_close_*.png`, `ep00_f0055_recovery_lift_*.png`.

Episodes 4 and 6 do the same thing at a later frame (inject at 40 and 42, regrasp close at 57 and 59). The inject frame is the only masked frame. Recovery actions after it keep `loss_mask` 1, including the close and the second lift.

The inject frame's arm command is still a large move (episode 0 `dz = +0.75`, episode 4 `dz = +1.00`). That command is not a training target. The next saved frame's arm command is small, and the hand is open. That is a plausible start of recovery, and it is trained.

## Observation / action timing

This is the check the files cannot finish.

Two pairings are possible:

1. The action is the next command for this picture (standard imitation).
2. The picture was taken after this action had already been applied (what `examples/faults/run_full_drop_recovery_pipeline.py` writes: it logs the observation returned by `env.step` together with the action just executed).

Checks that were run on all 60 episodes:

- Correlation of the xyz action with the end-effector move into this frame: Pearson dx 0.868, dy 0.944, dz 0.959, mean cosine 0.820 (5,893 pairs).
- Correlation with the move after this frame: Pearson dx 0.920, dy 0.982, dz 0.969, mean cosine 0.828 (same 5,893 pairs).
- Sign of `dz` versus the up/down command, ignoring inject frames and moves smaller than 4 mm: 2,695 / 2,741 (98.3%) agree with the next move, and 2,696 / 2,741 (98.4%) agree with the previous move.

Both pairings look good because neighboring actions point the same way. A gap of one pair, or a cosine gap of 0.008, does not pick a winner.

The drop frame is the sharp case. On episode 0 the open command appears on frame 32, which is the frame whose wrist picture already shows an open hand and the can leaving. Frame 31 still shows a closed command and a held can. So the open command is on the picture of the release, not on the previous held picture. That fits pairing 2 for the gripper. The same frame's upward arm command (`dz = +0.75`) does not match the tiny rise into frame 33 (z 0.181 to 0.186). That arm command is masked (`loss_mask` 0), so it is not a trained target.

`recording_stride: 2` means one control step between recorded rows is not stored. The dataset has no column that says whether the image was captured before or after the action.

**Verdict: not proven.**

## Duplicates and empty episodes

`audit_xy60.py` duplicate pass (1,770 episode pairs):

- No exact copy of a full state+action trajectory.
- No shared collector seed. No shared source video path.
- No exact match on the first 1, 5, 10, 20, or 30 frames.
- Three near pairs under that script's thresholds: episodes 6 and 43, 17 and 18, 48 and 55. The 48/12 split keeps each pair on the same side.
- Pilot episodes 0–7 are forced into train, and episode 43 stays with them because of the near pair with episode 6.

No empty episodes.

Most recorded soup start positions are the same point. Of 52 episodes that stored an initial xy, 51 round to `(-0.119094, -0.239781)` and one is different. The 8 pilot episodes have no initial xy in the registry. That is a limit on how much "where the can started" the model can learn. It is not a broken label.

## Video versus table

Every episode's video interval length matches its parquet length, for both cameras (the check inside `Phase1Dataset`, which `audit_xy60.py` runs).

Full decode of all 8 mp4s at 10 fps:

| File | Decoded frames | Frames implied by episode metadata |
| --- | --- | --- |
| `observation.images.image` file-000 | 895 | 895 |
| file-001 | 908 | 908 |
| file-002 | 111 | 111 |
| file-003 | 4039 | 4039 |
| `observation.images.image2` file-000 through file-003 | same four counts | same |

895 + 908 + 111 + 4039 = 5,953, for each camera.

## Verdict

| Claim | Evidence | Verdict |
| --- | --- | --- |
| 60 episodes, 20 nominal and 40 drops | info.json, episode parquet, `audit_report.json` | holds |
| 5,953 observations | parquet row count | holds |
| 48 train / 12 val | `audit_xy60.py` seed 42 on this extract. Not stored in `info.json` (`splits.train` is `0:60`). | holds |
| 40 invalid action targets | 40 inject frames, all `loss_mask` 0, no other mask 0 | holds |
| Failure columns exist | all six present | holds |
| Nominal episodes stay non-failure | 0 failure frames in 20 episodes | holds |
| Each drop has a real failure interval | one run per drop episode, length 10–20 | holds |
| Drop type is 1 | every failure frame is type 1; type 12 does not occur | holds |
| Recovery actions stay trainable | phase 3 is 2,976 frames, all `loss_mask` 1 | holds |
| No empty episodes, no exact duplicate trajectories | min length 62; duplicate pass found 0 exact copies | holds |
| Sampled video frame counts match the table | all 8 files, and per-episode intervals | holds |
| Each action is paired with the camera and state from the moment that action was meant for | both time lags match arm motion; the drop picture does not separate them | not proven |

## Decision waiting on Aviya

The rows, labels, masks, and videos agree with the training report. The before/after pairing of picture and action is not proven. The dataset was not changed.

Ways to treat that, if you want to choose:

- Leave the rows as they are and keep training the way Eran did. Pro: nothing is rewritten, and the bad inject action is already masked. Con: a failed rollout can still be this timing gap. `SIMULATION.md` already says the files cannot close it.
- Assume the picture is the input and the action is the next command. Pro: that is the usual imitation setup, and the next-frame arm motion matches the command about as well as the previous frame. Con: on the drop, the open-hand command is stored on the frame that already shows the open hand.
- Assume the picture is the result of the action. Pro: that is what the recording loop in `run_full_drop_recovery_pipeline.py` writes. Con: the parquet has no flag that these files were recorded that way, and the arm command also matches the following motion.

A recording pass that stores "image taken before the step" or "image taken after the step" would settle it. That recording was not started.

Track 3 (soup start positions) and checkpoint rollouts were not started.
