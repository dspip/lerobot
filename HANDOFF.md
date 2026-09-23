# HANDOFF — Failure-annotated LIBERO datasets (LeRobot Parquet)

Single source of truth for the **mid-air drop** workstream.

**Grasp-miss (next failure type):** [`HANDOFF_GRASP_MISS.md`](./HANDOFF_GRASP_MISS.md) — do not change drop latch semantics in this file.

## 1. Goal

Record LIBERO / MuJoCo episodes as LeRobot datasets (`data/**/*.parquet` + `videos/` + `meta/`) with **per-frame failure labels** from live simulator GT (not a JSONL join), then **fine-tune SmolVLA** so it can recover from drops without cloning the injector.

| Column | Meaning |
| ------ | ------- |
| `is_failure` | Physical failure this frame (held mid-air, now ungrasped, not in the basket). |
| `failure_onset` | Rising edge of `is_failure`. |
| `failure_type` | Category id (`0` none, `1` midair drop, …). Sticky after injector fire or spontaneous drop. |
| `injection_active` | Injector glitched **this** frame (independent of physics). |
| `phase` | `0` nominal, `1` injection, `2` post-fault, `3` recovery. Injection pulse wins over recovery on the glitch frame. |

SmolVLA camera/state/action keys are unchanged. Extra columns are ignored by the policy preprocessor until training uses `loss_mask` / `is_failure`.

**Hardware / extra sensors (depth, F/T, `lerobot-record`) are out of scope until sim labels + recovery recipe are honest.**

## 2. Architecture

**Recorders**

1. Eval: `lerobot_eval.py` `rollout()` when `--eval.recording=true` (pre-step images, post-step labels — same as `next.reward`).
2. Drop-recovery: `FaultRecoveryDatasetLogger` used by unified datagen (`examples/faults/run_drop_datagen.py` → `lerobot.faults.datagen`) (**post-step** obs + labels).
3. Hardware `lerobot-record`: not wired.

**Injection:** `maybe_wrap_env_tree` after `make_env`. Wrappers annotate **after** `env.step` and **before** `notify_dones`.

**Physics:** `src/lerobot/faults/annotation.py` + `sim/libero.py`.

**Training-grade recipe (not the library FaultInjectionConfig defaults):** `src/lerobot/faults/recovery/recording_recipe.py`.

## 3. Plan (do not skip)

| Step | Status | What |
| ---- | ------ | ---- |
| A. Frame-level Parquet labels | **Done** | Five columns from live GT; `loss_mask=0` only on injection frame |
| B. Training-grade **data recipe** | **Done (this change)** | Carry delay, no seat teleport, settle in Parquet |
| C. Re-verify **one** CUDA episode with the new recipe | **Done (user verified)** | `carry_steps=24`, `seat_assisted=false`, `n_settle_logged=5` on recipe run |
| D. Small mixed set (tens of episodes) | **In progress** | `examples/faults/run_drop_datagen.py` + `can_drop_datagen.json` experiment matrix |
| E. Smoke fine-tune | After D | Short SmolVLA run using `loss_mask`; eval with faults off then on |
| F. Scale | After E | Only if unaided recoveries look right on video **and** parquet |
| G. Jetson / extra sensors | After F | New schema; not a missing column in current verify |
| H. `grasp_miss` type | **Handoff written** | See [`HANDOFF_GRASP_MISS.md`](./HANDOFF_GRASP_MISS.md); do not fold into drop latch |

**Do not scale** from `outputs/failure_annotation_verify/` (delay=0, seat assist, settle not in parquet).

## 4. What shipped

Branch: `feature/failure-annotation-parquet`

### A — labels

| File | Role |
| ---- | ---- |
| `src/lerobot/faults/annotation.py` | Latch + schema + Gym info helpers |
| `src/lerobot/faults/wrappers.py` | All three wrappers stamp `info` |
| `src/lerobot/faults/recovery/dataset_logger.py` | Writes label columns |
| `src/lerobot/scripts/lerobot_eval.py` | Eval recording includes labels |
| injectors (`hold`, `burst`, `sensor_dropout`, `object_slip`, `eef_bump`) | `just_injected` pulse |
| `tests/faults/test_failure_annotation.py` | Latch / wrapper tests (no GPU) |
| `examples/faults/verify_failure_parquet.py` | Opens real Parquet |

### B — recording recipe

| File | Role |
| ---- | ---- |
| `src/lerobot/faults/recovery/recording_recipe.py` | Sample delay `[20,60]`, `seat_assist_enabled=False` |
| `examples/faults/run_drop_datagen.py` | Public CLI; matrix of controller × post-drop mode variants |
| `tests/faults/test_recording_recipe.py` | No-sim tests |

Library `FaultInjectionConfig` still defaults to `post_grasp_delay_steps=0` and `seat_assist_enabled=True` so injector unit tests stay valid. **Training datasets must go through the pipeline recipe (or equivalent kwargs).**

## 5. Results so far

### Labels (2026-09-16) — `outputs/failure_annotation_verify/`

- `uv run pytest tests/faults -q` → 207 passed (before recipe tests).
- Real LIBERO + SmolVLA + midair_drop (CUDA): 105 frames, 14 `is_failure`, 1 injection frame, cameras match drop.
- **Recipe problems (why we do not train on it):** drop on first grasp (`post_grasp_delay_steps=0`), `seat_assisted=true`, settle overlay frames not in Parquet.

### Recipe (2026-09-16, historical verify)

- Old single-episode pipeline used CLI flags such as `--post-grasp-delay-steps` and `--allow-seat-assist`. **Do not use those** — they belonged to removed recorders.
- Unified recording uses `examples/faults/recipes/can_drop_datagen.json` only; SmolVLA drop timing and XY bands live under `smolvla.*` in that file.

## 6. Risks / known mismatches

- Pipeline `grasp_flags` in `pipeline_log.json` are **pre-step**; annotator / parquet labels are **post-step**. Shift by one when joining.
- `midair_drop` starts recovery on the drop step → rarely `phase=2`.
- Image `stats.json` count may be 100 on a ~105-frame episode (LeRobot subsample).
- `evaluation_episode_id` in JSONL can be null.
- Basket keep-out: early-drop radius `min_drop_distance_from_basket_m=0.30` on the training recipe (drop when the carry reaches it). Hard pocket 22 cm — never inject inside that. `<= 0` disables.
- Unified datagen writes under `recording.output_dir` with one dataset tree per matrix variant; use a fresh recipe output dir for a clean run (see `RunDatasetWriter` / runner tests).

## 7. Commands

**Public recorder (only):**

```bash
export MUJOCO_GL=egl
uv run python examples/faults/run_drop_datagen.py \
  --recipe examples/faults/recipes/can_drop_datagen.json \
  --device cuda
```

**CLI overrides** (optional): `--base-seed`, `--output`, `--episodes`, `--device`, `--headless` / `--no-headless`. There is no `--policy-path`, `--post-grasp-delay-steps`, or `--allow-seat-assist` on this entry point.

**Configure in `can_drop_datagen.json` instead:**

| Key | Role |
| ----- | ---- |
| `experiment_matrix[]` | `{controller, post_drop_mode, episodes}` variants to record |
| `recording.base_seed`, `recording.output_dir`, `recording.dataset_fps` | Seeds, root output, logged FPS |
| `smolvla.policy_path` | Hugging Face policy id for SmolVLA rows |
| `smolvla.post_grasp_delay_steps` | Carry delay before drop (0 = XY-band recipe) |
| `smolvla.drop_xy_bands` | Mid-air drop bands for SmolVLA |
| `post_drop.dwell_steps` | Env steps to wait after drop before IK |
| `simple_ik.path_drop` | SimpleIK drop gates (phases, keepout) |

**CI-safe logger demo (no LIBERO):** `uv run python examples/faults/run_drop_recovery_demo.py --dry-run`

```bash
# Unit tests (no GPU)
uv run pytest tests/faults/test_failure_annotation.py tests/faults/test_recording_recipe.py tests/faults/test_mix_recording.py -q

# Verify Parquet under a variant dataset root (path depends on matrix entry)
uv run python examples/faults/verify_failure_parquet.py \
  --root outputs/can_drop_datagen/smolvla/continue_then_ik/dataset
```
