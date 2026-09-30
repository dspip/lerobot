# Single-Dataset Five Run Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record the approved five episode kinds into one LeRobot v3 dataset while preserving the existing frame annotations and storing mission success once per episode.

**Architecture:** Extend matrix rows with an explicit `drop` flag and carry that identity through seed manifests. Build one shared dataset logger for the entire matrix. Keep SimpleIK as the nominal controller. After drop: type 5 runs SmolVLA with IK suppressed; type 3 runs SmolVLA for `dwell_steps` then IK recovery. Pass `success` through the dataset writer into LeRobot episode metadata.

**Tech Stack:** Python 3, dataclasses/enums, NumPy, PyTorch policy inference, LeRobot v3 dataset writer, pytest.

## Global Constraints

- One matrix row produces one complete episode; rows are never spliced.
- All committed rows use `<output_dir>/dataset`.
- Frame features remain the existing Failure Annotated schema.
- `success` is episode metadata, not a frame feature.
- Types 1–3 and no-drop type 4 commit only on success.
- Type 5 commits after an injected drop even when `success=false`.
- Type 5 has `loss_mask=0` from injection through episode end.
- The same logical index shares `episode_seed`, `layout_seed`, and `drop_seed`.
- Trigger-pose replay for future recooks remains outside this implementation.

---

### Task 1: Matrix Schema and Stable Seed Identity

**Files:**
- Modify: `src/lerobot/faults/datagen/recipe.py`
- Modify: `src/lerobot/faults/datagen/paired_context.py`
- Modify: `tests/faults/test_datagen_recipe.py`

**Interfaces:**
- `MatrixVariant(controller, post_drop_mode, drop, episodes)`
- `ExpandedMatrixRun(..., drop: bool)`
- `EpisodeSeedManifest(..., drop: bool, variant_index: int)`
- `build_paired_episode_plan()` overrides the shared random decision with the row’s explicit `drop`.

- [ ] Write failing tests for the canonical five rows, `drop` type validation, duplicate `(controller, post_drop_mode, drop)` rejection, acceptance of SimpleIK reset and immediate-SmolVLA, and stable unique controller seeds for duplicate controller/mode rows.
- [ ] Run `uv run pytest tests/faults/test_datagen_recipe.py -q` and confirm failures identify missing `drop` / new mode support.
- [ ] Add `PostDropMode.IMMEDIATE_SMOLVLA`, add `drop` to matrix/run/manifest dataclasses, parse it strictly, remove the frozen old-pair requirement, and derive `controller_seed` from stable `variant_index` in addition to controller/mode.
- [ ] In `build_paired_episode_plan`, use `replace(decision, drop=manifest.drop, reason="matrix_drop"/"matrix_no_drop")`; preserve one shared `drop_u` draw for paired drop rows.
- [ ] Re-run the recipe tests and keep them green.

### Task 2: One Dataset and Episode-Level Success

**Files:**
- Modify: `src/lerobot/faults/datagen/dataset_writer.py`
- Modify: `src/lerobot/faults/recovery/dataset_logger.py`
- Modify: `src/lerobot/datasets/lerobot_dataset.py`
- Modify: `src/lerobot/datasets/dataset_writer.py`
- Modify: `tests/faults/test_datagen_dataset_writer.py`

**Interfaces:**
- `variant_dataset_directory(...) -> <output_dir>/dataset`
- `variant_repo_id(...) -> recipe.name`
- `DatagenEpisodeSession.commit(*, success: bool) -> int`
- `FaultRecoveryDatasetLogger.end_episode(episode_metadata={"success": bool})`
- Core `LeRobotDataset.save_episode(..., episode_metadata: dict | None = None)` forwards metadata without adding a frame feature.

- [ ] Write failing tests proving all manifests resolve to one dataset root/repo/logger and that `meta/episodes/**/*.parquet` contains one bool `success` per committed episode.
- [ ] Run the focused writer tests and confirm the path-count and missing-success assertions fail.
- [ ] Replace variant logger keys with one run key, pass success at commit, and extend the LeRobot writer call chain with optional episode metadata merged into `ep_metadata` before `DatasetMetadata.save_episode`.
- [ ] Keep `run_manifest.json` unchanged as diagnostics and ensure discarded episodes do not create episode metadata rows.
- [ ] Run focused writer and dataset round-trip tests.

### Task 3: Planned Type-5 Fault Mode and Mask

**Files:**
- Modify: `src/lerobot/faults/config.py`
- Modify: `src/lerobot/faults/recovery/midair_drop.py`
- Modify: `tests/faults/test_midair_drop_fault.py`
- Modify: `tests/faults/test_datagen_dataset_writer.py`

**Interfaces:**
- `POST_DROP_MODES` includes `immediate_smolvla`.
- On type 5 injection, `_EnvDropState.awaiting_manual_recovery=True`; the wrapper executes caller actions and never creates an IK planner.
- Existing post-fault mask logic returns zero while triggered and not recovering.

- [ ] Write failing tests that type 5 validates, never calls `_start_recovery_planner`, forwards subsequent proposed actions, reports phase 2, and keeps `loss_mask=0`.
- [ ] Run those tests and confirm expected failures.
- [ ] Implement the no-IK branch in `_trigger_drop`; do not alter the recovery behavior of the other three modes.
- [ ] Run the focused fault/mask tests.

### Task 4: SimpleIK Carry Followed by SmolVLA

**Files:**
- Create: `src/lerobot/faults/datagen/smolvla_action_provider.py`
- Modify: `src/lerobot/faults/datagen/controllers/simple_ik.py`
- Modify: `src/lerobot/faults/datagen/controllers/smolvla.py`
- Modify: `src/lerobot/faults/datagen/smolvla_resources.py`
- Modify: `tests/faults/test_simple_ik_episode_logging.py`
- Create: `tests/faults/test_smolvla_action_provider.py`

**Interfaces:**
- `SmolVLAActionProvider(resources, task).reset()`
- `SmolVLAActionProvider.select_action(observation) -> np.ndarray` reuses the same preprocessor/select/postprocessor pipeline as `run_pipeline`.
- `run_simple_ik_episode_loop(..., post_drop_action_provider=None)` uses SimpleIK before injection. After injection: type 5 uses the provider until done with no IK; type 3 uses the provider for dwell then existing IK recovery.

- [ ] Write failing unit tests for processor order, action shape, one provider reset per episode, and the controller handoff occurring only after the injection frame.
- [ ] Run the tests and confirm failure from the missing provider/interface.
- [ ] Extract the policy action conversion into the focused provider and use it from both the hybrid SimpleIK adapter and existing SmolVLA pipeline where practical.
- [ ] Load cached SmolVLA resources for type 3 and type 5. Type 5: provider until done, no IK. Type 3: `policy.reset()` at dwell start, provider during dwell, then IK regrasp/place.
- [ ] Do not reset the MuJoCo scene on `reset_then_ik`.
- [ ] Run the new provider and SimpleIK loop tests.

### Task 5: Keep Policy and Canonical Recipe

**Files:**
- Modify: `src/lerobot/faults/datagen/dataset_writer.py`
- Modify: `examples/faults/recipes/can_drop_datagen.json`
- Modify: `examples/faults/recipes/README.md`
- Modify: `HANDOFF_UNIFIED_DROP_DATAGEN.md`
- Modify: `tests/faults/test_datagen_dataset_writer.py`
- Modify: `tests/faults/test_datagen_runner.py`
- Modify: `tests/faults/test_datagen_final_review_contracts.py`

**Interfaces:**
- `evaluate_datagen_keep`: no-drop/types 1–3 require success; type 5 requires an actual injected-drop trigger and may have either success value.

- [ ] Write failing keep-policy tests for successful/failed rows 1–4 and both outcomes of type 5, including a type-5 no-drop rejection.
- [ ] Run the focused tests and confirm the type-5 expectations fail.
- [ ] Implement the keep policy from manifest `drop`, mode, `result.success`, and trigger evidence.
- [ ] Replace the canonical matrix with the five approved rows; set `smolvla.use_stock_layout=false`; update docs from “five datasets” to “one dataset, five episode kinds.”
- [ ] Update old fixed-matrix assertions and run all datagen tests.

### Task 6: Full Verification

**Files:**
- Check: all files above

- [ ] Run `uv run pytest tests/faults -q --tb=short`.
- [ ] Run `uv run ruff check src/lerobot/faults src/lerobot/datasets tests/faults examples/faults`.
- [ ] Run `uv run python examples/faults/run_drop_datagen.py --recipe examples/faults/recipes/can_drop_datagen.json --help` to confirm imports/CLI startup without GPU recording.
- [ ] Inspect `git diff --check` if git metadata is available; otherwise inspect lints for edited files.
- [ ] Re-read `docs/superpowers/specs/2026-09-24-single-dataset-five-run-types-design.md` and verify every in-scope requirement against tests.
