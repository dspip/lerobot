# Copyright 2026 Gangelia. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.faults.annotation import default_failure_frame
from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.datagen.controllers.simple_ik import SimpleIKDatagenAdapter
from lerobot.faults.datagen.controllers.smolvla import SmolVLADatagenAdapter
from lerobot.faults.datagen.dataset_writer import (
    DatagenEpisodeSession,
    RunDatasetFinalizeError,
    RunDatasetWriter,
    StaleRunOutputError,
    evaluate_datagen_keep,
    variant_dataset_directory,
    variant_repo_id,
)
from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult
from lerobot.faults.datagen.frame_logging import (
    POST_STEP_LOGGING_CONTRACT,
    log_fault_recovery_step,
    log_post_step_to_session,
    loss_mask_for_datagen_env,
)
from lerobot.faults.datagen.manifest import (
    EPISODE_METADATA_FIELDS,
    RunManifest,
    build_episode_metadata_row,
    read_run_manifest,
    write_run_manifest_atomic,
)
from lerobot.faults.datagen.paired_context import build_paired_episode_plan
from lerobot.faults.datagen.recipe import (
    DatagenController,
    load_drop_datagen_recipe,
    paired_episode_seed_manifests,
)
from lerobot.faults.datagen.runner import run_drop_datagen_matrix
from lerobot.faults.factory import make_midair_drop_fault

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"

_FAKE_LAYOUT = {
    "alphabet_soup_1": {
        "pos": [0.1, 0.2, 0.03],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
}


def _minimal_processed_frame() -> dict[str, np.ndarray]:
    return {
        "observation.state": np.zeros(8, dtype=np.float32),
        "observation.images.image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.images.image2": np.zeros((256, 256, 3), dtype=np.uint8),
    }


class _RecordingLogger:
    """Tracks commit/discard semantics similar to FaultRecoveryDatasetLogger."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.frames: list[dict[str, Any]] = []
        self._open = False
        self.committed = 0
        self.discarded = 0
        self.finalized = False
        self._total_episodes = 0

    def log_step(
        self,
        observation_dict: dict[str, Any],
        action: np.ndarray,
        task: str,
        loss_mask: float,
        phase: str | None = None,
        annotation: dict[str, Any] | None = None,
    ) -> None:
        self._open = True
        self.frames.append(
            {
                "loss_mask": float(loss_mask),
                "phase": phase,
                "annotation": dict(annotation or {}),
            }
        )

    @property
    def dataset(self) -> Any:
        return self

    @property
    def meta(self) -> Any:
        return self

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    def dataset_episode_index_on_commit(self) -> int:
        return self._total_episodes

    def end_episode(self, episode_data: dict[str, Any] | None = None, **kwargs: Any) -> int:
        del episode_data, kwargs
        index = self._total_episodes
        self._total_episodes += 1
        self.committed += 1
        self._open = False
        return index

    def clear_open_episode(self) -> None:
        self.frames.clear()
        self.discarded += 1
        self._open = False

    def finalize(self) -> None:
        if self._open:
            self.clear_open_episode()
        self.finalized = True


def _recipe_at(tmp_path: Path):
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    return recipe.__class__(
        **{
            **recipe.__dict__,
            "recording": recipe.recording.__class__(
                base_seed=9000,
                output_dir=str(tmp_path / "out"),
                dataset_fps=10,
            ),
        }
    )


def _request_and_result(
    recipe,
    manifest,
    *,
    success: bool,
    outcome: str,
    tmp_path: Path,
) -> tuple[EpisodeRequest, EpisodeResult]:
    plan = build_paired_episode_plan(
        recipe, manifest=manifest, object_name="alphabet_soup_1", num_init_states=50
    )
    request = EpisodeRequest(
        recipe=recipe,
        manifest=manifest,
        object_name="alphabet_soup_1",
        output_dir=tmp_path / "ep",
        paired_plan=plan,
        shared_layout=_FAKE_LAYOUT,
    )
    result = EpisodeResult.from_run(
        request,
        success=success,
        outcome=outcome,
        drop_trigger={"kind": "simple_ik_path", "drop": True},
        trigger_pose=[0.1, 0.2, 0.3],
        actual_dwell_steps=42,
    )
    return request, result


def test_variant_dataset_directories_partition_matrix(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    manifests = paired_episode_seed_manifests(recipe, logical_episode_index=0)
    roots = {variant_dataset_directory(recipe, m) for m in manifests}
    assert len(roots) == 5
    assert tmp_path / "out" / "simple_ik" / "immediate_ik" / "dataset" in roots


def test_refuses_non_empty_run_output_dir(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    root = Path(recipe.recording.output_dir)
    root.mkdir(parents=True)
    (root / "leftover.txt").write_text("prior run")
    with pytest.raises(StaleRunOutputError):
        RunDatasetWriter(recipe)


def test_record_keep_without_logged_frames_raises(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[0]
    writer = RunDatasetWriter(recipe, logger_factory=lambda root, repo_id, **_kw: _RecordingLogger(root))
    session = writer.open_episode_session(manifest)
    req, res = _request_and_result(recipe, manifest, success=True, outcome="ok", tmp_path=tmp_path)
    with pytest.raises(ValueError, match="no logged frames"):
        writer.record_episode_outcome(req, res, session)
    assert not session.is_open
    assert session.logger.committed == 0


def test_finalize_discards_open_episode_without_commit(tmp_path: Path) -> None:
    logger = _RecordingLogger(tmp_path / "ds")
    logger.log_step(_minimal_processed_frame(), np.zeros(7), "task", 1.0)
    assert logger._open
    logger.finalize()
    assert logger.committed == 0
    assert logger.discarded == 1
    assert not logger._open


def test_record_outcome_keep_then_reject(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[0]
    writer = RunDatasetWriter(recipe, logger_factory=lambda root, repo_id, **_kw: _RecordingLogger(root))

    session_keep = writer.open_episode_session(manifest)
    session_keep.log_step(_minimal_processed_frame(), np.zeros(7), "task", 1.0)
    req_ok, res_ok = _request_and_result(recipe, manifest, success=True, outcome="ok", tmp_path=tmp_path)
    row_keep = writer.record_episode_outcome(req_ok, res_ok, session_keep)
    assert row_keep.dataset_episode_index == 0
    assert session_keep.logger.committed == 1

    session_reject = writer.open_episode_session(manifest)
    session_reject.log_step(_minimal_processed_frame(), np.zeros(7), "task", 1.0)
    req_bad, res_bad = _request_and_result(
        recipe, manifest, success=False, outcome="failed", tmp_path=tmp_path
    )
    row_reject = writer.record_episode_outcome(req_bad, res_bad, session_reject)
    assert row_reject.dataset_episode_index is None
    assert session_reject.logger.discarded == 1
    assert session_keep.logger.committed == 1

    writer.finalize()
    loaded = read_run_manifest(tmp_path / "out" / "run_manifest.json")
    assert sum(1 for row in loaded.episodes if row.keep) == 1
    assert sum(1 for row in loaded.episodes if not row.keep) == 1


def test_failed_attempt_in_manifest_not_dataset(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[0]
    req, res = _request_and_result(
        recipe, manifest, success=False, outcome="recovery_finished_outside_basket", tmp_path=tmp_path
    )
    keep, reason = evaluate_datagen_keep(req, res)
    assert keep is False

    writer = RunDatasetWriter(recipe, logger_factory=lambda root, repo_id, **kw: _RecordingLogger(root))
    session = writer.open_episode_session(manifest)
    session.log_step(_minimal_processed_frame(), np.zeros(7), "task", 1.0)
    writer.record_episode_outcome(req, res, session)
    writer.finalize()
    loaded = read_run_manifest(tmp_path / "out" / "run_manifest.json")
    assert loaded.episodes[0].keep is False
    assert loaded.episodes[0].reject_reason
    assert loaded.episodes[0].dataset_episode_index is None
    assert session.logger.committed == 0
    assert session.logger.discarded == 1


def test_manifest_written_atomically(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    payload = RunManifest(
        recipe_name="can_drop_datagen",
        base_seed=1,
        output_dir=str(tmp_path),
        episodes=[],
    )
    write_run_manifest_atomic(path, payload)
    assert path.is_file()
    assert not path.with_suffix(".json.tmp").exists()
    assert read_run_manifest(path).recipe_name == "can_drop_datagen"


def test_episode_metadata_required_fields(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[0]
    req, res = _request_and_result(
        recipe, manifest, success=True, outcome="recovery_completed_in_basket", tmp_path=tmp_path
    )
    row = build_episode_metadata_row(
        req, res, keep=True, keep_reason="recovery_success", dataset_episode_index=0
    )
    data = row.to_dict()
    for field in EPISODE_METADATA_FIELDS:
        assert field in data, field
    assert data["dataset_episode_index"] == 0


def test_loss_mask_uses_env_loss_mask_not_helper_only() -> None:
    cfg = FaultInjectionConfig(
        enabled=True,
        type="midair_drop",
        probability=1.0,
        t_min=0,
        t_max=100,
        post_drop_dwell_steps=3,
        post_drop_mode="continue_then_ik",
    )
    fault = make_midair_drop_fault(cfg, num_envs=1)
    assert fault is not None
    state = fault._states[0]
    state.triggered = True
    state.drop_injection_step = True
    state.recovery_active = False

    class _Env:
        def __init__(self) -> None:
            self.fault = fault

        def loss_mask(self, env_idx: int = 0) -> float:
            return fault.loss_mask_for_env(env_idx)

        def failure_annotation(self, env_idx: int = 0) -> dict[str, Any]:
            return default_failure_frame()

    env = _Env()
    assert fault.loss_mask_for_env(0) == 0.0
    assert loss_mask_for_datagen_env(env, is_drop_episode=True) == env.loss_mask()

    state.drop_injection_step = False
    state.recovery_active = True
    assert loss_mask_for_datagen_env(env, is_drop_episode=True) == 1.0


def test_post_step_logging_contract_documented() -> None:
    assert "POST env.step()" in POST_STEP_LOGGING_CONTRACT


def test_log_post_step_routes_through_session_log_step() -> None:
    logger = _RecordingLogger(Path("/tmp/unused"))
    session = DatagenEpisodeSession(
        manifest=paired_episode_seed_manifests(_recipe_at(Path("/tmp")), logical_episode_index=0)[0],
        dataset_root=Path("/tmp"),
        repo_id="test/repo",
        logger=logger,
        policy_fps=10,
    )
    cfg = FaultInjectionConfig(enabled=True, type="midair_drop", probability=0.0, t_min=0, t_max=1)
    fault = make_midair_drop_fault(cfg, num_envs=1)
    assert fault is not None

    class _Env:
        last_executed_action = np.ones(7, dtype=np.float32)

        def __init__(self, fault_inject: Any) -> None:
            self.fault = fault_inject

        def loss_mask(self, env_idx: int = 0) -> float:
            return self.fault.loss_mask_for_env(env_idx)

        def failure_annotation(self, env_idx: int = 0) -> dict[str, Any]:
            ann = default_failure_frame()
            ann["ever_held_midair"] = np.array([True])
            return ann

    fault._states[0].triggered = True
    fault._states[0].drop_injection_step = True

    log_post_step_to_session(
        session,
        env=_Env(fault),
        post_step_observation={"observation.state": np.zeros(8)},
        executed_action=np.zeros(7),
        task="pick up can and place in basket",
        phase="recovery",
        is_drop_episode=True,
        observation_to_frame=lambda obs: _minimal_processed_frame(),
    )
    assert session.is_open
    assert logger.frames[0]["loss_mask"] == 0.0
    assert bool(logger.frames[0]["annotation"]["ever_held_midair"][0])


def test_smolvla_adapter_uses_injected_session_not_own_logger(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[2]
    plan = build_paired_episode_plan(
        recipe, manifest=manifest, object_name="alphabet_soup_1", num_init_states=50
    )
    logger = _RecordingLogger(tmp_path / "dataset")
    session = DatagenEpisodeSession(
        manifest=manifest,
        dataset_root=variant_dataset_directory(recipe, manifest),
        repo_id=variant_repo_id(recipe, manifest),
        logger=logger,
        policy_fps=10,
    )
    captured: dict[str, Any] = {}

    def _pipeline(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "success": True,
            "behavioral_success": True,
            "actual_dwell_steps": 17,
            "pre_drop_pose": [0.2, 0.1, 0.25],
            "triggered_at": 55,
            "drop_basket_xy_dist": 0.36,
            "drop_trigger_reason": "xy_band",
        }

    adapter = SmolVLADatagenAdapter(recipe, pipeline_runner=_pipeline)
    request = EpisodeRequest(
        recipe=recipe,
        manifest=manifest,
        object_name="alphabet_soup_1",
        output_dir=tmp_path / "episode",
        paired_plan=plan,
        shared_layout=_FAKE_LAYOUT,
        device="cpu",
        episode_session=session,
    )
    result = adapter.run_episode(request)
    assert captured.get("episode_session") is session
    assert "ds_logger" not in captured
    assert captured.get("defer_dataset_commit") is True
    assert captured.get("basket_name") == recipe.basket_name
    assert result.actual_dwell_steps == 17
    assert result.trigger_pose == [0.2, 0.1, 0.25]
    assert result.drop_trigger is not None
    assert result.drop_trigger.get("triggered_at") == 55


def test_simple_ik_adapter_receives_episode_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = _recipe_at(tmp_path)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[0]
    plan = build_paired_episode_plan(
        recipe, manifest=manifest, object_name="alphabet_soup_1", num_init_states=50
    )
    session = MagicMock(spec=DatagenEpisodeSession)
    seen: list[Any] = []

    def _fake_loop(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("episode_session"))
        from lerobot.faults.datagen.controllers import simple_ik as mod

        return mod.SimpleIKEpisodeFacts(True, "recovery_completed_in_basket", None, None, 0)

    monkeypatch.setattr(
        "lerobot.faults.datagen.controllers.simple_ik.run_simple_ik_episode_loop",
        _fake_loop,
    )
    monkeypatch.setattr(
        "lerobot.faults.datagen.controllers.simple_ik.make_env",
        lambda *a, **k: {"libero_object": {0: MagicMock()}},
    )
    mock_env = MagicMock()
    mock_env.fault = MagicMock()
    mock_env.fault.set_recovery_motion_profile = MagicMock()
    monkeypatch.setattr(
        "lerobot.faults.datagen.controllers.simple_ik.DropRecoveryEnvWrapper",
        lambda *a, **k: mock_env,
    )
    monkeypatch.setattr(
        "lerobot.faults.datagen.controllers.simple_ik.unwrap_libero_env",
        lambda v: MagicMock(_init_states=[0], init_state_id=0),
    )
    monkeypatch.setattr(
        "lerobot.faults.datagen.controllers.simple_ik.get_robosuite_env",
        lambda e, i=0: MagicMock(),
    )
    monkeypatch.setattr(
        "lerobot.faults.datagen.controllers.simple_ik.apply_serializable_layout",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "lerobot.faults.datagen.controllers.simple_ik.read_control_freq",
        lambda rs: 20,
    )
    monkeypatch.setattr(
        "lerobot.faults.datagen.controllers.simple_ik._new_planner",
        lambda *a, **k: MagicMock(phase_name="lift", carry_path=None, done=False),
    )

    adapter = SimpleIKDatagenAdapter(recipe)
    request = EpisodeRequest(
        recipe=recipe,
        manifest=manifest,
        object_name="alphabet_soup_1",
        output_dir=tmp_path / "episode",
        paired_plan=plan,
        shared_layout=_FAKE_LAYOUT,
        episode_session=session,
    )
    adapter.run_episode(request)
    assert seen == [session]


def test_runner_integrates_dataset_writer(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    writer = RunDatasetWriter(
        recipe,
        logger_factory=lambda root, repo_id, **_kw: _RecordingLogger(root),
    )

    class _FakeAdapter:
        def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
            assert request.episode_session is not None
            request.episode_session.log_step(
                _minimal_processed_frame(),
                np.zeros(7, dtype=np.float32),
                "task",
                1.0,
            )
            return EpisodeResult.from_run(request, success=True, outcome="ok")

    from lerobot.faults.datagen.layout_provider import LayoutProviderContext

    def _layout(_ctx: LayoutProviderContext) -> dict:
        return _FAKE_LAYOUT

    run_drop_datagen_matrix(
        recipe,
        logical_episode_indices=(0,),
        adapter_factories={
            DatagenController.SIMPLE_IK: lambda _r: _FakeAdapter(),
            DatagenController.SMOLVLA: lambda _r: _FakeAdapter(),
        },
        layout_provider=_layout,
        init_state_count_provider=lambda _r: 50,
        dataset_writer=writer,
    )
    manifest_path = tmp_path / "out" / "run_manifest.json"
    assert manifest_path.is_file()
    loaded = read_run_manifest(manifest_path)
    assert len(loaded.episodes) == 5


def test_finalize_raises_without_manifest_on_logger_failure(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)

    class _BadLogger(_RecordingLogger):
        def finalize(self) -> None:
            raise RuntimeError("encode failed")

    writer = RunDatasetWriter(recipe, logger_factory=lambda root, repo_id, **_kw: _BadLogger(root))
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[0]
    session = writer.open_episode_session(manifest)
    session.log_step(_minimal_processed_frame(), np.zeros(7), "task", 1.0)
    req, res = _request_and_result(recipe, manifest, success=True, outcome="ok", tmp_path=tmp_path)
    writer.record_episode_outcome(req, res, session)
    with pytest.raises(RunDatasetFinalizeError):
        writer.finalize()
    assert not (tmp_path / "out" / "run_manifest.json").exists()


def test_fault_recovery_logger_keep_reject_roundtrip(tmp_path: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    recipe = _recipe_at(tmp_path)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[0]
    writer = RunDatasetWriter(recipe)
    dataset_root = variant_dataset_directory(recipe, manifest)
    repo_id = variant_repo_id(recipe, manifest)

    session_keep = writer.open_episode_session(manifest)
    for mask in (1.0, 0.0, 1.0):
        session_keep.log_step(
            _minimal_processed_frame(),
            np.zeros(7, dtype=np.float32),
            "pick up alphabet_soup_1 and place it in basket_1",
            mask,
            annotation=default_failure_frame(),
        )
    req_ok, res_ok = _request_and_result(recipe, manifest, success=True, outcome="ok", tmp_path=tmp_path)
    kept = writer.record_episode_outcome(req_ok, res_ok, session_keep)
    assert kept.dataset_episode_index == 0

    session_reject = writer.open_episode_session(manifest)
    session_reject.log_step(_minimal_processed_frame(), np.zeros(7), "task", 1.0)
    req_bad, res_bad = _request_and_result(recipe, manifest, success=False, outcome="fail", tmp_path=tmp_path)
    rejected = writer.record_episode_outcome(req_bad, res_bad, session_reject)
    assert rejected.dataset_episode_index is None
    assert not session_reject.is_open

    writer.finalize()
    loaded = read_run_manifest(tmp_path / "out" / "run_manifest.json")
    assert len(loaded.episodes) == 2
    assert {row.dataset_episode_index for row in loaded.episodes if row.keep} == {0}
    assert all(row.dataset_episode_index is None for row in loaded.episodes if not row.keep)

    ds = LeRobotDataset(repo_id=repo_id, root=dataset_root, download_videos=False)
    assert ds.meta.total_episodes == 1
    episode = LeRobotDataset(repo_id=repo_id, root=dataset_root, episodes=[0], download_videos=False)
    masks = [float(np.asarray(episode[i]["loss_mask"]).reshape(-1)[0]) for i in range(len(episode))]
    assert masks == [1.0, 0.0, 1.0]
