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
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.faults.annotation import default_failure_frame
from lerobot.faults.datagen.controllers.simple_ik import SimpleIKDatagenAdapter
from lerobot.faults.datagen.controllers.smolvla import SmolVLADatagenAdapter
from lerobot.faults.datagen.dataset_writer import (
    DatagenEpisodeSession,
    RunDatasetWriter,
    evaluate_datagen_keep,
    variant_dataset_directory,
    variant_repo_id,
)
from lerobot.faults.datagen.drop_timing import DropDecision
from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult
from lerobot.faults.datagen.frame_logging import log_fault_recovery_step
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
    PostDropMode,
    load_drop_datagen_recipe,
    paired_episode_seed_manifests,
)
from lerobot.faults.datagen.runner import run_drop_datagen_matrix
from lerobot.faults.recovery.loss_mask import loss_mask_from_fault

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"

_FAKE_LAYOUT = {
    "alphabet_soup_1": {
        "pos": [0.1, 0.2, 0.03],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
}


class _RecordingLogger:
    """Minimal stand-in for FaultRecoveryDatasetLogger in unit tests."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.frames: list[dict[str, Any]] = []
        self._open = False
        self.committed = 0
        self.discarded = 0
        self.finalized = False
        self.last_episode_data: dict[str, Any] | None = None

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

    def end_episode(self, episode_data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if episode_data is None:
            episode_data = kwargs.get("episode_data")
        self.last_episode_data = episode_data
        self.committed += 1
        self._open = False

    def clear_open_episode(self) -> None:
        self.frames.clear()
        self.discarded += 1
        self._open = False

    def finalize(self) -> None:
        self.finalized = True

    @property
    def loss_mask_counts(self) -> dict[float, int]:
        counts = {0.0: 0, 1.0: 0}
        for frame in self.frames:
            key = 1.0 if frame["loss_mask"] >= 0.5 else 0.0
            counts[key] = counts.get(key, 0) + 1
        return counts


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


def test_variant_dataset_directories_partition_matrix(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    manifests = paired_episode_seed_manifests(recipe, logical_episode_index=0)
    roots = {variant_dataset_directory(recipe, m) for m in manifests}
    assert len(roots) == 5
    assert tmp_path / "out" / "simple_ik" / "immediate_ik" / "dataset" in roots
    assert tmp_path / "out" / "smolvla" / "reset_then_ik" / "dataset" in roots


def test_run_writer_begin_commit_discard(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[0]
    created: list[Path] = []

    def _factory(root: Path, repo_id: str, **_kwargs: Any) -> _RecordingLogger:
        created.append(Path(root))
        return _RecordingLogger(root)

    writer = RunDatasetWriter(recipe, logger_factory=_factory)
    session = writer.open_episode_session(manifest)
    session.log_step(
        {"observation.state": np.zeros(8, dtype=np.float32)},
        np.zeros(7, dtype=np.float32),
        "task",
        1.0,
        annotation=default_failure_frame(),
    )
    writer.commit_episode(session, keep=True, metadata={"episode_index": 0})
    session2 = writer.open_episode_session(manifest)
    session2.log_step(
        {"observation.state": np.zeros(8, dtype=np.float32)},
        np.zeros(7, dtype=np.float32),
        "task",
        0.0,
    )
    writer.commit_episode(session2, keep=False, metadata={"episode_index": 1})
    writer.finalize()
    assert len(created) == 1
    assert created[0] == variant_dataset_directory(recipe, manifest)
    assert session.logger.committed == 1
    assert session2.logger.discarded == 1
    assert session.logger.finalized is True


def test_failed_attempt_in_manifest_not_dataset(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[0]
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
        success=False,
        outcome="recovery_finished_outside_basket",
        drop_trigger={"kind": "simple_ik_path", "drop": True},
    )
    keep, reason = evaluate_datagen_keep(request, result)
    assert keep is False
    assert reason

    writer = RunDatasetWriter(recipe, logger_factory=lambda root, repo_id, **kw: _RecordingLogger(root))
    session = writer.open_episode_session(manifest)
    session.log_step(
        {"observation.state": np.zeros(8, dtype=np.float32)},
        np.zeros(7, dtype=np.float32),
        "task",
        1.0,
    )
    row = build_episode_metadata_row(request, result, keep=keep, keep_reason=reason)
    writer.commit_episode(session, keep=keep, metadata=row.to_dict())
    writer._episode_rows.append(row)
    writer.finalize()
    manifest_path = tmp_path / "out" / "run_manifest.json"
    write_run_manifest_atomic(
        manifest_path,
        RunManifest(
            recipe_name=recipe.name,
            base_seed=recipe.recording.base_seed,
            output_dir=str(recipe.recording.output_dir),
            episodes=[row],
        ),
    )
    loaded = read_run_manifest(manifest_path)
    assert loaded.episodes[0].keep is False
    assert loaded.episodes[0].reject_reason
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
        success=True,
        outcome="recovery_completed_in_basket",
        drop_trigger={"kind": "simple_ik_path", "drop": True, "target_t": 0.5},
        trigger_pose=[0.1, 0.2, 0.3],
        actual_dwell_steps=80,
    )
    row = build_episode_metadata_row(request, result, keep=True, keep_reason="recovery_success")
    data = row.to_dict()
    for field in EPISODE_METADATA_FIELDS:
        assert field in data, field
    assert data["controller"] == "simple_ik"
    assert data["init_state_id"] == plan.init_state_id
    assert data["shared_layout"] == _FAKE_LAYOUT


def test_frame_annotations_include_ever_held_midair() -> None:
    logger = _RecordingLogger(Path("/tmp/unused"))
    annotation = default_failure_frame()
    annotation["ever_held_midair"] = np.array([True])
    annotation["is_failure"] = np.array([True])
    log_fault_recovery_step(
        logger,
        observation_dict={
            "observation.state": np.zeros(8, dtype=np.float32),
            "observation.images.image": np.zeros((256, 256, 3), dtype=np.uint8),
            "observation.images.image2": np.zeros((256, 256, 3), dtype=np.uint8),
        },
        executed_action=np.zeros(7, dtype=np.float32),
        task="pick",
        phase="recovery",
        loss_mask=1.0,
        annotation=annotation,
    )
    assert bool(logger.frames[0]["annotation"]["ever_held_midair"][0])


def test_loss_masks_drop_dwell_zero_recovery_one() -> None:
    assert loss_mask_from_fault(triggered=True, drop_injection_step=True, recovery_active=True) == 0.0
    assert (
        loss_mask_from_fault(
            triggered=True,
            drop_injection_step=False,
            recovery_active=False,
            post_drop_dwell_step=True,
        )
        == 0.0
    )
    assert loss_mask_from_fault(triggered=True, drop_injection_step=False, recovery_active=True) == 1.0


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
        return {"success": True, "behavioral_success": True}

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
    adapter.run_episode(request)
    assert captured["ds_logger"] is logger
    assert captured["defer_dataset_commit"] is True
    assert captured["dataset_root"] == variant_dataset_directory(recipe, manifest)
    assert captured.get("wipe_output_dir") is False


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
                {"observation.state": np.zeros(8, dtype=np.float32)},
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
