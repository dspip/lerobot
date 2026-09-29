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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("datasets")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.faults.datagen.audit import audit_drop_run
from lerobot.faults.datagen.manifest import EpisodeMetadataRow, RunManifest, RunStatus, write_run_manifest_atomic
from tests.fixtures.constants import DUMMY_REPO_ID


def _manifest_row(logical: int, dataset_index: int, *, drop: bool) -> EpisodeMetadataRow:
    return EpisodeMetadataRow(
        controller="simple_ik",
        post_drop_mode="immediate_ik",
        object_name="alphabet_soup_1",
        logical_episode_index=logical,
        episode_index=0,
        episode_seed=1,
        layout_seed=2,
        drop_seed=3,
        controller_seed=4,
        init_state_id=0,
        shared_layout={},
        drop_decision={"drop": drop, "reason": "t", "drop_u": 0.1, "step": 0},
        drop_trigger=None,
        trigger_pose=None,
        configured_dwell_steps=0,
        actual_dwell_steps=None,
        outcome="ok",
        success=True,
        keep=True,
        dataset_episode_index=dataset_index,
    )


def _write_minimal_run(
    root: Path,
    *,
    hi_frames: list[dict],
    lo_frames: list[dict],
    manifest_rows: list[EpisodeMetadataRow],
) -> None:
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        "tick_index": {"dtype": "int64", "shape": (1,), "names": None},
        "drop_release": {"dtype": "bool", "shape": (1,), "names": None},
        "drop_event": {"dtype": "bool", "shape": (1,), "names": None},
        "attempt_index": {"dtype": "int64", "shape": (1,), "names": None},
        "is_failure": {"dtype": "bool", "shape": (1,), "names": None},
        "injection_active": {"dtype": "bool", "shape": (1,), "names": None},
        "phase": {"dtype": "string", "shape": (1,), "names": None},
        "loss_mask": {"dtype": "float32", "shape": (1,), "names": None},
    }

    def _fill_dataset(view_root: Path, repo_suffix: str, frames: list[dict], fps: int) -> None:
        ds = LeRobotDataset.create(
            f"{DUMMY_REPO_ID}_{repo_suffix}",
            fps=fps,
            features=features,
            root=view_root,
            use_videos=False,
        )
        for fr in frames:
            payload = {
                "observation.state": torch.as_tensor(fr["observation.state"], dtype=torch.float32),
                "action": torch.zeros(7),
                "tick_index": torch.tensor([fr["tick"]], dtype=torch.int64),
                "drop_release": torch.tensor([fr.get("drop_release", False)]),
                "drop_event": torch.tensor([fr.get("drop_event", False)]),
                "attempt_index": torch.tensor([fr.get("attempt_index", 0)], dtype=torch.int64),
                "is_failure": torch.tensor([fr.get("is_failure", False)]),
                "injection_active": torch.tensor([False]),
                "phase": "carry",
                "loss_mask": torch.tensor([1.0]),
                "task": "pick",
            }
            ds.add_frame(payload)
        ds.save_episode()
        ds.finalize()

    _fill_dataset(root / "dataset", "lo", lo_frames, fps=10)
    _fill_dataset(root / "dataset_20hz", "hi", hi_frames, fps=20)
    write_run_manifest_atomic(
        root / "run_manifest.json",
        RunManifest(
            recipe_name="can_drop_datagen",
            base_seed=1,
            output_dir=str(root),
            episodes=manifest_rows,
            run_status=RunStatus.COMPLETE,
        ),
    )


def test_audit_detects_pulse_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "run"
    # 20 Hz: release on tick 1; 10 Hz logs tick 0,2 — OR window for tick 2 should include tick 1
    hi = [
        {"tick": 0, "observation.state": np.zeros(8), "drop_release": False, "drop_event": False, "attempt_index": 0},
        {"tick": 1, "observation.state": np.zeros(8), "drop_release": True, "drop_event": True, "attempt_index": 0},
        {"tick": 2, "observation.state": np.zeros(8), "drop_release": False, "drop_event": True, "attempt_index": 0},
    ]
    lo = [
        {"tick": 0, "observation.state": np.zeros(8), "drop_release": False},
        {"tick": 2, "observation.state": np.zeros(8), "drop_release": False},  # wrong: should pulse True
    ]
    rows = [_manifest_row(0, 0, drop=True)]
    _write_minimal_run(root, hi_frames=hi, lo_frames=lo, manifest_rows=rows)
    report = audit_drop_run(root)
    assert not report.ok
    assert "pulse_mismatch_drop_release" in report.error_counts


def test_audit_detects_multiple_releases(tmp_path: Path) -> None:
    root = tmp_path / "run"
    hi = [
        {"tick": 0, "observation.state": np.zeros(8), "drop_release": True, "drop_event": True, "attempt_index": 0},
        {"tick": 1, "observation.state": np.zeros(8), "drop_release": True, "drop_event": True, "attempt_index": 0},
    ]
    lo = [{"tick": 0, "observation.state": np.zeros(8), "drop_release": True}]
    rows = [_manifest_row(0, 0, drop=True)]
    _write_minimal_run(root, hi_frames=hi, lo_frames=lo, manifest_rows=rows)
    report = audit_drop_run(root)
    assert "multiple_releases" in report.error_counts


def test_audit_detects_pair_divergence(tmp_path: Path) -> None:
    root = tmp_path / "run"
    state_a = np.zeros(8)
    state_b = np.ones(8)
    hi_drop = [
        {"tick": 0, "observation.state": state_a, "drop_release": False, "drop_event": False, "attempt_index": 0},
        {"tick": 1, "observation.state": state_a, "drop_release": True, "drop_event": True, "attempt_index": 0},
    ]
    hi_nodrop = [
        {"tick": 0, "observation.state": state_b, "drop_release": False, "drop_event": False, "attempt_index": 0},
    ]
    # Two episodes in one dataset — build manually with two save_episode calls
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        "tick_index": {"dtype": "int64", "shape": (1,), "names": None},
        "drop_release": {"dtype": "bool", "shape": (1,), "names": None},
        "drop_event": {"dtype": "bool", "shape": (1,), "names": None},
        "attempt_index": {"dtype": "int64", "shape": (1,), "names": None},
        "is_failure": {"dtype": "bool", "shape": (1,), "names": None},
        "injection_active": {"dtype": "bool", "shape": (1,), "names": None},
        "phase": {"dtype": "string", "shape": (1,), "names": None},
        "loss_mask": {"dtype": "float32", "shape": (1,), "names": None},
    }

    def add_ep(ds: LeRobotDataset, frames: list[dict]) -> None:
        for fr in frames:
            ds.add_frame(
                {
                    "observation.state": torch.as_tensor(fr["observation.state"], dtype=torch.float32),
                    "action": torch.zeros(7),
                    "tick_index": torch.tensor([fr["tick"]], dtype=torch.int64),
                    "drop_release": torch.tensor([fr.get("drop_release", False)]),
                    "drop_event": torch.tensor([fr.get("drop_event", False)]),
                    "attempt_index": torch.tensor([fr.get("attempt_index", 0)], dtype=torch.int64),
                    "is_failure": torch.tensor([False]),
                    "injection_active": torch.tensor([False]),
                    "phase": "carry",
                    "loss_mask": torch.tensor([1.0]),
                    "task": "pick",
                }
            )
        ds.save_episode()

    for view, suffix, fps, episodes in (
        ("dataset", "lo", 10, [hi_drop[:1], hi_nodrop]),
        ("dataset_20hz", "hi", 20, [hi_drop, hi_nodrop]),
    ):
        ds = LeRobotDataset.create(
            f"{DUMMY_REPO_ID}_{suffix}",
            fps=fps,
            features=features,
            root=root / view,
            use_videos=False,
        )
        for ep_frames in episodes:
            if view == "dataset":
                lo_ep = [
                    {
                        "tick": 0,
                        "observation.state": ep_frames[0]["observation.state"],
                        "drop_release": ep_frames[0].get("drop_release", False),
                    }
                ]
                add_ep(ds, lo_ep)
            else:
                add_ep(ds, ep_frames)
        ds.finalize()

    rows = [
        _manifest_row(7, 0, drop=True),
        _manifest_row(7, 1, drop=False),
    ]
    write_run_manifest_atomic(
        root / "run_manifest.json",
        RunManifest(
            recipe_name="can_drop_datagen",
            base_seed=1,
            output_dir=str(root),
            episodes=rows,
            run_status=RunStatus.COMPLETE,
        ),
    )
    report = audit_drop_run(root)
    assert "pair_diverges_before_release" in report.error_counts
