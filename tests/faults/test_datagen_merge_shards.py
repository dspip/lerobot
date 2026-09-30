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
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("datasets")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from fault_system.datagen.dataset_card import (
    DatasetCardStats,
    MatrixRowCount,
    ObjectMatrixCounts,
    SplitSizes,
    render_dataset_card,
)
from fault_system.datagen.failure_segments import FAILURE_SEGMENTS_REL_PATH, FailureSegmentRow
from fault_system.datagen.manifest import (
    EpisodeMetadataRow,
    RunManifest,
    RunStatus,
    write_run_manifest_atomic,
)
from fault_system.datagen.merge_shards import MergeShardsError, build_splits, merge_drop_datagen_shards
from tests.fixtures.constants import DUMMY_REPO_ID

_MIN_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
    "tick_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def _episode_row(
    *,
    logical: int,
    dataset_index: int | None,
    keep: bool,
    drop: bool,
    object_name: str = "alphabet_soup_1",
) -> EpisodeMetadataRow:
    return EpisodeMetadataRow(
        controller="simple_ik",
        post_drop_mode="immediate_ik",
        object_name=object_name,
        logical_episode_index=logical,
        episode_index=logical,
        episode_seed=1000 + logical,
        layout_seed=2000 + logical,
        drop_seed=3000 + logical,
        controller_seed=4000 + logical,
        init_state_id=0,
        shared_layout={},
        drop_decision={"drop": drop, "reason": "test", "drop_u": 0.5, "step": 1},
        drop_trigger=None,
        trigger_pose=None,
        configured_dwell_steps=0,
        actual_dwell_steps=None,
        outcome="recovery_success" if keep else "recovery_failed",
        success=keep,
        keep=keep,
        keep_reason="recovery_success" if keep else None,
        reject_reason=None if keep else "recovery_failed",
        dataset_episode_index=dataset_index if keep else None,
    )


def _write_shard(
    root: Path,
    *,
    logical_range: tuple[int, int],
    recipe_hash: str,
    rows: list[EpisodeMetadataRow],
    n_dataset_episodes: int,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ds_root = root / "dataset"
    ds = LeRobotDataset.create(
        f"{DUMMY_REPO_ID}_shard",
        fps=10,
        features=_MIN_FEATURES,
        root=ds_root,
        use_videos=False,
    )
    for _ in range(n_dataset_episodes):
        for _tick in (0, 2):
            ds.add_frame(
                {
                    "observation.state": torch.zeros(8),
                    "action": torch.zeros(7),
                    "tick_index": torch.tensor([_tick], dtype=torch.int64),
                    "task": "pick",
                }
            )
        ds.save_episode()
    ds.finalize()

    seg_rows = [
        FailureSegmentRow(
            episode_index=i,
            logical_episode_index=rows[i].logical_episode_index if i < len(rows) else i,
            object_name="alphabet_soup_1",
            task_id=0,
            controller="simple_ik",
            post_drop_mode="immediate_ik",
            is_drop_episode=False,
            attempt_index=0,
            attempt_start_frame=0,
            attempt_end_frame=1,
            has_drop=False,
            release_tick=None,
            landing_tick=None,
            release_frame=None,
            landing_frame=None,
            release_time_s=None,
            landing_time_s=None,
            object_z_at_release=None,
            episode_seed=0,
            layout_seed=0,
        )
        for i in range(n_dataset_episodes)
    ]
    seg_path = ds_root / FAILURE_SEGMENTS_REL_PATH
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in seg_rows]).to_parquet(seg_path, index=False)

    manifest = RunManifest(
        recipe_name="can_drop_datagen",
        base_seed=9000,
        output_dir=str(root),
        episodes=rows,
        run_status=RunStatus.COMPLETE,
        logical_range=logical_range,
        recipe_content_hash=recipe_hash,
    )
    write_run_manifest_atomic(root / "run_manifest.json", manifest)


def test_build_splits_keeps_paired_rows_together() -> None:
    rows = [
        _episode_row(logical=5, dataset_index=0, keep=True, drop=True),
        _episode_row(logical=5, dataset_index=1, keep=True, drop=False),
        _episode_row(logical=40, dataset_index=2, keep=True, drop=True),
        _episode_row(logical=40, dataset_index=3, keep=True, drop=False),
    ]
    splits = build_splits(rows, test_percent=15)
    # logical 5 -> 5 % 100 < 15 => test; logical 40 -> train
    assert set(splits["test_in_distribution"]) == {0, 1}
    assert set(splits["train"]) == {2, 3}


def test_merge_two_synthetic_shards(tmp_path: Path) -> None:
    recipe_hash = "abc123"
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    _write_shard(
        shard0,
        logical_range=(0, 1),
        recipe_hash=recipe_hash,
        rows=[
            _episode_row(logical=0, dataset_index=0, keep=True, drop=True),
            _episode_row(logical=0, dataset_index=None, keep=False, drop=False),
        ],
        n_dataset_episodes=1,
    )
    _write_shard(
        shard1,
        logical_range=(1, 2),
        recipe_hash=recipe_hash,
        rows=[
            _episode_row(logical=1, dataset_index=0, keep=True, drop=False),
        ],
        n_dataset_episodes=1,
    )
    out = tmp_path / "merged"
    merge_drop_datagen_shards([shard0, shard1], out, test_percent=15)

    manifest = json.loads((out / "run_manifest.json").read_text())
    kept = [e for e in manifest["episodes"] if e["keep"]]
    assert len(kept) == 2
    assert {e["dataset_episode_index"] for e in kept} == {0, 1}
    assert manifest["shard_logical_ranges"] == [[0, 1], [1, 2]]

    seg = pd.read_parquet(out / "dataset" / FAILURE_SEGMENTS_REL_PATH)
    assert set(seg["episode_index"].tolist()) == {0, 1}

    splits = json.loads((out / "dataset" / "meta" / "splits.json").read_text())
    assert "train" in splits and "test_in_distribution" in splits

    readme = (out / "dataset" / "README.md").read_text()
    assert "## Provenance" in readme
    assert "## Splits" in readme
    assert "## Columns" in readme


def test_merge_rejects_overlapping_ranges(tmp_path: Path) -> None:
    recipe_hash = "same"
    for name, lr in (("a", (0, 2)), ("b", (1, 3))):
        root = tmp_path / name
        _write_shard(
            root,
            logical_range=lr,
            recipe_hash=recipe_hash,
            rows=[_episode_row(logical=lr[0], dataset_index=0, keep=True, drop=False)],
            n_dataset_episodes=1,
        )
    with pytest.raises(MergeShardsError, match="overlap"):
        merge_drop_datagen_shards([tmp_path / "a", tmp_path / "b"], tmp_path / "out")


def test_render_dataset_card_contains_key_sections() -> None:
    stats = DatasetCardStats(
        recipe_name="can_drop_datagen",
        recipe_content_hash="deadbeef",
        view_name="dataset",
        dataset_fps=10,
        control_hz=20,
        object_counts=(
            ObjectMatrixCounts(
                object_name="alphabet_soup_1",
                task_id=0,
                rows=(
                    MatrixRowCount(
                        controller="simple_ik",
                        post_drop_mode="immediate_ik",
                        drop=True,
                        kept=1,
                        attempted=2,
                    ),
                ),
            ),
        ),
        release_heights_m=(0.42,),
        release_to_landing_durations_s=(0.5,),
        split_sizes=SplitSizes(train=3, test_in_distribution=1),
        objects_and_task_ids={"alphabet_soup_1": 0},
    )
    text = render_dataset_card(stats)
    assert "Recipe content hash" in text
    assert "test_in_distribution" in text
    assert "privileged" in text
