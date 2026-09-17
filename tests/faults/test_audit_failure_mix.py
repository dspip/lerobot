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

"""Audit failure-mix outputs (fixtures only, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot.faults.recovery.mix_audit import audit_failure_mix


def _write_drop_episode(
    output_dir: Path,
    *,
    seed: int,
    drop_basket_xy_dist: float,
    delay_steps: int = 40,
    triggered_at: int = 120,
    pre_drop_xy: tuple[float, float] | None = None,
    drop_trigger_reason: str = "delay_elapsed",
) -> None:
    ep = output_dir / "episodes" / f"seed_{seed}"
    ep.mkdir(parents=True)
    (ep / "videos").mkdir()
    (ep / "videos" / "full_pipeline.mp4").write_bytes(b"fake")
    pipeline = {
        "success": True,
        "episode_kind": "drop",
        "triggered_at": triggered_at,
        "carry_steps": 35,
        "seat_assisted": False,
        "pre_drop_pose": [
            (pre_drop_xy or (0.12, -0.08))[0],
            (pre_drop_xy or (0.12, -0.08))[1],
            0.25,
            0.0,
            0.0,
            0.0,
        ],
        "fault_config": {"post_grasp_delay_steps": delay_steps},
        "checks": {"fault_triggered": True, "object_in_basket": True},
    }
    (ep / "pipeline_log.json").write_text(json.dumps(pipeline), encoding="utf-8")
    event = {
        "event": "midair_drop",
        "status": "triggered",
        "drop_basket_xy_dist": drop_basket_xy_dist,
        "drop_trigger_reason": drop_trigger_reason,
        "episode_step": triggered_at,
    }
    (ep / "fault_events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")


def _write_mix_fixture(tmp_path: Path, *, seeds: list[int], dists: list[float]) -> Path:
    root = tmp_path / "mix"
    dataset = root / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "videos" / "chunk-000").mkdir(parents=True)
    (dataset / "videos" / "chunk-000" / "cam.mp4").write_bytes(b"v")
    kept = []
    for idx, (seed, dist) in enumerate(zip(seeds, dists, strict=True)):
        _write_drop_episode(
            root,
            seed=seed,
            drop_basket_xy_dist=dist,
            triggered_at=100 + idx * 17,
            delay_steps=40 + idx * 5,
            pre_drop_xy=(0.10 + 0.08 * idx, -0.05 - 0.06 * idx),
        )
        kept.append({"seed": seed, "kind": "drop", "delay_steps": 40 + idx * 5})
    mix_log = {
        "dataset_root": str(dataset),
        "kept_episodes": kept,
        "kept_drop_seeds": seeds,
        "kept_nominal_seeds": [],
    }
    (root / "mix_log.json").write_text(json.dumps(mix_log), encoding="utf-8")
    info = {"total_episodes": len(seeds), "features": {}}
    (dataset / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


def test_audit_passes_clean_fixture(tmp_path: Path) -> None:
    root = _write_mix_fixture(tmp_path, seeds=[2000, 2001], dists=[0.35, 0.42])
    result = audit_failure_mix(
        root,
        min_unique_triggered_at=2,
        verify_parquet_fn=None,
    )
    assert result.exit_code() == 0
    assert len(result.rows) == 2


def test_audit_fails_drop_too_close_to_basket(tmp_path: Path) -> None:
    root = _write_mix_fixture(tmp_path, seeds=[2000], dists=[0.10])
    result = audit_failure_mix(root, verify_parquet_fn=None)
    assert result.has_failures
    assert result.exit_code() == 1


def test_audit_fails_basket_keepout_edge_reason(tmp_path: Path) -> None:
    root = tmp_path / "mix"
    dataset = root / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "videos" / "chunk-000").mkdir(parents=True)
    (dataset / "videos" / "chunk-000" / "cam.mp4").write_bytes(b"v")
    _write_drop_episode(
        root,
        seed=2000,
        drop_basket_xy_dist=0.35,
        drop_trigger_reason="basket_keepout_edge",
    )
    mix_log = {
        "dataset_root": str(dataset),
        "kept_episodes": [{"seed": 2000, "kind": "drop", "delay_steps": 40}],
    }
    (root / "mix_log.json").write_text(json.dumps(mix_log), encoding="utf-8")
    (dataset / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1, "features": {}}),
        encoding="utf-8",
    )
    result = audit_failure_mix(root, verify_parquet_fn=None)
    assert result.has_failures
    assert any("basket_keepout_edge" in i.message for i in result.issues)


def test_audit_fails_episode_count_mismatch(tmp_path: Path) -> None:
    root = _write_mix_fixture(tmp_path, seeds=[2000], dists=[0.40])
    info_path = root / "dataset" / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_episodes"] = 99
    info_path.write_text(json.dumps(info), encoding="utf-8")
    result = audit_failure_mix(root, verify_parquet_fn=None)
    assert result.has_failures
    assert any("total_episodes" in i.message for i in result.issues)
