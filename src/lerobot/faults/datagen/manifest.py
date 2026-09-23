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

"""Run-level manifest and per-episode metadata rows for unified drop datagen."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult
from lerobot.faults.datagen.recipe import effective_post_drop_dwell_steps

__all__ = [
    "EPISODE_METADATA_FIELDS",
    "EpisodeMetadataRow",
    "RunManifest",
    "build_episode_metadata_row",
    "read_run_manifest",
    "write_run_manifest_atomic",
]

EPISODE_METADATA_FIELDS: tuple[str, ...] = (
    "controller",
    "post_drop_mode",
    "object_name",
    "logical_episode_index",
    "episode_index",
    "episode_seed",
    "layout_seed",
    "drop_seed",
    "controller_seed",
    "init_state_id",
    "shared_layout",
    "drop_decision",
    "drop_trigger",
    "trigger_pose",
    "configured_dwell_steps",
    "actual_dwell_steps",
    "outcome",
    "success",
    "keep",
    "keep_reason",
    "reject_reason",
    "fault_config",
    "output_dir",
    "dataset_episode_index",
)


@dataclass
class EpisodeMetadataRow:
    controller: str
    post_drop_mode: str
    object_name: str
    logical_episode_index: int
    episode_index: int
    episode_seed: int
    layout_seed: int
    drop_seed: int
    controller_seed: int
    init_state_id: int
    shared_layout: dict[str, Any]
    drop_decision: dict[str, Any]
    drop_trigger: dict[str, Any] | None
    trigger_pose: list[float] | None
    configured_dwell_steps: int
    actual_dwell_steps: int | None
    outcome: str
    success: bool
    keep: bool
    keep_reason: str | None = None
    reject_reason: str | None = None
    fault_config: dict[str, Any] = field(default_factory=dict)
    output_dir: str = ""
    dataset_episode_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    recipe_name: str
    base_seed: int
    output_dir: str
    episodes: list[EpisodeMetadataRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_name": self.recipe_name,
            "base_seed": self.base_seed,
            "output_dir": self.output_dir,
            "episodes": [row.to_dict() for row in self.episodes],
        }


def build_episode_metadata_row(
    request: EpisodeRequest,
    result: EpisodeResult,
    *,
    keep: bool,
    keep_reason: str | None,
    dataset_episode_index: int | None = None,
) -> EpisodeMetadataRow:
    manifest = request.manifest
    plan = request.paired_plan
    dwell = effective_post_drop_dwell_steps(request.recipe, manifest.post_drop_mode)
    drop_decision = {
        "drop": plan.drop_decision.drop,
        "reason": plan.drop_decision.reason,
        "drop_u": plan.drop_u,
        "step": plan.drop_decision.step,
    }
    fault_config = {
        "type": "midair_drop",
        "post_drop_mode": manifest.post_drop_mode.value,
        "post_drop_dwell_steps": dwell,
        "object_name": request.object_name,
        "basket_name": request.recipe.basket_name,
        "controller": manifest.controller.value,
    }
    reject_reason = None if keep else (keep_reason or result.outcome)
    return EpisodeMetadataRow(
        controller=manifest.controller.value,
        post_drop_mode=manifest.post_drop_mode.value,
        object_name=request.object_name,
        logical_episode_index=manifest.logical_episode_index,
        episode_index=manifest.episode_index,
        episode_seed=manifest.episode_seed,
        layout_seed=manifest.layout_seed,
        drop_seed=manifest.drop_seed,
        controller_seed=manifest.controller_seed,
        init_state_id=plan.init_state_id,
        shared_layout=request.shared_layout,
        drop_decision=drop_decision,
        drop_trigger=result.drop_trigger,
        trigger_pose=result.trigger_pose,
        configured_dwell_steps=dwell,
        actual_dwell_steps=result.actual_dwell_steps,
        outcome=result.outcome,
        success=result.success,
        keep=keep,
        keep_reason=keep_reason if keep else None,
        reject_reason=reject_reason,
        fault_config=fault_config,
        output_dir=str(request.output_dir),
        dataset_episode_index=dataset_episode_index if keep else None,
    )


def write_run_manifest_atomic(path: Path, manifest: RunManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(path)


def _row_from_dict(data: dict[str, Any]) -> EpisodeMetadataRow:
    return EpisodeMetadataRow(**{k: data[k] for k in EPISODE_METADATA_FIELDS if k in data})


def read_run_manifest(path: Path) -> RunManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    episodes = [_row_from_dict(item) for item in raw.get("episodes", [])]
    return RunManifest(
        recipe_name=str(raw["recipe_name"]),
        base_seed=int(raw["base_seed"]),
        output_dir=str(raw["output_dir"]),
        episodes=episodes,
    )
