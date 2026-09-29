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

"""LeRobot dataset card (README) generation for merged drop datagen runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["DatasetCardStats", "render_dataset_card"]


@dataclass(frozen=True)
class MatrixRowCount:
    controller: str
    post_drop_mode: str
    drop: bool
    kept: int
    attempted: int


@dataclass(frozen=True)
class ObjectMatrixCounts:
    object_name: str
    task_id: int
    rows: tuple[MatrixRowCount, ...]


@dataclass(frozen=True)
class SplitSizes:
    train: int
    test_in_distribution: int


@dataclass(frozen=True)
class DatasetCardStats:
    recipe_name: str
    recipe_content_hash: str
    view_name: str
    dataset_fps: int
    control_hz: int
    object_counts: tuple[ObjectMatrixCounts, ...]
    release_heights_m: tuple[float, ...]
    release_to_landing_durations_s: tuple[float, ...]
    split_sizes: SplitSizes
    objects_and_task_ids: dict[str, int] = field(default_factory=dict)


_COLUMN_LINES: tuple[str, ...] = (
    "observation.images.image — RGB wrist/agent camera (model input).",
    "observation.images.image2 — RGB second camera (model input).",
    "observation.state — 8-D robot/object state (model input).",
    "action — 7-D control command.",
    "task — natural-language task string.",
    "loss_mask — training loss weight scalar.",
    "tick_index — control tick index (label / privileged).",
    "drop_release — pulse: object release this interval (label / privileged).",
    "drop_event — mid-air drop active (label / privileged).",
    "attempt_index — grasp attempt index (label / privileged).",
    "is_failure — failure flag (label / privileged).",
    "ever_held_midair — object held mid-air (label / privileged).",
    "failure_onset — pulse: failure onset (label / privileged).",
    "failure_type — failure category (label / privileged).",
    "injection_active — fault injection active (label / privileged).",
    "phase — controller phase name (label / privileged).",
)


def _format_histogram(values: tuple[float, ...], *, unit: str, bins: int = 5) -> str:
    if not values:
        return f"(no data) — unit: {unit}"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"single value {lo:.4f} {unit} (n={len(values)})"
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / width) if width > 0 else 0)
        counts[idx] += 1
    lines = [f"range [{lo:.4f}, {hi:.4f}] {unit}, n={len(values)}"]
    for i, count in enumerate(counts):
        edge_lo = lo + i * width
        edge_hi = lo + (i + 1) * width
        lines.append(f"  [{edge_lo:.4f}, {edge_hi:.4f}): {count}")
    return "\n".join(lines)


def render_dataset_card(stats: DatasetCardStats) -> str:
    """Render a Hugging Face–style dataset README from precomputed stats."""
    lines: list[str] = [
        f"# {stats.recipe_name} ({stats.view_name})",
        "",
        "## Provenance",
        f"- Recipe: `{stats.recipe_name}`",
        f"- Recipe content hash: `{stats.recipe_content_hash}`",
        f"- Dataset FPS: {stats.dataset_fps}",
        f"- Control rate: {stats.control_hz} Hz",
        "",
        "## Splits",
        f"- train episodes: {stats.split_sizes.train}",
        f"- test_in_distribution episodes: {stats.split_sizes.test_in_distribution}",
        "",
        "## Objects and LIBERO task ids",
    ]
    for name, task_id in sorted(stats.objects_and_task_ids.items()):
        lines.append(f"- `{name}` → task id {task_id}")
    lines.extend(["", "## Counts per object × matrix row (kept / attempted)", ""])
    for block in stats.object_counts:
        lines.append(f"### {block.object_name} (task {block.task_id})")
        for row in block.rows:
            drop_label = "drop" if row.drop else "no_drop"
            lines.append(
                f"- {row.controller} × {row.post_drop_mode} ({drop_label}): "
                f"{row.kept} kept / {row.attempted} attempted"
            )
        lines.append("")
    lines.extend(
        [
            "## Drop release height (m, from failure_segments)",
            _format_histogram(stats.release_heights_m, unit="m"),
            "",
            "## Release → landing duration (s, from failure_segments)",
            _format_histogram(stats.release_to_landing_durations_s, unit="s"),
            "",
            "## Columns",
            "Model inputs: `observation.images.image`, `observation.images.image2`, `observation.state`.",
            "Labels / privileged (not model inputs):",
        ]
    )
    for col_line in _COLUMN_LINES:
        if "model input" in col_line:
            continue
        if col_line.startswith("observation.") or col_line.startswith("action"):
            continue
        lines.append(f"- {col_line}")
    lines.extend(
        [
            "",
            "## Model inputs (summary)",
            "- `observation.images.image`",
            "- `observation.images.image2`",
            "- `observation.state`",
            "",
        ]
    )
    return "\n".join(lines)
