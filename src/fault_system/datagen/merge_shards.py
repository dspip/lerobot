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

"""Merge sharded drop datagen runs into one LeRobot dataset + manifest."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from fault_system.datagen.dataset_card import (
    DatasetCardStats,
    MatrixRowCount,
    ObjectMatrixCounts,
    SplitSizes,
    render_dataset_card,
)
from fault_system.datagen.failure_segments import FAILURE_SEGMENTS_REL_PATH
from fault_system.datagen.libero_object_tasks import official_task_id
from fault_system.datagen.manifest import (
    EPISODE_METADATA_FIELDS,
    EpisodeMetadataRow,
    RunManifest,
    RunStatus,
    write_run_manifest_atomic,
)
from fault_system.datagen.shard_range import logical_range_from_manifest_raw, ranges_overlap
from fault_system.datagen.task_index import align_official_task_indices
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

__all__ = ["MergeShardsError", "build_splits", "merge_drop_datagen_shards"]

VIEW_DATASET = "dataset"
VIEW_MASTER = "dataset_20hz"


class MergeShardsError(ValueError):
    """Raised when shard manifests or datasets cannot be merged."""


def _manifest_path(shard_dir: Path) -> Path:
    return Path(shard_dir) / "run_manifest.json"


def _load_manifest_raw(shard_dir: Path) -> dict[str, Any]:
    path = _manifest_path(shard_dir)
    if not path.is_file():
        raise MergeShardsError(f"missing run_manifest.json under {shard_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _shard_logical_range(raw: dict[str, Any], shard_dir: Path) -> tuple[int, int]:
    parsed = logical_range_from_manifest_raw(raw)
    if parsed is not None:
        return parsed
    # Full-run shard without explicit range: infer from episode rows.
    indices = [
        int(row["logical_episode_index"])
        for row in raw.get("episodes", [])
        if isinstance(row, dict)
    ]
    if not indices:
        raise MergeShardsError(
            f"shard {shard_dir} has no logical_range and no episodes to infer range"
        )
    return min(indices), max(indices) + 1


def _views_present(shard_dir: Path) -> frozenset[str]:
    views: set[str] = set()
    if (shard_dir / VIEW_DATASET / "meta" / "info.json").is_file():
        views.add(VIEW_DATASET)
    if (shard_dir / VIEW_MASTER / "meta" / "info.json").is_file():
        views.add(VIEW_MASTER)
    return frozenset(views)


def _validate_shards(
    shard_dirs: list[Path],
) -> tuple[list[Path], list[dict[str, Any]], list[tuple[int, int]], frozenset[str]]:
    if len(shard_dirs) < 1:
        raise MergeShardsError("at least one shard directory is required")
    raws: list[dict[str, Any]] = []
    ranges: list[tuple[int, int]] = []
    views_intersection: frozenset[str] | None = None
    recipe_name: str | None = None
    recipe_hash: str | None = None
    base_seed: int | None = None

    for shard_dir in shard_dirs:
        raw = _load_manifest_raw(shard_dir)
        status = str(raw.get("run_status", RunStatus.COMPLETE.value))
        if status != RunStatus.COMPLETE.value:
            raise MergeShardsError(f"shard {shard_dir} run_status is {status!r}, expected complete")
        name = str(raw["recipe_name"])
        bseed = int(raw["base_seed"])
        rh = raw.get("recipe_content_hash")
        if rh is None:
            raise MergeShardsError(f"shard {shard_dir} missing recipe_content_hash (re-run shard with new CLI)")
        rh = str(rh)
        if recipe_name is None:
            recipe_name, base_seed, recipe_hash = name, bseed, rh
        else:
            if name != recipe_name:
                raise MergeShardsError(f"recipe_name mismatch: {name!r} vs {recipe_name!r}")
            if bseed != base_seed:
                raise MergeShardsError(f"base_seed mismatch across shards")
            if rh != recipe_hash:
                raise MergeShardsError("recipe_content_hash mismatch across shards")
        logical_range = _shard_logical_range(raw, shard_dir)
        views = _views_present(shard_dir)
        if not views:
            raise MergeShardsError(f"shard {shard_dir} has no finalized dataset view")
        views_intersection = views if views_intersection is None else views_intersection & views
        raws.append(raw)
        ranges.append(logical_range)

    assert views_intersection is not None
    if not views_intersection:
        raise MergeShardsError("no common dataset view across all shards")
    sorted_pairs = sorted(zip(shard_dirs, raws, ranges, strict=True), key=lambda item: item[2][0])
    shard_dirs = [p for p, _, _ in sorted_pairs]
    raws = [r for _, r, _ in sorted_pairs]
    ranges = [rg for _, _, rg in sorted_pairs]

    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            if ranges_overlap(ranges[i], ranges[j]):
                raise MergeShardsError(
                    f"logical ranges overlap: {ranges[i]} and {ranges[j]} "
                    f"({shard_dirs[i]} vs {shard_dirs[j]})"
                )
    return shard_dirs, raws, ranges, views_intersection


def _episode_row_from_dict(data: dict[str, Any]) -> EpisodeMetadataRow:
    return EpisodeMetadataRow(**{k: data[k] for k in EPISODE_METADATA_FIELDS if k in data})


def _remap_merged_episode_rows(
    raws: list[dict[str, Any]],
) -> list[EpisodeMetadataRow]:
    merged: list[EpisodeMetadataRow] = []
    next_dataset_index = 0
    for raw in raws:
        for item in raw.get("episodes", []):
            if not isinstance(item, dict):
                continue
            row = _episode_row_from_dict(item)
            if row.keep:
                row = replace(row, dataset_episode_index=next_dataset_index)
                next_dataset_index += 1
            else:
                row = replace(row, dataset_episode_index=None)
            merged.append(row)
    return merged


def build_splits(
    kept_rows: list[EpisodeMetadataRow],
    *,
    test_percent: int,
) -> dict[str, Any]:
    """Build ``meta/splits.json`` payload; pairs share split by logical episode index."""
    if not 0 <= test_percent <= 100:
        raise ValueError("test_percent must be in [0, 100]")
    logical_to_episodes: dict[int, list[int]] = {}
    objects: dict[str, int] = {}
    for row in kept_rows:
        if row.dataset_episode_index is None:
            continue
        logical_to_episodes.setdefault(row.logical_episode_index, []).append(row.dataset_episode_index)
        if row.object_name not in objects:
            objects[row.object_name] = official_task_id(row.object_name)
    train: list[int] = []
    test: list[int] = []
    for _logical, ep_indices in sorted(logical_to_episodes.items()):
        in_test = (int(_logical) % 100) < int(test_percent)
        bucket = test if in_test else train
        bucket.extend(sorted(ep_indices))
    train.sort()
    test.sort()
    return {
        "train": train,
        "test_in_distribution": test,
        "test_percent": int(test_percent),
        "objects": {name: objects[name] for name in sorted(objects)},
    }


def _concat_failure_segments(
    shard_dirs: list[Path],
    view: str,
    episode_offsets: list[int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for shard_dir, offset in zip(shard_dirs, episode_offsets, strict=True):
        path = shard_dir / view / FAILURE_SEGMENTS_REL_PATH
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        if offset and not df.empty:
            df = df.copy()
            df["episode_index"] = df["episode_index"] + int(offset)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _kept_episode_count(raw: dict[str, Any]) -> int:
    return sum(1 for row in raw.get("episodes", []) if isinstance(row, dict) and row.get("keep"))


def _dataset_episode_count(view_root: Path) -> int:
    meta = LeRobotDatasetMetadata("local", root=view_root)
    return int(meta.total_episodes)


def _verify_merged_view(
    view_root: Path,
    kept_rows: list[EpisodeMetadataRow],
    *,
    reference_rows: list[EpisodeMetadataRow] | None = None,
) -> None:
    n_kept = sum(1 for row in kept_rows if row.keep)
    n_ds = _dataset_episode_count(view_root)
    if n_ds != n_kept:
        raise MergeShardsError(
            f"{view_root}: dataset has {n_ds} episodes but manifest lists {n_kept} kept rows"
        )
    if reference_rows is not None:
        if len(reference_rows) != len(kept_rows):
            raise MergeShardsError("internal: kept row count mismatch between views")
        for i, (a, b) in enumerate(zip(reference_rows, kept_rows, strict=True)):
            if a.logical_episode_index != b.logical_episode_index:
                raise MergeShardsError(f"episode metadata order mismatch at {i}")
            if a.controller != b.controller or a.post_drop_mode != b.post_drop_mode:
                raise MergeShardsError(f"episode metadata mismatch at dataset index {i}")
            if bool(a.drop_decision.get("drop")) != bool(b.drop_decision.get("drop")):
                raise MergeShardsError(f"drop flag mismatch at dataset index {i}")


def _collect_card_stats(
    view_root: Path,
    view_name: str,
    manifest: RunManifest,
    splits: dict[str, Any],
    *,
    control_hz: int,
) -> DatasetCardStats:
    meta = LeRobotDatasetMetadata("local", root=view_root)
    seg_path = view_root / FAILURE_SEGMENTS_REL_PATH
    release_heights: list[float] = []
    durations: list[float] = []
    if seg_path.is_file():
        seg = pd.read_parquet(seg_path)
        for _, row in seg.iterrows():
            z = row.get("object_z_at_release")
            if z is not None and pd.notna(z) and row.get("has_drop"):
                release_heights.append(float(z))
            rt = row.get("release_time_s")
            lt = row.get("landing_time_s")
            if (
                pd.notna(rt)
                and pd.notna(lt)
                and row.get("has_drop")
                and row.get("attempt_index") == 0
            ):
                durations.append(float(lt) - float(rt))

    kept = [row for row in manifest.episodes if row.keep]
    counts: dict[tuple[str, str, str, bool], list[int]] = {}
    for row in manifest.episodes:
        key = (row.object_name, row.controller, row.post_drop_mode, bool(row.drop_decision.get("drop")))
        bucket = counts.setdefault(key, [0, 0])
        bucket[1] += 1
        if row.keep:
            bucket[0] += 1
    by_object: dict[str, list[MatrixRowCount]] = {}
    for (obj, ctrl, mode, drop), (kept_n, attempted) in sorted(counts.items()):
        by_object.setdefault(obj, []).append(
            MatrixRowCount(
                controller=ctrl,
                post_drop_mode=mode,
                drop=drop,
                kept=kept_n,
                attempted=attempted,
            )
        )
    object_counts = tuple(
        ObjectMatrixCounts(
            object_name=obj,
            task_id=official_task_id(obj),
            rows=tuple(by_object[obj]),
        )
        for obj in sorted(by_object)
    )
    objects_map = splits.get("objects") or {}
    return DatasetCardStats(
        recipe_name=manifest.recipe_name,
        recipe_content_hash=manifest.recipe_content_hash or "",
        view_name=view_name,
        dataset_fps=int(meta.fps),
        control_hz=int(control_hz),
        object_counts=object_counts,
        release_heights_m=tuple(release_heights),
        release_to_landing_durations_s=tuple(durations),
        split_sizes=SplitSizes(
            train=len(splits.get("train", [])),
            test_in_distribution=len(splits.get("test_in_distribution", [])),
        ),
        objects_and_task_ids={str(k): int(v) for k, v in objects_map.items()},
    )


def merge_drop_datagen_shards(
    shard_dirs: list[Path],
    output_dir: Path,
    *,
    test_percent: int = 15,
    control_hz: int = 20,
) -> Path:
    """Aggregate shard datasets, merge sidecars/manifest, write splits and dataset cards."""
    shard_dirs = [Path(p).resolve() for p in shard_dirs]
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MergeShardsError(f"output directory {output_dir} is not empty")

    shard_dirs, raws, ranges, views = _validate_shards(shard_dirs)
    merged_rows = _remap_merged_episode_rows(raws)
    kept_rows = [row for row in merged_rows if row.keep]

    episode_offsets: list[int] = []
    offset = 0
    for raw in raws:
        episode_offsets.append(offset)
        offset += _kept_episode_count(raw)

    recipe_hash = str(raws[0].get("recipe_content_hash", ""))
    merged_manifest = RunManifest(
        recipe_name=str(raws[0]["recipe_name"]),
        base_seed=int(raws[0]["base_seed"]),
        output_dir=str(output_dir),
        episodes=merged_rows,
        run_status=RunStatus.COMPLETE,
        recipe_content_hash=recipe_hash,
        shard_logical_ranges=tuple(ranges),
    )
    splits = build_splits(kept_rows, test_percent=test_percent)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest_atomic(output_dir / "run_manifest.json", merged_manifest)

    reference_kept: list[EpisodeMetadataRow] | None = None
    for view in sorted(views):
        view_out = output_dir / view
        repo_ids = [f"{merged_manifest.recipe_name}_shard_{i}" for i in range(len(shard_dirs))]
        roots = [shard / view for shard in shard_dirs]
        aggregate_datasets(
            repo_ids=repo_ids,
            aggr_repo_id=merged_manifest.recipe_name,
            roots=roots,
            aggr_root=view_out,
        )
        align_official_task_indices(view_out)
        seg_df = _concat_failure_segments(shard_dirs, view, episode_offsets)
        if not seg_df.empty:
            seg_path = view_out / FAILURE_SEGMENTS_REL_PATH
            seg_path.parent.mkdir(parents=True, exist_ok=True)
            seg_df.to_parquet(seg_path, index=False)

        _verify_merged_view(view_out, kept_rows, reference_rows=reference_kept)
        if reference_kept is None:
            reference_kept = kept_rows

        splits_path = view_out / "meta" / "splits.json"
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits_path.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        card_stats = _collect_card_stats(
            view_out,
            view,
            merged_manifest,
            splits,
            control_hz=control_hz,
        )
        readme = view_out / "README.md"
        readme.write_text(render_dataset_card(card_stats), encoding="utf-8")

    try:
        dir_fd = os.open(output_dir, os.O_DIRECTORY)
    except OSError:
        return output_dir
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return output_dir
