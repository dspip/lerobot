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

"""Audit helpers for XY-band checkpoint outputs (no GPU/sim)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.faults.annotation import PHASE_NOMINAL, PHASE_RECOVERY
from lerobot.faults.recovery.mix_recording import episode_dir_for_seed
from lerobot.faults.recovery.xy_band_checkpoint import XY_BANDS


@dataclass
class AuditIssue:
    level: str
    message: str


@dataclass
class CheckpointAuditResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    issues: list[AuditIssue] = field(default_factory=list)
    action_stats: dict[str, Any] = field(default_factory=dict)
    discarded: list[dict[str, Any]] = field(default_factory=list)

    def exit_code(self) -> int:
        if any(i.level == "fail" for i in self.issues):
            return 1
        return 0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _np_item(v: Any) -> Any:
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        try:
            return v[0]
        except Exception:
            return v
    return v


def _find_parquet_files(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.rglob("*.parquet"))


def _action_vector_stats(actions: Any) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return {"n": 0}
    qs = [0.01, 0.1, 0.5, 0.9, 0.99]
    out: dict[str, Any] = {
        "n": int(arr.shape[0]),
        "dims": int(arr.shape[1]) if arr.ndim == 2 else 1,
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "quantiles": {str(q): np.quantile(arr, q, axis=0).tolist() for q in qs},
        "near_zero_variance_dims": [
            int(i) for i, s in enumerate(arr.std(axis=0).tolist()) if s < 1e-6
        ],
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
    }
    return out


def compute_dataset_action_stats(dataset_root: Path) -> dict[str, Any]:
    import pandas as pd

    files = _find_parquet_files(dataset_root)
    if not files:
        return {"error": "no_parquet"}
    frames = [pd.read_parquet(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    if "action" not in df.columns:
        return {"error": "missing_action_column"}

    actions = df["action"].map(lambda v: [_np_item(x) for x in v]).tolist()
    stats_all = _action_vector_stats(actions)

    phase = df["phase"].map(lambda v: int(_np_item(v)))
    episode_kind = df.get("episode_kind")
    nominal_mask = phase == PHASE_NOMINAL
    if episode_kind is not None:
        nominal_mask = nominal_mask | episode_kind.map(
            lambda v: str(_np_item(v)) == "nominal" if v is not None else False
        )
    recovery_mask = phase == PHASE_RECOVERY

    stats_nominal_vla = _action_vector_stats(
        [a for a, m in zip(actions, nominal_mask, strict=False) if bool(m)]
    )
    stats_recovery = _action_vector_stats(
        [a for a, m in zip(actions, recovery_mask, strict=False) if bool(m)]
    )

    state_issues: list[str] = []
    if "observation.state" in df.columns:
        states = df["observation.state"].map(lambda v: [_np_item(x) for x in v]).tolist()
        import numpy as np

        sarr = np.asarray(states, dtype=np.float64)
        if np.isnan(sarr).any():
            state_issues.append("observation.state has NaN")
        if np.isinf(sarr).any():
            state_issues.append("observation.state has Inf")

    return {
        "all": stats_all,
        "phase0_or_nominal": stats_nominal_vla,
        "phase3_recovery": stats_recovery,
        "state_issues": state_issues,
    }


def _episode_annotation_counts(df: Any, episode_index: int) -> dict[str, int]:
    import pandas as pd

    ep = df["episode_index"].map(lambda v: int(_np_item(v)))
    sub = df.loc[ep == episode_index]
    if sub.empty:
        return {}
    inj = sub["injection_active"].map(lambda v: bool(_np_item(v))).sum()
    onset = sub["failure_onset"].map(lambda v: bool(_np_item(v))).sum()
    fail = sub["is_failure"].map(lambda v: bool(_np_item(v))).sum()
    return {
        "n_injection_active": int(inj),
        "n_failure_onset": int(onset),
        "n_is_failure": int(fail),
        "n_frames": int(len(sub)),
    }


def audit_xy_band_checkpoint(output_dir: Path) -> CheckpointAuditResult:
    output_dir = Path(output_dir)
    log_path = output_dir / "checkpoint_log.json"
    result = CheckpointAuditResult()
    if not log_path.is_file():
        result.issues.append(AuditIssue("fail", f"Missing {log_path}"))
        return result

    payload = _load_json(log_path)
    dataset_root = Path(payload.get("dataset_dir") or output_dir / "dataset")
    kept = payload.get("kept_registry") or []
    attempts = payload.get("attempts") or []
    result.discarded = [a for a in attempts if not a.get("kept")]

    import pandas as pd

    parquet_files = _find_parquet_files(dataset_root)
    df = None
    if parquet_files:
        df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)

    drop_by_band = {name: 0 for name, _, _ in XY_BANDS}
    nominal_count = 0

    for entry in kept:
        seed = int(entry["seed"])
        kind = str(entry.get("kind", "drop"))
        band = entry.get("band", "")
        ep_dir = episode_dir_for_seed(output_dir, seed)
        pipeline_path = ep_dir / "pipeline_log.json"
        pipeline = _load_json(pipeline_path) if pipeline_path.is_file() else {}

        triggered = pipeline.get("triggered_at")
        drop_dist = pipeline.get("drop_basket_xy_dist")
        initial_xy = pipeline.get("initial_soup_xy")
        landing = pipeline.get("landing_pos")
        video = pipeline.get("video_mp4") or entry.get("video_mp4")
        if video:
            video = str(Path(video).resolve().as_uri())

        ann: dict[str, int] = {}
        duration_s = None
        if df is not None and "episode_index" in df.columns:
            ep_ids = df["episode_index"].map(lambda v: int(_np_item(v))).unique()
            for epid in ep_ids:
                meta_path = dataset_root / "meta" / "episodes.jsonl"
                # Match by seed in pipeline only when single-file heuristic
                pass
        if df is not None:
            # Heuristic: row count for all episodes — per-seed mapping via episodes meta
            meta = dataset_root / "meta" / "episodes.jsonl"
            episode_id = None
            if meta.is_file():
                for line in meta.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("seed") == seed:
                        episode_id = int(row.get("episode_index", row.get("episode_index", 0)))
                        break
            if episode_id is None and len(df):
                episode_id = int(_np_item(df["episode_index"].iloc[0]))
            if episode_id is not None:
                ann = _episode_annotation_counts(df, episode_id)
                fps = 10.0
                info_path = dataset_root / "meta" / "info.json"
                if info_path.is_file():
                    fps = float(_load_json(info_path).get("fps", fps))
                duration_s = ann.get("n_frames", 0) / fps

        if kind == "drop":
            if band in drop_by_band:
                drop_by_band[band] += 1
            if ann:
                if ann.get("n_injection_active", 0) != 1:
                    result.issues.append(
                        AuditIssue(
                            "fail",
                            f"seed {seed}: expected 1 injection_active, got {ann.get('n_injection_active')}",
                        )
                    )
                if ann.get("n_failure_onset", 0) != 1:
                    result.issues.append(
                        AuditIssue(
                            "fail",
                            f"seed {seed}: expected 1 failure_onset, got {ann.get('n_failure_onset')}",
                        )
                    )
        elif kind == "nominal":
            nominal_count += 1
            if ann:
                if ann.get("n_injection_active", 0) != 0:
                    result.issues.append(
                        AuditIssue("fail", f"seed {seed}: nominal has injection_active")
                    )
                if ann.get("n_failure_onset", 0) != 0:
                    result.issues.append(
                        AuditIssue("fail", f"seed {seed}: nominal has failure_onset")
                    )
                if ann.get("n_is_failure", 0) != 0:
                    result.issues.append(
                        AuditIssue("fail", f"seed {seed}: nominal has is_failure")
                    )

        result.rows.append(
            {
                "episode_id": entry.get("episode_index"),
                "seed": seed,
                "type": kind,
                "band": band,
                "initial_soup_xy": initial_xy,
                "drop_frame": triggered,
                "drop_xy": pipeline.get("pre_drop_pose"),
                "landing_xy": landing,
                "drop_basket_xy_dist": drop_dist,
                "regrasp": pipeline.get("regrasped_after_drop"),
                "final_success": pipeline.get("success"),
                **ann,
                "duration_s": duration_s,
                "video_mp4": video,
            }
        )

    drop_total = sum(drop_by_band.values())
    if drop_total != 12:
        result.issues.append(
            AuditIssue("fail", f"Expected 12 drop episodes in registry, got {drop_total}")
        )
    for name, _, _ in XY_BANDS:
        if drop_by_band[name] != 3:
            result.issues.append(
                AuditIssue("fail", f"Band {name}: expected 3 drops, got {drop_by_band[name]}")
            )
    if nominal_count != 8:
        result.issues.append(
            AuditIssue("fail", f"Expected 8 nominal episodes, got {nominal_count}")
        )

    if dataset_root.is_dir():
        result.action_stats = compute_dataset_action_stats(dataset_root)
        all_stats = result.action_stats.get("all") or {}
        if all_stats.get("has_nan"):
            result.issues.append(AuditIssue("fail", "action column contains NaN"))
        if all_stats.get("has_inf"):
            result.issues.append(AuditIssue("fail", "action column contains Inf"))
        for dim in all_stats.get("near_zero_variance_dims") or []:
            result.issues.append(
                AuditIssue("warn", f"action dim {dim} has near-zero variance (std < 1e-6)")
            )
        for msg in result.action_stats.get("state_issues") or []:
            result.issues.append(AuditIssue("fail", msg))

    return result


def format_checkpoint_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "seed",
        "type",
        "band",
        "initial_soup_xy",
        "drop_frame",
        "drop_basket_xy_dist",
        "regrasp",
        "final_success",
        "n_frames",
        "video_mp4",
    ]
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append(
            "\t".join(
                str(row.get(h, "")) if h != "initial_soup_xy" else str(row.get(h, ""))
                for h in headers
            )
        )
    return "\n".join(lines)
