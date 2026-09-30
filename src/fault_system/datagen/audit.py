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

"""Audit finalized drop datagen datasets (single run or merged)."""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fault_system.datagen.failure_segments import FAILURE_SEGMENTS_REL_PATH
from fault_system.datagen.manifest import RunStatus

__all__ = ["AuditReport", "audit_drop_run"]

VIEW_DATASET = "dataset"
VIEW_MASTER = "dataset_20hz"
LEVEL_LABEL_COLUMNS = (
    "drop_event",
    "attempt_index",
    "is_failure",
    "injection_active",
    "phase",
    "loss_mask",
)
PULSE_COLUMNS = ("drop_release", "failure_onset")


class AuditError(RuntimeError):
    """Raised when audit preconditions fail."""


@dataclass
class AuditReport:
    run_dir: str
    error_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error_counts

    def record(self, kind: str, msg: str, *, max_notes: int = 3) -> None:
        self.error_counts[kind] = self.error_counts.get(kind, 0) + 1
        if self.error_counts[kind] <= max_notes:
            self.notes.append(f"{kind}: {msg}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "ok": self.ok,
            "error_counts": dict(self.error_counts),
            "notes": list(self.notes),
            "summary": dict(self.summary),
        }


def _scalar(col: pd.Series) -> np.ndarray:
    return np.array([np.asarray(v).reshape(-1)[0] for v in col])


def _load_view_parquet(run_dir: Path, view: str) -> pd.DataFrame:
    pattern = str(run_dir / view / "data" / "**" / "*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise AuditError(f"no parquet data under {run_dir / view / 'data'}")
    tmp_suffix = [f for f in files if f.endswith(".tmp") or ".tmp." in f]
    if tmp_suffix:
        raise AuditError(f"refusing in-progress parquet under {view}: {tmp_suffix[0]}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _assert_finalized_view(run_dir: Path, view: str) -> None:
    info = run_dir / view / "meta" / "info.json"
    if not info.is_file():
        raise AuditError(f"missing finalized meta/info.json for view {view!r}")


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        raise AuditError(f"missing {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    status = str(raw.get("run_status", RunStatus.COMPLETE.value))
    if status == RunStatus.IN_PROGRESS.value:
        raise AuditError("run_manifest run_status is in_progress; audit after finalize")
    return raw


def _audit_tick_spacing(
    report: AuditReport,
    df: pd.DataFrame,
    *,
    kind: str = "lo_tick_spacing",
) -> None:
    for ep in sorted(df.episode_index.unique()):
        frames = df[df.episode_index == ep]
        tl = _scalar(frames.tick_index)
        if tl.size == 0:
            continue
        if tl[0] != 0:
            report.record(kind, f"ep {ep} head {tl[:5]}")
            continue
        if tl.size == 1:
            continue
        diffs = np.unique(np.diff(tl))
        if len(diffs) != 1 or int(diffs[0]) not in (1, 2):
            report.record(kind, f"ep {ep} head {tl[:5]}")


def _audit_drop_labels(report: AuditReport, df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    by_ep: dict[int, dict[str, Any]] = {}
    for ep in sorted(df.episode_index.unique()):
        h = df[df.episode_index == ep]
        hs = np.stack(h["observation.state"].values)
        rel = np.flatnonzero(_scalar(h.drop_release).astype(bool)) if "drop_release" in h else np.array([])
        ev = _scalar(h.drop_event).astype(bool) if "drop_event" in h else np.array([], dtype=bool)
        att = _scalar(h.attempt_index) if "attempt_index" in h else np.array([])
        fail = _scalar(h.is_failure).astype(bool) if "is_failure" in h else np.array([], dtype=bool)
        if len(rel) == 0:
            if ev.any() or (len(att) and att.any()):
                report.record("nodrop_has_drop_labels", f"ep {ep}")
            if fail.any():
                report.record("nodrop_is_failure", f"ep {ep}")
            by_ep[int(ep)] = {"drop": False, "state": hs}
            continue
        if len(rel) != 1:
            report.record("multiple_releases", f"ep {ep} releases {rel}")
        r = int(rel[0])
        evi = np.flatnonzero(ev)
        if evi.size == 0 or evi[0] != r or not (np.diff(evi) == 1).all():
            report.record("drop_event_not_contiguous_from_release", f"ep {ep} rel {r}")
            by_ep[int(ep)] = {"drop": True, "state": hs, "release": r}
            continue
        land_end = int(evi[-1])
        if att[: land_end + 1].any() or not (att[land_end + 1 :] == 1).all():
            report.record("attempt_index", f"ep {ep} land_end {land_end}")
        if fail[:r].any():
            report.record("is_failure_before_release", f"ep {ep}")
        by_ep[int(ep)] = {"drop": True, "state": hs, "release": r, "fall_ticks": len(evi)}
    return by_ep


def _audit_cross_view(
    report: AuditReport,
    lo: pd.DataFrame,
    hi: pd.DataFrame,
    *,
    lo_stride: int = 2,
    hi_dt: float = 0.05,
    lo_dt: float = 0.1,
) -> dict[int, dict[str, Any]]:
    for ep in sorted(hi.episode_index.unique()):
        h = hi[hi.episode_index == ep]
        l = lo[lo.episode_index == ep]
        th, tl = _scalar(h.tick_index), _scalar(l.tick_index)
        if not (np.diff(th) == 1).all() or th[0] != 0:
            report.record("hi_tick_spacing", f"ep {ep}")
        if not (np.diff(tl) == lo_stride).all() or tl[0] != 0:
            report.record("lo_tick_spacing", f"ep {ep} head {tl[:5]}")
        if len(th) > 1 and not np.allclose(np.diff(_scalar(h.timestamp)), hi_dt, atol=1e-4):
            report.record("hi_timestamps", f"ep {ep}")
        if len(tl) > 1 and not np.allclose(np.diff(_scalar(l.timestamp)), lo_dt, atol=1e-4):
            report.record("lo_timestamps", f"ep {ep}")
        hmap = {int(t): i for i, t in enumerate(th)}
        if not set(tl.tolist()) <= set(hmap):
            report.record("lo_tick_missing_in_hi", f"ep {ep}")
            continue
        idx = [hmap[int(t)] for t in tl]
        hs = np.stack(h["observation.state"].values)
        ls = np.stack(l["observation.state"].values)
        if not np.array_equal(hs[idx], ls):
            report.record("state_mismatch", f"ep {ep}")
        ha = np.stack(h["action"].values)
        la = np.stack(l["action"].values)
        if not np.array_equal(ha[idx], la):
            report.record("action_mismatch", f"ep {ep}")
        for col in LEVEL_LABEL_COLUMNS:
            if col not in h.columns or col not in l.columns:
                continue
            if not np.array_equal(_scalar(h[col])[idx], _scalar(l[col])):
                report.record(f"level_mismatch_{col}", f"ep {ep}")
        for col in PULSE_COLUMNS:
            if col not in h.columns or col not in l.columns:
                continue
            hp = _scalar(h[col]).astype(bool)
            lp = _scalar(l[col]).astype(bool)
            expect = np.array(
                [hp[(th > (tl[k - 1] if k else -1)) & (th <= tl[k])].any() for k in range(len(tl))]
            )
            if not np.array_equal(expect, lp):
                report.record(f"pulse_mismatch_{col}", f"ep {ep}")

    return _audit_drop_labels(report, hi)


def _audit_pairs(
    report: AuditReport,
    by_ep: dict[int, dict[str, Any]],
    manifest_raw: dict[str, Any],
) -> None:
    kept = [e for e in manifest_raw.get("episodes", []) if isinstance(e, dict) and e.get("keep")]
    man_by_ep = {int(e["dataset_episode_index"]): e for e in kept if e.get("dataset_episode_index") is not None}
    pairs: dict[int, dict[bool, int]] = {}
    for ep, meta in man_by_ep.items():
        li = int(meta["logical_episode_index"])
        drop = bool(meta.get("drop_decision", {}).get("drop"))
        pairs.setdefault(li, {})[drop] = ep
    for li, mapping in pairs.items():
        if True not in mapping or False not in mapping:
            continue
        d = by_ep.get(mapping[True])
        n = by_ep.get(mapping[False])
        if d is None or n is None:
            report.record("pair_missing_episode", f"logical {li}")
            continue
        if not d.get("drop") or n.get("drop"):
            continue
        r = d.get("release")
        if r is None:
            continue
        ds, ns = d["state"], n["state"]
        upto = min(int(r), len(ds), len(ns))
        if upto > 0 and not np.array_equal(ds[:upto], ns[:upto]):
            diff = np.any(ds[:upto] != ns[:upto], axis=1)
            first = int(np.flatnonzero(diff)[0]) if diff.any() else -1
            report.record("pair_diverges_before_release", f"logical {li} tick {first} < release {r}")


def _audit_failure_segments(report: AuditReport, run_dir: Path, view: str, lo: pd.DataFrame) -> None:
    seg_path = run_dir / view / FAILURE_SEGMENTS_REL_PATH
    if not seg_path.is_file():
        report.record("missing_failure_segments", str(seg_path))
        return
    seg = pd.read_parquet(seg_path)
    for ep in sorted(lo.episode_index.unique()):
        ep_i = int(ep)
        frames = lo[lo.episode_index == ep_i]
        seg_ep = seg[seg.episode_index == ep_i]
        if frames.empty:
            continue
        if "drop_release" in frames:
            rel_frames = np.flatnonzero(_scalar(frames.drop_release).astype(bool))
            seg_rel = seg_ep[seg_ep.get("has_drop", False) == True]  # noqa: E712
            if len(rel_frames) >= 1 and seg_rel.empty:
                report.record("segment_missing_drop_row", f"ep {ep_i}")
            if len(rel_frames) == 1 and not seg_rel.empty:
                rf = int(rel_frames[0])
                sf = seg_rel.iloc[0].get("release_frame")
                if sf is not None and pd.notna(sf) and int(sf) != rf:
                    report.record("segment_release_frame", f"ep {ep_i} frame {rf} vs seg {sf}")
        if "attempt_index" in frames:
            att = _scalar(frames.attempt_index)
            if att.size and att.max() >= 1:
                has_att = "attempt_index" in seg_ep.columns and not seg_ep.empty
                seg_att = _scalar(seg_ep.attempt_index) if has_att else np.array([])
                if not (seg_att.size and seg_att.max() >= 1):
                    report.record("segment_missing_attempt1", f"ep {ep_i}")


def _audit_splits(report: AuditReport, run_dir: Path, view: str) -> None:
    splits_path = run_dir / view / "meta" / "splits.json"
    if not splits_path.is_file():
        return
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    manifest = _load_manifest(run_dir)
    train_eps = set(splits.get("train", []))
    test_eps = set(splits.get("test_in_distribution", []))
    overlap = train_eps & test_eps
    if overlap:
        report.record("split_episode_overlap", f"{sorted(overlap)[:5]}")
    logical_train: set[int] = set()
    logical_test: set[int] = set()
    kept = [e for e in manifest.get("episodes", []) if e.get("keep")]
    ep_to_logical = {
        int(e["dataset_episode_index"]): int(e["logical_episode_index"])
        for e in kept
        if e.get("dataset_episode_index") is not None
    }
    for ep in train_eps:
        li = ep_to_logical.get(int(ep))
        if li is not None:
            logical_train.add(li)
    for ep in test_eps:
        li = ep_to_logical.get(int(ep))
        if li is not None:
            logical_test.add(li)
    leaked = logical_train & logical_test
    if leaked:
        report.record("split_logical_leak", f"{sorted(leaked)[:5]}")


def _audit_heldout_objects(
    report: AuditReport,
    manifest_raw: dict[str, Any],
    heldout: frozenset[str],
) -> None:
    if not heldout:
        return
    for row in manifest_raw.get("episodes", []):
        if not isinstance(row, dict) or not row.get("keep"):
            continue
        obj = str(row.get("object_name", ""))
        if obj in heldout:
            report.record("heldout_object_in_dataset", obj)


def audit_drop_run(
    run_dir: Path,
    *,
    heldout_objects: frozenset[str] = frozenset(),
) -> AuditReport:
    """Run all audit checks; returns a report (``ok`` False when errors recorded)."""
    run_dir = Path(run_dir).resolve()
    report = AuditReport(run_dir=str(run_dir))
    manifest_raw = _load_manifest(run_dir)
    kept = [e for e in manifest_raw.get("episodes", []) if e.get("keep")]
    report.summary["kept_manifest_episodes"] = len(kept)

    has_lo = (run_dir / VIEW_DATASET / "meta" / "info.json").is_file()
    has_hi = (run_dir / VIEW_MASTER / "meta" / "info.json").is_file()
    if not has_lo:
        raise AuditError(f"missing view {VIEW_DATASET}")

    _assert_finalized_view(run_dir, VIEW_DATASET)
    lo = _load_view_parquet(run_dir, VIEW_DATASET)
    report.summary["dataset_episodes"] = int(lo.episode_index.nunique())

    by_ep: dict[int, dict[str, Any]] = {}
    if has_hi:
        _assert_finalized_view(run_dir, VIEW_MASTER)
        hi = _load_view_parquet(run_dir, VIEW_MASTER)
        report.summary["dataset_20hz_episodes"] = int(hi.episode_index.nunique())
        by_ep = _audit_cross_view(report, lo, hi)
        _audit_pairs(report, by_ep, manifest_raw)
    else:
        report.summary["cross_view_skipped"] = True
        _audit_tick_spacing(report, lo, kind="lo_tick_spacing")
        by_ep = _audit_drop_labels(report, lo)
        _audit_pairs(report, by_ep, manifest_raw)

    _audit_failure_segments(report, run_dir, VIEW_DATASET, lo)
    _audit_splits(report, run_dir, VIEW_DATASET)
    _audit_heldout_objects(report, manifest_raw, heldout_objects)

    if len(kept) != int(lo.episode_index.nunique()):
        report.record(
            "manifest_dataset_episode_count",
            f"kept {len(kept)} vs dataset {lo.episode_index.nunique()}",
        )
    return report
