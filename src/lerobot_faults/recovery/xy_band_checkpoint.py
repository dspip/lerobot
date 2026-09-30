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

"""Keep/reject helpers for XY-band checkpoint dataset recording (no GPU/sim)."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

XY_BANDS: list[tuple[str, float, float]] = [
    ("lift", 0.48, 0.52),
    ("early", 0.42, 0.46),
    ("mid", 0.34, 0.38),
    ("late", 0.30, 0.33),
]

PILOT_DROP_SEEDS = tuple(range(5100, 5108))
BOUNDARY_DIAG_SEED_MIN = 6200
BOUNDARY_DIAG_SEED_MAX = 6305

NEAR_DUPLICATE_XY_M = 0.005
TABLE_LANDING_Z_MAX = 0.06
TARGET_DROPS_PER_BAND = 10
TARGET_NOMINAL = 20
TARGET_DROPS = TARGET_DROPS_PER_BAND * len(XY_BANDS)


def is_blocked_checkpoint_seed(seed: int) -> bool:
    """Seeds reserved for pilot copies or boundary diagnostics."""
    s = int(seed)
    if s in PILOT_DROP_SEEDS:
        return True
    return BOUNDARY_DIAG_SEED_MIN <= s <= BOUNDARY_DIAG_SEED_MAX


def filter_checkpoint_seeds(seeds: list[int]) -> list[int]:
    return [s for s in seeds if not is_blocked_checkpoint_seed(s)]


def band_name_for_range(band_lo: float, band_hi: float) -> str | None:
    for name, lo, hi in XY_BANDS:
        if math.isclose(lo, band_lo) and math.isclose(hi, band_hi):
            return name
    return None


def _xy_from_summary(summary: dict[str, Any], key: str) -> tuple[float, float] | None:
    val = summary.get(key)
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return float(val[0]), float(val[1])
    return None


def drop_xy_from_summary(summary: dict[str, Any]) -> tuple[float, float] | None:
    pre = summary.get("pre_drop_pose")
    if isinstance(pre, (list, tuple)) and len(pre) >= 2:
        return float(pre[0]), float(pre[1])
    return _xy_from_summary(summary, "initial_soup_xy")


def landing_xy_from_summary(summary: dict[str, Any]) -> tuple[float, float] | None:
    landing = summary.get("landing_pos")
    if isinstance(landing, (list, tuple)) and len(landing) >= 2:
        return float(landing[0]), float(landing[1])
    metrics = summary.get("drop_metrics") or {}
    later = metrics.get("later_pos")
    if isinstance(later, (list, tuple)) and len(later) >= 2:
        return float(later[0]), float(later[1])
    return None


def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def used_checkpoint_seeds(
    output_dir: Path,
    kept_registry: list[dict[str, Any]] | None = None,
    attempts: list[dict[str, Any]] | None = None,
) -> set[int]:
    """Seeds already kept, attempted, or with an episode directory (do not rerun)."""
    used: set[int] = set()
    for row in kept_registry or []:
        if row.get("seed") is not None:
            used.add(int(row["seed"]))
    for row in attempts or []:
        if row.get("seed") is not None:
            used.add(int(row["seed"]))
    ep_root = Path(output_dir) / "episodes"
    if ep_root.is_dir():
        for path in ep_root.iterdir():
            if path.name.startswith("seed_"):
                try:
                    used.add(int(path.name.split("_", 1)[1]))
                except ValueError:
                    continue
    return used


def is_near_duplicate_drop(
    summary: dict[str, Any],
    existing_kept: list[dict[str, Any]],
    *,
    tol_m: float = NEAR_DUPLICATE_XY_M,
) -> bool:
    """Reject only near-clone table landings.

    With a stock soup start, drop XY in a given band is similar by construction;
    landing XY still separates distinct recoveries.
    """
    landing_xy = landing_xy_from_summary(summary)
    if landing_xy is None:
        return False
    for kept in existing_kept:
        k_land = kept.get("landing_xy")
        if k_land is None:
            k_land = landing_xy_from_summary(kept)
        if k_land is None:
            continue
        if isinstance(k_land, (list, tuple)):
            k_land = (float(k_land[0]), float(k_land[1]))
        if _xy_distance(landing_xy, k_land) <= tol_m:
            return True
    return False


def landing_on_table_not_basket(summary: dict[str, Any]) -> bool:
    if summary.get("drop_landed_in_basket"):
        return False
    landing = summary.get("landing_pos")
    if isinstance(landing, (list, tuple)) and len(landing) >= 3:
        return float(landing[2]) <= TABLE_LANDING_Z_MAX
    metrics = summary.get("drop_metrics") or {}
    later = metrics.get("later_pos")
    if isinstance(later, (list, tuple)) and len(later) >= 3:
        return float(later[2]) <= TABLE_LANDING_Z_MAX
    checks = summary.get("checks") or {}
    return bool(checks.get("drop_moved_object"))


def should_keep_checkpoint_drop(
    summary: dict[str, Any],
    *,
    band_lo: float,
    band_hi: float,
    existing_kept: list[dict[str, Any]] | None = None,
) -> bool:
    """Stricter keep gate for checkpoint drop episodes (table landing, no in-basket drop)."""
    if not summary.get("success"):
        return False
    checks = summary.get("checks") or {}
    if not checks.get("was_midair_at_drop"):
        return False
    if not checks.get("drop_moved_object"):
        return False
    if summary.get("drop_landed_in_basket"):
        return False
    if not landing_on_table_not_basket(summary):
        return False
    if not checks.get("object_in_view_after_drop", True):
        return False
    if not summary.get("regrasped_after_drop"):
        return False
    if not checks.get("object_in_basket"):
        return False
    if summary.get("seat_assisted") or checks.get("seat_assisted"):
        return False
    if not checks.get("seat_assist_ok", True):
        return False
    dist = summary.get("drop_basket_xy_dist")
    if dist is None:
        return False
    if not (float(band_lo) <= float(dist) <= float(band_hi)):
        return False
    if summary.get("drop_xy_band_ok") is False:
        return False
    kept = existing_kept or []
    if is_near_duplicate_drop(summary, kept):
        return False
    return True


def should_keep_checkpoint_nominal(summary: dict[str, Any]) -> bool:
    if not summary.get("success"):
        return False
    checks = summary.get("checks") or {}
    if checks.get("fault_triggered"):
        return False
    if summary.get("seat_assisted") or checks.get("seat_assisted"):
        return False
    if not checks.get("object_in_basket"):
        return False
    return True


def rejection_reason(
    summary: dict[str, Any],
    *,
    kind: str,
    band_lo: float | None = None,
    band_hi: float | None = None,
    existing_kept: list[dict[str, Any]] | None = None,
) -> str:
    if kind == "nominal":
        if should_keep_checkpoint_nominal(summary):
            return "keep"
        if not summary.get("success"):
            return "not_success"
        checks = summary.get("checks") or {}
        if checks.get("fault_triggered"):
            return "fault_triggered"
        if summary.get("seat_assisted") or checks.get("seat_assisted"):
            return "seat_assisted"
        if not checks.get("object_in_basket"):
            return "not_in_basket"
        return "reject_nominal"

    if band_lo is None or band_hi is None:
        return "missing_band"
    if should_keep_checkpoint_drop(
        summary, band_lo=band_lo, band_hi=band_hi, existing_kept=existing_kept
    ):
        return "keep"
    if not summary.get("success"):
        return "not_success"
    checks = summary.get("checks") or {}
    if not checks.get("was_midair_at_drop"):
        return "not_midair_grasp"
    if not checks.get("drop_moved_object"):
        return "no_physical_drop"
    if summary.get("drop_landed_in_basket"):
        return "drop_landed_in_basket"
    if not landing_on_table_not_basket(summary):
        return "landing_not_on_table"
    if not checks.get("object_in_view_after_drop", True):
        return "not_visible_after_drop"
    if not summary.get("regrasped_after_drop"):
        return "no_regrasp"
    if not checks.get("object_in_basket"):
        return "final_not_in_basket"
    if summary.get("seat_assisted") or checks.get("seat_assisted"):
        return "seat_assisted"
    dist = summary.get("drop_basket_xy_dist")
    if dist is None:
        return "missing_drop_xy_dist"
    if not (float(band_lo) <= float(dist) <= float(band_hi)):
        return "drop_xy_out_of_band"
    if is_near_duplicate_drop(summary, existing_kept or []):
        return "near_duplicate"
    return "reject_drop"


def summary_episode_fields(summary: dict[str, Any]) -> dict[str, Any]:
    drop_xy = drop_xy_from_summary(summary)
    landing_xy = landing_xy_from_summary(summary)
    return {
        "initial_soup_xy": summary.get("initial_soup_xy"),
        "drop_xy": list(drop_xy) if drop_xy else None,
        "landing_xy": list(landing_xy) if landing_xy else None,
        "drop_basket_xy_dist": summary.get("drop_basket_xy_dist"),
        "drop_landed_in_basket": summary.get("drop_landed_in_basket"),
        "video_mp4": summary.get("video_mp4"),
    }


def build_kept_registry_from_pilot_log(pilot_log: dict[str, Any]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for ep in pilot_log.get("kept_episodes") or []:
        seed = int(ep["seed"])
        if is_blocked_checkpoint_seed(seed) and seed not in PILOT_DROP_SEEDS:
            continue
        entry = {
            "seed": seed,
            "kind": "drop",
            "band": ep.get("band"),
            "band_lo": ep.get("band_lo"),
            "band_hi": ep.get("band_hi"),
            "source": "pilot",
            **{k: ep.get(k) for k in ("drop_basket_xy_dist", "video_mp4", "triggered_at")},
        }
        kept.append(entry)
    return kept


def copy_pilot_into_checkpoint(
    *,
    pilot_dir: Path,
    output_dir: Path,
    pilot_log_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Copy pilot dataset + seed_5100..5107 episode artifacts; return registry rows."""
    pilot_dir = Path(pilot_dir)
    output_dir = Path(output_dir)
    log_path = pilot_log_path or (pilot_dir / "pilot_log.json")
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing pilot log at {log_path}")
    pilot_log = json.loads(log_path.read_text(encoding="utf-8"))
    pilot_dataset = Path(pilot_log.get("dataset_dir") or pilot_dir / "dataset")
    out_dataset = output_dir / "dataset"
    if not out_dataset.exists():
        if not pilot_dataset.is_dir():
            raise FileNotFoundError(f"Missing pilot dataset at {pilot_dataset}")
        shutil.copytree(pilot_dataset, out_dataset)

    episodes_out = output_dir / "episodes"
    episodes_out.mkdir(parents=True, exist_ok=True)
    pilot_episodes = pilot_dir / "episodes"
    for seed in PILOT_DROP_SEEDS:
        src = pilot_episodes / f"seed_{seed}"
        if not src.is_dir():
            raise FileNotFoundError(f"Missing pilot episode dir {src}")
        dst = episodes_out / f"seed_{seed}"
        if not dst.exists():
            shutil.copytree(src, dst)

    registry = build_kept_registry_from_pilot_log(pilot_log)
    for entry in registry:
        seed = int(entry["seed"])
        pipe = episodes_out / f"seed_{seed}" / "pipeline_log.json"
        if not pipe.is_file():
            continue
        try:
            slog = json.loads(pipe.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        extra = summary_episode_fields(slog)
        entry.update({k: v for k, v in extra.items() if v is not None})
        if entry.get("drop_xy") is None:
            pre = slog.get("pre_drop_pose")
            if isinstance(pre, (list, tuple)) and len(pre) >= 2:
                entry["drop_xy"] = [float(pre[0]), float(pre[1])]
        if entry.get("landing_xy") is None:
            post = slog.get("post_drop_pose") or slog.get("landing_pos")
            if isinstance(post, (list, tuple)) and len(post) >= 2:
                entry["landing_xy"] = [float(post[0]), float(post[1])]
    registry_path = output_dir / "kept_registry.json"
    registry_path.write_text(json.dumps({"kept": registry}, indent=2), encoding="utf-8")
    return registry
