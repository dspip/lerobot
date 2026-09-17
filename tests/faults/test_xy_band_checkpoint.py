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
from pathlib import Path

from lerobot.faults.recovery.xy_band_checkpoint import (
    BOUNDARY_DIAG_SEED_MAX,
    BOUNDARY_DIAG_SEED_MIN,
    PILOT_DROP_SEEDS,
    build_kept_registry_from_pilot_log,
    copy_pilot_into_checkpoint,
    filter_checkpoint_seeds,
    is_blocked_checkpoint_seed,
    is_near_duplicate_drop,
    landing_on_table_not_basket,
    rejection_reason,
    should_keep_checkpoint_drop,
    should_keep_checkpoint_nominal,
)


def _drop_summary(**overrides) -> dict:
    base = {
        "success": True,
        "regrasped_after_drop": True,
        "drop_landed_in_basket": False,
        "drop_basket_xy_dist": 0.35,
        "drop_xy_band_ok": True,
        "pre_drop_pose": [0.1, 0.2, 0.25],
        "landing_pos": [0.12, 0.18, 0.038],
        "seat_assisted": False,
        "checks": {
            "was_midair_at_drop": True,
            "drop_moved_object": True,
            "object_in_view_after_drop": True,
            "object_in_basket": True,
            "seat_assisted": False,
            "seat_assist_ok": True,
        },
    }
    base.update(overrides)
    return base


def test_blocked_seeds() -> None:
    for s in PILOT_DROP_SEEDS:
        assert is_blocked_checkpoint_seed(s)
    assert is_blocked_checkpoint_seed(BOUNDARY_DIAG_SEED_MIN)
    assert is_blocked_checkpoint_seed(BOUNDARY_DIAG_SEED_MAX)
    assert not is_blocked_checkpoint_seed(7200)
    filtered = filter_checkpoint_seeds([5100, 6200, 7200, 8200])
    assert filtered == [7200, 8200]


def test_keep_table_landing_drop() -> None:
    summary = _drop_summary()
    assert should_keep_checkpoint_drop(summary, band_lo=0.34, band_hi=0.38)
    assert rejection_reason(summary, kind="drop", band_lo=0.34, band_hi=0.38) == "keep"


def test_reject_in_basket_landing() -> None:
    summary = _drop_summary(drop_landed_in_basket=True)
    assert not should_keep_checkpoint_drop(summary, band_lo=0.34, band_hi=0.38)
    assert rejection_reason(summary, kind="drop", band_lo=0.34, band_hi=0.38) == "drop_landed_in_basket"


def test_reject_near_duplicate() -> None:
    summary = _drop_summary()
    existing = [
        {
            "drop_xy": [0.1, 0.2],
            "landing_xy": [0.12, 0.18],
        }
    ]
    assert is_near_duplicate_drop(summary, existing)
    assert not should_keep_checkpoint_drop(
        summary, band_lo=0.34, band_hi=0.38, existing_kept=existing
    )
    assert (
        rejection_reason(
            summary,
            kind="drop",
            band_lo=0.34,
            band_hi=0.38,
            existing_kept=existing,
        )
        == "near_duplicate"
    )


def test_nominal_keep_reject() -> None:
    ok = {
        "success": True,
        "seat_assisted": False,
        "checks": {"fault_triggered": False, "object_in_basket": True, "seat_assisted": False},
    }
    assert should_keep_checkpoint_nominal(ok)
    bad = {
        "success": True,
        "checks": {"fault_triggered": True, "object_in_basket": True},
    }
    assert not should_keep_checkpoint_nominal(bad)


def test_landing_on_table_helper() -> None:
    assert landing_on_table_not_basket(_drop_summary())
    high = _drop_summary(landing_pos=[0.0, 0.0, 0.2], drop_landed_in_basket=False)
    assert not landing_on_table_not_basket(high)


def test_build_registry_from_pilot_log() -> None:
    pilot = {
        "kept_episodes": [
            {"seed": 5100, "band": "lift", "band_lo": 0.48, "band_hi": 0.52},
            {"seed": 5101, "band": "early", "band_lo": 0.42, "band_hi": 0.46},
        ]
    }
    kept = build_kept_registry_from_pilot_log(pilot)
    assert len(kept) == 2
    assert kept[0]["source"] == "pilot"


def test_copy_pilot_into_checkpoint(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot"
    out = tmp_path / "checkpoint"
    (pilot / "dataset" / "meta").mkdir(parents=True)
    (pilot / "dataset" / "meta" / "info.json").write_text('{"fps": 10}', encoding="utf-8")
    (pilot / "dataset" / "data").mkdir()
    (pilot / "dataset" / "data" / "chunk.parquet").write_bytes(b"")
    log = {
        "dataset_dir": str(pilot / "dataset"),
        "kept_episodes": [{"seed": 5100, "band": "lift", "band_lo": 0.48, "band_hi": 0.52}],
    }
    (pilot / "pilot_log.json").write_text(json.dumps(log), encoding="utf-8")
    for seed in PILOT_DROP_SEEDS:
        ep = pilot / "episodes" / f"seed_{seed}"
        ep.mkdir(parents=True)
        (ep / "pipeline_log.json").write_text("{}", encoding="utf-8")

    registry = copy_pilot_into_checkpoint(pilot_dir=pilot, output_dir=out)
    assert (out / "dataset" / "meta" / "info.json").is_file()
    assert (out / "episodes" / "seed_5100" / "pipeline_log.json").is_file()
    assert (out / "kept_registry.json").is_file()
    assert len(registry) == 1
