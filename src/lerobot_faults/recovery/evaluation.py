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

"""Behavioral success gates for mid-air-drop recovery rollouts."""

from __future__ import annotations

from typing import Any

MIN_DROP_BASKET_XY_DIST_M = 0.12


def evaluate_recovery_episode(
    *,
    grasped_before_drop: bool,
    was_midair_at_drop: bool,
    triggered_at: int | None,
    drop_moved: bool,
    object_visible_after_drop: bool,
    recovery_steps: int,
    drop_basket_xy_dist: float | None,
    drop_trigger_reason: str | None,
    landed_in_basket: bool,
    regrasp_step: int | None,
    seat_assisted: bool,
    forbid_seat_assist: bool,
    basket_place_ok: bool,
    min_drop_basket_xy_dist: float = MIN_DROP_BASKET_XY_DIST_M,
    min_recovery_steps: int = 10,
) -> tuple[bool, dict[str, Any]]:
    """Decide whether a mid-air-drop rollout is a usable recovery demonstration."""
    dropped_away_from_basket = (
        drop_basket_xy_dist is not None
        and float(drop_basket_xy_dist) >= float(min_drop_basket_xy_dist)
    )
    regrasped_after_drop = regrasp_step is not None
    seat_assist_ok = (not seat_assisted) if forbid_seat_assist else True

    checks: dict[str, Any] = {
        "grasped_before_drop": bool(grasped_before_drop),
        "was_midair_at_drop": bool(was_midair_at_drop),
        "fault_triggered": triggered_at is not None,
        "drop_moved_object": bool(drop_moved),
        "object_in_view_after_drop": bool(object_visible_after_drop),
        "recovery_steps": int(recovery_steps),
        "dropped_away_from_basket": bool(dropped_away_from_basket),
        "drop_basket_xy_dist": (
            None if drop_basket_xy_dist is None else float(drop_basket_xy_dist)
        ),
        "drop_trigger_reason": drop_trigger_reason,
        "landed_in_basket_no_recovery_needed": bool(landed_in_basket),
        "regrasped_after_drop": regrasped_after_drop,
        "regrasp_step": regrasp_step,
        "object_in_basket": bool(basket_place_ok),
        "seat_assisted": bool(seat_assisted),
        "seat_assist_ok": bool(seat_assist_ok),
    }
    success = bool(
        grasped_before_drop
        and was_midair_at_drop
        and triggered_at is not None
        and drop_moved
        and object_visible_after_drop
        and recovery_steps > int(min_recovery_steps)
        and dropped_away_from_basket
        and not landed_in_basket
        and regrasped_after_drop
        and seat_assist_ok
        and basket_place_ok
    )
    return success, checks
