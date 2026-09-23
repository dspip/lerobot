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

"""Shared, immutable geometry for shaped SimpleIK trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_RESOLUTION_STEPS = 32


@dataclass(frozen=True)
class PathSegment:
    name: str
    start_xyz: tuple[float, float, float]
    end_xyz: tuple[float, float, float]

    @property
    def length(self) -> float:
        return float(
            np.linalg.norm(np.asarray(self.end_xyz) - np.asarray(self.start_xyz))
        )


@dataclass(frozen=True)
class CarryPath:
    segments: tuple[PathSegment, ...]
    requested_transport_offset_m: float
    resolved_transport_offset_m: float
    fallback: bool


@dataclass(frozen=True)
class ResolvedPickupVia:
    position_xyz: tuple[float, float, float]
    requested_offset_xy_m: tuple[float, float]
    resolved_offset_xy_m: tuple[float, float]
    fallback: bool


def _legal_xy(
    xy: np.ndarray,
    *,
    workspace_xy_limit_m: float,
    basket_xy: np.ndarray,
    basket_keepout_m: float,
) -> bool:
    return bool(
        np.all(np.abs(xy) <= float(workspace_xy_limit_m))
        and np.linalg.norm(xy - basket_xy) >= float(basket_keepout_m)
    )


def _scales() -> np.ndarray:
    return np.linspace(1.0, 0.0, _RESOLUTION_STEPS)


def build_pickup_via(
    object_hover_xyz: np.ndarray,
    requested_offset_xy_m: tuple[float, float],
    *,
    workspace_xy_limit_m: float,
    basket_xy: np.ndarray,
    basket_keepout_m: float,
) -> ResolvedPickupVia:
    """Resolve the largest legal pickup bend along the requested direction."""
    hover = np.asarray(object_hover_xyz, dtype=np.float64).reshape(3)
    requested = np.asarray(requested_offset_xy_m, dtype=np.float64).reshape(2)
    basket = np.asarray(basket_xy, dtype=np.float64).reshape(2)
    for scale in _scales():
        resolved = requested * float(scale)
        candidate = hover.copy()
        candidate[:2] += resolved
        if _legal_xy(
            candidate[:2],
            workspace_xy_limit_m=workspace_xy_limit_m,
            basket_xy=basket,
            basket_keepout_m=basket_keepout_m,
        ):
            return ResolvedPickupVia(
                position_xyz=tuple(float(v) for v in candidate),
                requested_offset_xy_m=tuple(float(v) for v in requested),
                resolved_offset_xy_m=tuple(float(v) for v in resolved),
                fallback=bool(scale == 0.0 and np.linalg.norm(requested) > 0.0),
            )
    return ResolvedPickupVia(
        position_xyz=tuple(float(v) for v in hover),
        requested_offset_xy_m=tuple(float(v) for v in requested),
        resolved_offset_xy_m=(0.0, 0.0),
        fallback=True,
    )


def build_carry_path(
    lift_start_xyz: np.ndarray,
    lift_end_xyz: np.ndarray,
    basket_hover_xyz: np.ndarray,
    requested_transport_offset_m: float,
    *,
    workspace_xy_limit_m: float,
    basket_xy: np.ndarray,
    basket_keepout_m: float,
) -> CarryPath:
    """Build lift plus a legal bent (or straight) object-centric carry path."""
    start = np.asarray(lift_start_xyz, dtype=np.float64).reshape(3)
    lifted = np.asarray(lift_end_xyz, dtype=np.float64).reshape(3)
    basket_hover = np.asarray(basket_hover_xyz, dtype=np.float64).reshape(3)
    basket = np.asarray(basket_xy, dtype=np.float64).reshape(2)
    requested = float(requested_transport_offset_m)
    direct = basket_hover[:2] - lifted[:2]
    norm = float(np.linalg.norm(direct))
    perpendicular = (
        np.array([-direct[1], direct[0]], dtype=np.float64) / norm
        if norm > 1e-12
        else np.array([0.0, 0.0], dtype=np.float64)
    )
    if norm > 1e-12:
        keepout_boundary = basket_hover.copy()
        keepout_boundary[:2] = (
            basket_hover[:2] - direct / norm * float(basket_keepout_m)
        )
        midpoint = 0.5 * (lifted + keepout_boundary)
    else:
        midpoint = 0.5 * (lifted + basket_hover)

    resolved = 0.0
    via: np.ndarray | None = None
    for scale in _scales():
        candidate_offset = requested * float(scale)
        candidate = midpoint.copy()
        candidate[:2] += perpendicular * candidate_offset
        if _legal_xy(
            candidate[:2],
            workspace_xy_limit_m=workspace_xy_limit_m,
            basket_xy=basket,
            basket_keepout_m=basket_keepout_m,
        ):
            resolved = candidate_offset
            via = candidate
            break

    lift = PathSegment(
        name="lift",
        start_xyz=tuple(float(v) for v in start),
        end_xyz=tuple(float(v) for v in lifted),
    )
    fallback = via is None or (resolved == 0.0 and requested != 0.0)
    if via is None or abs(resolved) <= 1e-12:
        direct_segment = PathSegment(
            name="to_basket_hover",
            start_xyz=tuple(float(v) for v in lifted),
            end_xyz=tuple(float(v) for v in basket_hover),
        )
        return CarryPath(
            segments=(lift, direct_segment),
            requested_transport_offset_m=requested,
            resolved_transport_offset_m=0.0,
            fallback=fallback,
        )

    to_via = PathSegment(
        name="to_basket_via",
        start_xyz=tuple(float(v) for v in lifted),
        end_xyz=tuple(float(v) for v in via),
    )
    to_basket = PathSegment(
        name="to_basket_hover",
        start_xyz=tuple(float(v) for v in via),
        end_xyz=tuple(float(v) for v in basket_hover),
    )
    return CarryPath(
        segments=(lift, to_via, to_basket),
        requested_transport_offset_m=requested,
        resolved_transport_offset_m=resolved,
        fallback=False,
    )
