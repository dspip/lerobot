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

import numpy as np

from lerobot.faults.recovery.trajectory import build_carry_path, build_pickup_via


def _carry(offset: float):
    return build_carry_path(
        np.array([0.5, 0.0, 0.03]),
        np.array([0.5, 0.0, 0.25]),
        np.array([0.0, 0.0, 0.25]),
        offset,
        workspace_xy_limit_m=0.7,
        basket_xy=np.array([0.0, 0.0]),
        basket_keepout_m=0.20,
    )


def test_zero_pickup_offset_preserves_direct_approach():
    via = build_pickup_via(
        np.array([0.5, 0.0, 0.17]),
        (0.0, 0.0),
        workspace_xy_limit_m=0.7,
        basket_xy=np.array([0.0, 0.0]),
        basket_keepout_m=0.20,
    )

    assert via.position_xyz == (0.5, 0.0, 0.17)


def test_pickup_via_uses_safe_hover_height():
    via = build_pickup_via(
        np.array([0.5, 0.0, 0.17]),
        (0.02, -0.01),
        workspace_xy_limit_m=0.7,
        basket_xy=np.array([0.0, 0.0]),
        basket_keepout_m=0.20,
    )

    assert via.position_xyz[2] == 0.17


def test_transport_via_is_perpendicular_to_direct_route():
    path = _carry(0.06)
    via = np.asarray(path.segments[1].end_xyz)

    np.testing.assert_allclose(via[:2], [0.35, -0.06], atol=1e-12)


def test_positive_and_negative_offsets_bend_opposite_directions():
    positive = np.asarray(_carry(0.06).segments[1].end_xyz)
    negative = np.asarray(_carry(-0.06).segments[1].end_xyz)

    assert positive[1] == -negative[1]
    assert positive[1] != 0.0


def test_carry_path_preserves_lift_and_basket_endpoints():
    path = _carry(0.06)

    assert path.segments[0].start_xyz == (0.5, 0.0, 0.03)
    assert path.segments[0].end_xyz == (0.5, 0.0, 0.25)
    assert path.segments[-1].end_xyz == (0.0, 0.0, 0.25)


def test_via_resolution_rejects_workspace_violation_instead_of_clamping():
    via = build_pickup_via(
        np.array([0.69, 0.0, 0.17]),
        (0.03, 0.0),
        workspace_xy_limit_m=0.7,
        basket_xy=np.array([0.0, 0.0]),
        basket_keepout_m=0.20,
    )

    assert via.position_xyz[0] < 0.7
    assert via.position_xyz[0] != 0.7


def test_via_resolution_shrinks_magnitude_without_flipping_direction():
    via = build_pickup_via(
        np.array([0.69, 0.0, 0.17]),
        (0.03, 0.0),
        workspace_xy_limit_m=0.7,
        basket_xy=np.array([0.0, 0.0]),
        basket_keepout_m=0.20,
    )

    assert 0.0 <= via.resolved_offset_xy_m[0] < 0.03


def test_transport_falls_back_to_straight_when_via_cannot_clear_keepout():
    path = build_carry_path(
        np.array([0.10, 0.0, 0.03]),
        np.array([0.10, 0.0, 0.25]),
        np.array([0.0, 0.0, 0.25]),
        0.01,
        workspace_xy_limit_m=0.7,
        basket_xy=np.array([0.0, 0.0]),
        basket_keepout_m=0.20,
    )

    assert len(path.segments) == 2
    assert path.fallback
