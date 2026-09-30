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
import pytest

from fault_system.datagen.layout import ObjectPose2d, sample_layout
from fault_system.datagen.recipe import PlacementRecipe


def _placement(**kwargs) -> PlacementRecipe:
    values = {
        "xy_range_m": 0.12,
        "min_basket_clearance_m": 0.35,
        "distractor_basket_clearance_m": 0.15,
        "min_pairwise_clearance_m": 0.08,
        "yaw_range_deg": (-180.0, 180.0),
        "max_attempts": 200,
        "target_min_clearance_m": 0.08,
    }
    values.update(kwargs)
    if "target_min_clearance_m" not in kwargs:
        values["target_min_clearance_m"] = values["min_pairwise_clearance_m"]
    return PlacementRecipe(**values)


def test_sample_layout_keeps_away_from_basket():
    objects = [ObjectPose2d("can", np.array([0.4, 0.0]), 0.0)]

    result = sample_layout(
        objects=objects,
        basket_xy=np.array([0.0, 0.0]),
        table_xy_lim=0.7,
        rng=np.random.default_rng(0),
        placement=_placement(),
        target_name="can",
    )

    assert result is not None
    assert float(np.linalg.norm(result[0].xy)) >= 0.35


def test_target_uses_the_full_basket_keepout():
    # Nominal 0.30 m from the basket: only the far side of the jitter square
    # satisfies the 0.35 m target keep-out.
    objects = [ObjectPose2d("can", np.array([0.30, 0.0]), 0.0)]

    for seed in range(40):
        result = sample_layout(
            objects=objects,
            basket_xy=np.array([0.0, 0.0]),
            table_xy_lim=0.7,
            rng=np.random.default_rng(seed),
            placement=_placement(),
            target_name="can",
        )
        if result is not None:
            assert float(np.linalg.norm(result[0].xy)) >= 0.35


def test_distractors_use_the_smaller_basket_keepout():
    # Infeasible under the 0.35 m target keep-out, feasible under 0.15 m.
    objects = [ObjectPose2d("clutter", np.array([0.18, 0.0]), 0.0)]

    result = sample_layout(
        objects=objects,
        basket_xy=np.array([0.0, 0.0]),
        table_xy_lim=0.7,
        rng=np.random.default_rng(0),
        placement=_placement(xy_range_m=0.02),
        target_name="can",
    )

    assert result is not None
    assert float(np.linalg.norm(result[0].xy)) >= 0.15
    assert float(np.linalg.norm(result[0].xy)) < 0.35


def test_distractors_still_excluded_from_the_basket_footprint():
    objects = [ObjectPose2d("clutter", np.array([0.0, 0.0]), 0.0)]

    result = sample_layout(
        objects=objects,
        basket_xy=np.array([0.0, 0.0]),
        table_xy_lim=0.7,
        rng=np.random.default_rng(0),
        placement=_placement(xy_range_m=0.01, max_attempts=8),
        target_name="can",
    )

    assert result is None


def test_sample_layout_rejects_overlap():
    objects = [
        ObjectPose2d("a", np.array([0.5, 0.0]), 0.0),
        ObjectPose2d("b", np.array([0.5, 0.0]), 0.0),
    ]

    result = sample_layout(
        objects=objects,
        basket_xy=np.array([-1.0, -1.0]),
        table_xy_lim=0.7,
        rng=np.random.default_rng(1),
        placement=_placement(xy_range_m=0.0, max_attempts=5),
        target_name="a",
    )

    assert result is None


def test_sample_layout_failed_after_max_attempts():
    objects = [ObjectPose2d("can", np.array([0.0, 0.0]), 0.0)]

    result = sample_layout(
        objects=objects,
        basket_xy=np.array([0.0, 0.0]),
        table_xy_lim=0.7,
        rng=np.random.default_rng(0),
        placement=_placement(xy_range_m=0.01, max_attempts=8),
        target_name="can",
    )

    assert result is None


def test_sample_layout_is_reproducible():
    objects = [ObjectPose2d("can", np.array([0.4, 0.0]), 0.0)]

    first = sample_layout(
        objects=objects,
        basket_xy=np.array([0.0, 0.0]),
        table_xy_lim=0.7,
        rng=np.random.default_rng(17),
        placement=_placement(),
        target_name="can",
    )
    second = sample_layout(
        objects=objects,
        basket_xy=np.array([0.0, 0.0]),
        table_xy_lim=0.7,
        rng=np.random.default_rng(17),
        placement=_placement(),
        target_name="can",
    )

    assert first is not None and second is not None
    np.testing.assert_allclose(first[0].xy, second[0].xy)
    assert first[0].yaw_rad == second[0].yaw_rad


def test_stock_libero_object_scene_is_feasible():
    # Regression: the stock scene has distractors closer to the basket than the
    # target keep-out, which made the old all-objects keep-out near-infeasible.
    nominal = {
        "alphabet_soup_1": (-0.120, -0.240),
        "salad_dressing_1": (0.050, -0.100),
        "cream_cheese_1": (-0.151, 0.059),
        "milk_1": (0.099, -0.201),
        "tomato_sauce_1": (0.150, 0.030),
        "butter_1": (-0.204, -0.078),
    }
    objects = [ObjectPose2d(name, np.array(xy), 0.0) for name, xy in nominal.items()]
    basket_xy = np.array([-0.009, 0.249])

    failures = 0
    for seed in range(20):
        result = sample_layout(
            objects=objects,
            basket_xy=basket_xy,
            table_xy_lim=0.7,
            rng=np.random.default_rng(seed),
            placement=_placement(),
            target_name="alphabet_soup_1",
        )
        if result is None:
            failures += 1

    assert failures == 0


def test_target_min_clearance_rejects_distractor_at_ten_cm():
    objects = [
        ObjectPose2d("target", np.array([0.40, 0.0]), 0.0),
        ObjectPose2d("dist", np.array([0.50, 0.0]), 0.0),
    ]
    strict = _placement(target_min_clearance_m=0.14, xy_range_m=0.0)
    assert (
        sample_layout(
            objects=objects,
            basket_xy=np.array([-1.0, -1.0]),
            table_xy_lim=0.7,
            rng=np.random.default_rng(0),
            placement=strict,
            target_name="target",
        )
        is None
    )
    legacy = _placement(min_pairwise_clearance_m=0.08, target_min_clearance_m=0.08, xy_range_m=0.0)
    result = sample_layout(
        objects=objects,
        basket_xy=np.array([-1.0, -1.0]),
        table_xy_lim=0.7,
        rng=np.random.default_rng(0),
        placement=legacy,
        target_name="target",
    )
    assert result is not None
    assert float(np.linalg.norm(result[0].xy - result[1].xy)) == pytest.approx(0.10, abs=1e-6)


def test_default_target_min_clearance_matches_pairwise_for_seed():
    objects = [ObjectPose2d("can", np.array([0.4, 0.0]), 0.0)]
    explicit = _placement(min_pairwise_clearance_m=0.08, target_min_clearance_m=0.08)
    implicit = _placement(min_pairwise_clearance_m=0.08)
    for seed in (0, 17, 42):
        rng_a = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed)
        a = sample_layout(
            objects=objects,
            basket_xy=np.array([0.0, 0.0]),
            table_xy_lim=0.7,
            rng=rng_a,
            placement=explicit,
            target_name="can",
        )
        b = sample_layout(
            objects=objects,
            basket_xy=np.array([0.0, 0.0]),
            table_xy_lim=0.7,
            rng=rng_b,
            placement=implicit,
            target_name="can",
        )
        assert a is not None and b is not None
        np.testing.assert_allclose(a[0].xy, b[0].xy)
        assert a[0].yaw_rad == b[0].yaw_rad
