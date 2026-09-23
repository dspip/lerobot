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

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from lerobot.faults.datagen.motion_profile import sample_episode_motion_profile
from lerobot.faults.datagen.recipe import SimpleIKRecipe
from lerobot.faults.datagen.runtime import (
    movable_object_names,
    rotate_quat_about_world_z,
    stabilize_carry_action,
)
from lerobot.faults.recovery.planner import CARRY_PHASES


def test_motion_profile_is_independent_of_layout_rng_draws():
    recipe = SimpleIKRecipe(
        trajectory_randomization_enabled=True,
        pickup_via_offset_m=0.03,
        transport_via_offset_m=0.06,
        arm_posture_noise_deg=3.0,
        speed_multiplier_range=(0.8, 1.2),
        waypoint_blend_radius_m=0.04,
    )
    layout_rng = np.random.default_rng(np.random.SeedSequence([12, 0x4C41594F]))
    first = sample_episode_motion_profile(recipe, 12)
    layout_rng.random(10_000)
    second = sample_episode_motion_profile(recipe, 12)

    assert first == second


def test_stabilize_carry_action_slows_translation_and_closes_gripper():
    action = np.array([1.0, -0.5, 0.25, 0.2, -0.3, 0.4, -1.0])

    stabilized = stabilize_carry_action(action)

    np.testing.assert_allclose(stabilized[:3], [0.65, -0.325, 0.1625])
    np.testing.assert_allclose(stabilized[3:6], action[3:6])
    assert stabilized[6] == 1.0


def test_carry_phases_include_randomized_transport_via():
    assert CARRY_PHASES == frozenset(
        {"lift", "to_basket_via", "to_basket_hover"}
    )


def test_rotate_quat_about_world_z_from_identity():
    result = rotate_quat_about_world_z(np.array([1.0, 0.0, 0.0, 0.0]), np.pi / 2)

    np.testing.assert_allclose(
        result,
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        atol=1e-7,
    )


def test_movable_object_names_filters_basket_and_jointless_objects():
    can = SimpleNamespace(joints=["can_joint"])
    distractor = SimpleNamespace(joints=["other_joint"])
    basket = SimpleNamespace(joints=[])
    fixture = SimpleNamespace(joints=[])
    objects = {
        "alphabet_soup_1": can,
        "other_1": distractor,
        "basket_1": basket,
        "table": fixture,
    }
    env = MagicMock()
    env.objects = objects
    env.get_object.side_effect = objects.__getitem__

    names = movable_object_names(
        env,
        basket_name="basket_1",
        required_object_name="alphabet_soup_1",
    )

    assert names == ["alphabet_soup_1", "other_1"]


def test_movable_object_names_prefers_libero_objects_dict_keys():
    can = SimpleNamespace(name="alphabet_soup", joints=["can_joint"])
    other = SimpleNamespace(name="other", joints=["other_joint"])
    objects = {"alphabet_soup_1": can, "other_1": other}
    env = MagicMock()
    env.objects_dict = objects
    env.objects = list(objects.values())
    env.get_object.side_effect = objects.get

    names = movable_object_names(
        env,
        basket_name="basket_1",
        required_object_name="alphabet_soup_1",
    )

    assert names == ["alphabet_soup_1", "other_1"]
