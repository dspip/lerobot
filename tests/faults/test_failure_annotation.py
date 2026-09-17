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

"""Unit tests for physics-based failure annotation (no MuJoCo required)."""

from __future__ import annotations

import numpy as np

from lerobot.faults.annotation import (
    FAILURE_TYPE_MIDAIR_DROP,
    PHASE_INJECTION,
    PHASE_NOMINAL,
    PHASE_POST_FAULT,
    PHASE_RECOVERY,
    FailureAnnotator,
    PhysicsSnapshot,
    annotation_from_scripted_phase,
    clip_failure_to_first_interval,
    default_failure_frame,
    failure_frame_from_info,
    frames_to_info_arrays,
)
from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.wrappers import FaultEnvWrapper
from tests.faults.test_wrappers import _DummyEnv


def _snap(*, grasped: bool, held: bool, in_basket: bool = False) -> PhysicsSnapshot:
    return PhysicsSnapshot(
        grasped=grasped,
        held_midair=held,
        in_basket=in_basket,
        object_z=0.2 if held else 0.05,
        available=True,
    )


def test_successful_episode_all_flags_false():
    ann = FailureAnnotator(num_envs=1, injector_type_id=FAILURE_TYPE_MIDAIR_DROP)
    physics = [_snap(grasped=True, held=True)]
    frames = ann.update(None, injection_active=[False], physics=physics)
    f = frames[0]
    assert bool(f["is_failure"][0]) is False
    assert bool(f["failure_onset"][0]) is False
    assert bool(f["injection_active"][0]) is False
    assert int(f["failure_type"][0]) == 0
    assert int(f["phase"][0]) == PHASE_NOMINAL


def test_injection_without_lost_grasp_is_not_physical_failure():
    ann = FailureAnnotator(num_envs=1, injector_type_id=FAILURE_TYPE_MIDAIR_DROP)
    ann.update(None, injection_active=[False], physics=[_snap(grasped=True, held=True)])
    frames = ann.update(
        None,
        injection_active=[True],
        physics=[_snap(grasped=True, held=True)],
    )
    f = frames[0]
    assert bool(f["injection_active"][0]) is True
    assert bool(f["is_failure"][0]) is False
    assert bool(f["failure_onset"][0]) is False
    assert int(f["failure_type"][0]) == FAILURE_TYPE_MIDAIR_DROP
    assert int(f["phase"][0]) == PHASE_INJECTION


def test_physical_drop_onset_then_sustain_then_regrasp():
    ann = FailureAnnotator(num_envs=1, injector_type_id=FAILURE_TYPE_MIDAIR_DROP)
    ann.update(None, injection_active=[False], physics=[_snap(grasped=True, held=True)])
    drop = ann.update(
        None,
        injection_active=[True],
        physics=[_snap(grasped=False, held=False)],
    )[0]
    assert bool(drop["is_failure"][0]) is True
    assert bool(drop["failure_onset"][0]) is True
    assert bool(drop["injection_active"][0]) is True
    assert int(drop["phase"][0]) == PHASE_INJECTION

    post = ann.update(
        None,
        injection_active=[False],
        physics=[_snap(grasped=False, held=False)],
    )[0]
    assert bool(post["is_failure"][0]) is True
    assert bool(post["failure_onset"][0]) is False
    assert int(post["phase"][0]) == PHASE_POST_FAULT

    rec = ann.update(
        None,
        injection_active=[False],
        recovery_active=[True],
        physics=[_snap(grasped=True, held=True)],
    )[0]
    assert bool(rec["is_failure"][0]) is False
    assert int(rec["phase"][0]) == PHASE_RECOVERY
    assert int(rec["failure_type"][0]) == FAILURE_TYPE_MIDAIR_DROP


def test_recovery_release_is_not_a_second_onset():
    """Gripper-open over the basket after regrasp must not raise failure_onset again."""
    ann = FailureAnnotator(num_envs=1, injector_type_id=FAILURE_TYPE_MIDAIR_DROP)
    ann.update(None, injection_active=[False], physics=[_snap(grasped=True, held=True)])
    drop = ann.update(
        None,
        injection_active=[True],
        physics=[_snap(grasped=False, held=False)],
    )[0]
    assert bool(drop["failure_onset"][0]) is True
    rec = ann.update(
        None,
        injection_active=[False],
        recovery_active=[True],
        physics=[_snap(grasped=True, held=True)],
    )[0]
    assert bool(rec["is_failure"][0]) is False
    release = ann.update(
        None,
        injection_active=[False],
        recovery_active=[True],
        physics=[_snap(grasped=False, held=False, in_basket=False)],
    )[0]
    assert bool(release["is_failure"][0]) is False
    assert bool(release["failure_onset"][0]) is False


def test_clip_failure_to_first_interval_drops_one_frame_reentry():
    fail, onset = clip_failure_to_first_interval(
        [False, True, True, False, False, True, False]
    )
    assert fail.tolist() == [False, True, True, False, False, False, False]
    assert onset.tolist() == [False, True, False, False, False, False, False]


def test_in_basket_clears_physical_failure():
    ann = FailureAnnotator(num_envs=1)
    ann.update(None, injection_active=[False], physics=[_snap(grasped=True, held=True)])
    ann.update(None, injection_active=[True], physics=[_snap(grasped=False, held=False)])
    placed = ann.update(
        None,
        injection_active=[False],
        physics=[_snap(grasped=False, held=False, in_basket=True)],
    )[0]
    assert bool(placed["is_failure"][0]) is False


def test_never_held_cannot_be_a_drop_failure():
    ann = FailureAnnotator(num_envs=1)
    frames = ann.update(
        None,
        injection_active=[False],
        physics=[_snap(grasped=False, held=False)],
    )
    assert bool(frames[0]["is_failure"][0]) is False


def test_info_roundtrip_and_defaults():
    ann = FailureAnnotator(num_envs=2, injector_type_id=FAILURE_TYPE_MIDAIR_DROP)
    frames = ann.update(
        None,
        injection_active=[False, True],
        physics=[_snap(grasped=True, held=True), _snap(grasped=True, held=True)],
    )
    arrays = frames_to_info_arrays(frames)
    assert arrays["injection_active"].tolist() == [False, True]
    extracted = failure_frame_from_info(arrays, env_idx=1)
    assert bool(extracted["injection_active"][0]) is True
    zeros = failure_frame_from_info({}, env_idx=0)
    np.testing.assert_array_equal(zeros["failure_type"], default_failure_frame()["failure_type"])


def test_injection_phase_outranks_recovery_on_the_glitch_frame():
    ann = FailureAnnotator(num_envs=1, injector_type_id=FAILURE_TYPE_MIDAIR_DROP)
    ann.update(None, injection_active=[False], physics=[_snap(grasped=True, held=True)])
    f = ann.update(
        None,
        injection_active=[True],
        recovery_active=[True],
        physics=[_snap(grasped=False, held=False)],
    )[0]
    assert int(f["phase"][0]) == PHASE_INJECTION
    assert bool(f["injection_active"][0]) is True
    assert bool(f["is_failure"][0]) is True


def test_scripted_phase_helper():
    drop = annotation_from_scripted_phase("drop", onset=True)
    assert bool(drop["failure_onset"][0]) is True
    rec = annotation_from_scripted_phase("recovery")
    assert int(rec["phase"][0]) == PHASE_RECOVERY


def test_action_hold_wrapper_sets_injection_active_not_is_failure():
    cfg = FaultInjectionConfig(
        enabled=True,
        type="action_hold",
        trigger_step=1,
        duration=2,
        probability=1.0,
        seed=0,
        log_path=None,
    )
    env = FaultEnvWrapper(_DummyEnv(), cfg)
    env.reset()
    _, _, _, _, info0 = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert bool(np.asarray(info0["injection_active"]).reshape(-1)[0]) is False
    _, _, _, _, info1 = env.step(np.array([2.0, 2.0], dtype=np.float32))
    assert bool(np.asarray(info1["injection_active"]).reshape(-1)[0]) is True
    assert bool(np.asarray(info1["is_failure"]).reshape(-1)[0]) is False
