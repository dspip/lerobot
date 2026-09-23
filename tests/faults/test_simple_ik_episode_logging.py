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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.datagen.controllers.simple_ik import run_simple_ik_episode_loop
from lerobot.faults.datagen.dataset_writer import DatagenEpisodeSession
from lerobot.faults.datagen.drop_timing import DropDecision
from lerobot.faults.datagen.frame_logging import (
    DATAGEN_POST_STEP_ANNOTATION_KEY,
    DatagenPostStepLogContext,
    should_log_sim_step,
)
from lerobot.faults.datagen.recipe import load_drop_datagen_recipe, paired_episode_seed_manifests
from lerobot.faults.recovery.trajectory import CarryPath, PathSegment
from lerobot.faults.wrappers import DropRecoveryEnvWrapper
from tests.faults.test_datagen_dataset_writer import _minimal_processed_frame, _RecordingLogger
from tests.faults.test_midair_drop_fault import _setup_drop_mocks

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"

RECORDING_STRIDE = 2
CONFIGURED_DWELL_STEPS = 2
DROP_SIM_STEP = 1


class _MinimalInnerEnv:
    num_envs = 1

    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        del seed, options
        obs = {"observation.state": np.zeros(8, dtype=np.float32)}
        return obs, {}

    def step(self, action):
        del action
        obs = {"observation.state": np.zeros(8, dtype=np.float32)}
        return obs, 0.0, False, False, {}

    def close(self) -> None:
        return None


def _wrapped_continue_env(*, dwell: int = CONFIGURED_DWELL_STEPS) -> DropRecoveryEnvWrapper:
    cfg = FaultInjectionConfig(
        enabled=True,
        type="midair_drop",
        probability=0.0,
        post_drop_dwell_steps=dwell,
        post_drop_mode="continue_then_ik",
        object_name="alphabet_soup_1",
        require_grasp=False,
        t_min=0,
        t_max=0,
    )
    inner = _MinimalInnerEnv()
    wrapped = DropRecoveryEnvWrapper(inner, cfg)
    wrapped.reset()
    return wrapped


def _ctx_from_frame(frame: dict) -> DatagenPostStepLogContext:
    raw = frame["annotation"][DATAGEN_POST_STEP_ANNOTATION_KEY]
    return DatagenPostStepLogContext(**raw)


def test_simple_ik_loop_logs_dwell_and_recovery_masks_with_stride() -> None:
    held_midair_checks = {"n": 0}

    def _held_midair(*_args, **_kwargs) -> bool:
        held_midair_checks["n"] += 1
        return held_midair_checks["n"] > 1

    with (
        patch("lerobot.faults.recovery.midair_drop.get_place_destination") as mock_dest,
        patch("lerobot.faults.recovery.midair_drop.midair_drop") as mock_drop,
        patch("lerobot.faults.recovery.midair_drop.get_object_pose") as mock_obj_pose,
        patch("lerobot.faults.recovery.midair_drop.get_eef_pose") as mock_eef,
        patch("lerobot.faults.recovery.midair_drop.get_arm_qpos") as mock_arm_q,
        patch("lerobot.faults.recovery.midair_drop.get_robosuite_env") as mock_get_rs,
        patch("lerobot.faults.recovery.midair_drop.is_object_grasped") as mock_grasped,
        patch(
            "lerobot.faults.datagen.controllers.simple_ik._nominal_action",
            return_value=np.ones(7),
        ),
        patch("lerobot.faults.datagen.controllers.simple_ik.is_object_in_basket", return_value=False),
        patch("lerobot.faults.datagen.controllers.simple_ik.is_object_grasped", return_value=False),
        patch(
            "lerobot.faults.datagen.controllers.simple_ik.is_object_held_midair",
            side_effect=_held_midair,
        ),
        patch("lerobot.faults.datagen.controllers.simple_ik.get_place_destination") as mock_dest_simple,
        patch("lerobot.faults.datagen.controllers.simple_ik.get_object_pose") as mock_obj_pose_simple,
        patch("lerobot.envs.utils.preprocess_observation", side_effect=lambda obs: obs),
        patch(
            "lerobot.faults.recovery.dataset_logger.libero_obs_to_frame",
            side_effect=lambda obs: _minimal_processed_frame(),
        ),
    ):
        _setup_drop_mocks(
            mock_grasped, mock_get_rs, mock_arm_q, mock_eef, mock_obj_pose, mock_drop, mock_dest
        )
        mock_obj_pose_simple.return_value = {"pos": np.array([0.55, 0.0, 0.2])}
        mock_dest_simple.return_value = np.array([0.0, 0.0, 0.0])

        env = _wrapped_continue_env(dwell=CONFIGURED_DWELL_STEPS)
        fault = env.fault
        scheduled = MagicMock(wraps=fault.trigger_scheduled_drop)
        fault.trigger_scheduled_drop = scheduled

        rs_env = MagicMock()
        recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
        manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[0]
        logger = _RecordingLogger(Path("/tmp/simple_ik_logging_test"))
        session = DatagenEpisodeSession(
            manifest=manifest,
            dataset_root=Path("/tmp"),
            repo_id="test/simple_ik",
            logger=logger,
            policy_fps=10,
        )

        paired_plan = SimpleNamespace(
            drop_decision=DropDecision(True, None, "injected"),
            drop_u=0.05,
        )
        carry_path = CarryPath(
            segments=(PathSegment("lift", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
            requested_transport_offset_m=0.0,
            resolved_transport_offset_m=0.0,
            fallback=False,
        )
        planner = MagicMock(phase_name="lift", carry_path=carry_path, done=False)

        run_simple_ik_episode_loop(
            env,
            rs_env,
            fault=fault,
            planner=planner,
            recipe_drop=MagicMock(
                min_drop_distance_from_basket_m=0.3,
                hard_keepout_floor_m=0.22,
            ),
            object_name="alphabet_soup_1",
            basket_name="basket_1",
            q=1.0,
            drop_rng=np.random.default_rng(0),
            paired_plan=paired_plan,
            max_steps=24,
            gripper_settle_steps=0,
            episode_session=session,
            recording_stride=RECORDING_STRIDE,
        )

        scheduled.assert_called_once()
        assert session.is_open
        assert logger.frames, "expected session.log_step to record frames"

        logged_ctx = [_ctx_from_frame(frame) for frame in logger.frames]
        injection_logged = [ctx for ctx in logged_ctx if ctx.drop_injection_step]
        assert len(injection_logged) == 1
        assert injection_logged[0].sim_step == DROP_SIM_STEP
        assert injection_logged[0].sim_step % RECORDING_STRIDE == 1
        assert injection_logged[0].loss_mask == 0.0

        dwell_logged = [ctx for ctx in logged_ctx if ctx.post_drop_dwell_step]
        assert dwell_logged, "expected at least one logged dwell frame"
        assert all(ctx.loss_mask == 0.0 for ctx in dwell_logged)

        recovery_logged = [ctx for ctx in logged_ctx if ctx.recovery_active]
        assert recovery_logged, "expected logged recovery frames"
        assert all(ctx.loss_mask == 1.0 for ctx in recovery_logged)

        for ctx in logged_ctx:
            assert should_log_sim_step(
                ctx.sim_step,
                recording_stride=RECORDING_STRIDE,
                force_drop_injection=ctx.drop_injection_step,
            )

        injection_idx = next(i for i, ctx in enumerate(logged_ctx) if ctx.drop_injection_step)
        recovery_idx = next(i for i, ctx in enumerate(logged_ctx) if ctx.recovery_active)
        assert injection_idx < recovery_idx
        assert all(ctx.loss_mask == 0.0 for ctx in logged_ctx[injection_idx:recovery_idx])
        assert logged_ctx[injection_idx - 1].loss_mask == 1.0 if injection_idx > 0 else True
