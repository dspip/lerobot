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

import numpy as np

from lerobot.envs.configs import LiberoEnv
from lerobot.scripts.lerobot_eval import _build_raw_frame, _env_features_to_dataset_features
from lerobot.utils.feature_utils import _validate_feature_names


def test_libero_recording_features_have_no_slashes():
    env_cfg = LiberoEnv(
        camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}
    )
    features = _env_features_to_dataset_features(env_cfg.features, env_cfg.features_map)
    _validate_feature_names(features)
    assert "observation.images.camera1" in features
    assert "observation.images.camera2" in features
    assert "observation.state.eef_pos" in features
    assert not any("/" in key for key in features)


def test_build_raw_frame_maps_libero_pixels_and_state():
    env_cfg = LiberoEnv(
        camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}
    )
    raw_obs = {
        "pixels": {
            "camera1": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "camera2": np.ones((1, 8, 8, 3), dtype=np.uint8),
        },
        "robot_state": {
            "eef": {
                "pos": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
                "quat": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                "mat": np.eye(3, dtype=np.float32)[None, ...],
            },
            "gripper": {
                "qpos": np.array([[0.0, 0.0]], dtype=np.float32),
                "qvel": np.array([[0.0, 0.0]], dtype=np.float32),
            },
            "joints": {
                "pos": np.zeros((1, 7), dtype=np.float32),
                "vel": np.zeros((1, 7), dtype=np.float32),
            },
        },
    }
    frame = _build_raw_frame(
        raw_obs,
        0,
        np.zeros(7, dtype=np.float32),
        0.0,
        False,
        False,
        "pick up the alphabet soup and place it in the basket",
        env_cfg.features,
        features_map=env_cfg.features_map,
    )
    assert frame["observation.images.camera1"].shape == (8, 8, 3)
    assert frame["observation.images.camera2"].mean() == 1.0
    np.testing.assert_allclose(frame["observation.state.eef_pos"], [0.1, 0.2, 0.3])
    assert "next.success" in frame
