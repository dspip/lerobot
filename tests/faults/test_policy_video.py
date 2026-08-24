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

"""Unit tests for VLA policy-camera video recording."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lerobot.faults import BrightnessDropFault, FaultInjectionConfig
from lerobot.faults.observation.policy_video import PolicyCameraVideoRecorder, write_frame_video


def _obs(fill: int) -> dict:
    # camera1=agentview, camera2=wrist — recorder must use wrist.
    return {
        "pixels": {
            "camera1": np.full((32, 32, 3), 10, dtype=np.uint8),
            "camera2": np.full((32, 32, 3), fill, dtype=np.uint8),
        },
        "state": np.ones(3, dtype=np.float32),
    }


def test_write_frame_video_gif(tmp_path: Path):
    pytest.importorskip("PIL")
    frames = [np.full((16, 16, 3), i * 10, dtype=np.uint8) for i in range(5)]
    path = write_frame_video(frames, tmp_path / "clip", fps=5)
    assert path is not None
    assert path.suffix == ".gif"
    assert path.is_file()


def test_brightness_writes_clean_and_faulted_gifs(tmp_path: Path):
    pytest.importorskip("PIL")
    video_dir = tmp_path / "policy_videos"
    fault = BrightnessDropFault(
        FaultInjectionConfig(
            enabled=True,
            type="brightness_drop",
            trigger_step=2,
            duration=3,
            brightness_scale=0.25,
            seed=0,
            policy_video_dir=video_dir,
        ),
        num_envs=1,
    )
    for fill in (200, 200, 200, 200, 200, 200):
        fault.apply_obs(_obs(fill))
    fault.close()

    clean = list(video_dir.glob("vla_wrist_clean_ep*.gif"))
    faulted = list(video_dir.glob("vla_wrist_brightness_drop_ep*.gif"))
    assert len(clean) == 1
    assert len(faulted) == 1
    assert clean[0].stat().st_size > 0
    assert faulted[0].stat().st_size > 0


def test_recorder_flush_empty_is_noop(tmp_path: Path):
    rec = PolicyCameraVideoRecorder(tmp_path, fault_type="visual_blur")
    assert rec.flush() == []
