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

"""Unit tests for visual blur and brightness_drop faults."""

from __future__ import annotations

import numpy as np

from lerobot.faults import (
    BrightnessDropFault,
    FaultInjectionConfig,
    VisualBlurFault,
    make_obs_fault_injector,
)


def _obs() -> dict:
    img = np.zeros((24, 24, 3), dtype=np.uint8)
    img[8:16, 8:16] = 255
    return {"pixels": {"camera1": img}, "state": np.ones(3, dtype=np.float32)}


def test_blur_softens_edges():
    cfg = FaultInjectionConfig(
        enabled=True, type="visual_blur", trigger_step=0, duration=1, blur_sigma=2.0, seed=0
    )
    fault = VisualBlurFault(cfg, num_envs=1)
    out = fault.apply_obs(_obs())
    # Corner of bright square should no longer be pure 255/0 after blur
    assert 0 < int(out["pixels"]["camera1"][8, 7, 0]) < 255 or int(out["pixels"]["camera1"].mean()) < 255
    assert out["pixels"]["camera1"].std() > 0


def test_brightness_scales_down():
    cfg = FaultInjectionConfig(
        enabled=True,
        type="brightness_drop",
        trigger_step=0,
        duration=1,
        brightness_scale=0.25,
        seed=0,
    )
    fault = BrightnessDropFault(cfg, num_envs=1)
    base = _obs()
    out = fault.apply_obs(base)
    assert out["pixels"]["camera1"].max() <= 64  # ~255*0.25
    np.testing.assert_array_equal(out["state"], base["state"])


def test_factory_routes():
    assert isinstance(
        make_obs_fault_injector(
            FaultInjectionConfig(enabled=True, type="visual_blur", trigger_step=0, duration=1),
            num_envs=1,
        ),
        VisualBlurFault,
    )
    assert isinstance(
        make_obs_fault_injector(
            FaultInjectionConfig(
                enabled=True, type="brightness_drop", trigger_step=0, duration=1, brightness_scale=0.5
            ),
            num_envs=1,
        ),
        BrightnessDropFault,
    )


def test_blur_preserves_float_0_255_range():
    from lerobot.faults.observation.utils import apply_box_blur

    img = np.zeros((16, 16, 3), dtype=np.float32)
    img[4:12, 4:12] = 200.0
    apply_box_blur(img, 0, 1, radius=1)
    assert float(img.max()) > 1.5  # not crushed to [0,1]
    assert float(img.max()) <= 200.0 + 1e-3
