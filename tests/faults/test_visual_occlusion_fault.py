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

"""Unit tests for VisualOcclusionFault (no LeRobot / simulator)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.faults import (
    FaultEventLogger,
    FaultInjectionConfig,
    VisualOcclusionFault,
    make_obs_fault_injector,
)


def _cfg(**kwargs) -> FaultInjectionConfig:
    defaults = dict(
        enabled=True,
        type="visual_occlusion",
        trigger_step=1,
        duration=2,
        probability=1.0,
        seed=0,
        log_path=None,
        diag_dir=None,
    )
    defaults.update(kwargs)
    return FaultInjectionConfig(**defaults)


def _obs(fill: int = 200) -> dict:
    return {
        "pixels": {"camera1": np.full((32, 32, 3), fill, dtype=np.uint8)},
        "state": np.ones(4, dtype=np.float32),
    }


def test_no_occlusion_before_trigger():
    fault = VisualOcclusionFault(_cfg(trigger_step=2), num_envs=1)
    out = fault.apply_obs(_obs())
    assert out["pixels"]["camera1"].min() == 200


def test_occlusion_zeros_a_box_not_all_pixels():
    fault = VisualOcclusionFault(_cfg(trigger_step=0, duration=1, seed=1), num_envs=1)
    out = fault.apply_obs(_obs(fill=200))
    img = out["pixels"]["camera1"]
    assert img.min() == 0
    assert img.max() == 200  # not full blackout
    np.testing.assert_array_equal(out["state"], np.ones(4, dtype=np.float32))


def test_batched_single_env_layout_occludes_spatial_box():
    """Regression: num_envs=1 with leading batch dim (1,H,W,C) must paint H/W, not N."""
    fault = VisualOcclusionFault(
        _cfg(
            trigger_step=0,
            duration=1,
            seed=0,
            occlusion_h_frac_min=0.5,
            occlusion_h_frac_max=0.5,
            occlusion_w_frac_min=0.5,
            occlusion_w_frac_max=0.5,
        ),
        num_envs=1,
    )
    obs = {
        "pixels": {"camera1": np.full((1, 40, 40, 3), 180, dtype=np.uint8)},
        "state": np.ones((1, 3), dtype=np.float32),
    }
    out = fault.apply_obs(obs)
    img = out["pixels"]["camera1"][0]
    assert img.min() == 0
    assert img.max() == 180
    # At least ~25% of pixels black for a 0.5x0.5 box
    assert float((img == 0).mean()) > 0.2


def test_duration_and_factory(tmp_path: Path):
    pytest.importorskip("PIL")
    log_path = tmp_path / "occ.jsonl"
    cfg = _cfg(trigger_step=0, duration=2, log_path=log_path, diag_dir=tmp_path / "diag")
    fault = make_obs_fault_injector(cfg, num_envs=1)
    assert isinstance(fault, VisualOcclusionFault)
    fault.apply_obs(_obs())
    fault.apply_obs(_obs())
    third = fault.apply_obs(_obs())
    assert third["pixels"]["camera1"].min() == 200
    fault.event_logger.close()
    events = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    assert events[0]["event"] == "visual_occlusion"
    assert "occlusion_box" in events[0]
    assert any(tmp_path.joinpath("diag").glob("*.png"))
