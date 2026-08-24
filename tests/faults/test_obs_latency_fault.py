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

"""Unit tests for ObsLatencyFault."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.faults import FaultEventLogger, FaultInjectionConfig, ObsLatencyFault, make_obs_fault_injector


def _cfg(**kwargs) -> FaultInjectionConfig:
    defaults = dict(enabled=True, type="obs_latency", latency_steps=2, seed=0)
    defaults.update(kwargs)
    return FaultInjectionConfig(**defaults)


def _obs(step: int) -> dict:
    return {
        "pixels": {"camera1": np.full((8, 8, 3), step, dtype=np.uint8)},
        "state": np.array([float(step)], dtype=np.float32),
    }


def test_warm_up_then_stale():
    fault = ObsLatencyFault(_cfg(latency_steps=2), num_envs=1)
    o0 = fault.apply_obs(_obs(10))
    assert o0["pixels"]["camera1"][0, 0, 0] == 10
    o1 = fault.apply_obs(_obs(20))
    assert o1["pixels"]["camera1"][0, 0, 0] == 20
    o2 = fault.apply_obs(_obs(30))
    # buffer had [10,20], enqueue 30, pop 10
    assert o2["pixels"]["camera1"][0, 0, 0] == 10
    o3 = fault.apply_obs(_obs(40))
    assert o3["pixels"]["camera1"][0, 0, 0] == 20


def test_reset_clears_buffer():
    fault = ObsLatencyFault(_cfg(latency_steps=1), num_envs=1)
    fault.apply_obs(_obs(1))
    fault.apply_obs(_obs(2))
    fault.reset()
    out = fault.apply_obs(_obs(99))
    assert out["pixels"]["camera1"][0, 0, 0] == 99


def test_logging(tmp_path: Path):
    log_path = tmp_path / "lat.jsonl"
    logger = FaultEventLogger(log_path)
    fault = ObsLatencyFault(_cfg(latency_steps=1), num_envs=1, event_logger=logger)
    fault.apply_obs(_obs(1))
    fault.apply_obs(_obs(2))
    logger.close()
    events = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    assert events[0]["event"] == "obs_latency"
    assert events[0]["stale"] is True
    assert isinstance(make_obs_fault_injector(_cfg(), num_envs=1), ObsLatencyFault)


def test_obs_latency_writes_wrist_gifs(tmp_path: Path):
    pytest.importorskip("PIL")
    video_dir = tmp_path / "policy_videos"
    fault = ObsLatencyFault(
        _cfg(latency_steps=2, policy_video_dir=video_dir),
        num_envs=1,
    )
    for step in range(6):
        fault.apply_obs(
            {
                "pixels": {
                    "camera1": np.full((8, 8, 3), 1, dtype=np.uint8),
                    "camera2": np.full((8, 8, 3), step * 10, dtype=np.uint8),
                },
                "state": np.array([float(step)], dtype=np.float32),
            }
        )
    fault.close()
    assert list(video_dir.glob("vla_wrist_clean_ep*.gif"))
    assert list(video_dir.glob("vla_wrist_obs_latency_ep*.gif"))


def test_obs_latency_rejects_selective_env_ids_multi():
    import pytest

    with pytest.raises(ValueError, match="env_ids=None"):
        ObsLatencyFault(_cfg(env_ids=[1]), num_envs=2).apply_obs(
            {
                "pixels": {"camera1": np.zeros((2, 4, 4, 3), dtype=np.uint8)},
                "state": np.zeros((2, 1), dtype=np.float32),
            }
        )


def test_obs_latency_continues_after_env0_done():
    fault = ObsLatencyFault(_cfg(latency_steps=1), num_envs=2)
    obs = {
        "pixels": {"camera1": np.full((2, 4, 4, 3), 10, dtype=np.uint8)},
        "state": np.array([[10.0], [10.0]], dtype=np.float32),
    }
    fault.apply_obs(obs)  # warm-up
    fault.notify_dones(np.asarray([True, False]))
    # env0 finished; env1 still active — shared lag must still apply
    nxt = {
        "pixels": {"camera1": np.full((2, 4, 4, 3), 99, dtype=np.uint8)},
        "state": np.array([[99.0], [99.0]], dtype=np.float32),
    }
    out = fault.apply_obs(nxt)
    assert out["pixels"]["camera1"][0, 0, 0, 0] == 10

