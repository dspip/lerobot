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
# See the License for the License governing permissions and
# limitations under the License.

"""Unit tests for post-drop mode video banners and comparison ffmpeg graphs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_FAULTS = REPO_ROOT / "examples" / "faults"


def _load_module(name: str, filename: str):
    path = EXAMPLES_FAULTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pipeline_mod():
    from lerobot.faults.datagen import smolvla_pipeline

    return smolvla_pipeline


@pytest.fixture(scope="module")
def compare_mod():
    return _load_module("compare_post_drop_modes", "compare_post_drop_modes.py")


def test_drop_phase_banner_immediate_ik(pipeline_mod):
    banner = pipeline_mod._drop_phase_banner("immediate_ik", 0, 0)
    assert banner == "PHASE: DROP immediate_ik"


def test_drop_phase_banner_continue_mid_dwell(pipeline_mod):
    banner = pipeline_mod._drop_phase_banner("continue_then_ik", 23, 80)
    assert banner == "PHASE: DWELL continue_then_ik 23/80"


def test_drop_phase_banner_reset_mid_dwell(pipeline_mod):
    banner = pipeline_mod._drop_phase_banner("reset_then_ik", 23, 80)
    assert banner == "PHASE: DWELL reset_then_ik 23/80 (queue cleared)"


def test_build_filter_graph_two_inputs(compare_mod):
    graph = compare_mod.build_filter_graph(["immediate_ik", "continue_then_ik"])
    assert "immediate_ik" in graph
    assert "continue_then_ik" in graph
    assert "[0:v]" in graph
    assert "[1:v]" in graph
    assert "hstack=inputs=2" in graph
    assert "hstack=inputs=3" not in graph


def test_build_filter_graph_three_inputs(compare_mod):
    graph = compare_mod.build_filter_graph(["a", "b", "c"])
    assert graph.count("[0:v]") == 1
    assert "[2:v]" in graph
    assert "hstack=inputs=3" in graph
    assert "drawtext=text='a'" in graph
    assert "drawtext=text='c'" in graph


def test_build_filter_graph_requires_two_labels(compare_mod):
    with pytest.raises(ValueError, match="at least 2"):
        compare_mod.build_filter_graph(["only_one"])


def test_validate_episodes_too_few(compare_mod, tmp_path: Path):
    with pytest.raises(ValueError, match="At least 2"):
        compare_mod.validate_episodes([("a", tmp_path)])


def test_validate_episodes_missing_video(compare_mod, tmp_path: Path):
    ep_a = tmp_path / "a"
    ep_b = tmp_path / "b"
    ep_a.mkdir()
    ep_b.mkdir()
    (ep_a / "videos").mkdir()
    (ep_a / "videos" / "full_pipeline.mp4").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="full_pipeline.mp4"):
        compare_mod.validate_episodes([("a", ep_a), ("b", ep_b)])


def test_parse_episode_arg_invalid(compare_mod):
    with pytest.raises(ValueError, match="LABEL=PATH"):
        compare_mod.parse_episode_arg("noequals")


def test_inject_trailing_tpad(compare_mod):
    base = compare_mod.build_filter_graph(["x", "y"])
    padded = compare_mod.inject_trailing_tpad(base, [1.5, 0.0])
    assert "tpad=stop_mode=clone:stop_duration=1.500000[v0]" in padded
    first_chain, second_chain, _hstack = padded.split(";")
    assert "tpad" in first_chain
    assert "tpad" not in second_chain
