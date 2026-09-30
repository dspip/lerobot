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
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot_faults.datagen.controllers.simple_ik import run_simple_ik_episode_loop
from lerobot_faults.datagen.dataset_writer import RunDatasetWriter
from lerobot_faults.datagen.failure_segments import (
    AttemptTickSnapshot,
    attempt_rows_from_tick_snapshots,
    tick_index_at_or_after_frame,
)
from lerobot_faults.datagen.frame_logging import log_post_step_to_session
from lerobot_faults.datagen.recipe import RecipeError, load_drop_datagen_recipe, replace
from lerobot_faults.datagen.recording_views import (
    ViewRecordingState,
    build_datagen_frame_labels,
    offer_tick_to_views,
    should_log_view_tick,
)
from tests.faults.test_datagen_dataset_writer import _minimal_processed_frame, _RecordingLogger
from tests.faults.test_simple_ik_episode_logging import CAN_DROP_RECIPE


def test_should_log_view_tick_stride_two() -> None:
    logged = [t for t in range(10) if should_log_view_tick(t, stride=2)]
    assert logged == [0, 2, 4, 6, 8]


def test_pulse_or_accumulation_odd_release_on_even_frame() -> None:
    view = ViewRecordingState(name="dataset", stride=2)
    views = (view,)
    annotation = {"failure_onset": np.array([False])}
    raw = {
        "tick_index": np.array([1], dtype=np.int64),
        "drop_release": np.array([True]),
        "drop_event": np.array([True]),
        "attempt_index": np.array([0], dtype=np.int64),
    }
    offer_tick_to_views(views, tick=1, sim_time_s=0.05, annotation=annotation, raw_labels=raw)
    decisions = offer_tick_to_views(
        views,
        tick=2,
        sim_time_s=0.10,
        annotation={"failure_onset": np.array([False])},
        raw_labels={
            "tick_index": np.array([2], dtype=np.int64),
            "drop_release": np.array([False]),
            "drop_event": np.array([True]),
            "attempt_index": np.array([0], dtype=np.int64),
        },
    )
    assert decisions[0].log
    assert decisions[0].merged_annotation is not None
    assert bool(np.asarray(decisions[0].merged_annotation["drop_release"]).reshape(-1)[0])


def test_sim_time_decrease_raises() -> None:
    view = ViewRecordingState(name="dataset", stride=1)
    view.last_logged_sim_time_s = 1.0
    with pytest.raises(RuntimeError, match="sim time decreased"):
        offer_tick_to_views(
            (view,),
            tick=1,
            sim_time_s=0.5,
            annotation={},
            raw_labels={
                "tick_index": np.array([1], dtype=np.int64),
                "drop_release": np.array([False]),
                "drop_event": np.array([False]),
                "attempt_index": np.array([0], dtype=np.int64),
            },
        )


def test_drop_label_sequence_synthetic() -> None:
    # The real fault keeps drop_injection_step set for the whole fall.
    states = [
        SimpleNamespace(drop_injection_step=False, triggered=False, falling=False),
        SimpleNamespace(drop_injection_step=True, triggered=True, falling=True),
        SimpleNamespace(drop_injection_step=True, triggered=True, falling=True),
        SimpleNamespace(drop_injection_step=False, triggered=True, falling=False),
        SimpleNamespace(drop_injection_step=False, triggered=True, falling=False),
    ]
    labels = [
        build_datagen_frame_labels(
            states[i],
            tick=i,
            is_drop_episode=True,
            prev_triggered=bool(i and states[i - 1].triggered),
        )
        for i in range(len(states))
    ]
    assert bool(labels[1]["drop_release"][0])
    assert sum(bool(l["drop_release"][0]) for l in labels) == 1
    assert bool(labels[1]["drop_event"][0]) and bool(labels[2]["drop_event"][0])
    assert not bool(labels[3]["drop_event"][0])
    assert int(labels[0]["attempt_index"][0]) == 0
    assert int(labels[2]["attempt_index"][0]) == 0
    assert int(labels[3]["attempt_index"][0]) == 1


def test_no_drop_label_sequence() -> None:
    state = SimpleNamespace(drop_injection_step=False, triggered=False, falling=False)
    labels = build_datagen_frame_labels(state, tick=0, is_drop_episode=False)
    assert int(labels["attempt_index"][0]) == 0
    assert not bool(labels["drop_event"][0])


def test_failure_segments_tick_to_frame_stride() -> None:
    assert tick_index_at_or_after_frame(0, stride=2) == 0
    assert tick_index_at_or_after_frame(1, stride=2) == 1
    assert tick_index_at_or_after_frame(2, stride=2) == 1
    assert tick_index_at_or_after_frame(3, stride=2) == 2


def test_failure_segments_drop_episode_rows() -> None:
    snaps = [
        AttemptTickSnapshot(tick=0, drop_release=False, drop_event=False, attempt_index=0, object_z=0.2),
        AttemptTickSnapshot(tick=1, drop_release=True, drop_event=True, attempt_index=0, object_z=0.5),
        AttemptTickSnapshot(tick=2, drop_release=False, drop_event=True, attempt_index=0, object_z=0.4),
        AttemptTickSnapshot(tick=3, drop_release=False, drop_event=False, attempt_index=1, object_z=0.03),
        AttemptTickSnapshot(tick=4, drop_release=False, drop_event=False, attempt_index=1, object_z=0.03),
    ]
    rows = attempt_rows_from_tick_snapshots(
        snaps,
        stride=2,
        control_hz=20,
        episode_index=0,
        logical_episode_index=0,
        object_name="alphabet_soup_1",
        task_id=7,
        controller="simple_ik",
        post_drop_mode="continue_then_ik",
        is_drop_episode=True,
        episode_seed=1,
        layout_seed=2,
    )
    assert len(rows) == 2
    assert rows[0].has_drop and rows[0].release_tick == 1
    assert rows[0].landing_tick == 3
    assert rows[0].release_frame == 1
    assert rows[1].attempt_index == 1


def test_recipe_master_fps_must_match_control_hz(tmp_path: Path) -> None:
    import json

    raw = json.loads(CAN_DROP_RECIPE.read_text(encoding="utf-8"))
    raw["recording"]["master_fps"] = 10
    path = tmp_path / "bad_master_fps.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RecipeError, match="master_fps must equal control_hz"):
        load_drop_datagen_recipe(path)


def test_dual_view_commit_aligns_episode_indices(tmp_path: Path) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    recipe = replace(
        recipe,
        recording=replace(
            recipe.recording,
            output_dir=str(tmp_path / "run"),
            master_fps=recipe.control_hz,
        ),
    )
    writer = RunDatasetWriter(
        recipe,
        skip_fresh_output_check=True,
        logger_factory=lambda root, repo_id, **_kw: _RecordingLogger(root),
    )
    manifest = __import__(
        "lerobot_faults.datagen.recipe", fromlist=["paired_episode_seed_manifests"]
    ).paired_episode_seed_manifests(recipe, 0)[0]
    session = writer.open_episode_session(manifest)
    assert len(session.recording_views) == 2
    primary, master = session.recording_views
    frame = _minimal_processed_frame()
    session.log_step(frame, np.zeros(7), "task", 1.0, view_name=primary.name)
    session.log_step(frame, np.zeros(7), "task", 1.0, view_name=master.name)
    idx = session.commit(success=True)
    assert idx == 0
    assert primary.logger.committed == 1
    assert master.logger.committed == 1


def test_simple_ik_stops_env_step_after_done() -> None:
    """Further env.step after termination must not run (autoreset guard)."""
    from unittest.mock import patch

    import gymnasium as gym

    from lerobot_faults.config import FaultInjectionConfig
    from lerobot_faults.wrappers import DropRecoveryEnvWrapper

    step_calls = {"n": 0}

    class _TerminatingInner:
        num_envs = 1
        action_space = gym.spaces.Box(-1, 1, shape=(7,))

        def reset(self, *, seed=None, options=None):
            return {"observation.state": np.zeros(8)}, {}

        def step(self, action):
            step_calls["n"] += 1
            obs = {"observation.state": np.zeros(8)}
            if step_calls["n"] >= 3:
                return obs, 1.0, True, False, {"is_success": True}
            return obs, 0.0, False, False, {}

        def close(self):
            return None

    cfg = FaultInjectionConfig(
        enabled=True,
        type="midair_drop",
        probability=0.0,
        object_name="alphabet_soup_1",
    )
    env = DropRecoveryEnvWrapper(_TerminatingInner(), cfg)
    env.reset()
    paired_plan = SimpleNamespace(drop_decision=SimpleNamespace(drop=False, reason="skipped"))
    planner = MagicMock(phase_name="lift", carry_path=None, done=False)
    with (
        patch(
            "lerobot_faults.datagen.controllers.simple_ik._nominal_action",
            return_value=np.ones(7),
        ),
        patch(
            "lerobot_faults.datagen.controllers.simple_ik.get_object_pose",
            return_value={"pos": np.zeros(3)},
        ),
        patch(
            "lerobot_faults.datagen.controllers.simple_ik.get_eef_pose",
            return_value=(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
        ),
        patch(
            "lerobot_faults.datagen.controllers.simple_ik.get_place_destination",
            return_value=np.zeros(3),
        ),
        patch(
            "lerobot_faults.datagen.controllers.simple_ik.is_object_in_basket",
            return_value=True,
        ),
        patch(
            "lerobot_faults.datagen.controllers.simple_ik.is_object_held_midair",
            return_value=False,
        ),
        patch(
            "lerobot_faults.datagen.controllers.simple_ik.is_object_grasped",
            return_value=False,
        ),
    ):
        facts = run_simple_ik_episode_loop(
            env,
            MagicMock(),
            fault=env.fault,
            planner=planner,
            recipe_drop=MagicMock(min_drop_distance_from_basket_m=0.3, hard_keepout_floor_m=0.22),
            object_name="alphabet_soup_1",
            basket_name="basket_1",
            q=0.0,
            drop_rng=np.random.default_rng(0),
            paired_plan=paired_plan,
            max_steps=50,
        )
    assert step_calls["n"] == 3
    assert facts.success
