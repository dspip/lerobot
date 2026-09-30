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

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from lerobot.configs.default import EvalConfig
from lerobot.datasets.success_filter import (
    filter_successful_episodes,
    finish_eval_recorded_episode,
    list_successful_episode_indices,
)
from lerobot.utils.constants import SUCCESS


def test_eval_config_success_only_requires_recording():
    with pytest.raises(ValueError, match="recording_success_only"):
        EvalConfig(recording_success_only=True, n_episodes=1, batch_size=1)


def test_list_successful_episode_indices(tmp_path):
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1, 2, 2],
            SUCCESS: [[False], [True], [False], [False], [False], [True]],
        }
    )
    df.to_parquet(data_dir / "file-000.parquet")
    assert list_successful_episode_indices(tmp_path) == [0, 2]


def test_finish_eval_recorded_episode_success_only(tmp_path, empty_lerobot_dataset_factory):
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "rec", features=features)
    for _ in range(3):
        dataset.add_frame(
            {
                "action": np.zeros(6, dtype=np.float32),
                "observation.state": np.zeros(4, dtype=np.float32),
                "task": "pick",
            }
        )
    finish_eval_recorded_episode(dataset, succeeded=False, success_only=True)
    assert not dataset.has_pending_frames()
    assert dataset.meta.total_episodes == 0

    for _ in range(3):
        dataset.add_frame(
            {
                "action": np.ones(6, dtype=np.float32),
                "observation.state": np.ones(4, dtype=np.float32),
                "task": "pick",
            }
        )
    finish_eval_recorded_episode(dataset, succeeded=True, success_only=True)
    dataset.finalize()
    assert dataset.meta.total_episodes == 1


def test_filter_successful_episodes_caps_keep(tmp_path, empty_lerobot_dataset_factory):
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        SUCCESS: {"dtype": "bool", "shape": (1,), "names": None},
    }
    source = tmp_path / "source"
    dataset = empty_lerobot_dataset_factory(root=source, features=features)
    for ep_idx in range(3):
        for step in range(4):
            dataset.add_frame(
                {
                    "action": np.zeros(6, dtype=np.float32),
                    "observation.state": np.zeros(4, dtype=np.float32),
                    SUCCESS: np.array([ep_idx != 1 and step == 3], dtype=bool),
                    "task": "pick soup",
                }
            )
        dataset.save_episode()
    dataset.finalize()

    output = tmp_path / "kept"
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_download,
    ):
        mock_version.return_value = "v3.0"
        mock_download.side_effect = lambda *args, **kwargs: str(kwargs.get("local_dir") or output)
        result = filter_successful_episodes(source, output, max_keep=1, repo_id="eval_recording_success")

    assert result.source_episodes == 3
    assert result.successful_episodes == 2
    assert result.kept_episodes == 1
    assert result.kept_indices == (0,)
    assert result.discarded_failure_indices == (1,)
    assert result.discarded_overflow_indices == (2,)
