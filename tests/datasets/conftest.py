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

"""Fixtures used by the success-filter tests."""

from functools import partial

import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_REPO_ID


@pytest.fixture
def empty_lerobot_dataset_factory():
    """Return ``LeRobotDataset.create`` bound to a dummy repo id and fps."""
    return partial(LeRobotDataset.create, repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS)
