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

from examples.faults import run_drop_datagen as cli

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"


def test_cli_rejects_non_positive_episodes_override() -> None:
    code = cli.main(
        ["--recipe", str(CAN_DROP_RECIPE), "--episodes", "0"],
    )
    assert code == 2


def test_smolvla_pipeline_run_pipeline_is_in_package() -> None:
    from lerobot.faults.datagen.smolvla_pipeline import run_pipeline

    assert callable(run_pipeline)
