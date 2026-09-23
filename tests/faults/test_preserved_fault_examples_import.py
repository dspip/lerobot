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

"""Import preserved example scripts without running sim/GPU/UI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_FAULTS = REPO_ROOT / "examples" / "faults"


def _load_example_module(stem: str) -> ModuleType:
    path = EXAMPLES_FAULTS / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "stem",
    [
        "run_drop_recovery_demo",
        "run_xy_band_boundary_diag",
        "run_soup_offset_probe",
    ],
)
def test_preserved_example_script_imports(stem: str) -> None:
    mod = _load_example_module(stem)
    assert callable(getattr(mod, "main", None))


def test_drop_recovery_demo_exposes_dry_run_entry() -> None:
    mod = _load_example_module("run_drop_recovery_demo")
    assert callable(mod.run_dry_run)
