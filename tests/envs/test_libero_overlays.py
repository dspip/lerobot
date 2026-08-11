# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import pytest
import yaml

from lerobot.envs.libero_overlays import apply_overlay, load_overlay_config
from lerobot.envs.libero_overlays.bddl import patch_bddl
from lerobot.envs.libero_overlays.config import AddObjectSpec, LiberoOverlayConfig

SAMPLE_BDDL = """\
(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language Pick the tomato sauce and place it in the basket)
    (:regions
      (bin_region
          (:target floor)
          (:ranges (
              (-0.01 0.25 0.01 0.27)
            )
          )
      )
      (target_object_region
          (:target floor)
          (:ranges (
              (0.025 -0.125 0.075 -0.075)
            )
          )
      )
    )

  (:fixtures
    floor - floor
  )

  (:objects
    tomato_sauce_1 - tomato_sauce
    basket_1 - basket
  )

  (:obj_of_interest
    tomato_sauce_1
    basket_1
  )

  (:init
    (On tomato_sauce_1 floor_target_object_region)
    (On basket_1 floor_bin_region)
  )

  (:goal
    (And (In tomato_sauce_1 basket_1_contain_region))
  )

)
"""


def test_replace_rewrites_category_instances_and_language():
    cfg = LiberoOverlayConfig(
        mode="replace",
        replacements={"tomato_sauce": "red_cube"},
        language="Pick the red cube and place it in the basket",
    )
    patched = patch_bddl(SAMPLE_BDDL, cfg)
    assert "tomato_sauce" not in patched
    assert "red_cube_1 - red_cube" in patched
    assert "(In red_cube_1 basket_1_contain_region)" in patched
    assert "(:language Pick the red cube and place it in the basket)" in patched
    assert "basket_1 - basket" in patched


def test_add_injects_object_region_and_init(tmp_path: Path):
    cfg = LiberoOverlayConfig(
        mode="add",
        add_objects=(
            AddObjectSpec(
                category="red_cube",
                ranges=(0.05, 0.10, 0.10, 0.15),
                target="floor",
            ),
        ),
    )
    patched = patch_bddl(SAMPLE_BDDL, cfg)
    assert "red_cube_1 - red_cube" in patched
    assert "red_cube_init_region" in patched
    assert "(On red_cube_1 floor_red_cube_init_region)" in patched
    # Original objects remain.
    assert "tomato_sauce_1 - tomato_sauce" in patched


def test_load_overlay_yaml_replace(tmp_path: Path):
    yaml_path = tmp_path / "replace.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "mode": "replace",
                "suite": "libero_object",
                "task_id": 5,
                "replacements": {"tomato_sauce": "red_cube"},
                "language": "Pick the red cube and place it in the basket",
            }
        ),
        encoding="utf-8",
    )
    cfg = load_overlay_config(yaml_path)
    assert cfg.mode == "replace"
    assert cfg.replacements == {"tomato_sauce": "red_cube"}
    assert cfg.suite == "libero_object"
    assert cfg.task_id == 5
    assert cfg.requires_disable_init_states() is False


def test_add_mode_disables_init_states_by_default():
    cfg = LiberoOverlayConfig(
        mode="add",
        add_objects=(AddObjectSpec(category="red_cube", ranges=(0.0, 0.0, 0.1, 0.1)),),
    )
    assert cfg.requires_disable_init_states() is True


def test_apply_overlay_writes_new_file_without_touching_base(tmp_path: Path):
    base = tmp_path / "base.bddl"
    base.write_text(SAMPLE_BDDL, encoding="utf-8")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "mode": "replace",
                "replacements": {"tomato_sauce": "red_cube"},
                "language": "Pick the red cube and place it in the basket",
            }
        ),
        encoding="utf-8",
    )

    applied = apply_overlay(base, overlay, output_dir=tmp_path / "out")
    assert applied.bddl_path.exists()
    assert applied.bddl_path != base
    assert base.read_text(encoding="utf-8") == SAMPLE_BDDL
    assert "red_cube_1 - red_cube" in applied.bddl_path.read_text(encoding="utf-8")
    assert applied.disable_init_states is False


def test_apply_overlay_rejects_missing_source_category(tmp_path: Path):
    base = tmp_path / "base.bddl"
    base.write_text(SAMPLE_BDDL, encoding="utf-8")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump({"mode": "replace", "replacements": {"missing_obj": "red_cube"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not contain categories"):
        apply_overlay(base, overlay, output_dir=tmp_path / "out")


def test_example_overlay_yamls_load():
    root = Path(__file__).resolve().parents[2] / "examples" / "libero_overlays"
    replace_cfg = load_overlay_config(root / "replace_tomato_with_red_cube.yaml")
    add_cfg = load_overlay_config(root / "add_red_cube.yaml")
    assert replace_cfg.mode == "replace"
    assert replace_cfg.replacements["tomato_sauce"] == "red_cube"
    assert add_cfg.mode == "add"
    assert add_cfg.add_objects[0].category == "red_cube"


def test_libero_env_config_default_has_no_overlay():
    from lerobot.envs.configs import LiberoEnv

    cfg = LiberoEnv(task="libero_object", task_ids=[5])
    assert cfg.overlay is None
