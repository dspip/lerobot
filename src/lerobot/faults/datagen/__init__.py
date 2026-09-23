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

"""Unified drop-datagen helpers (recipe, runner, controllers)."""

from lerobot.faults.datagen.drop_timing import (
    DropDecision,
    TraceFrame,
    eligible_indices,
    sample_drop,
)
from lerobot.faults.datagen.events import DatagenEventLog, LogEvent
from lerobot.faults.datagen.layout import ObjectPose2d, sample_layout
from lerobot.faults.datagen.recipe import (
    DatagenController,
    DropDatagenRecipe,
    DropRecipe,
    DropXYBand,
    EpisodeSeedManifest,
    ExpandedMatrixRun,
    MatrixVariant,
    PlacementRecipe,
    PostDropMode,
    PostDropRecipe,
    RecipeError,
    RecordingRecipe,
    SimpleIKPathDropRecipe,
    SimpleIKRecipe,
    SmolVLARecipe,
    effective_post_drop_dwell_steps,
    expand_experiment_matrix,
    legacy_drop_recipe,
    load_drop_datagen_recipe,
    paired_episode_seed_manifests,
    validate_controller_mode_pair,
    validate_experiment_matrix_entries,
)

__all__ = [
    "DatagenController",
    "DatagenEventLog",
    "DropDatagenRecipe",
    "DropDecision",
    "DropRecipe",
    "DropXYBand",
    "EpisodeSeedManifest",
    "ExpandedMatrixRun",
    "LogEvent",
    "MatrixVariant",
    "ObjectPose2d",
    "PlacementRecipe",
    "PostDropMode",
    "PostDropRecipe",
    "RecipeError",
    "RecordingRecipe",
    "SimpleIKPathDropRecipe",
    "SimpleIKRecipe",
    "SmolVLARecipe",
    "TraceFrame",
    "effective_post_drop_dwell_steps",
    "eligible_indices",
    "expand_experiment_matrix",
    "legacy_drop_recipe",
    "load_drop_datagen_recipe",
    "paired_episode_seed_manifests",
    "sample_drop",
    "sample_layout",
    "validate_controller_mode_pair",
    "validate_experiment_matrix_entries",
]
