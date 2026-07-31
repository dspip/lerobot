#!/usr/bin/env python

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

import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES


def make_config(**overrides) -> LingBotVAConfig:
    kwargs = {"device": "cpu"}
    kwargs.update(overrides)
    return LingBotVAConfig(**kwargs)


def test_registered_in_choice_registry() -> None:
    assert "lingbot_va" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("lingbot_va") is LingBotVAConfig


def test_type_property() -> None:
    assert make_config().type == "lingbot_va"


def test_chunk_size_and_action_steps() -> None:
    cfg = make_config(frame_chunk_size=4, action_per_frame=4)
    assert cfg.chunk_size == 16
    assert cfg.n_action_steps == 16
    assert cfg.action_delta_indices == list(range(16))
    # 4 * (frame_chunk_size - 1) + 1 frames: exactly what AutoencoderKLWan._encode consumes to
    # produce frame_chunk_size latent frames. Loading 16 decoded 3 frames the VAE never read.
    assert cfg.observation_delta_indices == list(range(13))
    assert cfg.reward_delta_indices is None


def test_observation_delta_indices_are_all_consumed_by_the_vae() -> None:
    """Every requested frame must survive ``iter_ = 1 + (n - 1) // 4``, and F must be preserved."""
    for frame_chunk_size, action_per_frame in ((1, 4), (2, 16), (4, 4), (3, 8)):
        cfg = make_config(frame_chunk_size=frame_chunk_size, action_per_frame=action_per_frame)
        deltas = cfg.observation_delta_indices
        num_latent_frames = 1 + (len(deltas) - 1) // 4
        assert num_latent_frames == frame_chunk_size, (frame_chunk_size, deltas)
        assert len(deltas) == 4 * (num_latent_frames - 1) + 1, (frame_chunk_size, deltas)
        assert deltas[0] == 0
        assert len(set(deltas)) == len(deltas)


def test_optimizer_and_scheduler_presets() -> None:
    cfg = make_config()
    opt = cfg.get_optimizer_preset()
    assert opt.lr == cfg.optimizer_lr
    sched = cfg.get_scheduler_preset()
    assert sched.num_warmup_steps == cfg.scheduler_warmup_steps


def test_validate_features_sets_action_feature() -> None:
    cfg = make_config()
    cfg.input_features = {f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))}
    cfg.output_features = {}
    cfg.validate_features()
    assert ACTION in cfg.output_features
    assert cfg.output_features[ACTION].shape == (len(cfg.used_action_channel_ids),)


def test_validate_features_no_visual_raises() -> None:
    cfg = make_config()
    cfg.input_features = {}
    cfg.output_features = {}
    with pytest.raises(ValueError, match="at least one visual input feature"):
        cfg.validate_features()


def test_invalid_attn_mode_raises() -> None:
    with pytest.raises(ValueError, match="attn_mode"):
        make_config(attn_mode="banana")
