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

"""Increment-encoded conditioning for the action KV cache under relative actions.

LingBot-VA's action stream doubles as its memory -- `_compute_kv_cache` feeds the previous chunk's
actions back as clean context. Under `use_relative_actions` those values are cumulative offsets from
the chunk's own anchor, so the cached stream resets to zero displacement at every boundary; the
model, whose only cross-frame supervision is "continue from the previous clean action frame",
continues instead of restarting and the postprocessor double-counts the displacement.

The conversion applies to the conditioning ONLY. The predicted actions stay cumulative offsets, so
the processors are untouched and nothing has to invert an accumulated stream.
"""

from __future__ import annotations

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES

USED_CHANNELS = [21, 22, 23, 24, 25, 26, 27, 29, 14, 15, 16, 17, 18, 19, 20, 28]


def _make_config(*, use_relative_actions: bool = True):
    cfg = LingBotVAConfig(
        device="cpu",
        action_per_frame=16,
        frame_chunk_size=2,
        used_action_channel_ids=list(USED_CHANNELS),
        use_relative_actions=use_relative_actions,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.QUANTILES,
        },
    )
    cfg.input_features = {f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))}
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,))}
    return cfg


def _policy(cfg) -> LingBotVAPolicy:
    """A bare policy: the conversion reads only `config`, so the 5B transformer is irrelevant here."""
    policy = LingBotVAPolicy.__new__(LingBotVAPolicy)
    policy.config = cfg
    policy.dtype = torch.float32
    return policy


def _ramp_chunk(cfg, *, start: float = 0.0) -> torch.Tensor:
    """A chunk of cumulative offsets growing from `start`, shaped [1, action_dim, F, apf, 1]."""
    horizon = cfg.frame_chunk_size * cfg.action_per_frame
    values = start + torch.arange(horizon, dtype=torch.float32) * 0.05
    chunk = torch.zeros(1, cfg.action_dim, cfg.frame_chunk_size, cfg.action_per_frame, 1)
    for channel in USED_CHANNELS:
        chunk[0, channel, :, :, 0] = values.reshape(cfg.frame_chunk_size, cfg.action_per_frame)
    return chunk


def test_increments_are_the_first_difference_along_the_horizon() -> None:
    """The horizon runs frame-major across F * action_per_frame, not per frame independently."""
    cfg = _make_config()
    chunk = _ramp_chunk(cfg)
    out = _policy(cfg)._to_action_increments(chunk)

    flat_in = chunk[0, USED_CHANNELS[0]].reshape(-1)
    flat_out = out[0, USED_CHANNELS[0]].reshape(-1)
    assert flat_out[0] == flat_in[0]  # element 0 is passed through
    torch.testing.assert_close(flat_out[1:], flat_in[1:] - flat_in[:-1], atol=1e-6, rtol=0)
    # the frame boundary is a normal step, not a restart
    boundary = cfg.action_per_frame
    torch.testing.assert_close(
        flat_out[boundary], flat_in[boundary] - flat_in[boundary - 1], atol=1e-6, rtol=0
    )


def test_conditioning_is_independent_of_the_chunk_anchor() -> None:
    """The property that removes the sawtooth: the same motion encodes identically in any chunk."""
    cfg = _make_config()
    policy = _policy(cfg)
    early = policy._to_action_increments(_ramp_chunk(cfg, start=0.0))
    late = policy._to_action_increments(_ramp_chunk(cfg, start=0.6))  # a later chunk, same motion

    # cumulative offsets differ everywhere; increments differ only at element 0, which carries the
    # anchor offset by design (it is the command-vs-measured error, ~2 deg, not a chunk of motion).
    assert not torch.allclose(_ramp_chunk(cfg, start=0.0), _ramp_chunk(cfg, start=0.6))
    torch.testing.assert_close(early[:, :, :, 1:], late[:, :, :, 1:], atol=1e-6, rtol=0)


def test_preprocess_action_state_applies_the_conversion() -> None:
    cfg = _make_config()
    chunk = _ramp_chunk(cfg)
    out = _policy(cfg)._preprocess_action_state(chunk)
    torch.testing.assert_close(out, _policy(cfg)._to_action_increments(chunk), atol=1e-6, rtol=0)


def test_absolute_action_runs_are_untouched() -> None:
    """The one hard compatibility constraint: an absolute-action checkpoint must be unaffected.

    Its action stream is already globally consistent (the values are joint positions, not offsets
    from a per-chunk anchor), so there is nothing to re-encode.
    """
    cfg = _make_config(use_relative_actions=False)
    chunk = _ramp_chunk(cfg)
    out = _policy(cfg)._preprocess_action_state(chunk)
    torch.testing.assert_close(out, chunk, atol=0, rtol=0)


def test_conversion_is_invertible_so_no_information_is_lost() -> None:
    """cumsum recovers the original chunk -- the conditioning is a re-encoding, not a truncation."""
    cfg = _make_config()
    chunk = _ramp_chunk(cfg, start=0.3)
    increments = _policy(cfg)._to_action_increments(chunk)
    b, c, f, n, _ = chunk.shape
    recovered = torch.cumsum(increments.reshape(b, c, f * n), dim=-1).reshape(b, c, f, n, 1)
    torch.testing.assert_close(recovered, chunk, atol=1e-5, rtol=0)


def test_unused_channels_stay_zero() -> None:
    cfg = _make_config()
    out = _policy(cfg)._to_action_increments(_ramp_chunk(cfg))
    unused = [c for c in range(cfg.action_dim) if c not in USED_CHANNELS]
    assert torch.count_nonzero(out[0, unused]) == 0


def test_predicted_actions_are_not_converted() -> None:
    """Only the conditioning changes; the output path still emits cumulative offsets.

    Guards the reason this is safe to do inside the model: normalization is affine with a non-zero
    constant, so cumsum does not commute with unnormalization. If the *prediction* were differenced,
    the postprocessor would drift by (n-1)*(q99+q01)/2 per chunk -- 0 on a symmetric channel but
    ~118 deg on right_joint_6.
    """
    import inspect

    source = inspect.getsource(LingBotVAPolicy.predict_action_chunk)
    assert "_to_action_increments" not in source
    # the conversion is reachable only from the KV-cache path
    assert "_to_action_increments" in inspect.getsource(LingBotVAPolicy._preprocess_action_state)
    assert "_preprocess_action_state" in inspect.getsource(LingBotVAPolicy._compute_kv_cache)
