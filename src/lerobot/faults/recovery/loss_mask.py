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

"""Dataset loss-mask helpers for mid-air drop recovery logging."""

from __future__ import annotations


def loss_mask_for_step(step: int, t_fault: int, *, drop_duration: int = 1) -> float:
    """Return dataset ``loss_mask`` for a scripted episode step."""
    del drop_duration
    if step < t_fault:
        return 1.0
    if step == t_fault:
        return 0.0
    return 1.0


def loss_mask_from_fault(*, triggered: bool, drop_injection_step: bool, recovery_active: bool) -> float:
    """Derive ``loss_mask`` from midair_drop fault state when available."""
    del triggered, recovery_active
    if drop_injection_step:
        return 0.0
    return 1.0
