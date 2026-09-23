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

"""Basket-distance target crossing helpers (no datagen package imports)."""


def basket_distance_target_reached(
    *,
    prev_m: float,
    curr_m: float,
    target_m: float,
    band_min_m: float,
    band_max_m: float,
    held_midair: bool,
) -> bool:
    if not held_midair:
        return False
    if prev_m <= curr_m:
        return False
    if curr_m <= target_m <= prev_m:
        return True
    if band_min_m <= curr_m <= band_max_m and curr_m <= target_m:
        return True
    return False
