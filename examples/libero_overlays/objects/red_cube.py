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

"""Register a simple red cube for LIBERO overlay demos.

Importing this module registers ``red_cube`` in LIBERO's object registry without
overwriting any stock classes (e.g. ``TomatoSauce``).
"""

from __future__ import annotations

import os
import re

from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import register_object

_XML_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "red_cube", "red_cube.xml")
)


@register_object
class RedCube(MujocoXMLObject):
    def __init__(self, name: str = "red_cube", obj_name: str = "red_cube"):
        del obj_name  # kept for symmetry with LIBERO HopeObject constructors
        super().__init__(
            _XML_PATH,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}
