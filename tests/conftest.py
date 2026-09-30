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

"""Session hooks for the fault-system tests."""

import functools
import os

import draccus.wrappers.docstring as _draccus_docstring

# Config parses re-read dataclass source for help text. Memoize it for the session.
_draccus_docstring.get_attribute_docstring = functools.cache(_draccus_docstring.get_attribute_docstring)


def pytest_collection_finish() -> None:
    """Print the device the session is using."""
    device = os.environ.get("DEVICE", "cpu")
    print(f"\nTesting with DEVICE={device!r}")
