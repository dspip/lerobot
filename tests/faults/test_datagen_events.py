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

import json
from pathlib import Path

from lerobot.faults.datagen.events import (
    DatagenEventLog,
    LogEvent,
    format_event_line,
    should_autoscroll,
)


def test_emit_writes_jsonl_stdout_and_listeners(tmp_path: Path, capsys):
    path = tmp_path / "events.jsonl"
    seen = []
    log = DatagenEventLog(path, stdout=True)
    log.add_listener(seen.append)

    event = log.emit("drop", "injected", step=12, q=0.5)
    log.close()

    assert event.kind == "drop"
    assert event.step == 12
    assert seen == [event]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["message"] == "injected"
    assert payload["payload"]["q"] == 0.5
    assert "[drop] injected" in capsys.readouterr().out


def test_listener_failure_does_not_hide_event_from_other_sinks(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    seen = []
    log = DatagenEventLog(path, stdout=False)

    def broken_listener(_event):
        raise RuntimeError("viewer closed")

    log.add_listener(broken_listener)
    log.add_listener(seen.append)

    event = log.emit("layout_ok", "layout accepted")
    log.close()

    assert seen == [event]
    assert "layout accepted" in path.read_text(encoding="utf-8")


def test_should_autoscroll_only_when_view_is_at_bottom():
    assert should_autoscroll(1.0) is True
    assert should_autoscroll(0.999) is True
    assert should_autoscroll(0.5) is False


def test_format_event_line_includes_step_kind_message_and_payload():
    event = LogEvent(
        ts=1234.5,
        step=12,
        kind="drop",
        message="injected",
        payload={"q": 0.5},
    )

    line = format_event_line(event)

    assert "step=12" in line
    assert "[drop] injected" in line
    assert '"q": 0.5' in line
