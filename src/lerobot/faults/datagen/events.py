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

"""Structured event fan-out for JSONL, stdout, and the live viewer."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class LogEvent:
    ts: float
    step: int | None
    kind: str
    message: str
    payload: dict[str, Any]


class DatagenEventLog:
    """Write each event to persistent and live sinks in emission order."""

    def __init__(self, jsonl_path: Path | None = None, *, stdout: bool = True) -> None:
        self._stdout = bool(stdout)
        self._listeners: list[Callable[[LogEvent], None]] = []
        self._file: TextIO | None = None
        if jsonl_path is not None:
            path = Path(jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")

    def add_listener(self, fn: Callable[[LogEvent], None]) -> None:
        self._listeners.append(fn)

    def emit(
        self,
        kind: str,
        message: str,
        *,
        step: int | None = None,
        **payload: Any,
    ) -> LogEvent:
        event = LogEvent(
            ts=time.time(),
            step=step,
            kind=str(kind),
            message=str(message),
            payload=payload,
        )
        if self._file is not None:
            self._file.write(json.dumps(asdict(event), default=_json_default) + "\n")
            self._file.flush()
        if self._stdout:
            suffix = f" step={step}" if step is not None else ""
            print(f"[{event.kind}] {event.message}{suffix}", flush=True)
        for listener in tuple(self._listeners):
            try:
                listener(event)
            except Exception:
                continue
        return event

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def should_autoscroll(yview_hi: float) -> bool:
    """Keep following new events only while the user is at the log bottom."""
    return float(yview_hi) >= 0.999


def format_event_line(event: LogEvent) -> str:
    """Format one event identically for the viewer's scrollable log."""
    clock = time.strftime("%H:%M:%S", time.localtime(event.ts))
    step = "-" if event.step is None else str(event.step)
    payload = f" {json.dumps(event.payload, default=_json_default, sort_keys=True)}"
    if not event.payload:
        payload = ""
    return f"{clock} step={step} [{event.kind}] {event.message}{payload}"


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")
