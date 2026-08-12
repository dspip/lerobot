#!/usr/bin/env python3
"""HTTP action server for smolVLA on LIBERO (LeRobot venv).

Run from the LeRobot repo (Python 3.13 + smolvla extra). GraspGenX connects
via ``SmolVLAClient`` — no cross-venv imports.

Example:
  cd ~/Projects/lerobot
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
    uv run python scripts/smolvla_action_server.py \\
      --policy.path=lerobot/smolvla_libero \\
      --policy.device=cuda \\
      --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import threading
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("smolvla_action_server")


def _decode_png_b64(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _as_batched(x: Any, *, trailing: tuple[int, ...]) -> np.ndarray:
    """Ensure array is writable with leading batch dim for SyncVectorEnv-style obs."""
    arr = np.array(x, dtype=np.float64, copy=True)
    expected_ndim = 1 + len(trailing)
    if arr.shape == trailing:
        arr = arr.reshape((1, *trailing))
    elif arr.ndim == expected_ndim and arr.shape[0] == 1 and arr.shape[1:] == trailing:
        pass
    else:
        raise ValueError(
            f"Expected shape {trailing} or {(1, *trailing)}, got {arr.shape}"
        )
    return arr


def _build_observation(payload: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a single-env LIBERO observation dict for preprocessors.

    ``LiberoProcessorStep._quat2axisangle`` requires batched quaternions ``(B, 4)``.
    VectorEnv rollouts already batch; this server must add the batch dim itself.
    """
    # np.array(..., copy=True) so torch.from_numpy gets a writable buffer.
    obs: dict[str, Any] = {
        "pixels": {
            "camera1": np.array(_decode_png_b64(payload["agentview_b64"]), copy=True),
            "camera2": np.array(_decode_png_b64(payload["wrist_b64"]), copy=True),
        },
    }
    robot_state = payload.get("robot_state")
    if robot_state is not None:
        rs = robot_state
        eef = rs["eef"]
        obs["robot_state"] = {
            "eef": {
                "pos": _as_batched(eef["pos"], trailing=(3,)),
                "quat": _as_batched(eef["quat"], trailing=(4,)),
                "mat": _as_batched(eef["mat"], trailing=(3, 3)),
            },
            "gripper": {
                "qpos": _as_batched(rs["gripper"]["qpos"], trailing=(2,)),
                "qvel": _as_batched(rs["gripper"]["qvel"], trailing=(2,)),
            },
            "joints": {
                "pos": _as_batched(rs["joints"]["pos"], trailing=(7,)),
                "vel": _as_batched(rs["joints"]["vel"], trailing=(7,)),
            },
        }
    return obs


class SmolVLAService:
    """Load policy + processors once; serve actions from synthetic observations."""

    def __init__(
        self,
        *,
        policy_path: str,
        device: str,
        empty_cameras: int,
        use_amp: bool,
        suite: str,
        camera_height: int,
        camera_width: int,
    ) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.utils.constants import ACTION
        from lerobot.utils.device_utils import get_safe_torch_device

        self._preprocess_observation = preprocess_observation
        self._ACTION = ACTION

        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = Path(policy_path)
        policy_cfg.device = device
        policy_cfg.empty_cameras = empty_cameras
        policy_cfg.use_amp = use_amp

        env_cfg = LiberoEnv(
            task=suite,
            obs_type="pixels_agent_pos",
            observation_height=camera_height,
            observation_width=camera_width,
            camera_name_mapping={
                "agentview_image": "camera1",
                "robot0_eye_in_hand_image": "camera2",
            },
        )

        self.device = get_safe_torch_device(device, log=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        log.info("Loading policy from %s on %s", policy_path, self.device)
        self.policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        self.policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": str(self.policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(policy_cfg.pretrained_path),
            preprocessor_overrides=preprocessor_overrides,
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=policy_cfg
        )
        self.use_amp = use_amp
        self._lock = threading.Lock()
        log.info("smolVLA action server ready (empty_cameras=%d)", empty_cameras)

    def reset(self) -> None:
        with self._lock:
            self.policy.reset()
        log.info("Policy action queue reset")

    @torch.no_grad()
    def act(self, payload: dict[str, Any]) -> list[float]:
        task = payload.get("task", "")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Missing or empty 'task' string in /act payload")

        obs = _build_observation(payload)
        observation = self._preprocess_observation(obs)
        observation["task"] = [task]

        with self._lock:
            ctx = torch.autocast(device_type=self.device.type) if self.use_amp else nullcontext()
            with ctx:
                observation = self.env_preprocessor(observation)
                observation = self.preprocessor(observation)
                action = self.policy.select_action(observation)
                action = self.postprocessor(action)
                action_transition = {self._ACTION: action}
                action_transition = self.env_postprocessor(action_transition)
                action = action_transition[self._ACTION]

        action_np = action.to("cpu").numpy()
        if action_np.ndim == 2:
            action_np = action_np[0]
        action_np = np.asarray(action_np, dtype=np.float64).reshape(-1)
        if action_np.shape[0] != 7:
            raise ValueError(f"Policy returned action shape {action_np.shape}, expected (7,)")
        return action_np.tolist()


def make_handler(service: SmolVLAService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # noqa: D401
            log.debug(fmt, *args)

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path == "/reset":
                    service.reset()
                    self._send_json(200, {"status": "reset"})
                    return
                if self.path == "/act":
                    payload = self._read_json()
                    action = service.act(payload)
                    self._send_json(200, {"action": action})
                    return
                self._send_json(404, {"error": "not found"})
            except Exception as exc:
                log.exception("Request failed: %s", self.path)
                self._send_json(500, {"error": str(exc)})

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--policy.path",
        dest="policy_path",
        default="lerobot/smolvla_libero",
        help="HuggingFace repo id or local path for smolVLA weights.",
    )
    ap.add_argument(
        "--policy.device",
        dest="policy_device",
        default="cuda",
        help="Torch device for inference (cuda or cpu).",
    )
    ap.add_argument(
        "--policy.empty_cameras",
        dest="empty_cameras",
        type=int,
        default=1,
        help="Padded camera slots expected by smolvla_libero.",
    )
    ap.add_argument(
        "--policy.use_amp",
        dest="use_amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use autocast during inference.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--env.task",
        dest="env_task",
        default="libero_object",
        help="LIBERO suite name for env feature shapes (default libero_object).",
    )
    ap.add_argument("--camera-height", type=int, default=256)
    ap.add_argument("--camera-width", type=int, default=256)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    service = SmolVLAService(
        policy_path=args.policy_path,
        device=args.policy_device,
        empty_cameras=args.empty_cameras,
        use_amp=args.use_amp,
        suite=args.env_task,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )
    handler = make_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    log.info("Listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
