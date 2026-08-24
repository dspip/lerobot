# Copyright 2026 Gangelia. All rights reserved.
"""Monkeypatch hooks for LIBERO env construction (zero-fork)."""

from __future__ import annotations

from typing import Any

from lerobot.faults.recovery.fps import configure_libero_control_freq, get_intended_libero_control_freq

_ORIGINAL_ENSURE_ENV: Any = None
_HOOK_INSTALLED = False


def install_libero_control_freq_hook(control_freq: int) -> bool:
    """Patch ``LiberoEnv._ensure_env`` to pass ``control_freq`` into ``OffScreenRenderEnv``.

    Returns ``True`` when the hook was installed, ``False`` when LeRobot/LIBERO
    imports are unavailable (callers should fall back to stride subsampling).

    Warning: For SmolVLA on LIBERO, pass ``DEFAULT_LIBERO_CONTROL_FREQ`` (20).
    Forcing ``control_freq=10`` breaks pick-and-place (empirically verified).
    """
    configure_libero_control_freq(control_freq)
    global _ORIGINAL_ENSURE_ENV, _HOOK_INSTALLED

    if _HOOK_INSTALLED:
        return True

    try:
        from lerobot.envs.libero import LiberoEnv
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        return False

    _ORIGINAL_ENSURE_ENV = LiberoEnv._ensure_env

    def _ensure_env_with_control_freq(self: Any) -> None:
        if self._env is not None:
            return
        kwargs: dict[str, Any] = {
            "bddl_file_name": self._task_bddl_file,
            "camera_heights": self.observation_height,
            "camera_widths": self.observation_width,
        }
        freq = get_intended_libero_control_freq()
        if freq is not None:
            kwargs["control_freq"] = freq
        env = OffScreenRenderEnv(**kwargs)
        env.reset()
        self._env = env

    LiberoEnv._ensure_env = _ensure_env_with_control_freq  # type: ignore[method-assign]
    _HOOK_INSTALLED = True
    return True


def uninstall_libero_control_freq_hook() -> None:
    """Restore the original ``LiberoEnv._ensure_env`` if patched."""
    global _ORIGINAL_ENSURE_ENV, _HOOK_INSTALLED
    if not _HOOK_INSTALLED or _ORIGINAL_ENSURE_ENV is None:
        return
    try:
        from lerobot.envs.libero import LiberoEnv

        LiberoEnv._ensure_env = _ORIGINAL_ENSURE_ENV  # type: ignore[method-assign]
    except ImportError:
        pass
    _ORIGINAL_ENSURE_ENV = None
    _HOOK_INSTALLED = False
