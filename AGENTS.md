This file provides guidance to AI agents when working with code in this repository.

User-facing run steps are in [`HOW_TO_RUN.md`](./HOW_TO_RUN.md). The project overview is [`README.md`](./README.md).

## What this repo is

Fault injection, recovery datagen, and eval recording on a pinned LeRobot commit. The LeRobot library is a dependency. It is not in this tree.

## Setup

```bash
uv sync --locked
```

## Commands

```bash
uv run pytest tests -q --maxfail=10
pre-commit run --all-files
```

## Layout

- `src/fault_system/` — faults, wrappers, eval CLI, recording, datagen
- `packages/lerobot_env_libero_overlay/` — LIBERO scene plugin (`--env.type=libero_overlay`)
- `tests/faults/` — unit tests. Overlay and recording tests live under `tests/envs/` and `tests/scripts/`
- `examples/faults/`, `examples/libero_overlays/` — run scripts and scene YAML
- `scripts/run_fault_smoke.sh` — one-episode SmolVLA smoke
- `docs/source/fault_injection.mdx` — fault eval notes

## Notes

- New source files use `Copyright 2026 Gangelia`. Do not rewrite Hugging Face headers on files you only edit.
- Do not add fault flags, overlay fields, or recording arguments to the pinned LeRobot classes. See `.cursor/rules/fork-boundary.mdc`.
- The overlay distribution must keep the `lerobot_env_` prefix so upstream plugin discovery imports it.
- Run Python with `uv run`.
