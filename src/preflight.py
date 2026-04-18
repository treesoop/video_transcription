"""Dependency preflight check for video_transcription."""
from __future__ import annotations

import os
import shutil
from typing import TypedDict


class DependencyError(Exception):
    pass


class Missing(TypedDict):
    name: str
    install: str


AGENT_INSTALL = {
    "claude": "curl -fsSL https://claude.ai/install.sh | bash   # then run `claude` to log in",
    "codex": "npm install -g @openai/codex   # then `codex login`",
    "gemini": "npm install -g @google/gemini-cli   # then run `gemini` to log in",
}

BASE_INSTALL = {
    "ffmpeg": "brew install ffmpeg",
    "mlx_whisper": "pip3 install mlx-whisper",
    "whispermlx": "uv tool install whispermlx --python 3.12",
}


def _which_mlx_whisper() -> str | None:
    """mlx_whisper may live in ~/Library/Python/3.9/bin (non-PATH by default)."""
    known = os.path.expanduser("~/Library/Python/3.9/bin/mlx_whisper")
    if os.path.isfile(known):
        return known
    return shutil.which("mlx_whisper")


def check_dependencies(agent: str, diarize: bool) -> list[Missing]:
    if agent not in AGENT_INSTALL:
        raise DependencyError(
            f"Unknown agent '{agent}'. Choose from: {', '.join(AGENT_INSTALL)}"
        )

    missing: list[Missing] = []

    if not shutil.which("ffmpeg"):
        missing.append({"name": "ffmpeg", "install": BASE_INSTALL["ffmpeg"]})

    if not _which_mlx_whisper():
        missing.append({"name": "mlx_whisper", "install": BASE_INSTALL["mlx_whisper"]})

    if diarize and not shutil.which("whispermlx"):
        missing.append({"name": "whispermlx", "install": BASE_INSTALL["whispermlx"]})

    if not shutil.which(agent):
        missing.append({"name": agent, "install": AGENT_INSTALL[agent]})

    return missing


def print_report(missing: list[Missing]) -> None:
    if not missing:
        print("[preflight] All dependencies present.")
        return
    print("[preflight] Missing dependencies:")
    for m in missing:
        print(f"  - {m['name']}")
        print(f"      install: {m['install']}")
