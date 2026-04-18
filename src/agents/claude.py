from __future__ import annotations

import subprocess
from pathlib import Path

from .base import BaseAgent


class ClaudeAgent(BaseAgent):
    name = "claude"

    def describe_frame(
        self,
        image_path: Path,
        audio_context: str,
        system_prompt: str,
        timeout_sec: int = 120,
    ) -> str:
        abs_path = Path(image_path).resolve()
        prompt = (
            "You are given a still frame from a video. "
            "Use the Read tool to open the image file, then follow the instructions below.\n\n"
            f"AUDIO CONTEXT (transcript around this frame):\n{audio_context or '(none)'}\n\n"
            f"IMAGE FILE: {abs_path}\n\n"
            "Output only the description text. No preamble, no tool trace."
        )
        cmd = [
            "claude",
            "-p", prompt,
            "--append-system-prompt", system_prompt,
            "--permission-mode", "bypassPermissions",
        ]
        return self._run_with_retry(cmd, timeout_sec)

    @staticmethod
    def _run_with_retry(cmd: list[str], timeout_sec: int) -> str:
        from .base import AgentInvocationError
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=True,
                )
                return proc.stdout.strip()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                last_err = e
        assert last_err is not None
        raise AgentInvocationError("claude", cmd, last_err)
