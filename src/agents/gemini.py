from __future__ import annotations

import subprocess
from pathlib import Path

from .base import BaseAgent


class GeminiAgent(BaseAgent):
    name = "gemini"

    def describe_frame(
        self,
        image_path: Path,
        audio_context: str,
        system_prompt: str,
        timeout_sec: int = 120,
    ) -> str:
        abs_path = Path(image_path).resolve()
        prompt = (
            f"{system_prompt}\n\n"
            f"AUDIO CONTEXT (transcript around this frame):\n{audio_context or '(none)'}\n\n"
            f"Describe this frame concisely. Output only the description text.\n\n"
            f"@{abs_path}"
        )
        cmd = ["gemini", "-p", prompt]
        return self._run(cmd, timeout_sec)

    @staticmethod
    def _run(cmd: list[str], timeout_sec: int) -> str:
        """Retry-once runner. Mirrors ClaudeAgent._run_with_retry / CodexAgent._run
        but uses this module's subprocess so tests patching
        src.agents.gemini.subprocess.run intercept correctly."""
        last_err: Exception | None = None
        for _ in range(2):
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=timeout_sec, check=True,
                )
                return proc.stdout.strip()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                last_err = e
        assert last_err is not None
        raise last_err
