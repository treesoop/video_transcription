from __future__ import annotations

import subprocess
from pathlib import Path

from .base import BaseAgent


class CodexAgent(BaseAgent):
    name = "codex"

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
            "Describe the attached image. Output only the description text, no preamble."
        )
        cmd = [
            "codex", "exec",
            "-i", str(abs_path),
            prompt,
        ]
        return self._run(cmd, timeout_sec)

    @staticmethod
    def _run(cmd: list[str], timeout_sec: int) -> str:
        """Retry-once runner. Mirrors ClaudeAgent._run_with_retry but calls
        subprocess.run from this module's namespace so tests patching
        src.agents.codex.subprocess.run intercept correctly."""
        from .base import AgentInvocationError
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
        raise AgentInvocationError("codex", cmd, last_err)
