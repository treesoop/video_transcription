"""Abstract agent interface. All adapters implement describe_frame."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def describe_frame(
        self,
        image_path: Path,
        audio_context: str,
        system_prompt: str,
        timeout_sec: int = 120,
    ) -> str:
        """Return a text description of the given frame."""
        ...


class AgentInvocationError(RuntimeError):
    """Raised when an agent's subprocess fails after all retry attempts."""

    def __init__(self, agent_name: str, cmd: list[str], cause: Exception):
        stderr = getattr(cause, "stderr", None)
        if isinstance(stderr, bytes):
            stderr_str = stderr.decode(errors="replace").strip()
        elif isinstance(stderr, str):
            stderr_str = stderr.strip()
        else:
            stderr_str = str(cause)
        super().__init__(
            f"{agent_name} failed after 2 attempts (cmd={cmd[0]} ...): {stderr_str}"
        )
        self.agent_name = agent_name
        self.cause = cause
