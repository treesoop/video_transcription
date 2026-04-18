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
