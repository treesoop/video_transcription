from pathlib import Path
from .base import BaseAgent


class CodexAgent(BaseAgent):
    name = "codex"

    def describe_frame(self, image_path: Path, audio_context: str,
                       system_prompt: str, timeout_sec: int = 120) -> str:
        raise NotImplementedError
