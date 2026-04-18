from pathlib import Path
from .base import BaseAgent


class ClaudeAgent(BaseAgent):
    name = "claude"

    def describe_frame(self, image_path: Path, audio_context: str,
                       system_prompt: str, timeout_sec: int = 120) -> str:
        raise NotImplementedError
