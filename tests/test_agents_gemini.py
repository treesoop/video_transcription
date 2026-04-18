from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agents.gemini import GeminiAgent


def test_gemini_describe_frame_mentions_file_with_at_symbol(tmp_path):
    img = tmp_path / "frame.png"
    img.write_bytes(b"")
    agent = GeminiAgent()
    fake = MagicMock(returncode=0, stdout="Scene description.\n", stderr="")
    with patch("src.agents.gemini.subprocess.run", return_value=fake) as run:
        out = agent.describe_frame(img, "ctx", "sys prompt")
    assert out == "Scene description."
    cmd = run.call_args[0][0]
    assert cmd[0] == "gemini"
    assert "-p" in cmd
    prompt_idx = cmd.index("-p") + 1
    assert f"@{img.resolve()}" in cmd[prompt_idx]
    assert "ctx" in cmd[prompt_idx]
    assert "sys prompt" in cmd[prompt_idx]
