from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agents.claude import ClaudeAgent


def test_claude_describe_frame_passes_read_tool_prompt(tmp_path):
    img = tmp_path / "frame_00001.png"
    img.write_bytes(b"")
    agent = ClaudeAgent()

    fake = MagicMock(returncode=0, stdout="A slide titled Q1 Review.\n", stderr="")
    with patch("src.agents.claude.subprocess.run", return_value=fake) as run:
        out = agent.describe_frame(
            image_path=img,
            audio_context="Speaker 1: Let's start Q1 review.",
            system_prompt="You describe frames concisely.",
        )
    assert out == "A slide titled Q1 Review."
    cmd = run.call_args[0][0]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    # The prompt string should include the absolute image path (for Read tool).
    prompt_idx = cmd.index("-p") + 1
    assert str(img) in cmd[prompt_idx]
    assert "Read" in cmd[prompt_idx]
    # System prompt appended via --append-system-prompt
    assert "--append-system-prompt" in cmd


def test_claude_retries_once_on_failure(tmp_path):
    import subprocess as sp
    img = tmp_path / "frame.png"
    img.write_bytes(b"")
    agent = ClaudeAgent()

    calls = {"n": 0}
    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sp.CalledProcessError(1, a[0], stderr=b"boom")
        return MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("src.agents.claude.subprocess.run", side_effect=flaky):
        out = agent.describe_frame(img, "", "")
    assert out == "ok"
    assert calls["n"] == 2
