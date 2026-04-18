from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agents.codex import CodexAgent


def test_codex_describe_frame_uses_exec_i_flag(tmp_path):
    img = tmp_path / "frame.png"
    img.write_bytes(b"")
    agent = CodexAgent()
    fake = MagicMock(returncode=0, stdout="Describing frame.\n", stderr="")
    with patch("src.agents.codex.subprocess.run", return_value=fake) as run:
        out = agent.describe_frame(img, "ctx", "sys prompt")
    assert out == "Describing frame."
    cmd = run.call_args[0][0]
    assert cmd[:2] == ["codex", "exec"]
    assert "-i" in cmd
    i_idx = cmd.index("-i")
    assert cmd[i_idx + 1] == str(img.resolve())
    # prompt is the final positional arg
    assert "ctx" in cmd[-1]
    assert "sys prompt" in cmd[-1]
