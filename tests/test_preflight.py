from unittest.mock import patch
import pytest

from src.preflight import check_dependencies, DependencyError


def _which(available):
    def fn(name):
        return f"/usr/local/bin/{name}" if name in available else None
    return fn


def test_returns_ok_when_all_present():
    available = {"ffmpeg", "mlx_whisper", "claude"}
    with patch("src.preflight.shutil.which", side_effect=_which(available)):
        missing = check_dependencies(agent="claude", diarize=False)
    assert missing == []


def test_reports_missing_ffmpeg():
    available = {"mlx_whisper", "claude"}
    with patch("src.preflight.shutil.which", side_effect=_which(available)):
        missing = check_dependencies(agent="claude", diarize=False)
    names = [m["name"] for m in missing]
    assert "ffmpeg" in names


def test_reports_missing_agent_only():
    available = {"ffmpeg", "mlx_whisper"}
    with patch("src.preflight.shutil.which", side_effect=_which(available)):
        missing = check_dependencies(agent="codex", diarize=False)
    names = [m["name"] for m in missing]
    assert names == ["codex"]


def test_diarize_requires_whispermlx():
    available = {"ffmpeg", "mlx_whisper", "claude"}
    with patch("src.preflight.shutil.which", side_effect=_which(available)):
        missing = check_dependencies(agent="claude", diarize=True)
    names = [m["name"] for m in missing]
    assert "whispermlx" in names


def test_unknown_agent_raises():
    with pytest.raises(DependencyError):
        check_dependencies(agent="bogus", diarize=False)
