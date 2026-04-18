from pathlib import Path
import pytest

from src.audio import extract_audio_from_video

TEST_VIDEO = Path("/Users/dion/Downloads/ep02-final.mp4")


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test video not present")
def test_extract_audio_from_video_produces_wav(tmp_path):
    out = tmp_path / "audio.wav"
    result = extract_audio_from_video(str(TEST_VIDEO), str(out))
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0
