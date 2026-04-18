import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.frames import extract_frames


def _fake_ffmpeg_produces(paths: list[Path]):
    """Return a side_effect that creates empty files at the given paths."""
    def run(cmd, *a, **kw):
        for p in paths:
            p.write_bytes(b"")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")
    return run


def test_interval_mode_emits_one_frame_per_interval(tmp_path):
    out_dir = tmp_path / "frames"
    out_dir.mkdir()
    # 90s video at interval=30 => frames at t=0, 30, 60 (3 files)
    produced = [out_dir / f"frame_{i:05d}.png" for i in (1, 2, 3)]

    with patch("src.frames.subprocess.run", side_effect=_fake_ffmpeg_produces(produced)), \
         patch("src.frames._video_duration_sec", return_value=90.0):
        frames = extract_frames(
            video_path=tmp_path / "fake.mp4",
            out_dir=out_dir,
            mode="interval",
            interval=30,
            min_interval=10,
            max_interval=60,
            max_frames=200,
            scene_threshold=0.3,
        )

    assert [f["timestamp"] for f in frames] == [0.0, 30.0, 60.0]
    assert all(f["image_path"].exists() for f in frames)


FAKE_SHOWINFO_STDERR = b"""
[Parsed_showinfo_1 @ 0x600001] n:   0 pts:    1024 pts_time:12.5   pos:  1024
[Parsed_showinfo_1 @ 0x600002] n:   1 pts:    4096 pts_time:47.8   pos:  4096
[Parsed_showinfo_1 @ 0x600003] n:   2 pts:   10240 pts_time:119.25 pos: 10240
"""


def test_scene_mode_parses_timestamps_from_showinfo(tmp_path):
    out_dir = tmp_path / "frames"
    out_dir.mkdir()
    produced = [out_dir / f"frame_{i:05d}.png" for i in (1, 2, 3)]

    def fake_run(cmd, *a, **kw):
        for p in produced:
            p.write_bytes(b"")
        result = MagicMock(returncode=0, stdout=b"")
        result.stderr = FAKE_SHOWINFO_STDERR
        return result

    with patch("src.frames.subprocess.run", side_effect=fake_run):
        frames = extract_frames(
            video_path=tmp_path / "fake.mp4",
            out_dir=out_dir,
            mode="scene",
            interval=30,
            min_interval=10,
            max_interval=60,
            max_frames=200,
            scene_threshold=0.3,
        )

    assert [f["timestamp"] for f in frames] == [12.5, 47.8, 119.25]
