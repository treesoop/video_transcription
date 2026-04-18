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


from src.frames import _apply_hybrid_rules


def test_hybrid_merges_timestamps_closer_than_min_interval():
    # Scene candidates 0, 5, 12, 18, 40  with min_interval=10
    # Keep 0, 12 (5 drops, 18 drops because 12 just kept), 40
    result = _apply_hybrid_rules(
        scene_times=[0.0, 5.0, 12.0, 18.0, 40.0],
        duration=60.0,
        min_interval=10,
        max_interval=60,
        max_frames=200,
    )
    assert result == [0.0, 12.0, 40.0]


def test_hybrid_fills_gaps_larger_than_max_interval():
    # Only two scene hits at 0 and 200, 180s gap > max_interval=60
    # Insert filler frames every ~60s: 60, 120, 180 (rough midpoints)
    result = _apply_hybrid_rules(
        scene_times=[0.0, 200.0],
        duration=210.0,
        min_interval=10,
        max_interval=60,
        max_frames=200,
    )
    # Expect filler timestamps strictly between 0 and 200, each spaced <= 60s
    assert result[0] == 0.0
    assert result[-1] == 200.0
    # Every consecutive pair respects max_interval
    for a, b in zip(result, result[1:]):
        assert b - a <= 60 + 1e-6
    # And min_interval preserved
    for a, b in zip(result, result[1:]):
        assert b - a >= 10 - 1e-6


def test_hybrid_downsamples_to_max_frames():
    # 500 scene hits at 0..499 seconds, min_interval=1 so nothing merges
    scene_times = [float(i) for i in range(500)]
    result = _apply_hybrid_rules(
        scene_times=scene_times,
        duration=500.0,
        min_interval=1,
        max_interval=1000,
        max_frames=50,
    )
    assert len(result) == 50
    # First and last preserved
    assert result[0] == 0.0
    assert result[-1] == 499.0


def test_hybrid_pads_empty_scene_list():
    # No scenes detected at all: fall back to max_interval cadence
    result = _apply_hybrid_rules(
        scene_times=[],
        duration=120.0,
        min_interval=10,
        max_interval=60,
        max_frames=200,
    )
    assert result[0] == 0.0
    for a, b in zip(result, result[1:]):
        assert b - a <= 60 + 1e-6


def test_hybrid_max_frames_one_returns_single_anchor():
    result = _apply_hybrid_rules(
        scene_times=[5.0, 10.0, 20.0],
        duration=30.0,
        min_interval=1,
        max_interval=60,
        max_frames=1,
    )
    assert result == [0.0]


def test_hybrid_max_frames_zero_returns_empty():
    result = _apply_hybrid_rules(
        scene_times=[5.0, 10.0],
        duration=30.0,
        min_interval=1,
        max_interval=60,
        max_frames=0,
    )
    assert result == []


def test_hybrid_drops_scene_times_beyond_duration():
    result = _apply_hybrid_rules(
        scene_times=[10.0, 150.0, 500.0],  # 150 and 500 both > duration=120
        duration=120.0,
        min_interval=1,
        max_interval=60,
        max_frames=200,
    )
    # 500 and 150 must not appear; 10 may appear if kept after merge
    assert all(t <= 120.0 for t in result)
