"""Frame extraction from video files via ffmpeg."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Literal, TypedDict


class Frame(TypedDict):
    timestamp: float
    image_path: Path


Mode = Literal["interval", "scene", "hybrid"]


def _video_duration_sec(video_path: Path) -> float:
    """Probe duration with ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def _extract_at_interval(video_path: Path, out_dir: Path, interval: int) -> None:
    """Run ffmpeg to dump one frame every `interval` seconds into out_dir."""
    pattern = str(out_dir / "frame_%05d.png")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps=1/{interval}",
        "-vsync", "vfr",
        pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


_PTS_TIME_RE = re.compile(rb"pts_time:([\d.]+)")


def _extract_scenes(
    video_path: Path, out_dir: Path, threshold: float
) -> list[float]:
    """Run ffmpeg with scene filter + showinfo. Returns list of timestamps (seconds)."""
    pattern = str(out_dir / "frame_%05d.png")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        pattern,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    timestamps = [float(m.group(1)) for m in _PTS_TIME_RE.finditer(proc.stderr)]
    return timestamps


def extract_frames(
    video_path: Path,
    out_dir: Path,
    mode: Mode,
    interval: int,
    min_interval: int,
    max_interval: int,
    max_frames: int,
    scene_threshold: float,
) -> list[Frame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "interval":
        duration = _video_duration_sec(video_path)
        _extract_at_interval(video_path, out_dir, interval)
        timestamps = []
        t = 0.0
        while t < duration:
            timestamps.append(t)
            t += interval
    elif mode == "scene":
        timestamps = _extract_scenes(video_path, out_dir, scene_threshold)
    else:
        raise NotImplementedError(f"mode={mode!r} not implemented yet")

    images = sorted(out_dir.glob("frame_*.png"))
    n = min(len(timestamps), len(images))
    return [{"timestamp": timestamps[i], "image_path": images[i]} for i in range(n)]
