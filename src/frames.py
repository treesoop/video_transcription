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


def _apply_hybrid_rules(
    scene_times: list[float],
    duration: float,
    min_interval: int,
    max_interval: int,
    max_frames: int,
) -> list[float]:
    """
    1. Ensure the list starts at 0.0 (anchor the opening frame).
    2. Drop any candidate closer than min_interval to the previously kept one.
    3. For any gap > max_interval between consecutive kept timestamps,
       insert evenly-spaced fillers so each sub-gap <= max_interval AND
       each sub-gap >= min_interval (caller guarantees min_interval <= max_interval).
    4. If total > max_frames, evenly downsample while preserving first & last.
    """
    # Step 1: prepend 0.0 if missing
    candidates = sorted({0.0, *scene_times})

    # Step 2: min_interval merge
    merged: list[float] = []
    for t in candidates:
        if not merged or (t - merged[-1]) >= min_interval:
            merged.append(t)

    # Step 3: max_interval gap-fill
    filled: list[float] = []
    for i, t in enumerate(merged):
        filled.append(t)
        if i + 1 < len(merged):
            nxt = merged[i + 1]
            gap = nxt - t
            if gap > max_interval:
                # Number of filler frames so each sub-gap <= max_interval
                n_fillers = int(gap // max_interval)
                if gap % max_interval == 0:
                    n_fillers -= 1
                step = gap / (n_fillers + 1)
                for k in range(1, n_fillers + 1):
                    filled.append(t + step * k)
    # Handle tail gap from last merged to duration (opening-to-end coverage)
    if filled and (duration - filled[-1]) > max_interval:
        gap = duration - filled[-1]
        n_fillers = int(gap // max_interval)
        if gap % max_interval == 0:
            n_fillers -= 1
        step = gap / (n_fillers + 1)
        last = filled[-1]
        for k in range(1, n_fillers + 1):
            filled.append(last + step * k)

    # Step 4: downsample to max_frames preserving endpoints
    if len(filled) > max_frames:
        step = (len(filled) - 1) / (max_frames - 1)
        idx = [round(i * step) for i in range(max_frames)]
        # dedup (rounding can collide)
        seen = set()
        downs = []
        for i in idx:
            if i not in seen:
                seen.add(i)
                downs.append(filled[i])
        filled = downs

    return filled


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
        images = sorted(out_dir.glob("frame_*.png"))
        n = min(len(timestamps), len(images))
        return [{"timestamp": timestamps[i], "image_path": images[i]} for i in range(n)]

    if mode == "scene":
        timestamps = _extract_scenes(video_path, out_dir, scene_threshold)
        images = sorted(out_dir.glob("frame_*.png"))
        n = min(len(timestamps), len(images))
        return [{"timestamp": timestamps[i], "image_path": images[i]} for i in range(n)]

    if mode == "hybrid":
        duration = _video_duration_sec(video_path)
        scene_ts = _extract_scenes(video_path, out_dir, scene_threshold)
        # ffmpeg only emitted frames at scene points; clean up and re-extract at final timestamps.
        for p in out_dir.glob("frame_*.png"):
            p.unlink()
        final_ts = _apply_hybrid_rules(
            scene_times=scene_ts,
            duration=duration,
            min_interval=min_interval,
            max_interval=max_interval,
            max_frames=max_frames,
        )
        _extract_at_timestamps(video_path, out_dir, final_ts)
        images = sorted(out_dir.glob("frame_*.png"))
        n = min(len(final_ts), len(images))
        return [{"timestamp": final_ts[i], "image_path": images[i]} for i in range(n)]

    raise ValueError(f"unknown mode {mode!r}")


def _extract_at_timestamps(
    video_path: Path, out_dir: Path, timestamps: list[float]
) -> None:
    """One ffmpeg invocation per timestamp — simpler and robust for <= few hundred frames."""
    for idx, ts in enumerate(timestamps, start=1):
        out = out_dir / f"frame_{idx:05d}.png"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
