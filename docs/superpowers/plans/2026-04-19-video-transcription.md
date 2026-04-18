# Video Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that takes a video file, transcribes the audio with mlx_whisper, extracts representative frames, and lets a pluggable CLI agent (claude / codex / gemini) describe each frame, then merges everything into a timestamped Markdown + JSON transcript.

**Architecture:** Python 3.9+ CLI. Audio pipeline reuses `whisper_transcription/transcribe.py` (ported as `src/audio.py`). Frames extracted via ffmpeg with hybrid scene-detection + interval guardrails. Agent calls are subprocess-based adapters behind a common `BaseAgent` interface. Concurrent frame description with `ThreadPoolExecutor`. Output in Markdown (default) and/or JSON.

**Tech Stack:** Python (stdlib only for core), ffmpeg (external binary), mlx_whisper (external CLI), pytest for tests, subprocess-based agent adapters (`claude -p`, `codex exec`, `gemini -p`).

**Working directory:** `/Users/dion/potenlab/our_project/video_transcription`
**Test video:** `/Users/dion/Downloads/ep02-final.mp4` (5m49s, 1280x720, H.264+AAC)

---

## Task 1: Repo manual-fork + directory skeleton

Manual fork `treesoop/whisper_transcription` git history into this directory, create the new GitHub repo, and scaffold the initial folder structure. Preserve the design/plans docs that already exist here.

**Files:**
- Modify (incoming from fork): `transcribe.py`, `README.md`, `.gitignore` (fork defaults)
- Create: `src/__init__.py`, `src/agents/__init__.py`, `tests/__init__.py`, `prompts/.gitkeep`, `examples/.gitkeep`
- Create: `requirements.txt`, `pytest.ini`

- [ ] **Step 1: Stash existing docs outside the target dir**

```bash
cd /Users/dion/potenlab/our_project/video_transcription
mv docs /tmp/vt_docs_stash
ls -la  # expect: empty
```

Expected: only `.` and `..` remain.

- [ ] **Step 2: Clone the source repo into a temp location and pull .git + seed files**

```bash
git clone git@github-treesoop:treesoop/whisper_transcription.git /tmp/vt_fork_src
cp -r /tmp/vt_fork_src/.git /Users/dion/potenlab/our_project/video_transcription/
cp /tmp/vt_fork_src/transcribe.py /Users/dion/potenlab/our_project/video_transcription/
cp /tmp/vt_fork_src/README.md /Users/dion/potenlab/our_project/video_transcription/
# .gitignore may or may not exist in source — copy if present
[ -f /tmp/vt_fork_src/.gitignore ] && cp /tmp/vt_fork_src/.gitignore /Users/dion/potenlab/our_project/video_transcription/
cd /Users/dion/potenlab/our_project/video_transcription
git status
```

Expected: `git status` shows "On branch main" with clean tree, and `ls` shows `transcribe.py`, `README.md`, `.git/`.

- [ ] **Step 3: Restore the design docs**

```bash
mv /tmp/vt_docs_stash docs
git status
```

Expected: `docs/` appears as untracked.

- [ ] **Step 4: Point the remote at the NEW repo and create it on GitHub**

```bash
git remote -v
# should show origin = git@github-treesoop:treesoop/whisper_transcription.git
git remote set-url origin git@github-treesoop:treesoop/video_transcription.git
gh repo create treesoop/video_transcription --public --description "Video transcription: mlx_whisper audio + agent-based frame description" --source . --push=false
git remote -v
```

Expected: origin points to `treesoop/video_transcription.git`. `gh repo create` prints the new URL.

- [ ] **Step 5: Create skeleton directories + placeholder files**

```bash
mkdir -p src/agents tests prompts examples
touch src/__init__.py src/agents/__init__.py tests/__init__.py prompts/.gitkeep examples/.gitkeep
```

- [ ] **Step 6: Write `requirements.txt`**

Create `/Users/dion/potenlab/our_project/video_transcription/requirements.txt`:

```
mlx-whisper>=0.3
```

(ffmpeg and the various CLI agents are external binaries, not pip deps. `whispermlx` is a uv tool and doesn't belong in requirements.txt.)

- [ ] **Step 7: Write `pytest.ini`**

Create `/Users/dion/potenlab/our_project/video_transcription/pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v
```

- [ ] **Step 8: Update `.gitignore`**

Overwrite `/Users/dion/potenlab/our_project/video_transcription/.gitignore`:

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/
.DS_Store
# video/audio artifacts
*.mp4
*.m4a
*.wav
*.mp3
# tool output
out/
output/
tmp/
```

- [ ] **Step 9: Initial commit + push**

```bash
git add -A
git status
git commit -m "chore: scaffold video_transcription (fork of whisper_transcription)

Manual fork from treesoop/whisper_transcription, preserving git history.
Adds skeleton (src/, tests/, prompts/, examples/), requirements.txt,
pytest.ini, expanded .gitignore, and the design + plan docs."
git push -u origin main
```

Expected: push succeeds. Branch tracking `origin/main`.

---

## Task 2: Preflight dependency checker

A single module that verifies all binaries/CLIs needed for a given `--agent` choice are installed. Used by `--check` flag and at the start of every normal run.

**Files:**
- Create: `src/preflight.py`
- Create: `tests/test_preflight.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/dion/potenlab/our_project/video_transcription/tests/test_preflight.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /Users/dion/potenlab/our_project/video_transcription
python -m pytest tests/test_preflight.py -v
```

Expected: all 5 tests FAIL with `ModuleNotFoundError: No module named 'src.preflight'`.

- [ ] **Step 3: Implement `src/preflight.py`**

Create `/Users/dion/potenlab/our_project/video_transcription/src/preflight.py`:

```python
"""Dependency preflight check for video_transcription."""
from __future__ import annotations

import os
import shutil
from typing import TypedDict


class DependencyError(Exception):
    pass


class Missing(TypedDict):
    name: str
    install: str


AGENT_INSTALL = {
    "claude": "curl -fsSL https://claude.ai/install.sh | bash   # then run `claude` to log in",
    "codex": "npm install -g @openai/codex   # then `codex login`",
    "gemini": "npm install -g @google/gemini-cli   # then run `gemini` to log in",
}

BASE_INSTALL = {
    "ffmpeg": "brew install ffmpeg",
    "mlx_whisper": "pip3 install mlx-whisper",
    "whispermlx": "uv tool install whispermlx --python 3.12",
}


def _which_mlx_whisper() -> str | None:
    """mlx_whisper may live in ~/Library/Python/3.9/bin (non-PATH by default)."""
    known = os.path.expanduser("~/Library/Python/3.9/bin/mlx_whisper")
    if os.path.isfile(known):
        return known
    return shutil.which("mlx_whisper")


def check_dependencies(agent: str, diarize: bool) -> list[Missing]:
    if agent not in AGENT_INSTALL:
        raise DependencyError(
            f"Unknown agent '{agent}'. Choose from: {', '.join(AGENT_INSTALL)}"
        )

    missing: list[Missing] = []

    if not shutil.which("ffmpeg"):
        missing.append({"name": "ffmpeg", "install": BASE_INSTALL["ffmpeg"]})

    if not _which_mlx_whisper():
        missing.append({"name": "mlx_whisper", "install": BASE_INSTALL["mlx_whisper"]})

    if diarize and not shutil.which("whispermlx"):
        missing.append({"name": "whispermlx", "install": BASE_INSTALL["whispermlx"]})

    if not shutil.which(agent):
        missing.append({"name": agent, "install": AGENT_INSTALL[agent]})

    return missing


def print_report(missing: list[Missing]) -> None:
    if not missing:
        print("[preflight] All dependencies present.")
        return
    print("[preflight] Missing dependencies:")
    for m in missing:
        print(f"  - {m['name']}")
        print(f"      install: {m['install']}")
```

- [ ] **Step 4: Run tests to verify pass**

```bash
python -m pytest tests/test_preflight.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/preflight.py src/__init__.py tests/test_preflight.py tests/__init__.py
git commit -m "feat(preflight): dependency checker for ffmpeg, mlx_whisper, agent CLI"
```

---

## Task 3: Frame extraction — `interval` mode

Simplest of the three modes. `ffmpeg -vf fps=1/N` pulls one frame every N seconds. Establishes the file layout + timestamp calculation that later modes extend.

**Files:**
- Create: `src/frames.py`
- Create: `tests/test_frames.py`

- [ ] **Step 1: Write failing test**

Create `/Users/dion/potenlab/our_project/video_transcription/tests/test_frames.py`:

```python
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
```

- [ ] **Step 2: Run test to verify failure**

```bash
python -m pytest tests/test_frames.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.frames'`.

- [ ] **Step 3: Implement `src/frames.py` (interval mode only, for now)**

Create `/Users/dion/potenlab/our_project/video_transcription/src/frames.py`:

```python
"""Frame extraction from video files via ffmpeg."""
from __future__ import annotations

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
        # timestamps: 0, interval, 2*interval, ... < duration
        timestamps: list[float] = []
        t = 0.0
        while t < duration:
            timestamps.append(t)
            t += interval

        images = sorted(out_dir.glob("frame_*.png"))
        # Pair by index (ffmpeg order matches timestamps order)
        n = min(len(timestamps), len(images))
        return [{"timestamp": timestamps[i], "image_path": images[i]} for i in range(n)]

    raise NotImplementedError(f"mode={mode!r} not implemented yet")
```

- [ ] **Step 4: Run test to verify pass**

```bash
python -m pytest tests/test_frames.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/frames.py tests/test_frames.py
git commit -m "feat(frames): interval mode extraction via ffmpeg"
```

---

## Task 4: Frame extraction — `scene` mode

ffmpeg scene filter emits keyframes at visual discontinuities. We parse the `showinfo` stderr to recover timestamps.

**Files:**
- Modify: `src/frames.py`
- Modify: `tests/test_frames.py`

- [ ] **Step 1: Add failing test for scene mode**

Append to `/Users/dion/potenlab/our_project/video_transcription/tests/test_frames.py`:

```python
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
```

- [ ] **Step 2: Run test to verify failure**

```bash
python -m pytest tests/test_frames.py::test_scene_mode_parses_timestamps_from_showinfo -v
```

Expected: FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement scene mode**

Edit `/Users/dion/potenlab/our_project/video_transcription/src/frames.py`. Add new helpers and extend `extract_frames`:

```python
import re

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
```

And replace the body of `extract_frames` so it handles both modes:

```python
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
```

- [ ] **Step 4: Run all frame tests**

```bash
python -m pytest tests/test_frames.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/frames.py tests/test_frames.py
git commit -m "feat(frames): scene mode extraction via ffmpeg scene filter"
```

---

## Task 5: Frame extraction — `hybrid` mode

Hybrid is the default and the only mode with real logic: scene candidates first, then `min_interval` merge, then `max_interval` gap-fill, then `max_frames` downsample. Pure timestamp math — perfect for unit tests.

**Files:**
- Modify: `src/frames.py`
- Modify: `tests/test_frames.py`

- [ ] **Step 1: Write failing tests for each hybrid sub-behavior**

Append to `/Users/dion/potenlab/our_project/video_transcription/tests/test_frames.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

```bash
python -m pytest tests/test_frames.py -v -k hybrid
```

Expected: 4 tests FAIL (ImportError for `_apply_hybrid_rules`).

- [ ] **Step 3: Implement `_apply_hybrid_rules` and wire into `extract_frames`**

Edit `/Users/dion/potenlab/our_project/video_transcription/src/frames.py`. Add:

```python
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
    if scene_times and scene_times[-1] < duration:
        # Don't force an end anchor — the natural end is just "duration".
        pass

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
    # Handle tail gap from last merged to duration (if we want opening-to-end coverage)
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
```

And extend `extract_frames` to handle hybrid. Replace its body with:

```python
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
    duration = _video_duration_sec(video_path)

    if mode == "interval":
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
```

- [ ] **Step 4: Run all frame tests**

```bash
python -m pytest tests/test_frames.py -v
```

Expected: all tests PASS. Scene-mode and interval-mode tests continue to pass; hybrid tests all pass.

- [ ] **Step 5: Commit**

```bash
git add src/frames.py tests/test_frames.py
git commit -m "feat(frames): hybrid mode with min/max_interval guards and max_frames cap"
```

---

## Task 6: Port `transcribe.py` to `src/audio.py` + add `extract_audio_from_video`

Preserve git history via `git mv`, then add a helper that extracts the audio track of a video file to a wav using ffmpeg.

**Files:**
- Rename: `transcribe.py` → `src/audio.py`
- Modify: `src/audio.py` (add `extract_audio_from_video`, drop the old `main()` CLI section)
- Create: `tests/test_audio.py`

- [ ] **Step 1: Move with history**

```bash
cd /Users/dion/potenlab/our_project/video_transcription
git mv transcribe.py src/audio.py
git status
```

Expected: `renamed: transcribe.py -> src/audio.py`.

- [ ] **Step 2: Remove the CLI `main()` from the module (not needed; orchestration moves to `video_transcribe.py`)**

Edit `/Users/dion/potenlab/our_project/video_transcription/src/audio.py`:

- Delete the entire `def main():` function and the `if __name__ == "__main__":` block at the bottom.
- Delete the top `import argparse` and `import sys` if they become unused.
- Keep: `find_mlx_whisper`, `find_whispermlx`, `transcribe_mlx`, `diarize_whispermlx`, `assign_speakers`, `format_timestamp`, `write_output` (the last one we'll still expose for plain-text transcript output if someone wants it, but it's optional).

- [ ] **Step 3: Add `extract_audio_from_video`**

Append to `/Users/dion/potenlab/our_project/video_transcription/src/audio.py`:

```python
def extract_audio_from_video(video_path: str, out_wav: str) -> str:
    """Extract the audio track of a video into a 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
        out_wav,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_wav
```

- [ ] **Step 4: Write smoke test (only runs if test video is present)**

Create `/Users/dion/potenlab/our_project/video_transcription/tests/test_audio.py`:

```python
import os
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
```

- [ ] **Step 5: Run the smoke test**

```bash
python -m pytest tests/test_audio.py -v
```

Expected: 1 test PASSes (or SKIPs if test video missing).

- [ ] **Step 6: Commit**

```bash
git add src/audio.py tests/test_audio.py
git commit -m "refactor(audio): move transcribe.py -> src/audio.py, add video->wav extractor"
```

---

## Task 7: `BaseAgent` interface + adapter registry

One abstract class that all three adapters implement, plus a registry mapping agent names to classes for `video_transcribe.py` to import.

**Files:**
- Create: `src/agents/base.py`
- Modify: `src/agents/__init__.py`
- Create: `tests/test_agents_base.py`

- [ ] **Step 1: Write failing test for the registry contract**

Create `/Users/dion/potenlab/our_project/video_transcription/tests/test_agents_base.py`:

```python
import pytest

from src.agents import get_agent, AGENT_NAMES
from src.agents.base import BaseAgent


def test_agent_names_covers_three_expected_adapters():
    assert set(AGENT_NAMES) == {"claude", "codex", "gemini"}


def test_get_agent_returns_baseagent_subclass():
    for name in AGENT_NAMES:
        agent = get_agent(name)
        assert isinstance(agent, BaseAgent)


def test_get_agent_unknown_raises():
    with pytest.raises(KeyError):
        get_agent("bogus")
```

- [ ] **Step 2: Run test to verify failure**

```bash
python -m pytest tests/test_agents_base.py -v
```

Expected: FAIL (ImportError — `get_agent` / `AGENT_NAMES` don't exist).

- [ ] **Step 3: Implement `src/agents/base.py`**

Create `/Users/dion/potenlab/our_project/video_transcription/src/agents/base.py`:

```python
"""Abstract agent interface. All adapters implement describe_frame."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def describe_frame(
        self,
        image_path: Path,
        audio_context: str,
        system_prompt: str,
        timeout_sec: int = 120,
    ) -> str:
        """Return a text description of the given frame."""
        ...
```

- [ ] **Step 4: Update `src/agents/__init__.py`**

Overwrite `/Users/dion/potenlab/our_project/video_transcription/src/agents/__init__.py`:

```python
"""Registry of pluggable CLI agent adapters."""
from __future__ import annotations

from .base import BaseAgent
from .claude import ClaudeAgent
from .codex import CodexAgent
from .gemini import GeminiAgent

_REGISTRY: dict[str, type[BaseAgent]] = {
    "claude": ClaudeAgent,
    "codex": CodexAgent,
    "gemini": GeminiAgent,
}

AGENT_NAMES = tuple(_REGISTRY.keys())


def get_agent(name: str) -> BaseAgent:
    return _REGISTRY[name]()


__all__ = ["BaseAgent", "get_agent", "AGENT_NAMES"]
```

- [ ] **Step 5: Create stub adapter files (full impls come in Tasks 8-10)**

Create `/Users/dion/potenlab/our_project/video_transcription/src/agents/claude.py`:

```python
from pathlib import Path
from .base import BaseAgent


class ClaudeAgent(BaseAgent):
    name = "claude"

    def describe_frame(self, image_path: Path, audio_context: str,
                       system_prompt: str, timeout_sec: int = 120) -> str:
        raise NotImplementedError
```

Create `/Users/dion/potenlab/our_project/video_transcription/src/agents/codex.py`:

```python
from pathlib import Path
from .base import BaseAgent


class CodexAgent(BaseAgent):
    name = "codex"

    def describe_frame(self, image_path: Path, audio_context: str,
                       system_prompt: str, timeout_sec: int = 120) -> str:
        raise NotImplementedError
```

Create `/Users/dion/potenlab/our_project/video_transcription/src/agents/gemini.py`:

```python
from pathlib import Path
from .base import BaseAgent


class GeminiAgent(BaseAgent):
    name = "gemini"

    def describe_frame(self, image_path: Path, audio_context: str,
                       system_prompt: str, timeout_sec: int = 120) -> str:
        raise NotImplementedError
```

- [ ] **Step 6: Run test to verify pass**

```bash
python -m pytest tests/test_agents_base.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agents/ tests/test_agents_base.py
git commit -m "feat(agents): BaseAgent interface + registry + adapter stubs"
```

---

## Task 8: Claude adapter — `claude -p` + Read tool

Claude Code's `-p` mode has no image flag, but the Read tool (enabled by default in print mode) can load images as multimodal input. We craft a prompt that tells Claude to Read the image path and describe it.

**Files:**
- Modify: `src/agents/claude.py`
- Create: `tests/test_agents_claude.py`

- [ ] **Step 1: Write failing test (mock subprocess)**

Create `/Users/dion/potenlab/our_project/video_transcription/tests/test_agents_claude.py`:

```python
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
```

- [ ] **Step 2: Run test to verify failure**

```bash
python -m pytest tests/test_agents_claude.py -v
```

Expected: FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement Claude adapter**

Overwrite `/Users/dion/potenlab/our_project/video_transcription/src/agents/claude.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

from .base import BaseAgent


class ClaudeAgent(BaseAgent):
    name = "claude"

    def describe_frame(
        self,
        image_path: Path,
        audio_context: str,
        system_prompt: str,
        timeout_sec: int = 120,
    ) -> str:
        abs_path = Path(image_path).resolve()
        prompt = (
            "You are given a still frame from a video. "
            "Use the Read tool to open the image file, then follow the instructions below.\n\n"
            f"AUDIO CONTEXT (transcript around this frame):\n{audio_context or '(none)'}\n\n"
            f"IMAGE FILE: {abs_path}\n\n"
            "Output only the description text. No preamble, no tool trace."
        )
        cmd = [
            "claude",
            "-p", prompt,
            "--append-system-prompt", system_prompt,
            "--permission-mode", "acceptEdits",
        ]
        return self._run_with_retry(cmd, timeout_sec)

    @staticmethod
    def _run_with_retry(cmd: list[str], timeout_sec: int) -> str:
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=True,
                )
                return proc.stdout.strip()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                last_err = e
        assert last_err is not None
        raise last_err
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_agents_claude.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agents/claude.py tests/test_agents_claude.py
git commit -m "feat(agents/claude): claude -p adapter using Read tool for image input"
```

---

## Task 9: Codex adapter — `codex exec -i`

Codex CLI has a first-class `-i <IMAGE>` flag on the `exec` subcommand — much cleaner than Claude.

**Files:**
- Modify: `src/agents/codex.py`
- Create: `tests/test_agents_codex.py`

- [ ] **Step 1: Write failing test**

Create `/Users/dion/potenlab/our_project/video_transcription/tests/test_agents_codex.py`:

```python
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
```

- [ ] **Step 2: Run test to verify failure**

```bash
python -m pytest tests/test_agents_codex.py -v
```

Expected: FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement Codex adapter**

Overwrite `/Users/dion/potenlab/our_project/video_transcription/src/agents/codex.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

from .base import BaseAgent
from .claude import ClaudeAgent  # reuse retry helper


class CodexAgent(BaseAgent):
    name = "codex"

    def describe_frame(
        self,
        image_path: Path,
        audio_context: str,
        system_prompt: str,
        timeout_sec: int = 120,
    ) -> str:
        abs_path = Path(image_path).resolve()
        prompt = (
            f"{system_prompt}\n\n"
            f"AUDIO CONTEXT (transcript around this frame):\n{audio_context or '(none)'}\n\n"
            "Describe the attached image. Output only the description text, no preamble."
        )
        cmd = [
            "codex", "exec",
            "-i", str(abs_path),
            prompt,
        ]
        return ClaudeAgent._run_with_retry(cmd, timeout_sec)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_agents_codex.py -v
```

Expected: test PASSes.

- [ ] **Step 5: Commit**

```bash
git add src/agents/codex.py tests/test_agents_codex.py
git commit -m "feat(agents/codex): codex exec -i adapter"
```

---

## Task 10: Gemini adapter — `gemini -p "... @path"`

Gemini CLI uses `@` mention for file attachment inside the prompt.

**Files:**
- Modify: `src/agents/gemini.py`
- Create: `tests/test_agents_gemini.py`

- [ ] **Step 1: Write failing test**

Create `/Users/dion/potenlab/our_project/video_transcription/tests/test_agents_gemini.py`:

```python
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
```

- [ ] **Step 2: Run test to verify failure**

```bash
python -m pytest tests/test_agents_gemini.py -v
```

Expected: FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement Gemini adapter**

Overwrite `/Users/dion/potenlab/our_project/video_transcription/src/agents/gemini.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

from .base import BaseAgent
from .claude import ClaudeAgent


class GeminiAgent(BaseAgent):
    name = "gemini"

    def describe_frame(
        self,
        image_path: Path,
        audio_context: str,
        system_prompt: str,
        timeout_sec: int = 120,
    ) -> str:
        abs_path = Path(image_path).resolve()
        prompt = (
            f"{system_prompt}\n\n"
            f"AUDIO CONTEXT (transcript around this frame):\n{audio_context or '(none)'}\n\n"
            f"Describe this frame concisely. Output only the description text.\n\n"
            f"@{abs_path}"
        )
        cmd = ["gemini", "-p", prompt]
        return ClaudeAgent._run_with_retry(cmd, timeout_sec)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_agents_gemini.py -v
```

Expected: test PASSes.

- [ ] **Step 5: Full agents test sweep**

```bash
python -m pytest tests/test_agents_base.py tests/test_agents_claude.py tests/test_agents_codex.py tests/test_agents_gemini.py -v
```

Expected: all agent tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agents/gemini.py tests/test_agents_gemini.py
git commit -m "feat(agents/gemini): gemini -p @path adapter"
```

---

## Task 11: Timeline merge + Markdown/JSON output

Pure functions: take audio segments + frame descriptions, produce a timeline, render to Markdown or JSON.

**Files:**
- Create: `src/merge.py`
- Create: `tests/test_merge.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/dion/potenlab/our_project/video_transcription/tests/test_merge.py`:

```python
from src.merge import build_timeline, to_markdown, to_json_dict, format_timestamp


def test_format_timestamp_hours_minutes_seconds():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(75) == "01:15"
    assert format_timestamp(3725) == "1:02:05"


def test_build_timeline_interleaves_audio_and_frames_in_order():
    audio = [
        {"start": 0.0, "end": 10.0, "speaker": "A", "text": "Hello."},
        {"start": 10.0, "end": 20.0, "speaker": "B", "text": "Hi."},
    ]
    frames = [
        {"timestamp": 5.0, "image_path": "frames/f1.png", "description": "Opening slide."},
        {"timestamp": 15.0, "image_path": "frames/f2.png", "description": "Chart."},
    ]
    events = build_timeline(audio, frames)
    types = [e["type"] for e in events]
    # Expected order: audio[0], frame@5, audio[1], frame@15
    assert types == ["audio", "frame", "audio", "frame"]


def test_to_markdown_contains_timestamps_and_descriptions():
    audio = [{"start": 0.0, "end": 10.0, "speaker": "A", "text": "Hi."}]
    frames = [
        {"timestamp": 5.0, "image_path": "frames/f1.png", "description": "A slide."},
    ]
    md = to_markdown("video.mp4", "claude", 10.0, audio, frames)
    assert "video.mp4" in md
    assert "claude" in md
    assert "A slide." in md
    assert "00:05" in md
    assert "[A]" in md or "**A**" in md  # speaker shown


def test_to_json_dict_has_expected_keys():
    d = to_json_dict(
        source="video.mp4", agent="claude", duration_sec=10.0,
        audio_segments=[], frame_descriptions=[],
    )
    assert set(d.keys()) >= {"source", "agent", "duration_sec", "audio_segments", "frame_descriptions", "generated_at"}
```

- [ ] **Step 2: Run tests to verify failure**

```bash
python -m pytest tests/test_merge.py -v
```

Expected: 4 tests FAIL (module missing).

- [ ] **Step 3: Implement `src/merge.py`**

Create `/Users/dion/potenlab/our_project/video_transcription/src/merge.py`:

```python
"""Build merged timelines of audio + frame events and render output."""
from __future__ import annotations

from datetime import datetime, timezone


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_timeline(audio_segments: list[dict], frames: list[dict]) -> list[dict]:
    """Interleave audio segments and frame descriptions by time."""
    events: list[dict] = []
    for seg in audio_segments:
        events.append({"type": "audio", "start": seg["start"], "data": seg})
    for fr in frames:
        events.append({"type": "frame", "start": fr["timestamp"], "data": fr})
    events.sort(key=lambda e: e["start"])
    return events


def to_markdown(
    source: str,
    agent: str,
    duration_sec: float,
    audio_segments: list[dict],
    frame_descriptions: list[dict],
) -> str:
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append(f"# {source} — Transcript")
    lines.append("")
    lines.append(
        f"**Duration:** {format_timestamp(duration_sec)} · "
        f"**Frames analyzed:** {len(frame_descriptions)} · "
        f"**Agent:** {agent} · "
        f"**Generated:** {generated}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    events = build_timeline(audio_segments, frame_descriptions)
    for ev in events:
        if ev["type"] == "audio":
            s = ev["data"]
            ts = format_timestamp(s["start"])
            speaker = s.get("speaker")
            text = s.get("text", "").strip()
            if speaker:
                lines.append(f"**[{speaker}]** {ts} {text}")
            else:
                lines.append(f"{ts} {text}")
        else:
            f = ev["data"]
            ts = format_timestamp(f["timestamp"])
            lines.append("")
            lines.append(f"> 🖼️ **[{ts}]** {f['description']}")
            lines.append("")
    lines.append("")
    return "\n".join(lines)


def to_json_dict(
    source: str,
    agent: str,
    duration_sec: float,
    audio_segments: list[dict],
    frame_descriptions: list[dict],
) -> dict:
    return {
        "source": source,
        "agent": agent,
        "duration_sec": duration_sec,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "audio_segments": audio_segments,
        "frame_descriptions": [
            {
                "timestamp": f["timestamp"],
                "image": str(f["image_path"]),
                "description": f["description"],
            }
            for f in frame_descriptions
        ],
    }
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_merge.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/merge.py tests/test_merge.py
git commit -m "feat(merge): timeline builder + markdown/json renderers"
```

---

## Task 12: Default prompt + example recipes

**Files:**
- Create: `prompts/default.md`
- Create: `examples/meeting_recipe.md`
- Create: `examples/lecture_recipe.md`

- [ ] **Step 1: Write `prompts/default.md`**

Create `/Users/dion/potenlab/our_project/video_transcription/prompts/default.md`:

```
You are a visual describer for a video transcription pipeline. Given a still frame and the surrounding audio transcript, describe the frame concisely.

Output rules:
- 3 to 5 lines, plain text, no preamble, no headings
- Line 1: overall scene (layout, people, setting)
- Line 2 to 3: any visible text (UI, slides, captions) verbatim — quote exactly
- Line 4 to 5 (optional): how the frame relates to the audio context, only if clearly related

Do not speculate beyond what is visible. Do not add disclaimers or apologies.
```

- [ ] **Step 2: Write `examples/meeting_recipe.md`**

Create `/Users/dion/potenlab/our_project/video_transcription/examples/meeting_recipe.md`:

```
You are describing frames from a business meeting recording (Zoom, Google Meet, or similar).

Focus on:
- What slide or shared screen is visible, including slide title, section, page number if shown
- Any data being discussed: charts, tables, numbers — quote visible numbers exactly
- Demo software or product screens — name the app if identifiable
- Participant tiles only if they are a central focus (e.g. presenter in spotlight)

Output 3-5 lines, plain text, no preamble.
```

- [ ] **Step 3: Write `examples/lecture_recipe.md`**

Create `/Users/dion/potenlab/our_project/video_transcription/examples/lecture_recipe.md`:

```
You are describing frames from a lecture or technical tutorial recording.

Focus on:
- Slide title + key bullet points (quote verbatim when short)
- Code snippets visible on screen — transcribe the code exactly, or summarize if too long
- Diagrams: describe structure (nodes, arrows, labels)
- Whiteboard or handwritten content if that's the main focus

Output 3-5 lines, plain text, no preamble.
```

- [ ] **Step 4: Commit**

```bash
git add prompts/ examples/
git commit -m "docs(prompts): default frame prompt + meeting/lecture example recipes"
```

---

## Task 13: `video_transcribe.py` — main CLI orchestrator

The full pipeline: argparse, preflight, audio extract + transcribe, frame extract, parallel agent calls, merge, write outputs.

**Files:**
- Create: `video_transcribe.py`

- [ ] **Step 1: Write the CLI**

Create `/Users/dion/potenlab/our_project/video_transcription/video_transcribe.py`:

```python
#!/usr/bin/env python3
"""video_transcribe.py — video → audio transcript + frame descriptions → md/json."""
from __future__ import annotations

import argparse
import bisect
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src import preflight
from src.audio import (
    extract_audio_from_video,
    transcribe_mlx,
    diarize_whispermlx,
    assign_speakers,
)
from src.frames import extract_frames
from src.agents import get_agent, AGENT_NAMES
from src.merge import to_markdown, to_json_dict


DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "default.md"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Video transcription: audio + frame descriptions merged into a timeline."
    )
    p.add_argument("video_path", nargs="?", help="Input video file")
    p.add_argument("-o", "--output-dir", default=".", help="Output directory (default: cwd)")
    p.add_argument("--agent", choices=AGENT_NAMES, default="claude")
    p.add_argument("--mode", choices=["hybrid", "interval", "scene"], default="hybrid")
    p.add_argument("--interval", type=int, default=30)
    p.add_argument("--min-interval", type=int, default=10)
    p.add_argument("--max-interval", type=int, default=60)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--scene-threshold", type=float, default=0.3)
    p.add_argument("--diarize", action="store_true")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_PATH))
    p.add_argument("--context-window", type=int, default=30,
                   help="Seconds of audio before/after each frame as context")
    p.add_argument("--format", choices=["md", "json", "both"], default="md")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--check", action="store_true",
                   help="Run preflight dependency check and exit")
    return p


def slice_audio_context(
    audio_segments: list[dict], center: float, window: int
) -> str:
    """Join audio segments whose mid-point falls within [center-window, center+window]."""
    lo, hi = center - window, center + window
    parts: list[str] = []
    for seg in audio_segments:
        mid = (seg["start"] + seg["end"]) / 2
        if lo <= mid <= hi:
            speaker = seg.get("speaker")
            text = seg.get("text", "").strip()
            parts.append(f"[{speaker}] {text}" if speaker else text)
    return "\n".join(parts)


def describe_all_frames(
    agent, frames: list[dict], audio_segments: list[dict],
    system_prompt: str, window: int, parallel: int,
) -> list[dict]:
    results: list[dict | None] = [None] * len(frames)

    def work(idx: int, frame: dict) -> tuple[int, dict]:
        ctx = slice_audio_context(audio_segments, frame["timestamp"], window)
        try:
            desc = agent.describe_frame(frame["image_path"], ctx, system_prompt)
        except Exception as e:
            desc = f"[frame description failed: {e}]"
        return idx, {
            "timestamp": frame["timestamp"],
            "image_path": frame["image_path"],
            "description": desc,
        }

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = [ex.submit(work, i, f) for i, f in enumerate(frames)]
        for fut in as_completed(futures):
            i, out = fut.result()
            results[i] = out
    return [r for r in results if r is not None]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Preflight
    missing = preflight.check_dependencies(agent=args.agent, diarize=args.diarize)
    if args.check:
        preflight.print_report(missing)
        return 1 if missing else 0
    if missing:
        preflight.print_report(missing)
        return 1

    if not args.video_path:
        print("error: video_path is required (omit only with --check)", file=sys.stderr)
        return 2
    if args.diarize and not args.hf_token:
        print("error: --diarize requires --hf-token or HF_TOKEN env var", file=sys.stderr)
        return 2

    video = Path(args.video_path).resolve()
    if not video.exists():
        print(f"error: video not found: {video}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frames_dir = (out_dir / "frames") if args.keep_frames else (tmp_dir / "frames")
        wav_path = tmp_dir / "audio.wav"

        # 1. Extract audio
        print("[1/4] Extracting audio...")
        extract_audio_from_video(str(video), str(wav_path))

        # 2. Transcribe
        print("[2/4] Transcribing audio (mlx_whisper)...")
        transcription = transcribe_mlx(str(wav_path))
        if args.diarize:
            print("      + diarizing (whispermlx)...")
            diar = diarize_whispermlx(str(wav_path), args.hf_token)
            audio_segments = assign_speakers(transcription, diar)
        else:
            audio_segments = [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in transcription["segments"]
            ]

        # 3. Extract frames
        print(f"[3/4] Extracting frames (mode={args.mode})...")
        frames = extract_frames(
            video_path=video,
            out_dir=frames_dir,
            mode=args.mode,
            interval=args.interval,
            min_interval=args.min_interval,
            max_interval=args.max_interval,
            max_frames=args.max_frames,
            scene_threshold=args.scene_threshold,
        )
        print(f"      {len(frames)} frames selected.")

        # 4. Describe frames
        system_prompt = Path(args.prompt_file).read_text()
        agent = get_agent(args.agent)
        print(f"[4/4] Describing frames with '{args.agent}' (parallel={args.parallel})...")
        frame_descs = describe_all_frames(
            agent=agent,
            frames=frames,
            audio_segments=audio_segments,
            system_prompt=system_prompt,
            window=args.context_window,
            parallel=args.parallel,
        )

        # 5. Merge + write output
        from src.frames import _video_duration_sec
        duration = _video_duration_sec(video)
        base = out_dir / video.stem

        if args.format in ("md", "both"):
            md = to_markdown(video.name, args.agent, duration, audio_segments, frame_descs)
            (base.with_suffix(".md")).write_text(md)
            print(f"      wrote {base.with_suffix('.md')}")
        if args.format in ("json", "both"):
            payload = to_json_dict(
                source=video.name, agent=args.agent, duration_sec=duration,
                audio_segments=audio_segments, frame_descriptions=frame_descs,
            )
            (base.with_suffix(".json")).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
            print(f"      wrote {base.with_suffix('.json')}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Sanity check `--check` works without a video**

```bash
cd /Users/dion/potenlab/our_project/video_transcription
python video_transcribe.py --check --agent claude
```

Expected: prints either "All dependencies present." or a list of missing items. Exits 0 if all ok, 1 otherwise.

- [ ] **Step 3: Sanity check argparse**

```bash
python video_transcribe.py --help
```

Expected: prints the flag list cleanly.

- [ ] **Step 4: Commit**

```bash
git add video_transcribe.py
git commit -m "feat(cli): main orchestrator with preflight, audio, frames, agents, merge"
```

---

## Task 14: README rewrite

Replace `whisper_transcription`'s README with a video-transcription-centric one. Credit original.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Overwrite `README.md`**

Overwrite `/Users/dion/potenlab/our_project/video_transcription/README.md`:

```markdown
# video_transcription

Mac(Apple Silicon) 로컬에서 **비디오 파일을 오디오 전사 + 프레임 시각 해석**으로 통합 transcript를 만드는 CLI.

오디오는 `mlx_whisper`, 프레임 해석은 로컬에 설치된 CLI agent(Claude Code / Codex / Gemini) 중 하나를 subprocess로 호출해서 처리.

> Based on [treesoop/whisper_transcription](https://github.com/treesoop/whisper_transcription). 오디오 전사 파이프라인을 그대로 가져와서 프레임 해석 레이어를 얹음.

## How it works

```
video.mp4
  ├─ ffmpeg → audio.wav  → mlx_whisper  → 오디오 transcript
  └─ ffmpeg scene detect → frames/*.png  → agent 해석  → 프레임 설명
                                             ↓
                                    병합된 타임라인 (md + json)
```

## Setup

### 1. 공통 의존성

```bash
brew install ffmpeg
pip3 install -r requirements.txt
```

### 2. 오디오 전사 백엔드

```bash
pip3 install mlx-whisper
# 화자 분리 쓸 때만
uv tool install whispermlx --python 3.12
```

### 3. Agent 하나 선택해서 설치

**Claude (기본)**
```bash
curl -fsSL https://claude.ai/install.sh | bash
# 또는 npm install -g @anthropic-ai/claude-code
claude   # 첫 실행 OAuth 로그인 (Claude Pro/Max 구독 또는 API key)
```

**Codex**
```bash
npm install -g @openai/codex
codex login
```

**Gemini**
```bash
npm install -g @google/gemini-cli
gemini   # Google 계정 로그인 또는 `export GEMINI_API_KEY=...`
```

### 4. 환경 검증

```bash
python video_transcribe.py --check --agent claude
```

없는 의존성이 있으면 설치 명령을 안내해줌.

## Usage

### 기본

```bash
python video_transcribe.py meeting.mp4
```

출력: `meeting.md`

### 주요 옵션

```bash
python video_transcribe.py video.mp4 \
    --agent codex \
    --mode hybrid \
    --min-interval 10 --max-interval 60 --max-frames 200 \
    --scene-threshold 0.3 \
    --parallel 4 \
    --format both \
    -o ./out
```

| flag | 기본값 | 설명 |
|---|---|---|
| `--agent` | `claude` | `claude` \| `codex` \| `gemini` |
| `--mode` | `hybrid` | 프레임 추출 전략 |
| `--interval` | `30` | `interval` 모드의 고정 간격 |
| `--min-interval` | `10` | hybrid: 최소 프레임 간격 |
| `--max-interval` | `60` | hybrid: 최대 공백 → 강제 프레임 |
| `--max-frames` | `200` | 전체 프레임 상한 |
| `--scene-threshold` | `0.3` | ffmpeg scene 민감도 |
| `--diarize` | off | 화자 분리 (--hf-token 필수) |
| `--prompt-file` | `prompts/default.md` | 프레임 해석 프롬프트 override |
| `--context-window` | `30` | 프레임 주변 오디오 맥락(±초) |
| `--format` | `md` | `md` \| `json` \| `both` |
| `--parallel` | `4` | 동시 agent 호출 수 |
| `--keep-frames` | off | 프레임 이미지 유지 |
| `--check` | — | 의존성 검증만 하고 종료 |

### 커스텀 프롬프트

`examples/meeting_recipe.md`, `examples/lecture_recipe.md` 참고:

```bash
python video_transcribe.py meeting.mp4 --prompt-file examples/meeting_recipe.md
```

### 화자 분리 포함

```bash
python video_transcribe.py meeting.mp4 --diarize --hf-token $HF_TOKEN
```

## 출력 예시 (Markdown)

```markdown
# meeting.mp4 — Transcript

**Duration:** 1:23:18 · **Frames analyzed:** 42 · **Agent:** claude · **Generated:** 2026-04-19T10:00:00+09:00

---

**[SPEAKER_00]** 00:00 네 오늘은 1분기 리뷰 시작하겠습니다.
**[SPEAKER_01]** 00:08 준비됐습니다.

> 🖼️ **[00:15]** Zoom 화면 공유 상태. "Q1 Review" 슬라이드 표지. 상단에 참가자 썸네일 4명.
```

## Architecture

- `video_transcribe.py` — CLI entrypoint
- `src/audio.py` — mlx_whisper 래퍼 + 비디오→오디오 추출
- `src/frames.py` — ffmpeg 프레임 추출 (interval / scene / hybrid)
- `src/agents/{claude,codex,gemini}.py` — subprocess adapter
- `src/merge.py` — 타임라인 병합 + md/json 렌더
- `src/preflight.py` — 의존성 검증

## License

MIT (same as upstream `whisper_transcription`).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for video_transcription"
```

---

## Task 15: End-to-end smoke test with real video + fix any bugs

Run the real pipeline on `ep02-final.mp4` with Claude, inspect output, fix issues.

**Files:**
- Potentially modify: any of the above, based on what breaks

- [ ] **Step 1: Run full pipeline with Claude**

```bash
cd /Users/dion/potenlab/our_project/video_transcription
mkdir -p out/ep02_claude
python video_transcribe.py /Users/dion/Downloads/ep02-final.mp4 \
    --agent claude \
    --format both \
    --keep-frames \
    --parallel 4 \
    -o out/ep02_claude
```

Expected:
- preflight passes
- progress lines print for each of the 4 stages
- `out/ep02_claude/ep02-final.md` and `.json` produced
- `out/ep02_claude/frames/*.png` with several frame images

- [ ] **Step 2: Spot-check the Markdown**

```bash
cat out/ep02_claude/ep02-final.md | head -60
```

Expected: timestamps, Korean transcript text, inline 🖼️ frame blocks with short descriptions.

- [ ] **Step 3: Spot-check the JSON**

```bash
python -c "import json; d=json.load(open('out/ep02_claude/ep02-final.json')); print('audio segs:', len(d['audio_segments'])); print('frames:', len(d['frame_descriptions'])); print('agent:', d['agent']); print('first frame:', d['frame_descriptions'][0] if d['frame_descriptions'] else None)"
```

Expected: non-zero counts, first frame has a real (non-error) description.

- [ ] **Step 4: Repeat with Codex and Gemini if installed, otherwise skip**

```bash
# Only run if the CLI is installed and logged in.
python video_transcribe.py --check --agent codex && \
    python video_transcribe.py /Users/dion/Downloads/ep02-final.mp4 --agent codex --parallel 2 -o out/ep02_codex

python video_transcribe.py --check --agent gemini && \
    python video_transcribe.py /Users/dion/Downloads/ep02-final.mp4 --agent gemini --parallel 2 -o out/ep02_gemini
```

Expected: each produces its own .md in the corresponding out dir.

- [ ] **Step 5: If anything broke in Steps 1-4, fix inline and re-run**

Typical issues to check:
- ffmpeg `-ss` placement for frame extraction — we put `-ss` before `-i` for speed; may need `-ss` after `-i` for frame accuracy. Swap if frames look wrong.
- Claude stdout may include tool traces; if so, strip leading/trailing non-description lines or switch to `--output-format json` in the adapter and parse `.result`.
- Unicode issues on JSON dump: `ensure_ascii=False` already set.
- Paths with spaces: ensure all passed as list args (no shell string interpolation) — already covered.

Commit any fixes with focused messages, e.g.:

```bash
git add <file>
git commit -m "fix(frames): move -ss after -i for accurate seek on hybrid timestamps"
```

- [ ] **Step 6: Final push**

```bash
git push origin main
```

Expected: push succeeds.

- [ ] **Step 7: Final test sweep**

```bash
python -m pytest -v
```

Expected: all tests PASS.

---

## Self-review results

**Spec coverage:** each §1–§13 section of the design has a covering task.
- §2 Repo relationship → Task 1
- §4.1 CLI → Task 13
- §4.2 audio → Task 6
- §4.3 frames → Tasks 3, 4, 5
- §4.4 agents → Tasks 7, 8, 9, 10
- §4.5 merge → Task 11
- §4.6 preflight → Task 2
- §6 default prompt → Task 12
- §7 output formats → Task 11 + Task 13 (wiring)
- §8 error handling → Task 13 (per-frame failure placeholder) + Task 2 (preflight)
- §9 file layout → Tasks 1–14 create every file listed
- §10 prerequisites → Task 14 (README)
- §11 testing → Task 15 (real smoke test) + unit tests in Tasks 2/3/4/5/7/8/9/10/11

**No placeholders:** no TBD / TODO / "implement later" / "similar to Task N" strings. Every code step has complete code. All commands have expected outputs or expected behaviors.

**Type consistency:**
- `Frame = {timestamp, image_path}` used identically in Tasks 3, 5, 13
- `audio_segments` items = `{start, end, speaker?, text}` used identically in audio.py, merge.py, CLI
- `describe_frame(image_path, audio_context, system_prompt, timeout_sec=120)` signature matches across BaseAgent and all 3 adapters
- `get_agent(name)` / `AGENT_NAMES` names consistent (Task 7, 13)
