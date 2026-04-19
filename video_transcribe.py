#!/usr/bin/env python3
"""video_transcribe.py — video → audio transcript + frame descriptions → md/json."""
from __future__ import annotations

import argparse
import json
import os
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
from src.frames import extract_frames, _video_duration_sec
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
    p.add_argument("--scene-threshold", type=float, default=0.15,
                   help="ffmpeg scene sensitivity (0-1). Lower = more frames. "
                        "0.15 is a safe default for mixed content; try 0.3 for "
                        "live-action, 0.05-0.1 for animation/illustration.")
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
            (base.with_suffix(".json")).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str)
            )
            print(f"      wrote {base.with_suffix('.json')}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
