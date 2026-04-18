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
