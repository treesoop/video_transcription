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
