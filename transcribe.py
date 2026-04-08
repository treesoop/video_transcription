#!/usr/bin/env python3
"""
mlx_whisper 전사 + whispermlx 화자 분리 병합 파이프라인

mlx_whisper의 우수한 전사 퀄리티(hallucination 방지 포함)와
whispermlx의 pyannote 화자 분리를 결합.

Usage:
    # 전사만
    python transcribe.py meeting.m4a

    # 화자 분리 포함
    python transcribe.py meeting.m4a --diarize --hf-token YOUR_TOKEN

    # 출력 디렉토리 지정
    python transcribe.py meeting.m4a --diarize --hf-token YOUR_TOKEN -o ./output
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def find_mlx_whisper() -> str:
    """mlx_whisper CLI 경로 찾기"""
    known = os.path.expanduser("~/Library/Python/3.9/bin/mlx_whisper")
    if os.path.isfile(known):
        return known
    import shutil
    found = shutil.which("mlx_whisper")
    if found:
        return found
    raise FileNotFoundError("mlx_whisper를 찾을 수 없습니다. pip install mlx-whisper")


def find_whispermlx() -> str:
    """whispermlx CLI 경로 찾기"""
    import shutil
    found = shutil.which("whispermlx")
    if found:
        return found
    raise FileNotFoundError("whispermlx를 찾을 수 없습니다. uv tool install whispermlx")


def transcribe_mlx(audio_path: str) -> dict:
    """mlx_whisper로 전사 (JSON 출력) — hallucination 방지 옵션 포함"""
    mlx_whisper = find_mlx_whisper()
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            mlx_whisper,
            "--model", "mlx-community/whisper-large-v3-turbo",
            "--condition-on-previous-text", "False",
            "--hallucination-silence-threshold", "1",
            "-f", "json",
            "-o", tmp,
            audio_path,
        ]
        print("[1/2] 전사 중 (mlx_whisper)...")
        subprocess.run(cmd, check=True)

        basename = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(tmp, f"{basename}.json")
        with open(json_path) as f:
            return json.load(f)


def diarize_whispermlx(audio_path: str, hf_token: str) -> list:
    """whispermlx로 화자 분리만 수행, JSON 결과에서 화자 정보 추출"""
    whispermlx = find_whispermlx()
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            whispermlx,
            audio_path,
            "--model", "mlx-community/whisper-large-v3-turbo",
            "--diarize",
            "--hf_token", hf_token,
            "--output_dir", tmp,
        ]
        print("[2/2] 화자 분리 중 (whispermlx)...")
        subprocess.run(cmd, check=True, capture_output=True)

        basename = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(tmp, f"{basename}.json")
        with open(json_path) as f:
            data = json.load(f)

        # whispermlx JSON에서 화자+타임스탬프 추출
        diar_segments = []
        for seg in data.get("segments", []):
            speaker = seg.get("speaker", "UNKNOWN")
            diar_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": speaker,
            })
        return diar_segments


def assign_speakers(transcription: dict, diarization: list) -> list:
    """mlx_whisper 전사 세그먼트에 whispermlx 화자 배정"""
    result = []
    for seg in transcription["segments"]:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # 오버랩이 가장 큰 화자 구간 찾기
        speaker = "UNKNOWN"
        best_overlap = 0
        for d in diarization:
            overlap = max(0, min(seg_end, d["end"]) - max(seg_start, d["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                speaker = d["speaker"]

        result.append({
            "start": seg_start,
            "end": seg_end,
            "speaker": speaker,
            "text": seg["text"].strip(),
        })
    return result


def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def write_output(segments: list, output_path: str, has_speakers: bool):
    with open(output_path, "w") as f:
        for seg in segments:
            ts = format_timestamp(seg["start"])
            if has_speakers:
                f.write(f"[{seg['speaker']}] {ts} {seg['text']}\n")
            else:
                f.write(f"{ts} {seg['text']}\n")
    print(f"\n완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="mlx_whisper + whispermlx 화자 분리")
    parser.add_argument("audio", help="오디오 파일 경로")
    parser.add_argument("--diarize", action="store_true", help="화자 분리 활성화")
    parser.add_argument("--hf-token", help="HuggingFace 토큰 (화자 분리 시 필수)")
    parser.add_argument("-o", "--output-dir", default=".", help="출력 디렉토리")
    args = parser.parse_args()

    if args.diarize and not args.hf_token:
        parser.error("--diarize 사용 시 --hf-token 필수")

    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.audio))[0]
    output_path = os.path.join(args.output_dir, f"{basename}.txt")

    # 1. mlx_whisper 전사
    transcription = transcribe_mlx(args.audio)

    if args.diarize:
        # 2. whispermlx 화자 분리
        diar_segments = diarize_whispermlx(args.audio, args.hf_token)
        # 3. 병합
        segments = assign_speakers(transcription, diar_segments)
        write_output(segments, output_path, has_speakers=True)
    else:
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in transcription["segments"]
        ]
        write_output(segments, output_path, has_speakers=False)


if __name__ == "__main__":
    main()
