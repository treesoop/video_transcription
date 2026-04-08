# Whisper Transcription Guide (Mac, MLX)

Apple Silicon Mac에서 `mlx_whisper`를 사용해 미팅 노트를 전사하는 최적 설정 가이드.

## Setup

### 1. 설치

```bash
pip install mlx-whisper
```

### 2. 모델

```
mlx-community/whisper-large-v3-turbo
```

첫 실행 시 자동 다운로드됨. GPU(Metal) 가속으로 빠르게 동작.

## Usage

### 기본 명령어

```bash
mlx_whisper \
  --model mlx-community/whisper-large-v3-turbo \
  --condition-on-previous-text False \
  --hallucination-silence-threshold 1 \
  "your_audio.m4a"
```

### 핵심 옵션 설명

| 옵션 | 값 | 이유 |
|------|-----|------|
| `--condition-on-previous-text` | `False` | 이전 텍스트에 의존하지 않아 hallucination 전파 방지 |
| `--hallucination-silence-threshold` | `1` | 무음 구간에서 "좀 더", "thank you" 같은 반복 생성 차단 |

### 하지 말아야 할 것

- **`--language` 옵션 사용 금지** — 자동 감지가 훨씬 정확함. 특히 한국어+영어가 섞인 미팅에서 언어를 고정하면 오히려 품질이 떨어짐.

## Tips

### 긴 오디오는 30분 단위로 분할

Whisper는 오디오가 길어질수록 후반부 품질이 급격히 떨어짐. 1시간짜리 미팅이면 반드시 분할 후 전사.

```bash
# ffmpeg으로 30분 단위 분할
ffmpeg -i meeting.m4a -f segment -segment_time 1800 -c copy meeting_%03d.m4a
```

### 출력 형식

기본은 `.txt`. SRT 자막이 필요하면:

```bash
mlx_whisper \
  --model mlx-community/whisper-large-v3-turbo \
  --condition-on-previous-text False \
  --hallucination-silence-threshold 1 \
  --output_format srt \
  "your_audio.m4a"
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.9+
- `mlx-whisper` (`pip install mlx-whisper`)
