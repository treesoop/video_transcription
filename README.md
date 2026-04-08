# Whisper Transcription (Mac, Apple Silicon)

Apple Silicon Mac에서 미팅 녹음을 로컬로 전사하는 도구.
Notta 같은 유료 서비스 없이, 더 높은 퀄리티로 전사 + 화자 분리까지 가능.

## How it works

두 개의 오픈소스 도구를 조합해서 각각의 장점만 사용:

| 단계 | 도구 | 역할 |
|------|------|------|
| 전사 | [mlx-whisper](https://github.com/ml-explore/mlx-examples) | Metal GPU 가속, hallucination 방지 옵션 |
| 화자 분리 | [whispermlx](https://github.com/kalebjs/whispermlx) | pyannote 기반 speaker diarization |

> **왜 이렇게?**
> whispermlx 단독 사용 시 한국어에서 hallucination("Jelly Jelly..." 반복)이 발생.
> mlx-whisper는 `--hallucination-silence-threshold` 옵션으로 이를 방지할 수 있지만 화자 분리가 없음.
> 그래서 **전사는 mlx-whisper, 화자 분리는 whispermlx**에서 가져와 병합.

## Setup

### 1. mlx-whisper 설치

```bash
pip3 install mlx-whisper
```

설치 후 확인:
```bash
mlx_whisper --help
```

### 2. whispermlx 설치 (화자 분리 사용 시)

Python 3.10+ 필요. [uv](https://github.com/astral-sh/uv)로 설치하는 게 가장 간단:

```bash
# uv 없으면 먼저 설치
brew install uv

# whispermlx 설치
uv tool install whispermlx --python 3.12
```

설치 후 확인:
```bash
whispermlx --help
```

### 3. HuggingFace 토큰 발급 (화자 분리 사용 시)

화자 분리에 사용되는 pyannote 모델이 HuggingFace 인증을 요구합니다.

1. https://huggingface.co/join 에서 회원가입
2. https://huggingface.co/settings/tokens 에서 `New token` 클릭 → 이름 입력 → `Read` 권�� 선택 → 생성
3. 생성된 `hf_xxxx...` 토큰을 복사해두기
4. 아래 두 모델 페이지에 각각 들어가서 **"Agree and access repository"** 버튼 클릭 (즉시 승인됨):
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0

이후 `--hf-token` 옵션에 토큰을 넣어 사용:
```bash
python3 transcribe.py meeting.m4a --diarize --hf-token hf_xxxx...
```

## Usage

### 전사만 (화자 분리 없이)

```bash
python3 transcribe.py meeting.m4a
```

출력: `./meeting.txt`

```
00:00 네. 저희 그래서 30일에 먼저 태국으로 가세요?
00:15 스미님은?
00:17 예정상으로는 그렇긴 합니다.
```

### 전사 + 화자 분리

```bash
python3 transcribe.py meeting.m4a --diarize --hf-token YOUR_TOKEN
```

출력: `./meeting.txt`

```
[SPEAKER_03] 00:00 네. 저희 그래서 30일에 먼저 태국으로 가세요?
[SPEAKER_03] 00:15 스미님은?
[SPEAKER_02] 00:17 예정상으로는 그렇긴 합니다.
```

### 출력 디렉토리 지정

```bash
python3 transcribe.py meeting.m4a --diarize --hf-token YOUR_TOKEN -o ./output
```

## Tips

### 긴 오디오는 30분 단위로 분할

Whisper는 오디오가 길어질수록 후반부 품질이 떨어짐. 1시간 이상이면 분할 후 전사.

```bash
ffmpeg -i meeting.m4a -f segment -segment_time 1800 -c copy meeting_%03d.m4a
```

### `--language` 옵션 사용하지 않기

자동 감지가 훨씬 정확. 한국어+영어 섞인 미팅에서 언어를 고정하면 오히려 품질이 떨어짐.

### 첫 실행은 느림

모델을 다운로드해야 해서 첫 실행에 시간이 걸림 (약 3GB). 이후에는 캐시되어 빠르게 동작.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.9+ (mlx-whisper용)
- [uv](https://github.com/astral-sh/uv) (whispermlx 설치용)
- ffmpeg (오디오 분할 시): `brew install ffmpeg`

## Benchmarks

20분 한국어 미팅 기준:

| | whisper_transcription | Notta (유료) |
|---|---|---|
| **전사 정확도** | 높음 | 보통 |
| **Hallucination** | 없음 | 없음 |
| **화자 분리** | 있음 (짧은 추임새 간혹 흔들림) | 있음 (안정적) |
| **고유명사/약어** | 정확 | 누락/오인식 많음 |
| **문장 분리** | 한 줄씩 깔끔 | 긴 덩어리 |
| **한국어** | 강함 | 보통 |
| **영어** | 강함 | 강함 |
| **비용** | 무료 (로컬) | 유료 |
