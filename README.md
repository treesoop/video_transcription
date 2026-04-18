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

핵심 아이디어: 오디오를 먼저 전사한 다음, 각 프레임을 agent에 던질 때 해당 시점의 대화 맥락을 함께 전달해서 더 정확한 시각 해석을 얻음.

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
python3 video_transcribe.py --check --agent claude
```

없는 의존성이 있으면 설치 명령을 안내해줌.

## Usage

### 기본

```bash
python3 video_transcribe.py meeting.mp4
```

출력: `./meeting.md`

### 주요 옵션

```bash
python3 video_transcribe.py video.mp4 \
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
| `--interval` | `30` | `interval` 모드의 고정 간격(초) |
| `--min-interval` | `10` | hybrid: 최소 프레임 간격 |
| `--max-interval` | `60` | hybrid: 최대 공백 → 강제 프레임 삽입 |
| `--max-frames` | `200` | 전체 프레임 상한 |
| `--scene-threshold` | `0.3` | ffmpeg scene 민감도 (0~1) |
| `--diarize` | off | 화자 분리 (`--hf-token` 필수) |
| `--hf-token` | env `HF_TOKEN` | HuggingFace 토큰 |
| `--prompt-file` | `prompts/default.md` | 프레임 해석 프롬프트 override |
| `--context-window` | `30` | 프레임 주변 오디오 맥락(±초) |
| `--format` | `md` | `md` \| `json` \| `both` |
| `--parallel` | `4` | 동시 agent 호출 수 |
| `--keep-frames` | off | 프레임 이미지 유지 (기본은 삭제) |
| `--check` | — | 의존성 검증만 하고 종료 |

### 커스텀 프롬프트

`examples/meeting_recipe.md`, `examples/lecture_recipe.md` 참고:

```bash
python3 video_transcribe.py meeting.mp4 --prompt-file examples/meeting_recipe.md
```

### 화자 분리 포함

```bash
python3 video_transcribe.py meeting.mp4 --diarize --hf-token $HF_TOKEN
```

화자 분리용 HuggingFace 토큰은 [whisper_transcription README](https://github.com/treesoop/whisper_transcription#3-huggingface-토큰-발급-화자-분리-사용-시)의 절차를 따라 발급.

## 출력 예시 (Markdown)

```markdown
# meeting.mp4 — Transcript

**Duration:** 1:23:18 · **Frames analyzed:** 42 · **Agent:** claude · **Generated:** 2026-04-19T10:00:00+09:00

---

**[SPEAKER_00]** 00:00 네 오늘은 1분기 리뷰 시작하겠습니다.
**[SPEAKER_01]** 00:08 준비됐습니다.

> 🖼️ **[00:15]** Zoom 화면 공유 상태. "Q1 Review" 슬라이드 표지. 상단에 참가자 썸네일 4명.
```

## Prompt 팁

- 용도별 프롬프트를 `examples/`에 두거나 직접 작성. 범용 기본값은 `prompts/default.md`.
- 회의 녹화·강의·튜토리얼 등 타겟 컨텐츠에 맞춘 recipe를 쓰면 프레임 해석 품질이 눈에 띄게 향상됨.
- 프롬프트는 일반 텍스트. `--prompt-file`로 경로 지정.

## Architecture

- `video_transcribe.py` — CLI entrypoint (argparse + 파이프라인 orchestration)
- `src/audio.py` — mlx_whisper 래퍼 + 비디오→오디오 추출 (whisper_transcription 이식)
- `src/frames.py` — ffmpeg 프레임 추출 (interval / scene / hybrid)
- `src/agents/{claude,codex,gemini}.py` — subprocess 기반 pluggable adapter
- `src/merge.py` — 타임라인 병합 + md/json 렌더
- `src/preflight.py` — 의존성 검증 (`--check`)

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.9+
- ffmpeg
- 다음 중 하나: Claude Code / Codex CLI / Gemini CLI

## License

MIT (upstream `whisper_transcription`과 동일).
