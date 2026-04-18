# Video Transcription — Design

Date: 2026-04-19
Owner: Dion (treesoop)
Status: draft (awaiting review)

## 1. Goal

입력된 비디오 파일에서 **오디오 전사**와 **타임스탬프별 프레임 시각 해석**을 병합해 단일 타임라인 transcript를 생성하는 CLI 도구. 프레임 해석은 로컬에 설치된 CLI 에이전트(Claude Code / Codex / Gemini) 중 하나를 subprocess로 호출해 수행한다. GitHub 공개 배포 (`treesoop/video_transcription`).

핵심 아이디어: 오디오를 **먼저** 전사한 뒤, 각 프레임을 agent에 던질 때 해당 시점의 대화 맥락을 함께 전달해 더 정확한 시각 해석을 얻는다.

## 2. Repository relationship

- `treesoop/whisper_transcription`의 Git 히스토리를 수동 fork로 가져와 `treesoop/video_transcription` 신규 레포로 독립 운영.
- GitHub 기본 fork 기능은 동일 owner 간 사용 불가하므로 수동 절차 사용:
  ```bash
  cd /Users/dion/potenlab/our_project/video_transcription
  git clone git@github-treesoop:treesoop/whisper_transcription.git .
  gh repo create treesoop/video_transcription --public --source . --remote origin --push
  # (또는 repo 수동 생성 후 git remote set-url + push)
  ```
- upstream fork network 없음. README에 `whisper_transcription` 크레딧 명시.

## 3. Architecture

```
video.mp4
  │
  ├─ ffmpeg → audio.wav ─→ mlx_whisper → audio_transcript.json
  │                         (옵션) whispermlx diarize → speaker merge
  │
  └─ ffmpeg scene detect → frames/frame_XXXXX.png (+ timestamps)
                            │
                            ▼
                         pluggable agent adapter
                         (claude / codex / gemini)
                            │
                            ▼
                         frame_descriptions

audio_transcript + frame_descriptions
  → timeline merge
  → output.md (기본) + output.json (옵션)
```

## 4. Components

### 4.1 `video_transcribe.py` — 메인 CLI

argparse 기반 엔트리포인트. flags:

| flag | default | 설명 |
|---|---|---|
| `video_path` (positional) | — | 입력 비디오 |
| `-o`, `--output-dir` | `./` | 출력 디렉토리 |
| `--agent {claude,codex,gemini}` | `claude` | 프레임 해석 에이전트 |
| `--mode {hybrid,interval,scene}` | `hybrid` | 프레임 추출 전략 |
| `--interval` | `30` (sec) | `interval` 모드의 고정 간격 |
| `--min-interval` | `10` | hybrid: 프레임 간 최소 시간 |
| `--max-interval` | `60` | hybrid: 공백 구간에 강제 프레임 삽입 간격 |
| `--max-frames` | `200` | 전체 프레임 수 상한 |
| `--scene-threshold` | `0.3` | ffmpeg scene 민감도 (0-1) |
| `--diarize` | off | 오디오 화자 분리 활성화 |
| `--hf-token` | env `HF_TOKEN` | 화자 분리용 HuggingFace 토큰 |
| `--prompt-file` | 내장 기본값 | 프레임 해석 시스템 프롬프트 override |
| `--context-window` | `30` (sec) | 프레임 주변 오디오 context 범위 (±N초) |
| `--format {md,json,both}` | `md` | 출력 포맷 |
| `--parallel` | `4` | agent 동시 호출 수 |
| `--keep-frames` | off | 프레임 이미지 삭제 방지 (기본은 삭제) |
| `--check` | — | preflight 검증만 수행 후 exit |

### 4.2 `src/audio.py` — 오디오 파이프라인

`whisper_transcription/transcribe.py`의 함수를 그대로 이식 (fork로 히스토리 가져오니 파일 자체를 `src/audio.py`로 move + refactor):

- `extract_audio_from_video(video_path, out_wav) -> Path` *(신규)* — ffmpeg로 오디오 트랙 추출
- `transcribe_mlx(audio_path) -> dict` *(기존)*
- `diarize_whispermlx(audio_path, hf_token) -> list` *(기존)*
- `merge_transcript_with_speakers(segments, diar_segments) -> list` *(기존)*

한국어 hallucination 방지용 `--condition-on-previous-text False --hallucination-silence-threshold 1` 옵션 유지.

### 4.3 `src/frames.py` — 프레임 추출

단일 엔트리:
```python
def extract_frames(
    video_path: Path,
    out_dir: Path,
    mode: Literal["hybrid", "interval", "scene"],
    interval: int,
    min_interval: int,
    max_interval: int,
    max_frames: int,
    scene_threshold: float,
) -> list[dict]:  # [{"timestamp": float, "image_path": Path}]
```

로직:
- `mode="interval"`: `ffmpeg -vf fps=1/{interval}` — 가장 단순
- `mode="scene"`: `ffmpeg -vf "select='gt(scene,{threshold})',showinfo"` + stderr 파싱으로 pts_time 추출
- `mode="hybrid"` (기본):
  1. scene 후보 프레임 + 타임스탬프 추출
  2. `min_interval` 내 인접한 후보는 뒤쪽 제거
  3. 인접 프레임 간격 > `max_interval`이면 중간 지점에 강제 프레임 삽입
  4. 총 개수 > `max_frames`면 타임스탬프 기준 균등 downsample

### 4.4 `src/agents/` — pluggable adapter

```
src/agents/
  __init__.py   # REGISTRY = {"claude": ClaudeAdapter, "codex": CodexAdapter, "gemini": GeminiAdapter}
  base.py       # BaseAgent(ABC)
  claude.py
  codex.py
  gemini.py
```

공통 인터페이스:
```python
class BaseAgent(ABC):
    @abstractmethod
    def describe_frame(
        self,
        image_path: Path,
        audio_context: str,
        system_prompt: str,
    ) -> str: ...

    @abstractmethod
    def check_installed(self) -> tuple[bool, str]: ...
    # returns (ok, install_instruction_if_not)
```

호출 방식:
- **Claude**: `claude -p "<system_prompt>\n\nAUDIO CONTEXT:\n<ctx>\n\nRead the image at <abs_path> using the Read tool, then describe it."` — Read tool이 print 모드에서 default로 enabled이고 멀티모달 이미지 읽기 지원. `--append-system-prompt` 활용 가능.
- **Codex**: `codex exec -i <abs_path> "<system_prompt>\n\nAUDIO CONTEXT:\n<ctx>\n\nDescribe the attached image."`
- **Gemini**: `gemini -p "<system_prompt>\n\nAUDIO CONTEXT:\n<ctx>\n\nDescribe this frame. @<abs_path>"`

공통 처리:
- `subprocess.run` + `capture_output=True`, timeout (기본 120초)
- stdout 순수 텍스트만 추출 (agent별 banner 제거 로직은 adapter 내부)
- 실패 시 1회 retry. 그래도 실패하면 해당 프레임 설명을 `[frame description failed: <reason>]`로 남기고 계속.
- 동시성: `concurrent.futures.ThreadPoolExecutor(max_workers=parallel)`

### 4.5 `src/merge.py` — 타임라인 병합

- input: `audio_segments` (list), `frame_descriptions` (list)
- 각 frame을 timestamp가 포함되는 audio segment 구간 끝에 삽입
- `to_markdown(events)` — 보기 좋은 transcript
- `to_json(events)` — 구조화 데이터

### 4.6 `src/preflight.py` — `--check`

확인 대상:
- `ffmpeg` (필수)
- `mlx_whisper` (필수; `~/Library/Python/3.9/bin/mlx_whisper` fallback)
- `whispermlx` (diarize 옵션 켰을 때만)
- 선택된 agent CLI (`claude --version` / `codex --version` / `gemini --version`)

각 누락 항목에 대해 설치 명령을 콘솔에 출력. 하나라도 없으면 exit code 1.

## 5. Data flow

1. preflight 통과
2. video → tmp/audio.wav (ffmpeg)
3. mlx_whisper 전사 → `audio_segments`
4. (옵션) whispermlx diarize → speaker merge
5. video → tmp/frames/*.png + timestamps (hybrid)
6. 각 프레임 병렬 처리:
   - timestamp ± `context_window`초 오디오 transcript slice를 `audio_context`로
   - agent.describe_frame(image, audio_context, system_prompt) 호출
7. merge → timeline events
8. output.md (+ output.json)
9. `--keep-frames` 아니면 tmp 정리

## 6. Frame interpretation prompt (기본)

`prompts/default.md`:

```
You are a visual describer for a video transcription pipeline. Given a single frame and the surrounding audio transcript, describe the frame concisely.

Output rules:
- 3–5 lines, plain text, no preamble, no headings
- Line 1: overall scene (layout, people, setting)
- Line 2–3: any visible text (UI, slides, captions) verbatim — quote exactly
- Line 4–5 (optional): how the frame relates to the audio context, only if clearly related

Do not speculate beyond what is visible. Do not moralize. Do not add disclaimers.
```

사용자가 `--prompt-file my_prompt.md`로 override 가능. `examples/`에 회의 녹화용·강의 녹화용 샘플 프롬프트 포함.

## 7. Output formats

### 7.1 Markdown (기본)

```markdown
# meeting.mp4 — Transcript

**Duration:** 1h 23m 18s
**Frames analyzed:** 42
**Agent:** claude
**Generated:** 2026-04-19T10:00:00+09:00

---

## 00:00–00:28

**SPEAKER_00:** 네, 오늘은 1분기 리뷰 시작하겠습니다.
**SPEAKER_01:** 준비됐습니다.

> 🖼️ **[00:15]** Zoom 화면 공유 상태. "Q1 Review" 슬라이드 표지, 우측 하단 회사 로고. 참가자 썸네일 4명 상단 바.

## 00:28–01:05
...
```

### 7.2 JSON

```json
{
  "source": "meeting.mp4",
  "agent": "claude",
  "duration_sec": 4998.3,
  "generated_at": "2026-04-19T10:00:00+09:00",
  "audio_segments": [
    {"start": 0.0, "end": 12.5, "speaker": "SPEAKER_00", "text": "..."}
  ],
  "frame_descriptions": [
    {"timestamp": 15.0, "image": "frames/frame_00001.png", "description": "..."}
  ]
}
```

## 8. Error handling

| 상황 | 동작 |
|---|---|
| ffmpeg 실패 | abort + stderr 출력 |
| mlx_whisper 실패 | abort |
| agent 호출 실패 (retry 후) | 해당 프레임 placeholder 삽입, 계속 진행 |
| video에 audio track 없음 | audio 단계 skip, 프레임 해석만 수행, audio_context는 빈 문자열 |
| 프레임 0개 (너무 짧은 영상 등) | audio 전사만 출력 |
| 모든 agent 호출 실패 | 경고 출력 + audio transcript만 저장 + exit code 1 |

## 9. File layout

```
video_transcription/
├── README.md                          # 사용법, prerequisites, 각 agent 세팅 가이드
├── requirements.txt
├── video_transcribe.py                # 메인 엔트리포인트
├── src/
│   ├── __init__.py
│   ├── audio.py
│   ├── frames.py
│   ├── merge.py
│   ├── preflight.py
│   └── agents/
│       ├── __init__.py
│       ├── base.py
│       ├── claude.py
│       ├── codex.py
│       └── gemini.py
├── prompts/
│   └── default.md
├── examples/
│   ├── meeting_recipe.md
│   └── lecture_recipe.md
├── docs/
│   └── superpowers/specs/2026-04-19-video-transcription-design.md  (이 문서)
└── .gitignore
```

## 10. Prerequisites (README 요약)

```bash
# 필수
brew install ffmpeg
python3 --version   # 3.9+
pip3 install -r requirements.txt

# 오디오 전사 백엔드 (whisper_transcription 그대로)
pip3 install mlx-whisper
uv tool install whispermlx --python 3.12   # --diarize 쓸 때만

# Agent 하나 선택해서 설치
## Claude (기본)
curl -fsSL https://claude.ai/install.sh | bash
claude   # 첫 실행 OAuth 로그인

## Codex
npm install -g @openai/codex
codex login

## Gemini
npm install -g @google/gemini-cli
gemini   # Google 계정 로그인 또는 GEMINI_API_KEY

# 환경 검증
python video_transcribe.py --check --agent claude
```

## 11. Testing strategy (초기)

**End-to-end smoke test 영상:** `/Users/dion/Downloads/ep02-final.mp4`
- 1280x720, H.264 + AAC, 약 5분 49초, 43MB
- hybrid 기본값(30/10/60/200)으로 실행 시 6~15 프레임 정도 예상
- 3개 agent 각각으로 돌려보고 출력 결과 비교 (특히 프레임 해석 품질 + 병렬 호출 안정성)
- 기대 산출물: `ep02-final.md`, `ep02-final.json`, (옵션) `frames/*.png`

**추가 검증:**
- `src/frames.py`의 hybrid 로직은 단위 테스트 작성 (min/max interval 병합 로직이 까다로움)
- preflight 단독 테스트 (which mocking)
- 전체 pytest 스위트는 초기 버전엔 YAGNI — 사용자 피드백 들어오면 추가

## 12. Out of scope (v0.1)

- GUI / 웹 UI
- 실시간 스트리밍 전사
- 번역
- 배치 모드 (여러 비디오)
- Anthropic/OpenAI/Google API 직접 호출 adapter — 나중에 `agents/anthropic_api.py` 등으로 플러그인 추가
- PyPI 배포
- Docker 이미지
- 화자별 색상 지정 등 출력 커스터마이즈

## 13. Milestones (implementation plan에서 구체화)

1. 레포 수동 fork + 기본 구조 셋업
2. preflight + frames 추출 (hybrid 로직)
3. audio 파이프라인 리팩터링 (`transcribe.py` → `src/audio.py` + 비디오에서 오디오 추출 함수)
4. agent adapter 3종 + 공통 인터페이스
5. 병렬 호출 + merge + 출력 포맷 2종
6. README + 예시 프롬프트
7. 샘플 비디오로 end-to-end 검증
