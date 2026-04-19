<div align="center">

# 🎬 video_transcription

**Turn any video into a rich, timestamped transcript — audio + on-screen visuals, merged.**

Local. Fast. No API keys required (if you already have Claude Code / Codex / Gemini CLI).

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-black.svg)](https://support.apple.com/en-us/116943)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Built on mlx_whisper](https://img.shields.io/badge/audio-mlx__whisper-orange.svg)](https://github.com/ml-explore/mlx-examples/tree/main/whisper)

[Features](#-features) · [Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Agents](#-pick-your-agent) · [Example Output](#-example-output)

</div>

---

## 💡 Why?

Traditional transcription tools give you **text only**. You lose everything on screen — slides, charts, code, UI screenshots. You end up with a transcript that says *"as you can see here..."* and no way to know what *here* is.

`video_transcription` fixes that by running two passes:

1. **Audio** is transcribed with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (hallucination-guarded, Metal-accelerated).
2. **Frames** are pulled via scene detection + hybrid interval rules, then handed to **your local AI CLI agent** (Claude Code, Codex, or Gemini) which describes each frame *with the surrounding audio as context* — so it understands what it's looking at.

The result: one interleaved timeline you can read, search, or feed back into another LLM.

## ✨ Features

- 🎯 **Audio + visual in one transcript** — not two separate files you have to align yourself
- 🔌 **Pluggable AI agents** — swap between `claude`, `codex`, `gemini` with one flag
- 🧠 **Context-aware frame descriptions** — each frame is described *knowing* what was just said
- ⚡ **Metal-accelerated** — `mlx-whisper` on Apple Silicon runs transcription faster than realtime
- 🎛️ **Smart frame selection** — hybrid mode combines scene-change detection with min/max interval guards, plus bitwise dedup for static scenes
- 🗣️ **Speaker diarization (optional)** — via pyannote through `whispermlx`
- 🌐 **Multilingual** — Whisper auto-detects language (tested on Korean, Indonesian, English; works for 99 languages)
- 📄 **Markdown + JSON output** — human-readable and machine-parseable
- 🔒 **Runs locally** — your video never leaves your machine (your CLI agent's cloud calls aside)

## 🚀 Quick Start

```bash
# 1. Clone
git clone git@github.com:treesoop/video_transcription.git
cd video_transcription

# 2. Install common deps
brew install ffmpeg
pip3 install -r requirements.txt mlx-whisper

# 3. Install ONE agent CLI (Claude is the default)
curl -fsSL https://claude.ai/install.sh | bash
claude   # first run: OAuth sign-in

# 4. Verify
python3 video_transcribe.py --check --agent claude

# 5. Transcribe
python3 video_transcribe.py my_video.mp4
# → my_video.md
```

That's it. Drop in a video, get a transcript back.

## ⚙️ How It Works

```
                         ┌─────────────────────────────────────────┐
                         │             video.mp4                    │
                         └────────────────┬────────────────────────┘
                                          │
                ┌─────────────────────────┴─────────────────────────┐
                ▼                                                   ▼
    ┌───────────────────────┐                      ┌────────────────────────────┐
    │   ffmpeg → audio.wav  │                      │  ffmpeg scene detection +  │
    │                       │                      │  hybrid interval guards    │
    │   mlx_whisper         │                      │  + MD5 dedup               │
    │   (hallucination-safe)│                      └──────────────┬─────────────┘
    └──────────┬────────────┘                                     │
               │                                                  ▼
    (opt) whispermlx diarize                          ┌──────────────────────┐
               │                                      │  your local agent    │
               ▼                                      │  CLI (parallel x4)   │
    ┌──────────────────────┐                          │  claude | codex |    │
    │ audio_segments:      │  ◀── audio ±30s context  │  gemini              │
    │  [start, end, text,  │  ─────────────────────▶  │                      │
    │   speaker?]          │                          │  describes each      │
    └──────────┬───────────┘                          │  frame contextually  │
               │                                      └──────────┬───────────┘
               │                                                 │
               └────────────────────┬────────────────────────────┘
                                    ▼
                     ┌──────────────────────────────┐
                     │   timeline merge (by time)   │
                     └─────────────┬────────────────┘
                                   │
                      ┌────────────┴────────────┐
                      ▼                         ▼
                 output.md                 output.json
              (human-friendly)          (machine-parseable)
```

The clever bit is the **audio → context → frame** flow. Instead of asking an agent to describe a frame cold, we slice the audio transcript around that moment (±30s by default) and feed it in. The agent stops guessing and starts grounding.

## 🤖 Pick Your Agent

| Agent | How it's called | Cost model | Install |
|---|---|---|---|
| **Claude Code** (default) | `claude -p` + Read tool | Claude Pro/Max subscription or API key | `curl -fsSL https://claude.ai/install.sh \| bash` |
| **Codex** | `codex exec -i <image>` | ChatGPT Plus/Pro or OpenAI API key | `npm install -g @openai/codex` |
| **Gemini** | `gemini -p "@<image>"` | Free tier available + Google API | `npm install -g @google/gemini-cli` |

Swap with a flag:

```bash
python3 video_transcribe.py my_video.mp4 --agent codex
python3 video_transcribe.py my_video.mp4 --agent gemini
```

Every adapter implements the same `BaseAgent.describe_frame(image, audio_context, system_prompt)` interface, so adding another CLI (Aider, Cursor headless, your own wrapper) is a 30-line file in `src/agents/`.

## 📺 Example Output

From an Indonesian psychology explainer video (5m 49s, mlx_whisper auto-detected the language, Claude described each frame):

```markdown
# ep02-final.mp4 — Transcript

**Duration:** 05:49 · **Frames analyzed:** 6 · **Agent:** claude

---

00:00 Kalau ada seseorang yang diam-diam mengendalikan pikiranmu...

> 🖼️ **[00:00]** Dark stage with a giant hand holding strings from above,
> a small paper-puppet boy hanging in the center, a small lantern to the
> right. The scene visually mirrors the audio's metaphor of "social
> influence techniques" — an unseen manipulator pulling the strings.

00:26 Dalam psikologi, ini disebut teknik pengaruh sosial.
00:33 Hari ini, aku akan bongkar lima trik yang paling sering kamu alami.
00:42 Pertama, teknik kaki di pintu.

> 🖼️ **[03:04]** Illustrated phone screen showing the Tokopedia shopping
> app. Top bar: "tokopedia". Product page for "Sepatu Kasual Pria" priced
> at "Rp 1.200.000". Bottom tabs: Beranda / Feed / Official Store /
> Wishlist / Transaksi. Buttons: "Beli Langsung" and "Keranjang".
> Exactly matches the audio's anchoring example — the Rp1.2M original
> price being shown before the flash-sale reveal.
```

The frame descriptions *know* what the narrator is talking about because the audio context was passed alongside the image.

## 🎛️ CLI Reference

```bash
python3 video_transcribe.py VIDEO_PATH [options]
```

| Flag | Default | Description |
|---|---|---|
| `--agent` | `claude` | `claude` \| `codex` \| `gemini` |
| `--mode` | `hybrid` | Frame extraction strategy: `hybrid`, `interval`, or `scene` |
| `--interval` | `30` | Fixed seconds between frames (for `interval` mode) |
| `--min-interval` | `10` | Hybrid: minimum seconds between kept frames |
| `--max-interval` | `60` | Hybrid: force-insert a frame in gaps longer than this |
| `--max-frames` | `200` | Hard cap on total frames analyzed |
| `--scene-threshold` | `0.3` | ffmpeg scene sensitivity (0–1) |
| `--diarize` | off | Enable speaker diarization (requires `--hf-token`) |
| `--hf-token` | `$HF_TOKEN` | HuggingFace token for pyannote |
| `--prompt-file` | `prompts/default.md` | Override the frame-description system prompt |
| `--context-window` | `30` | ±seconds of audio context fed to each frame |
| `--format` | `md` | `md` \| `json` \| `both` |
| `--parallel` | `4` | Concurrent agent invocations |
| `--keep-frames` | off | Preserve extracted PNGs (default: deleted after run) |
| `--check` | — | Preflight check (verifies all deps) and exit |
| `-o`, `--output-dir` | `.` | Where to write the transcript |

## 🎨 Custom Prompts

Tailor the frame-description style for your use case. Two recipes ship with the repo:

```bash
# Business meeting recordings (Zoom, Meet, etc.)
python3 video_transcribe.py meeting.mp4 --prompt-file examples/meeting_recipe.md

# Lectures / technical tutorials
python3 video_transcribe.py lecture.mp4 --prompt-file examples/lecture_recipe.md
```

Or write your own — prompts are just plain text. The system prompt is sent to each frame call, so describe *what you want the agent to focus on* (slides, code, diagrams, product UI, etc.).

## 🗣️ Speaker Diarization

Need "who said what"? Add `--diarize`:

```bash
python3 video_transcribe.py meeting.mp4 --diarize --hf-token $HF_TOKEN
```

Output becomes:

```markdown
**[SPEAKER_00]** 00:00 Let's start today's review.
**[SPEAKER_01]** 00:08 Sounds good, I'm ready.
```

(HuggingFace token for [pyannote diarization model](https://huggingface.co/pyannote/speaker-diarization-community-1) is free to get — just click "Agree" on the model page.)

## 🏗️ Architecture

```
video_transcription/
├── video_transcribe.py       # CLI orchestrator
├── src/
│   ├── audio.py              # mlx_whisper wrapper + video→wav extract
│   ├── frames.py             # ffmpeg frame extraction (3 modes + dedup)
│   ├── merge.py              # timeline builder + md/json renderers
│   ├── preflight.py          # dependency check (--check)
│   └── agents/
│       ├── base.py           # BaseAgent(ABC) + AgentInvocationError
│       ├── claude.py         # claude -p + Read-tool image pattern
│       ├── codex.py          # codex exec -i
│       └── gemini.py         # gemini -p @path
├── prompts/
│   └── default.md            # built-in frame-description prompt
├── examples/
│   ├── meeting_recipe.md
│   └── lecture_recipe.md
└── tests/                    # 31 unit tests, all pure-function where possible
```

**Design principles:**

- 🎯 **One responsibility per file.** `frames.py` only knows frames. `merge.py` only knows timelines. Adapters only know their one CLI.
- 🧪 **TDD on pure functions.** The hybrid frame-selection algorithm (min/max interval + downsample + dedup) has 10 unit tests and zero subprocess dependencies.
- 🚫 **No shell=True.** Everything is a `list[str]` to `subprocess.run`. Zero injection surface.
- 🔁 **Retry-once on CLI failures.** Transient network errors don't kill the whole run — just that one frame gets a `[frame description failed: ...]` placeholder and the pipeline continues.

## 🧪 Dev

```bash
python3 -m pytest -v    # 31 tests
```

## 📚 Built On

- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Metal-accelerated Whisper
- [whispermlx](https://github.com/kalebjs/whispermlx) — diarization via pyannote
- [ffmpeg](https://ffmpeg.org/) — audio/frame extraction
- [treesoop/whisper_transcription](https://github.com/treesoop/whisper_transcription) — the audio pipeline started here, got forked and expanded

## 🤔 Comparison

| | **video_transcription** | paid SaaS (Notta, etc.) | whisper + custom scripts |
|---|---|---|---|
| Audio transcript | ✅ | ✅ | ✅ |
| **Frame descriptions** | ✅ | ❌ | ❌ (you'd build it) |
| **Context-aware frames** | ✅ (audio → agent) | — | — |
| Speaker diarization | ✅ (optional) | ✅ | ⚠️ (setup pain) |
| Runs locally | ✅ | ❌ | ✅ |
| Hallucination guarded | ✅ | ✅ | ⚠️ (default settings are lossy) |
| Multilingual | ✅ 99 langs | ✅ | ✅ |
| Price | Free | Subscription | Free (but your time) |
| Swap AI provider | ✅ (one flag) | ❌ | — |

## 📜 License

MIT — same as upstream. Do whatever you want, attribution appreciated.

## 🙌 Contributing

PRs welcome. Good places to start:

- Add a new agent adapter (Aider, Cursor headless, local LM Studio, etc.) — look at `src/agents/gemini.py` for the ~30-line template
- Imagehash-based visual dedup (current MD5 dedup only catches bitwise-identical PNGs; visually-similar frames still get described multiple times)
- Single-pass ffmpeg `select` filter for `_extract_at_timestamps` (current: one ffmpeg call per frame, fine up to ~100 frames)
- Windows / Linux support (currently macOS-only because of mlx_whisper)

---

<div align="center">

Made by [Dion](https://github.com/treesoop) · Part of [Treesoop](https://treesoop.com)

**If this is useful, please ⭐ the repo — it's how I know what's worth building next.**

</div>
