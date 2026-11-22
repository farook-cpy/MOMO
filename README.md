# ZIA — Terminal-Only Personal AI Companion

Headless, voice-first Python personal AI companion designed to run inside **Termux** on a **Sony Xperia 1 III** (no custom Android app, no APK, no GUI).

## Implemented Capabilities
- **Phase 1 — Termux Audio Feasibility (`docs/AUDIO_FEASIBILITY.md`, `scripts/probe-audio.sh`):** Documented `termux-microphone-record` vs PulseAudio (`module-sles-source` + `parec`), Android 11–13 background mic & Phantom Process Killer constraints, Sony STAMINA mitigations, and hardware test matrix.
- **Phase 2 — Modular Cloud & Offline AI (`intelligence/`):** Async HTTPX clients for **Groq** (primary) and **Google Gemini** (fallback), retry with exponential backoff, HTTP 429 rate-limit handling, daily request/token quota caps, SSE streaming, secret redaction in logs, and deterministic offline fallback.
- **Phase 3 — Persistent SQLite + FTS5 Memory (`memory/`):** 10 normalized SQLite tables (`conversations`, `messages`, `personal_preferences`, `important_facts`, `projects`, `tasks`, `events`, `reminders`, `conversation_summaries`, `memory_metadata`) + FTS5 `memory_fts` index, BM25 + importance + recency ranking, duplicate detection, sensitive-data approval guard, extractive summaries, backup/restore, and JSON export.
- **Phase 4–7 Foundations (`audio/`, `core/`, `automation/`, `system/`):** Half-duplex voice state machine, pure-Python energy VAD, PulseAudio `parec` & `termux-microphone-record` adapters, `whisper.cpp` / `vosk` / `openWakeWord` adapters, `termux-tts-speak` / `espeak-ng` TTS, 14 validated local tools, timezone-aware reminder scheduler with missed-reminder recovery, watchdog supervisor, `zia doctor`, and full CLI.

## Quick Start
```bash
pip install -r requirements.txt
python main.py doctor
python main.py chat
```

See [docs/SETUP.md](docs/SETUP.md), [docs/AUDIO_FEASIBILITY.md](docs/AUDIO_FEASIBILITY.md), [docs/HEADLESS.md](docs/HEADLESS.md), and [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
