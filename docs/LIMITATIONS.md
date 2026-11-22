# Known Limitations & Architectural Trade-offs

This document tracks verified technical constraints of running a terminal-only AI assistant in Termux on an unrooted Sony Xperia 1 III.

## 1. Android Background & Microphone Restrictions
- **No bypass of Android permissions:** Termux cannot override Android's runtime permission prompts, SELinux audio policies, or locked-device sensor rules without user authorization.
- **No guaranteed post-reboot mic auto-start:** `Termux:Boot` can launch `sshd` and ZIA after reboot once unlocked/booted, but Android 11+ restricts background microphone initialization if the device has not been unlocked after reboot (Direct Boot / While-In-Use policy).
- **Phantom Process Killer (Android 12/13):** Without ADB `max_phantom_processes` adjustment or Developer Options toggle, Android may send `SIGKILL` (`Signal 9`) to child processes (`pulseaudio`, `parec`, `python`) after the screen turns off.
- **Sony STAMINA Mode:** Must be turned off; otherwise wake locks and background network sockets are throttled.

## 2. Audio Stack Limitations
- **`termux-microphone-record` cannot stream:** It writes compressed/container files via Android `MediaRecorder` and is unsuitable for continuous wake-word detection.
- **`parec` (`module-sles-source`) is unverified on target ROM:** Must be tested on the physical Xperia 1 III with the screen off and locked before claiming always-on wake-word support.
- **Half-duplex only initially:** Full-duplex barge-in (interrupting ZIA mid-sentence while speaker output is playing) requires hardware acoustic echo cancellation (AEC), which is not exposed reliably to `parec`. ZIA operates half-duplex with a post-TTS echo-guard delay.

## 3. Native Python Wheel Compatibility on Termux ARM64
- PyPI wheels built against `glibc` (`manylinux_aarch64`) — notably `onnxruntime` (used by `openWakeWord`) and sometimes `vosk` — may fail to install on Termux (`Bionic libc`).
- `httpx` and stdlib `sqlite3` (with FTS5), `tomllib`, and `asyncio` work cleanly on Termux Python 3.11+ without compilation issues.
- `whisper.cpp` is compiled natively on-device with `clang` + `cmake`/`make` using ARM NEON rather than relying on a fragile Python wheel.

## 4. Free-Tier LLM Constraints
- Groq and Gemini free tiers enforce strict RPM, TPM, RPD, and TPD limits (`HTTP 429`).
- **Gemini Free Tier privacy note:** Prompts sent to the Gemini API free tier may be reviewed/used by Google per AI Studio terms; avoid sending sensitive secrets, and enforce ZIA's sensitive-memory approval guard.

## 5. Broken Display Constraint
- With a physically broken screen, initial Android permissions, F-Droid installs, and battery exclusions require either an external USB-C HDMI monitor + USB mouse/keyboard or previously authorized ADB + `scrcpy`.
