# Phase 1: Termux Audio Pipeline Feasibility Report (Sony Xperia 1 III)

> **Status:** Architecture & API research verified; **physical Xperia 1 III hardware validation pending** (all live microphone/speaker/screen-off claims below explicitly separate verified platform behavior from tests that must be executed on the device).

## 1. Target Device & Constraints

- **Device:** Sony Xperia 1 III (Snapdragon 888, 12 GB RAM, stereo speakers, built-in mics, physically broken display)
- **OS:** Android 11 (launch) up to Android 13 (final official Xperia 1 III release)
- **Hard Constraint:** Terminal-only Python application inside **Termux** — no custom Android APK, no custom Android foreground service code, no GUI.

---

## 2. Feasibility Questions & Findings

### 2.1 Microphone Recording Interfaces Available in Termux

Standard Linux audio interfaces (**PyAudio**, **PortAudio**, **ALSA `/dev/snd`**) **cannot** access Android microphone hardware directly in unrooted Termux due to Android SELinux sandboxing and the Android Audio HAL.

Two terminal-accessible interfaces exist in Termux:

| Interface | Underlying Android Layer | Streaming PCM? | Suitability |
| :--- | :--- | :--- | :--- |
| **`termux-microphone-record`** (`Termux:API`) | Android Java `MediaRecorder` | **No** (file-based `.m4a`/`.amr`/`.opus`, requires `-q` to finalize file) | Unsuitable for continuous low-latency wake-word detection. Viable only as a half-duplex fallback recorder for push-to-talk or fixed-window capture. |
| **PulseAudio + `module-sles-source` + `parec`** | Android native C/NDK **OpenSL ES** | **Yes** (`parec --format=s16le --rate=16000 --channels=1` streams raw 16-bit PCM to stdout) | **Primary candidate** for real-time VAD + wake-word frame ingestion in Python (`asyncio.create_subprocess_exec`). |

### 2.2 Continuous Low-Latency Audio Frames

- With `pulseaudio --start --exit-idle-time=-1` and `pactl load-module module-sles-source`, `parec` produces a continuous raw PCM byte stream on stdout.
- At `16000 Hz`, `s16le`, `1 channel`, each 80 ms frame is `16000 * 0.08 * 2 = 2560 bytes` (1280 samples), matching `openWakeWord` (80 ms chunks) and `Vosk` / `webrtcvad` framing.
- **Known failure mode:** `module-sles-source` fails to load or yields silence if `Termux:API` has not been granted Android `RECORD_AUDIO` permission or if Android blocks background sensor initialization.

### 2.3 Screen-Off & Locked-Device Behavior (Critical Risk Area)

On Android 11–13, three separate OS mechanisms can break screen-off microphone capture:

1. **Android 11+ While-In-Use Microphone Restriction:** Background apps without an active foreground service of type `microphone` are denied microphone access or receive zeroed audio frames when initiated from the background. Termux's persistent notification (activated via `termux-wake-lock`) acts as Termux's foreground service context on most ROMs, **but** starting capture while the screen is already off or after Android revokes background mic access can still fail or go silent.
2. **Android 12/13 Phantom Process Killer:** Kills background child processes spawned by Termux (such as `pulseaudio`, `parec`, or `python`) when `max_phantom_processes` (default 32) or background freezer limits trigger. Mitigation requires ADB (`adb shell settings put global activity_manager_constants max_phantom_processes=2147483647` and disabling background process freezer / Developer Options *Disable child process restrictions*).
3. **Sony STAMINA Mode & Doze:** Sony's STAMINA mode aggressively suspends background CPU, wake locks, and network sockets. STAMINA Mode and Adaptive Battery must be disabled, and Termux + Termux:API + Termux:Boot must be set to **Unrestricted** battery usage.

**Verdict:** Screen-off continuous capture via `parec` + `termux-wake-lock` is **architecturally plausible** on Xperia 1 III, **not guaranteed** across 8-hour screen-off / lock transitions without hardware testing using `scripts/probe-audio.sh`.

### 2.4 Local Wake-Word Engine Compatibility (`openWakeWord` vs Alternatives)

- **`openWakeWord`** depends on `onnxruntime` (or `tflite-runtime`) + `numpy` + `scipy`.
- **ARM64 Bionic libc caveat:** Standard PyPI `onnxruntime` wheels target `manylinux` (`glibc`), whereas Termux uses Android **Bionic libc**. `pip install onnxruntime` frequently fails in Termux unless a Termux-native build/wheel is used or `onnxruntime` is built from source.
- **Fallback wake-word options if `onnxruntime` fails or streaming stalls:**
  1. **Vosk keyword spotting** (`vosk` grammar restricted to `["hey zia", "[unk]"]`) — verify `vosk` wheel/so compatibility on Termux `aarch64` first.
  2. **Termux:API / SSH / local socket trigger** (`zia trigger` or hardware volume/media button listener) + `termux-microphone-record` half-duplex capture if continuous background mic streaming proves unstable on the Xperia ROM.

### 2.5 Speaker Playback (Simultaneous vs Sequential)

- **Android TTS (`termux-tts-speak -s MUSIC`):** Uses Android's system TTS engine (e.g., Google Speech Services). Routing to `MUSIC` avoids `NOTIFICATION` DND suppression and plays through Xperia stereo speakers while the screen is off (with `termux-wake-lock`).
- **Offline Native TTS (`espeak-ng`, `piper`):** `espeak-ng` is available directly via `pkg install espeak-ng` and can synthesize `.wav` files played via `paplay` or `mpv` or `termux-media-player`.
- **Sequential (Half-Duplex) First:** Simultaneously keeping the mic active during loudspeaker playback causes acoustic echo (ZIA hears its own voice). ZIA must **mute/discard mic frames during `SPEAKER_OUTPUT` plus a configurable echo-cooldown window (default 400 ms)** before transitioning to `FOLLOW_UP_LISTENING` or `WAKE_WORD_LISTENING`.
- **Post-TTS Mic Recovery:** Some Android Audio HALs suspend or preempt an OpenSL ES capture stream when another audio stream (`MediaPlayer` / Android `TextToSpeech`) opens audio focus. Hardware testing must verify whether `parec` resumes cleanly after `termux-tts-speak` or requires restarting the `parec` subprocess.

### 2.6 Required Initial Android Permissions & Setup

Because the Xperia 1 III display is physically broken, initial setup requires either:
- USB-C to HDMI external display + USB mouse/keyboard (Xperia 1 III supports DP Alt Mode), **or**
- Previously authorized `adb` + `scrcpy` over USB from the Windows PC.

Required on-device prerequisites:
1. Install **Termux**, **Termux:API**, and **Termux:Boot** all from **F-Droid** (same signing key; never mix Google Play and F-Droid builds).
2. Open **Termux:API** and **Termux:Boot** once manually so Android registers their receivers/permissions.
3. Grant **Microphone** permission to **Termux:API** (*Allow while using the app*, disable *Remove permissions if app is unused*).
4. Disable **Battery Optimization** (*Unrestricted*) for **Termux**, **Termux:API**, and **Termux:Boot**.
5. Disable **Sony STAMINA Mode** and **Adaptive Battery**.
6. On Android 12/13, disable Phantom Process Killer via ADB.
7. Run `termux-wake-lock` before starting the audio daemon.

---

## 3. Test Matrix: Windows vs Physical Sony Xperia 1 III

| Component / Test | Windows Dev Machine | Physical Xperia 1 III Required |
| :--- | :---: | :---: |
| Config loading, secret redaction, CLI parsing | Yes | - |
| SQLite schema, FTS5 search, ranking, CRUD, export | Yes | - |
| Groq & Gemini API clients, fallback, backoff, token caps | Yes | - |
| State machine transitions, local tools, offline fallback | Yes | - |
| Reminder scheduler & missed-reminder recovery | Yes | - |
| `termux-microphone-record` file capture | - | **Yes** |
| `pulseaudio` + `module-sles-source` + `parec` 16kHz PCM stream | - | **Yes** |
| Mic capture with screen off (10 min / 1 hr / 8 hr) | - | **Yes** |
| Mic capture while phone locked | - | **Yes** |
| `termux-tts-speak` screen-off playback & post-TTS mic resume | - | **Yes** |
| `openWakeWord` / `onnxruntime` / `vosk` / `whisper.cpp` ARM64 install & CPU load | - | **Yes** |
| Incoming phone call / audio focus interruption recovery | - | **Yes** |
| Termux:Boot auto-start after device reboot | - | **Yes** |
| Thermal & charging 24/7 stability | - | **Yes** |

---

## 4. Next Step for Hardware Verification

Copy the project to Termux on the Xperia 1 III and run:
```bash
bash scripts/probe-audio.sh
```
This script verifies `termux-battery-status`, `termux-tts-speak`, `termux-microphone-record`, `pulseaudio` + `module-sles-source` + `parec` frame rate, and a 60-second screen-off capture test before enabling always-listening mode.
