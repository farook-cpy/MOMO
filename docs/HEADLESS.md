# Headless 24/7 Operation & Xperia Hardware Verification Procedures

> Never mark any hardware scenario below as passed until executed and observed on the physical Sony Xperia 1 III.

## 1. Startup & Supervision
- Run `bash scripts/setup-boot.sh` to register `~/.termux/boot/start-zia`.
- Normal headless start over SSH:
  ```sh
  bash scripts/start.sh
  zia status
  zia logs -n 50
  ```

## 2. Physical Xperia 1 III Verification Procedures

### A. Screen Off — 10 Minutes
1. Verify `termux-wake-lock` notification is active and `zia start --daemon` is running.
2. Turn display off (`input keyevent 26` or power button).
3. Wait 10 minutes without touching the phone.
4. Over SSH: check `zia status`, `ps aux | grep parec`, and trigger `zia test-mic` or speak wake word. Record whether frames/RMS remain non-zero.

### B. Screen Off — 1 Hour
1. Leave screen off and locked for 60 minutes (both on charger and on battery).
2. Over SSH: inspect `zia logs -n 100` for watchdog warnings, SQLite integrity, and PulseAudio stream continuity.
3. Speak wake word or run `zia test-speaker`. Verify audio output occurs without waking/unlocking screen.

### C. Screen Off — 8 Hours (Overnight)
1. Leave device on charger overnight (8 hours) with screen off.
2. Check morning logs for process uptime (`zia status`), `parec` restarts, Wi-Fi drops, and morning reminder delivery.

### D. Wi-Fi Disconnection & Recovery
1. Disable router/Wi-Fi temporarily while ZIA is running.
2. Ask ZIA `"What time is it?"` and `"List my reminders"` -> confirm offline fallback answers cleanly.
3. Re-enable Wi-Fi -> confirm next conversational query reaches Groq/Gemini without restarting ZIA.

### E. API Failure / Quota Exhaustion
1. Temporarily set invalid `GROQ_API_KEY` -> verify automatic failover to Gemini.
2. Set both keys invalid or `offline_mode = true` -> verify graceful offline command fallback without crash.

### F. Incoming Phone Calls & Microphone Interruption
1. Place a cellular/VoIP call to the Xperia while `parec` is streaming.
2. Observe whether Android audio focus preempts `parec` (causing `EOF` or silence).
3. End the call and verify whether ZIA's watchdog/mic restart recovers capture cleanly.

### G. Repeated Wake-Word Activation & Half-Duplex Echo Guard
1. Trigger 10 consecutive voice turns back-to-back.
2. Verify ZIA's spoken response does not self-trigger the wake word or STT during `echo_cooldown_ms`.

### H. Device Reboot
1. Reboot the Xperia (`adb reboot`).
2. Wait 60 seconds without unlocking the screen; check whether SSH (`8022`) and `zia status` come up via `Termux:Boot`.
3. Test whether microphone access requires screen unlock before starting voice mode.

### I. Charging & Thermal Stress
1. Keep device on USB-PD charger for 4+ hours running continuous wake-word + scheduler.
2. Monitor `termux-battery-status` temperature and `zia logs` for `thermal_warn_celsius` warnings.
