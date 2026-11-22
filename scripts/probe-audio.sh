#!/data/data/com.termux/files/usr/bin/sh
# Phase 1 On-Device Audio Feasibility Probe for Sony Xperia 1 III (Termux)
set -u

echo "=== ZIA Audio Feasibility Probe ==="
echo "Date: $(date)"
echo "Android SDK: $(getprop ro.build.version.sdk 2>/dev/null || echo unknown)"
echo "Device: $(getprop ro.product.model 2>/dev/null || uname -m)"

echo "\n[1] Acquiring wake lock..."
termux-wake-lock || echo "WARN: termux-wake-lock failed"

echo "\n[2] Checking Termux:API battery telemetry..."
termux-battery-status || echo "FAIL: termux-battery-status failed (check Termux:API F-Droid install)"

echo "\n[3] Testing speaker TTS playback (MUSIC stream)..."
termux-tts-speak -s MUSIC "Zia audio probe starting." || echo "FAIL: termux-tts-speak failed"

echo "\n[4] Testing termux-microphone-record (3s file capture)..."
TMP_REC="$HOME/zia_probe_mic.m4a"
rm -f "$TMP_REC"
termux-microphone-record -f "$TMP_REC" -l 3 -r 16000 -c 1 || echo "FAIL: termux-microphone-record start"
sleep 4
termux-microphone-record -q >/dev/null 2>&1 || true
if [ -s "$TMP_REC" ]; then
    echo "OK: recorded $(wc -c < "$TMP_REC") bytes to $TMP_REC"
else
    echo "FAIL: empty or missing recording (grant Microphone permission to Termux:API)"
fi

echo "\n[5] Testing PulseAudio + module-sles-source + parec continuous PCM..."
pulseaudio --start --exit-idle-time=-1 || echo "WARN: pulseaudio start returned non-zero"
pactl load-module module-sles-source 2>/dev/null || true
pactl list sources short || true

RAW_PCM="$HOME/zia_probe.raw"
rm -f "$RAW_PCM"
timeout 4 parec --format=s16le --rate=16000 --channels=1 > "$RAW_PCM" 2>/dev/null || true
if [ -s "$RAW_PCM" ]; then
    BYTES="$(wc -c < "$RAW_PCM")"
    echo "OK: parec captured $BYTES bytes in ~4s (expected ~128000 bytes for 16kHz s16le mono)"
else
    echo "FAIL: parec captured 0 bytes"
fi

echo "\n[6] Screen-off test instructions:"
echo "Turn OFF the screen (power button) and run:"
echo "  sleep 10 && timeout 5 parec --format=s16le --rate=16000 --channels=1 > ~/zia_screenoff.raw && ls -lh ~/zia_screenoff.raw"
echo "Then inspect ~/zia_screenoff.raw size & amplitude to confirm non-zero audio with screen off."
