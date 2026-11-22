"""Pure-Python Voice Activity Detection (VAD) with adaptive noise floor & ZCR."""

from __future__ import annotations

import math
import struct


def pcm16_rms(frame: bytes) -> float:
    """Compute RMS amplitude of little-endian 16-bit signed PCM bytes."""
    count = len(frame) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", frame[: count * 2])
    sum_sq = sum(s * s for s in samples)
    return math.sqrt(sum_sq / count)


def pcm16_zcr(frame: bytes) -> float:
    """Compute zero-crossing rate (0.0 to 1.0) to distinguish speech from low-frequency hum."""
    count = len(frame) // 2
    if count < 2:
        return 0.0
    samples = struct.unpack(f"<{count}h", frame[: count * 2])
    crossings = sum(
        1 for i in range(1, count) if (samples[i - 1] >= 0) != (samples[i] >= 0)
    )
    return crossings / (count - 1)


class EnergyVAD:
    """Adaptive energy + ZCR VAD with minimum-speech-frame gating."""

    def __init__(
        self,
        energy_threshold: float = 260.0,
        silence_timeout_seconds: float = 1.8,
        frame_duration_ms: int = 80,
        min_speech_frames: int = 2,
        min_zcr: float = 0.015,
        max_zcr: float = 0.48,
    ) -> None:
        self.energy_threshold = energy_threshold
        self.frame_duration_ms = max(10, frame_duration_ms)
        self.silence_frames_limit = max(
            1, int((silence_timeout_seconds * 1000.0) / self.frame_duration_ms)
        )
        self.min_speech_frames = max(1, min_speech_frames)
        self.min_zcr = min_zcr
        self.max_zcr = max_zcr
        self.noise_floor = energy_threshold * 0.4
        self.speech_started = False
        self.speech_frames = 0
        self.silent_frames = 0

    def reset(self) -> None:
        self.speech_started = False
        self.speech_frames = 0
        self.silent_frames = 0

    def process_frame(self, frame: bytes) -> tuple[bool, bool]:
        """Return (is_speech, end_of_utterance)."""
        rms = pcm16_rms(frame)
        zcr = pcm16_zcr(frame)
        effective_threshold = max(self.energy_threshold, self.noise_floor * 1.8)
        in_speech_band = self.min_zcr <= zcr <= self.max_zcr
        is_speech = rms >= effective_threshold and in_speech_band

        if is_speech:
            self.speech_frames += 1
            if self.speech_frames >= self.min_speech_frames:
                self.speech_started = True
            self.silent_frames = 0
            return True, False

        # Slowly adapt noise floor on quiet non-speech frames
        if not self.speech_started and rms < effective_threshold:
            self.noise_floor = (0.9 * self.noise_floor) + (0.1 * rms)

        if self.speech_started:
            self.silent_frames += 1
            if self.silent_frames >= self.silence_frames_limit:
                return False, True
        else:
            # Reset transient single-frame clicks
            self.speech_frames = 0
        return False, False


def has_sustained_speech(
    pcm_bytes: bytes,
    frame_bytes: int,
    energy_threshold: float = 450.0,
    min_speech_frames: int = 3,
) -> bool:
    """Return True only if PCM buffer contains sustained speech-band frames."""
    vad = EnergyVAD(
        energy_threshold=energy_threshold,
        min_speech_frames=min_speech_frames,
    )
    speech_count = 0
    for i in range(0, len(pcm_bytes), frame_bytes):
        chunk = pcm_bytes[i : i + frame_bytes]
        if len(chunk) < frame_bytes:
            break
        is_sp, _ = vad.process_frame(chunk)
        if is_sp:
            speech_count += 1
            if speech_count >= min_speech_frames:
                return True
    return False
