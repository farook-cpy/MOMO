"""Termux microphone capture backends (PulseAudio OpenSL ES & Termux:API fallback)."""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import time
import wave
from collections import deque
from pathlib import Path
from typing import AsyncIterator

from audio.vad import EnergyVAD, pcm16_rms
from core.config import AudioConfig

logger = logging.getLogger("zia.audio.mic")


def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return path


class PulseAudioMicrophone:
    """Continuous 16kHz PCM streaming via PulseAudio + module-sles-source + parec."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._proc: asyncio.subprocess.Process | None = None
        self.frame_bytes = int(
            config.sample_rate * (config.frame_duration_ms / 1000.0) * 2 * config.channels
        )

    @staticmethod
    def is_available() -> bool:
        return bool(shutil.which("parec") and shutil.which("pulseaudio") and shutil.which("pactl"))

    def ensure_sles_source(self) -> tuple[bool, str]:
        if not self.is_available():
            return False, "PulseAudio binaries (pulseaudio, pactl, parec) not found on PATH."
        try:
            subprocess.run(
                ["pulseaudio", "--start", "--exit-idle-time=-1"],
                capture_output=True,
                timeout=8,
                check=False,
            )
            sources = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if "OpenSL_ES_source" not in sources.stdout and "sles" not in sources.stdout.lower():
                load = subprocess.run(
                    ["pactl", "load-module", "module-sles-source"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if load.returncode != 0:
                    return False, f"Failed to load module-sles-source: {load.stderr.strip()}"
            return True, "PulseAudio OpenSL ES source ready."
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"PulseAudio initialization error: {exc}"

    async def start(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        ok, msg = self.ensure_sles_source()
        if not ok:
            raise RuntimeError(msg)

        cmd = [
            "parec",
            "--format=s16le",
            f"--rate={self.config.sample_rate}",
            f"--channels={self.config.channels}",
            "--latency-msec=80",
        ]
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info("Started parec microphone stream (%d Hz).", self.config.sample_rate)

    async def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._proc.kill()
        self._proc = None

    async def read_frame(self) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Microphone stream is not started.")
        chunk = await self._proc.stdout.readexactly(self.frame_bytes)
        return chunk

    async def frames(self) -> AsyncIterator[bytes]:
        await self.start()
        try:
            while True:
                yield await self.read_frame()
        except asyncio.IncompleteReadError:
            logger.warning("parec stream ended unexpectedly.")
        finally:
            await self.stop()

    async def record_utterance(self, timeout_seconds: float | None = None) -> bytes:
        """Record speech with pre-roll buffer so opening syllables at normal volume aren't clipped."""
        await self.start()
        vad = EnergyVAD(
            energy_threshold=self.config.vad_energy_threshold,
            silence_timeout_seconds=self.config.silence_timeout_seconds,
            frame_duration_ms=self.config.frame_duration_ms,
        )
        max_sec = timeout_seconds or self.config.max_record_seconds
        deadline = time.monotonic() + max_sec
        pre_roll: deque[bytes] = deque(maxlen=4)  # ~320ms pre-speech buffer
        buf = bytearray()
        captured_start = False
        try:
            while time.monotonic() < deadline:
                frame = await asyncio.wait_for(self.read_frame(), timeout=2.0)
                is_speech, ended = vad.process_frame(frame)
                if not captured_start:
                    pre_roll.append(frame)
                    if is_speech or vad.speech_started:
                        captured_start = True
                        for pf in pre_roll:
                            buf.extend(pf)
                else:
                    buf.extend(frame)
                    if ended:
                        break
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            pass
        return bytes(buf)


class TermuxRecordMicrophone:
    """File-oriented half-duplex fallback using termux-microphone-record."""

    @staticmethod
    def is_available() -> bool:
        return bool(shutil.which("termux-microphone-record"))

    async def record_to_file(self, output_path: Path, seconds: int = 5, sample_rate: int = 16000) -> Path:
        if not self.is_available():
            raise RuntimeError("termux-microphone-record is not installed.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        start_proc = await asyncio.create_subprocess_exec(
            "termux-microphone-record",
            "-f",
            str(output_path),
            "-l",
            str(max(1, seconds)),
            "-r",
            str(sample_rate),
            "-c",
            "1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await start_proc.communicate()
        await asyncio.sleep(seconds + 0.4)
        quit_proc = await asyncio.create_subprocess_exec(
            "termux-microphone-record",
            "-q",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await quit_proc.communicate()
        return output_path


async def test_microphone_stream(config: AudioConfig, duration_seconds: float = 3.0) -> dict[str, object]:
    """Quick diagnostic measuring captured frames and RMS energy."""
    if config.mic_backend == "pulseaudio" and PulseAudioMicrophone.is_available():
        mic = PulseAudioMicrophone(config)
        await mic.start()
        frames = 0
        max_rms = 0.0
        end = time.monotonic() + duration_seconds
        try:
            while time.monotonic() < end:
                frame = await asyncio.wait_for(mic.read_frame(), timeout=2.0)
                frames += 1
                max_rms = max(max_rms, pcm16_rms(frame))
        finally:
            await mic.stop()
        return {
            "backend": "pulseaudio",
            "ok": frames > 0,
            "frames": frames,
            "peak_rms": round(max_rms, 2),
        }

    if TermuxRecordMicrophone.is_available():
        rec = TermuxRecordMicrophone()
        tmp = config.sample_rate
        out = Path("data/mic_test.m4a")
        await rec.record_to_file(out, seconds=max(1, int(duration_seconds)), sample_rate=tmp)
        size = out.stat().st_size if out.exists() else 0
        return {
            "backend": "termux-microphone-record",
            "ok": size > 0,
            "file": str(out),
            "bytes": size,
        }

    return {"backend": "none", "ok": False, "error": "No Termux microphone backend available on this host."}
