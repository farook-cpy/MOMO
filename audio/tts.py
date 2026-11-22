"""Text-to-speech playback backends (Kokoro-82M streaming, Termux TTS, espeak-ng)."""

from __future__ import annotations

import asyncio
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from core.config import AudioConfig

logger = logging.getLogger("zia.audio.tts")


class BaseTTS(ABC):
    @abstractmethod
    async def speak(self, text: str, rate: float | None = None, pitch: float | None = None) -> bool:
        """Speak text aloud. Returns True if playback succeeded."""


class KokoroTTS(BaseTTS):
    """Real-time streaming Kokoro-82M ONNX TTS with pre-warmed model."""

    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        voice: str = "af_sky",
        speed: float = 1.0,
    ) -> None:
        try:
            from kokoro_onnx import Kokoro  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("kokoro-onnx package is not installed.") from exc

        if not model_path.exists():
            raise FileNotFoundError(f"Kokoro model not found: {model_path}")
        if not voices_path.exists():
            raise FileNotFoundError(f"Kokoro voices file not found: {voices_path}")

        self.voice = voice
        self.speed = speed
        # Pre-warm ONNX session once at startup for sub-second first-chunk latency
        self._kokoro = Kokoro(str(model_path), str(voices_path))

    async def speak(self, text: str, rate: float | None = None, pitch: float | None = None) -> bool:
        clean = text.strip()
        if not clean:
            return True

        eff_speed = rate if rate is not None else self.speed
        stream = self._kokoro.create_stream(clean, voice=self.voice, speed=eff_speed, lang="en-us")

        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError:
            sd = None

        if sd is not None:
            async for samples, sample_rate in stream:
                await asyncio.to_thread(sd.play, samples, sample_rate)
                await asyncio.to_thread(sd.wait)
            return True

        # Termux/Linux fallback: stream raw float32le PCM directly to PulseAudio paplay
        if shutil.which("paplay"):
            proc = None
            try:
                async for samples, sample_rate in stream:
                    if proc is None:
                        proc = await asyncio.create_subprocess_exec(
                            "paplay",
                            "--raw",
                            "--format=float32le",
                            f"--rate={sample_rate}",
                            "--channels=1",
                            stdin=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                    if proc.stdin is not None:
                        proc.stdin.write(samples.astype("float32").tobytes())
                        await proc.stdin.drain()
                if proc is not None and proc.stdin is not None:
                    proc.stdin.close()
                    await proc.wait()
                return True
            except Exception as exc:
                logger.warning("paplay streaming failed: %s", exc)
                if proc is not None and proc.returncode is None:
                    proc.kill()
                return False

        logger.warning("No audio playback backend (sounddevice or paplay) available for KokoroTTS.")
        return False


class TermuxTTS(BaseTTS):
    """Uses Android system TTS via termux-tts-speak (MUSIC stream by default)."""

    def __init__(self, stream: str = "MUSIC", rate: float = 1.0, pitch: float = 1.0) -> None:
        self.stream = stream
        self.rate = rate
        self.pitch = pitch

    async def speak(self, text: str, rate: float | None = None, pitch: float | None = None) -> bool:
        if not text.strip():
            return True
        if not shutil.which("termux-tts-speak"):
            logger.warning("termux-tts-speak not found.")
            return False
        proc = await asyncio.create_subprocess_exec(
            "termux-tts-speak",
            "-s",
            self.stream,
            "-r",
            str(rate if rate is not None else self.rate),
            "-p",
            str(pitch if pitch is not None else self.pitch),
            text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        if proc.returncode != 0:
            logger.warning("termux-tts-speak failed: %s", stderr.decode("utf-8", errors="replace"))
            return False
        return True


class EspeakTTS(BaseTTS):
    """Offline native TTS using espeak-ng."""

    async def speak(self, text: str, rate: float | None = None, pitch: float | None = None) -> bool:
        if not text.strip():
            return True
        if not shutil.which("espeak-ng"):
            logger.warning("espeak-ng not found.")
            return False
        proc = await asyncio.create_subprocess_exec(
            "espeak-ng",
            text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=60.0)
        return proc.returncode == 0


class NullTTS(BaseTTS):
    """Silent TTS adapter for headless text-only testing."""

    async def speak(self, text: str, rate: float | None = None, pitch: float | None = None) -> bool:
        return True


def build_tts(config: AudioConfig, root_dir: Path | None = None) -> BaseTTS:
    backend = config.tts_backend.lower()
    if backend == "kokoro":
        base = root_dir or Path.cwd()
        mpath = Path(config.kokoro_model)
        vpath = Path(config.kokoro_voices)
        if not mpath.is_absolute():
            mpath = base / mpath
        if not vpath.is_absolute():
            vpath = base / vpath
        try:
            return KokoroTTS(
                model_path=mpath,
                voices_path=vpath,
                voice=config.kokoro_voice,
                speed=config.tts_rate,
            )
        except Exception as exc:
            logger.warning("KokoroTTS unavailable (%s); falling back.", exc)

    if backend in {"kokoro", "termux_tts"} and shutil.which("termux-tts-speak"):
        return TermuxTTS(
            stream=config.tts_stream,
            rate=config.tts_rate,
            pitch=config.tts_pitch,
        )
    if backend in {"kokoro", "termux_tts", "espeak"} and shutil.which("espeak-ng"):
        return EspeakTTS()
    return NullTTS()
