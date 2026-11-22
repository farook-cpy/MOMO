"""Speech-to-text adapters (whisper.cpp CLI and Vosk offline STT)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from core.config import AudioConfig

logger = logging.getLogger("zia.audio.stt")

_HALLUCINATION_TAGS = re.compile(r"\[[^\]]+\]|\([^)]+\)|\*[^*]+\*")
_NOISE_PHRASES = frozenset(
    {
        "", ".", "..", "...", "you", "thank you", "thanks", "thanks for watching",
        "bye", "mm", "hmm", "uh", "um", "oh", "ah",
    }
)


def clean_transcript(raw_text: str) -> str:
    """Strip non-speech tags and reject common STT noise hallucinations."""
    cleaned = _HALLUCINATION_TAGS.sub(" ", raw_text)
    cleaned = " ".join(cleaned.split()).strip()
    if len(re.sub(r"[^a-zA-Z0-9]", "", cleaned)) < 2:
        return ""
    if cleaned.lower().rstrip(".!?,") in _NOISE_PHRASES:
        return ""
    return cleaned


class BaseSTT(ABC):
    @abstractmethod
    async def transcribe_wav(self, wav_path: Path) -> str:
        """Transcribe a 16kHz mono WAV file to text."""


class WhisperCppSTT(BaseSTT):
    """Invokes locally compiled whisper.cpp binary (whisper-cli / main)."""

    def __init__(self, binary: str, model_path: Path, prompt: str = "") -> None:
        self.binary = binary
        self.model_path = model_path
        self.prompt = prompt.strip()

    def is_ready(self) -> tuple[bool, str]:
        if not shutil.which(self.binary) and not Path(self.binary).exists():
            return False, f"whisper.cpp binary not found: {self.binary}"
        if not self.model_path.exists():
            return False, f"whisper.cpp model not found: {self.model_path}"
        return True, "ok"

    async def transcribe_wav(self, wav_path: Path) -> str:
        ok, msg = self.is_ready()
        if not ok:
            raise RuntimeError(msg)
        cmd = [
            self.binary,
            "-m",
            str(self.model_path),
            "-f",
            str(wav_path),
            "-l",
            "en",
            "-nt",  # no timestamps
            "-np",  # no extra prints
        ]
        if self.prompt:
            cmd.extend(["--prompt", self.prompt])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        if proc.returncode != 0:
            raise RuntimeError(f"whisper.cpp failed: {stderr.decode('utf-8', errors='replace').strip()}")
        return clean_transcript(stdout.decode("utf-8", errors="replace"))


class VoskSTT(BaseSTT):
    """Offline Vosk STT for WAV audio."""

    def __init__(self, model_dir: Path, sample_rate: int = 16000) -> None:
        try:
            import vosk  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("vosk package is not installed.") from exc
        if not model_dir.exists():
            raise FileNotFoundError(f"Vosk model directory not found: {model_dir}")
        self._vosk = vosk
        self._model = vosk.Model(str(model_dir))
        self.sample_rate = sample_rate

    async def transcribe_wav(self, wav_path: Path) -> str:
        import wave

        with wave.open(str(wav_path), "rb") as wf:
            rec = self._vosk.KaldiRecognizer(self._model, wf.getframerate())
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                rec.AcceptWaveform(data)
            res = json.loads(rec.FinalResult() or "{}")
            return clean_transcript(str(res.get("text", "")))


def build_stt(config: AudioConfig, root_dir: Path) -> BaseSTT | None:
    backend = config.stt_backend.lower()
    if backend == "whisper_cpp":
        mpath = Path(config.whisper_model)
        if not mpath.is_absolute():
            mpath = root_dir / mpath
        return WhisperCppSTT(config.whisper_bin, mpath, prompt=config.whisper_prompt)
    if backend == "vosk":
        mdir = Path(config.vosk_model_dir)
        if not mdir.is_absolute():
            mdir = root_dir / mdir
        return VoskSTT(mdir, sample_rate=config.sample_rate)
    return None
