"""Wake-word detector abstractions and optional local engines."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from core.config import AudioConfig

logger = logging.getLogger("zia.audio.wakeword")


class BaseWakeWordDetector(ABC):
    @abstractmethod
    def process_frame(self, pcm16_frame: bytes) -> bool:
        """Return True if wake word was detected."""

    def reset(self) -> None:
        return None


class ManualTriggerWakeWord(BaseWakeWordDetector):
    """Fallback mode when continuous wake-word detection is disabled or unavailable."""

    def process_frame(self, pcm16_frame: bytes) -> bool:
        return False


class VoskKeywordDetector(BaseWakeWordDetector):
    """Grammar-restricted Vosk keyword spotter (requires vosk + model on ARM64)."""

    def __init__(self, model_dir: Path, phrase: str = "hey zia", sample_rate: int = 16000) -> None:
        try:
            import vosk  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("vosk package is not installed.") from exc

        if not model_dir.exists():
            raise FileNotFoundError(f"Vosk model directory not found: {model_dir}")
        self.phrase = phrase.lower().strip()
        self.variants = {self.phrase, "zia", "hey sia", "hey xia", "hey see ya"}
        model = vosk.Model(str(model_dir))
        grammar = json.dumps(sorted(self.variants) + ["[unk]"])
        self._recognizer = vosk.KaldiRecognizer(model, sample_rate, grammar)

    def process_frame(self, pcm16_frame: bytes) -> bool:
        if self._recognizer.AcceptWaveform(pcm16_frame):
            res = json.loads(self._recognizer.Result() or "{}")
            text = str(res.get("text", "")).lower().strip()
            return any(v in text for v in self.variants)
        partial = json.loads(self._recognizer.PartialResult() or "{}")
        ptext = str(partial.get("partial", "")).lower().strip()
        return any(v in ptext for v in self.variants)

    def reset(self) -> None:
        self._recognizer.Reset()


class OpenWakeWordDetector(BaseWakeWordDetector):
    """openWakeWord adapter (requires openwakeword + onnxruntime/tflite on Termux ARM64)."""

    def __init__(self, threshold: float = 0.5) -> None:
        try:
            import numpy as np  # type: ignore[import-not-found]
            from openwakeword.model import Model  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "openwakeword (and numpy/onnxruntime) not installed on this environment."
            ) from exc
        self._np = np
        self.threshold = threshold
        self._model = Model()

    def process_frame(self, pcm16_frame: bytes) -> bool:
        arr = self._np.frombuffer(pcm16_frame, dtype=self._np.int16)
        preds = self._model.predict(arr)
        return any(float(score) >= self.threshold for score in preds.values())

    def reset(self) -> None:
        self._model.reset()


def build_wakeword_detector(config: AudioConfig, root_dir: Path) -> BaseWakeWordDetector:
    backend = config.wakeword_backend.lower()
    if backend == "openwakeword":
        return OpenWakeWordDetector(threshold=config.wakeword_threshold)
    if backend == "vosk_keyword":
        model_dir = Path(config.vosk_model_dir)
        if not model_dir.is_absolute():
            model_dir = root_dir / model_dir
        return VoskKeywordDetector(
            model_dir=model_dir,
            phrase=config.wakeword_phrase,
            sample_rate=config.sample_rate,
        )
    return ManualTriggerWakeWord()
