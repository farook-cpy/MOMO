"""Configuration loader and secret management for ZIA."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class ProviderEndpointConfig:
    base_url: str
    model: str
    fast_model: str
    daily_request_limit: int = 900
    daily_token_limit: int = 450_000


@dataclass(slots=True)
class LLMConfig:
    primary_provider: str = "groq"
    fallback_provider: str = "gemini"
    request_timeout_seconds: float = 25.0
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    temperature: float = 0.6
    max_tokens: int = 512
    stream_responses: bool = True
    groq: ProviderEndpointConfig = field(
        default_factory=lambda: ProviderEndpointConfig(
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            fast_model="llama-3.1-8b-instant",
        )
    )
    gemini: ProviderEndpointConfig = field(
        default_factory=lambda: ProviderEndpointConfig(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-2.5-flash",
            fast_model="gemini-2.5-flash",
        )
    )


@dataclass(slots=True)
class MemoryConfig:
    short_term_turns: int = 8
    max_retrieved_memories: int = 8
    max_context_chars: int = 3500
    summary_trigger_messages: int = 24
    duplicate_similarity_threshold: float = 0.85


@dataclass(slots=True)
class AudioConfig:
    mic_backend: str = "pulseaudio"
    sample_rate: int = 16000
    channels: int = 1
    frame_duration_ms: int = 80
    vad_energy_threshold: float = 260.0
    silence_timeout_seconds: float = 1.8
    max_record_seconds: float = 18.0
    followup_timeout_seconds: float = 6.0
    echo_cooldown_ms: int = 400
    tts_backend: str = "kokoro"
    tts_stream: str = "MUSIC"
    tts_rate: float = 1.0
    tts_pitch: float = 1.0
    kokoro_model: str = "models/kokoro-v1.0.onnx"
    kokoro_voices: str = "models/voices-v1.0.bin"
    kokoro_voice: str = "af_sky"
    stt_backend: str = "whisper_cpp"
    whisper_bin: str = "whisper-cli"
    whisper_model: str = "models/ggml-base.en.bin"
    whisper_prompt: str = "Natural Indian English conversation with Zia."
    vosk_model_dir: str = "models/vosk-model-small-en-in-0.4"
    wakeword_backend: str = "manual"
    wakeword_phrase: str = "hey zia"
    wakeword_threshold: float = 0.45


@dataclass(slots=True)
class SchedulerConfig:
    poll_interval_seconds: int = 15
    morning_summary_time: str = "08:00"
    evening_summary_time: str = "21:00"


@dataclass(slots=True)
class SystemConfig:
    log_dir: Path = PROJECT_ROOT / "data" / "logs"
    log_level: str = "INFO"
    pid_file: Path = PROJECT_ROOT / "data" / "zia.pid"
    watchdog_interval_seconds: int = 30
    battery_low_percent: int = 15
    thermal_warn_celsius: float = 45.0


@dataclass(slots=True)
class ZiaConfig:
    root_dir: Path = PROJECT_ROOT
    name: str = "ZIA"
    timezone: str = "America/New_York"
    personality_file: Path = PROJECT_ROOT / "config" / "personality.md"
    offline_mode: bool = False
    db_path: Path = PROJECT_ROOT / "data" / "zia.db"
    backup_dir: Path = PROJECT_ROOT / "data" / "backups"
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    groq_api_key: str = ""
    gemini_api_key: str = ""
    permission_warnings: list[str] = field(default_factory=list)

    def load_personality(self) -> str:
        if self.personality_file.exists():
            return self.personality_file.read_text(encoding="utf-8").strip()
        return "You are ZIA, a calm, concise, intelligent voice-first personal AI companion."


def default_secrets_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "zia" / "secrets.env"


def _parse_env_file(path: Path) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    values: dict[str, str] = {}
    if not path.exists():
        return values, warnings

    if os.name == "posix":
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            warnings.append(
                f"Secrets file {path} has permissive mode {oct(mode)}; recommend chmod 600."
            )

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
    return values, warnings


def _resolve_path(base: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base / p)


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> None:
    for key, val in data.items():
        if hasattr(instance, key) and not isinstance(val, dict):
            setattr(instance, key, val)


def load_config(
    settings_path: Path | None = None,
    secrets_path: Path | None = None,
) -> ZiaConfig:
    """Load settings.toml and external secrets without storing keys in source/config."""
    root = PROJECT_ROOT
    cfg_path = settings_path or (root / "config" / "settings.toml")
    sec_path = secrets_path or default_secrets_path()

    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("rb") as fh:
            raw = tomllib.load(fh)

    cfg = ZiaConfig(root_dir=root)

    assistant_sec = raw.get("assistant", {})
    if "name" in assistant_sec:
        cfg.name = str(assistant_sec["name"])
    if "timezone" in assistant_sec:
        cfg.timezone = str(assistant_sec["timezone"])
    if "personality_file" in assistant_sec:
        cfg.personality_file = _resolve_path(root, str(assistant_sec["personality_file"]))
    if "offline_mode" in assistant_sec:
        cfg.offline_mode = bool(assistant_sec["offline_mode"])

    db_sec = raw.get("database", {})
    if "path" in db_sec:
        cfg.db_path = _resolve_path(root, str(db_sec["path"]))
    if "backup_dir" in db_sec:
        cfg.backup_dir = _resolve_path(root, str(db_sec["backup_dir"]))

    llm_sec = raw.get("llm", {})
    _merge_dataclass(cfg.llm, llm_sec)
    if "groq" in llm_sec and isinstance(llm_sec["groq"], dict):
        _merge_dataclass(cfg.llm.groq, llm_sec["groq"])
    if "gemini" in llm_sec and isinstance(llm_sec["gemini"], dict):
        _merge_dataclass(cfg.llm.gemini, llm_sec["gemini"])

    _merge_dataclass(cfg.memory, raw.get("memory", {}))
    _merge_dataclass(cfg.audio, raw.get("audio", {}))
    _merge_dataclass(cfg.scheduler, raw.get("scheduler", {}))

    sys_sec = raw.get("system", {})
    _merge_dataclass(cfg.system, sys_sec)
    if "log_dir" in sys_sec:
        cfg.system.log_dir = _resolve_path(root, str(sys_sec["log_dir"]))
    if "pid_file" in sys_sec:
        cfg.system.pid_file = _resolve_path(root, str(sys_sec["pid_file"]))

    env_secrets, warnings = _parse_env_file(sec_path)
    cfg.permission_warnings.extend(warnings)

    cfg.groq_api_key = os.environ.get("GROQ_API_KEY") or env_secrets.get("GROQ_API_KEY", "")
    cfg.gemini_api_key = os.environ.get("GEMINI_API_KEY") or env_secrets.get("GEMINI_API_KEY", "")
    return cfg
