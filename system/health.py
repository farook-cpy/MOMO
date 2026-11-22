"""Device health, battery, thermal, uptime, and `zia doctor` diagnostics."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import ZiaConfig
from memory.database import ZiaDatabase
from system.connectivity import get_network_status

_PROCESS_START = time.monotonic()


def get_battery_status() -> dict[str, Any]:
    """Query battery via termux-battery-status or Linux sysfs fallback."""
    if shutil.which("termux-battery-status"):
        try:
            proc = subprocess.run(
                ["termux-battery-status"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                return {
                    "source": "termux-battery-status",
                    "percentage": data.get("percentage"),
                    "status": data.get("status"),
                    "plugged": data.get("plugged"),
                    "temperature_c": data.get("temperature"),
                    "health": data.get("health"),
                }
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass

    # Fallback: Linux power_supply sysfs (if readable)
    ps_dir = Path("/sys/class/power_supply")
    if ps_dir.exists():
        for entry in ps_dir.iterdir():
            cap = entry / "capacity"
            if cap.exists():
                try:
                    pct = int(cap.read_text(encoding="utf-8").strip())
                    st_file = entry / "status"
                    st = st_file.read_text(encoding="utf-8").strip() if st_file.exists() else "UNKNOWN"
                    return {"source": str(entry), "percentage": pct, "status": st}
                except OSError:
                    continue

    return {"source": "unavailable", "percentage": None, "status": "UNAVAILABLE"}


def get_thermal_celsius() -> float | None:
    """Read maximum CPU/battery thermal zone temperature in Celsius if exposed."""
    thermal_dir = Path("/sys/class/thermal")
    if not thermal_dir.exists():
        return None
    temps: list[float] = []
    for zone in thermal_dir.glob("thermal_zone*/temp"):
        try:
            raw = float(zone.read_text(encoding="utf-8").strip())
            c = raw / 1000.0 if raw > 200 else raw
            if 0.0 < c < 130.0:
                temps.append(c)
        except (OSError, ValueError):
            continue
    return round(max(temps), 1) if temps else None


def get_system_uptime() -> dict[str, Any]:
    """Return OS uptime (/proc/uptime when available) and ZIA process uptime."""
    proc_uptime = Path("/proc/uptime")
    os_uptime_seconds: float | None = None
    if proc_uptime.exists():
        try:
            first = proc_uptime.read_text(encoding="utf-8").split()[0]
            os_uptime_seconds = float(first)
        except (OSError, ValueError, IndexError):
            os_uptime_seconds = None

    return {
        "process_uptime_seconds": round(time.monotonic() - _PROCESS_START, 1),
        "os_uptime_seconds": round(os_uptime_seconds, 1) if os_uptime_seconds is not None else None,
        "platform": sys.platform,
    }


@dataclass(slots=True)
class DoctorCheck:
    name: str
    status: str  # "OK" | "WARN" | "FAIL"
    detail: str


def run_doctor(config: ZiaConfig, db: ZiaDatabase) -> list[DoctorCheck]:
    """Verify Python, dependencies, APIs, audio tooling, models, DB, storage, and network."""
    checks: list[DoctorCheck] = []

    # 1. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 10)
    checks.append(DoctorCheck("python", "OK" if py_ok else "FAIL", f"Python {py_ver}"))

    # 2. Core & optional dependencies
    for mod in ("httpx", "sqlite3"):
        found = importlib.util.find_spec(mod) is not None
        checks.append(DoctorCheck(f"dep:{mod}", "OK" if found else "FAIL", "installed" if found else "missing"))
    for opt in ("numpy", "vosk", "openwakeword"):
        found = importlib.util.find_spec(opt) is not None
        checks.append(
            DoctorCheck(
                f"opt:{opt}",
                "OK" if found else "WARN",
                "installed" if found else "not installed (optional until audio phase)",
            )
        )

    # 3. API configuration
    checks.append(
        DoctorCheck(
            "api:groq",
            "OK" if config.groq_api_key else "WARN",
            f"key present (model={config.llm.groq.model})" if config.groq_api_key else "GROQ_API_KEY not set",
        )
    )
    checks.append(
        DoctorCheck(
            "api:gemini",
            "OK" if config.gemini_api_key else "WARN",
            f"key present (model={config.llm.gemini.model})" if config.gemini_api_key else "GEMINI_API_KEY not set",
        )
    )
    for warn in config.permission_warnings:
        checks.append(DoctorCheck("security:secrets_perm", "WARN", warn))

    # 4. Microphone & speaker binaries
    mic_bins = [b for b in ("parec", "pulseaudio", "pactl", "termux-microphone-record") if shutil.which(b)]
    checks.append(
        DoctorCheck(
            "audio:microphone_tools",
            "OK" if mic_bins else "WARN",
            f"found: {', '.join(mic_bins)}" if mic_bins else "no Termux mic binaries on PATH (expected on Windows)",
        )
    )
    spk_bins = [b for b in ("termux-tts-speak", "espeak-ng", "mpv", "paplay") if shutil.which(b)]
    checks.append(
        DoctorCheck(
            "audio:speaker_tools",
            "OK" if spk_bins else "WARN",
            f"found: {', '.join(spk_bins)}" if spk_bins else "no Termux TTS/speaker binaries on PATH",
        )
    )

    # 5. Speech recognition & wake-word models
    whisper_path = Path(config.audio.whisper_model)
    if not whisper_path.is_absolute():
        whisper_path = config.root_dir / whisper_path
    vosk_path = Path(config.audio.vosk_model_dir)
    if not vosk_path.is_absolute():
        vosk_path = config.root_dir / vosk_path

    has_stt = whisper_path.exists() or vosk_path.exists()
    checks.append(
        DoctorCheck(
            "models:stt",
            "OK" if has_stt else "WARN",
            f"whisper={whisper_path.exists()}, vosk={vosk_path.exists()}",
        )
    )
    checks.append(
        DoctorCheck(
            "models:wakeword",
            "OK" if config.audio.wakeword_backend != "manual" else "WARN",
            f"backend={config.audio.wakeword_backend}",
        )
    )

    # 6. Database integrity + FTS5
    ok_db, msg_db = db.integrity_check()
    ok_fts = db.fts_supported()
    checks.append(
        DoctorCheck(
            "database:integrity",
            "OK" if (ok_db and ok_fts) else "FAIL",
            f"integrity={msg_db}, fts5={ok_fts}",
        )
    )

    # 7. Storage availability
    storage_root = config.db_path.parent if config.db_path.parent.exists() else config.root_dir
    usage = shutil.disk_usage(storage_root)
    free_mb = usage.free // (1024 * 1024)
    checks.append(
        DoctorCheck(
            "storage",
            "OK" if free_mb >= 200 else "WARN",
            f"{free_mb} MB free at {storage_root}",
        )
    )

    # 8. Network connectivity
    net = get_network_status()
    checks.append(
        DoctorCheck(
            "network",
            "OK" if net["online"] else "WARN",
            f"online={net['online']}, interface={net['interface']}",
        )
    )

    return checks
