"""Process lifecycle, PID management, Termux wake-lock helper, and watchdog loop."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Awaitable, Callable

from core.config import ZiaConfig
from memory.database import ZiaDatabase
from system.health import get_battery_status, get_thermal_celsius

logger = logging.getLogger("zia.watchdog")


def acquire_termux_wakelock() -> bool:
    if not shutil.which("termux-wake-lock"):
        return False
    try:
        res = subprocess.run(["termux-wake-lock"], capture_output=True, timeout=5, check=False)
        return res.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def release_termux_wakelock() -> bool:
    if not shutil.which("termux-wake-unlock"):
        return False
    try:
        res = subprocess.run(["termux-wake-unlock"], capture_output=True, timeout=5, check=False)
        return res.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def write_pid_file(pid_file: Path) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")


def read_pid_file(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def remove_pid_file(pid_file: Path) -> None:
    if pid_file.exists():
        pid_file.unlink(missing_ok=True)


def stop_process_by_pid(pid_file: Path, timeout_seconds: float = 5.0) -> bool:
    pid = read_pid_file(pid_file)
    if pid is None:
        return False
    if not is_pid_running(pid):
        remove_pid_file(pid_file)
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False

    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not is_pid_running(pid):
            remove_pid_file(pid_file)
            return True
        time.sleep(0.2)
    return not is_pid_running(pid)


class WatchdogSupervisor:
    """Monitors database integrity, battery, thermal limits, and supervised worker restarts."""

    def __init__(self, config: ZiaConfig, db: ZiaDatabase) -> None:
        self.config = config
        self.db = db
        self._stop_event = asyncio.Event()

    async def check_health(self) -> None:
        ok, msg = self.db.integrity_check()
        if not ok:
            logger.error("SQLite integrity check failed: %s", msg)

        batt = get_battery_status()
        pct = batt.get("percentage")
        if isinstance(pct, (int, float)) and pct <= self.config.system.battery_low_percent:
            logger.warning("Low battery warning: %s%% (%s)", pct, batt.get("status"))

        temp = get_thermal_celsius()
        if temp is not None and temp >= self.config.system.thermal_warn_celsius:
            logger.warning("High device thermal reading: %.1f C", temp)

    async def supervise_task(
        self,
        name: str,
        coro_factory: Callable[[], Awaitable[None]],
        max_restarts: int = 10,
    ) -> None:
        """Restart recoverable worker failures with bounded backoff."""
        restarts = 0
        while not self._stop_event.is_set():
            try:
                await coro_factory()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                restarts += 1
                logger.exception("Supervised worker %s crashed (%d/%d): %s", name, restarts, max_restarts, exc)
                if restarts >= max_restarts:
                    logger.error("Worker %s exceeded max restarts; stopping supervision.", name)
                    return
                await asyncio.sleep(min(30.0, 2.0 * restarts))

    async def run(self) -> None:
        self._stop_event.clear()
        while not self._stop_event.is_set():
            await self.check_health()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=max(5, self.config.system.watchdog_interval_seconds),
                )
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        self._stop_event.set()
