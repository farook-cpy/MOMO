"""Proactive conversation manager with quiet hours and daily limits."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from core.persona import PersonalityMode
from memory.database import ZiaDatabase
from memory.retrieval import MemoryManager


def _parse_hhmm(hhmm: str) -> tuple[int, int]:
    parts = hhmm.strip().split(":")
    return int(parts[0]), int(parts[1])


def is_in_quiet_hours(now_local: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    sh, sm = _parse_hhmm(start_hhmm)
    eh, em = _parse_hhmm(end_hhmm)
    cur = now_local.hour * 60 + now_local.minute
    start = sh * 60 + sm
    end = eh * 60 + em
    if start <= end:
        return start <= cur < end
    return cur >= start or cur < end


class ProactiveManager:
    """Generates occasional, non-intrusive check-ins respecting quiet hours & caps."""

    def __init__(
        self,
        db: ZiaDatabase,
        memory: MemoryManager,
        timezone_name: str = "UTC",
        enabled: bool = True,
        quiet_start: str = "23:00",
        quiet_end: str = "07:30",
        max_per_day: int = 4,
    ) -> None:
        self.db = db
        self.memory = memory
        self.timezone_name = timezone_name
        self.enabled = enabled
        self.quiet_start = quiet_start
        self.quiet_end = quiet_end
        self.max_per_day = max_per_day

    def _local_now(self) -> datetime:
        try:
            tz = ZoneInfo(self.timezone_name)
        except Exception:
            tz = timezone.utc
        return datetime.now(tz)

    def maybe_build_prompt(self, mode: PersonalityMode, now_local: datetime | None = None) -> str | None:
        if not self.enabled or mode in {PersonalityMode.QUIET, PersonalityMode.FOCUS}:
            return None

        now = now_local or self._local_now()
        if is_in_quiet_hours(now, self.quiet_start, self.quiet_end):
            return None

        day_key = f"proactive:{now.strftime('%Y-%m-%d')}"
        count = int(self.db.get_meta(day_key, "0") or "0")
        if count >= self.max_per_day:
            return None

        open_tasks = self.memory.list_tasks("open")
        if not open_tasks:
            return None

        top = open_tasks[0]
        self.db.set_meta(day_key, str(count + 1))
        return f"Quick check-in: you still have '{top['title']}' open. Want to tackle that or push it?"
