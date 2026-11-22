"""Background async scheduler with startup missed-reminder recovery."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from automation.reminders import ReminderManager

logger = logging.getLogger("zia.scheduler")


class ReminderScheduler:
    """Polls SQLite reminders, recovers missed ones, and delivers via callback."""

    def __init__(
        self,
        reminders: ReminderManager,
        poll_interval_seconds: int = 15,
        on_due: Callable[[dict[str, Any]], Awaitable[bool]] | None = None,
    ) -> None:
        self.reminders = reminders
        self.poll_interval_seconds = max(1, poll_interval_seconds)
        self.on_due = on_due
        self._stop_event = asyncio.Event()

    async def check_once(self, now_utc: datetime | None = None) -> list[dict[str, Any]]:
        """Check due reminders once. Marks undeliverable reminders as 'missed' for later."""
        ref = now_utc or datetime.now(timezone.utc)
        due_list = self.reminders.fetch_due_reminders(ref)
        delivered: list[dict[str, Any]] = []

        for item in due_list:
            rid = int(item["id"])
            if self.on_due is None:
                self.reminders.mark_missed(rid)
                continue
            try:
                ok = await self.on_due(item)
            except Exception as exc:
                logger.warning("Reminder #%d delivery failed: %s", rid, exc)
                ok = False

            if ok:
                self.reminders.mark_delivered(rid, ref)
                delivered.append(item)
            else:
                self.reminders.mark_missed(rid)

        return delivered

    def pop_missed_announcements(self) -> list[dict[str, Any]]:
        """Return missed/overdue reminders to announce at next user interaction."""
        return self.reminders.fetch_due_reminders(datetime.now(timezone.utc))

    async def run(self) -> None:
        self._stop_event.clear()
        logger.info("Scheduler started (interval=%ds).", self.poll_interval_seconds)
        while not self._stop_event.is_set():
            try:
                await self.check_once()
            except Exception as exc:  # pragma: no cover
                logger.error("Scheduler iteration error: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval_seconds)
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        self._stop_event.set()
