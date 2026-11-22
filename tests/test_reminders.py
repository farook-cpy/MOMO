"""Tests for persistent timezone-aware reminders and missed-reminder recovery."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from automation.reminders import ReminderManager
from automation.scheduler import ReminderScheduler
from memory.database import ZiaDatabase


def test_create_list_and_cancel_reminder(test_db: ZiaDatabase) -> None:
    rm = ReminderManager(test_db, default_timezone="America/New_York")
    created = rm.create_reminder("Check Xperia thermal logs", "in 30 minutes")
    assert created["id"] > 0
    assert created["status"] == "pending"

    pending = rm.list_reminders("pending")
    assert len(pending) == 1

    res = rm.cancel_reminder(reminder_id=created["id"])
    assert res["cancelled"] is True
    assert rm.list_reminders("pending") == []


@pytest.mark.asyncio
async def test_missed_reminder_preserved_and_delivered_on_recovery(test_db: ZiaDatabase) -> None:
    rm = ReminderManager(test_db, default_timezone="UTC")
    past = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    created = rm.create_reminder("Stretch break", past, schedule_type="daily")

    # Simulate failed speaker delivery -> reminder becomes 'missed' instead of being lost
    async def failing_delivery(_item: dict) -> bool:
        return False

    sched = ReminderScheduler(rm, on_due=failing_delivery)
    delivered = await sched.check_once()
    assert delivered == []
    missed = rm.list_reminders("missed")
    assert len(missed) == 1
    assert missed[0]["id"] == created["id"]

    # Simulate recovery after restart: pop missed or deliver via working TTS
    assert len(sched.pop_missed_announcements()) == 1

    async def ok_delivery(_item: dict) -> bool:
        return True

    sched_ok = ReminderScheduler(rm, on_due=ok_delivery)
    delivered_ok = await sched_ok.check_once()
    assert len(delivered_ok) == 1

    # Because schedule_type='daily', next occurrence is rescheduled into the future
    pending_after = rm.list_reminders("pending")
    assert len(pending_after) == 1
    next_due = datetime.fromisoformat(pending_after[0]["due_at"])
    assert next_due > datetime.now(timezone.utc)
