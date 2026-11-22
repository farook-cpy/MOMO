"""Persistent timezone-aware reminders and recurring routines in SQLite."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from memory.database import ZiaDatabase, utc_now_iso

VALID_SCHEDULE_TYPES = frozenset(
    {"once", "daily", "weekly", "routine", "deadline", "morning_summary", "evening_summary"}
)


def parse_due_timestamp(raw: str, tz_name: str = "UTC") -> datetime:
    """Parse ISO timestamp or relative expression ('in 15 minutes', 'in 2 hours', 'HH:MM')."""
    text = raw.strip()
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        local_tz = timezone.utc

    now_local = datetime.now(local_tz)
    lower = text.lower()

    if lower.startswith("in "):
        parts = lower[3:].split()
        if len(parts) >= 2 and parts[0].isdigit():
            amount = int(parts[0])
            unit = parts[1]
            if unit.startswith("sec"):
                return (now_local + timedelta(seconds=amount)).astimezone(timezone.utc)
            if unit.startswith("min"):
                return (now_local + timedelta(minutes=amount)).astimezone(timezone.utc)
            if unit.startswith("hour") or unit.startswith("hr"):
                return (now_local + timedelta(hours=amount)).astimezone(timezone.utc)
            if unit.startswith("day"):
                return (now_local + timedelta(days=amount)).astimezone(timezone.utc)

    # Support HH:MM (today or tomorrow if already passed)
    if len(text) == 5 and text[2] == ":" and text[:2].isdigit() and text[3:].isdigit():
        hh, mm = int(text[:2]), int(text[3:])
        target = now_local.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target <= now_local:
            target += timedelta(days=1)
        return target.astimezone(timezone.utc)

    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=local_tz)
    return dt.astimezone(timezone.utc)


def compute_next_occurrence(current_due: datetime, schedule_type: str, recurrence_rule: str = "") -> datetime | None:
    """Compute next due datetime for recurring reminders, or None for one-shot."""
    if schedule_type in {"once", "deadline"}:
        return None
    if schedule_type in {"daily", "morning_summary", "evening_summary"}:
        return current_due + timedelta(days=1)
    if schedule_type == "weekly":
        return current_due + timedelta(weeks=1)
    if schedule_type == "routine":
        rule = recurrence_rule.strip().lower()
        if rule.startswith("every ") and rule.endswith("h") and rule[6:-1].isdigit():
            return current_due + timedelta(hours=int(rule[6:-1]))
        if rule.startswith("every ") and rule.endswith("m") and rule[6:-1].isdigit():
            return current_due + timedelta(minutes=int(rule[6:-1]))
        return current_due + timedelta(days=1)
    return None


class ReminderManager:
    """CRUD and due-resolution logic for reminders."""

    def __init__(self, db: ZiaDatabase, default_timezone: str = "UTC") -> None:
        self.db = db
        self.default_timezone = default_timezone

    def create_reminder(
        self,
        title: str,
        due_at: str,
        message: str = "",
        schedule_type: str = "once",
        recurrence_rule: str = "",
        tz_name: str | None = None,
    ) -> dict[str, Any]:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("Reminder title cannot be empty.")
        if schedule_type not in VALID_SCHEDULE_TYPES:
            raise ValueError(f"Unsupported schedule_type: {schedule_type}")

        effective_tz = tz_name or self.default_timezone
        due_utc = parse_due_timestamp(due_at, effective_tz)
        now = utc_now_iso()
        clean_msg = message.strip() or clean_title

        with self.db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO reminders (
                    title, message, schedule_type, due_at, recurrence_rule,
                    timezone, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?);
                """,
                (
                    clean_title,
                    clean_msg,
                    schedule_type,
                    due_utc.isoformat(),
                    recurrence_rule.strip(),
                    effective_tz,
                    now,
                    now,
                ),
            )
            rid = int(cur.lastrowid)
            row = conn.execute("SELECT * FROM reminders WHERE id = ?;", (rid,)).fetchone()
            if not row:
                raise RuntimeError("Reminder persistence verification failed.")
        return dict(row)

    def list_reminders(self, status: str | None = "pending") -> list[dict[str, Any]]:
        conn = self.db.connect()
        if status and status != "all":
            rows = conn.execute(
                "SELECT * FROM reminders WHERE status = ? ORDER BY due_at ASC;",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM reminders ORDER BY due_at ASC;").fetchall()
        return [dict(r) for r in rows]

    def cancel_reminder(self, reminder_id: int | None = None, title_query: str | None = None) -> dict[str, Any]:
        conn = self.db.connect()
        row = None
        if reminder_id is not None:
            row = conn.execute("SELECT * FROM reminders WHERE id = ?;", (reminder_id,)).fetchone()
        elif title_query:
            row = conn.execute(
                """
                SELECT * FROM reminders
                WHERE status IN ('pending', 'missed') AND lower(title) LIKE ?
                ORDER BY due_at ASC LIMIT 1;
                """,
                (f"%{title_query.strip().lower()}%",),
            ).fetchone()

        if not row:
            return {"cancelled": False, "reason": "Matching reminder not found."}

        rid = int(row["id"])
        now = utc_now_iso()
        with self.db.transaction() as tx:
            tx.execute(
                "UPDATE reminders SET status = 'cancelled', updated_at = ? WHERE id = ?;",
                (now, rid),
            )
        return {"cancelled": True, "id": rid, "title": str(row["title"])}

    def fetch_due_reminders(self, now_utc: datetime | None = None) -> list[dict[str, Any]]:
        """Return all pending or missed reminders whose due_at <= now_utc."""
        ref = now_utc or datetime.now(timezone.utc)
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT * FROM reminders WHERE status IN ('pending', 'missed') ORDER BY due_at ASC;"
        ).fetchall()
        due: list[dict[str, Any]] = []
        for row in rows:
            try:
                due_dt = datetime.fromisoformat(str(row["due_at"]))
                if due_dt.tzinfo is None:
                    due_dt = due_dt.replace(tzinfo=timezone.utc)
                if due_dt <= ref:
                    due.append(dict(row))
            except ValueError:
                continue
        return due

    def mark_missed(self, reminder_id: int) -> None:
        now = utc_now_iso()
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE reminders SET status = 'missed', updated_at = ? WHERE id = ? AND status = 'pending';",
                (now, reminder_id),
            )

    def mark_delivered(self, reminder_id: int, now_utc: datetime | None = None) -> None:
        ref = now_utc or datetime.now(timezone.utc)
        now = ref.isoformat()
        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM reminders WHERE id = ?;", (reminder_id,)).fetchone()
            if not row:
                return
            sched_type = str(row["schedule_type"])
            rule = str(row["recurrence_rule"])
            due_dt = datetime.fromisoformat(str(row["due_at"]))
            if due_dt.tzinfo is None:
                due_dt = due_dt.replace(tzinfo=timezone.utc)

            next_dt = compute_next_occurrence(due_dt, sched_type, rule)
            while next_dt is not None and next_dt <= ref:
                next_dt = compute_next_occurrence(next_dt, sched_type, rule)

            if next_dt is not None:
                conn.execute(
                    """
                    UPDATE reminders
                    SET due_at = ?, status = 'pending', last_triggered_at = ?, updated_at = ?
                    WHERE id = ?;
                    """,
                    (next_dt.isoformat(), now, now, reminder_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE reminders
                    SET status = 'triggered', last_triggered_at = ?, updated_at = ?
                    WHERE id = ?;
                    """,
                    (now, now, reminder_id),
                )
