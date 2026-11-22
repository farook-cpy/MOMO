"""SQLite + FTS5 persistence layer for ZIA long-term memory and schedules."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now_iso() -> str:
    """Return current timezone-aware UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    token_estimate INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conv_created
    ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS personal_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    importance REAL NOT NULL DEFAULT 0.7,
    sensitive INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS important_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    importance REAL NOT NULL DEFAULT 0.6,
    source TEXT NOT NULL DEFAULT 'conversation',
    sensitive INTEGER NOT NULL DEFAULT 0,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'archived')),
    priority INTEGER NOT NULL DEFAULT 2,
    importance REAL NOT NULL DEFAULT 0.8,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    title TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'done', 'cancelled')),
    priority INTEGER NOT NULL DEFAULT 2,
    due_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    event_at TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.6,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    schedule_type TEXT NOT NULL DEFAULT 'once'
        CHECK (schedule_type IN ('once', 'daily', 'weekly', 'routine', 'deadline', 'morning_summary', 'evening_summary')),
    due_at TEXT NOT NULL,
    recurrence_rule TEXT NOT NULL DEFAULT '',
    timezone TEXT NOT NULL DEFAULT 'UTC',
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'triggered', 'missed', 'completed', 'cancelled')),
    last_triggered_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reminders_status_due
    ON reminders(status, due_at);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER,
    summary TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    importance REAL NOT NULL DEFAULT 0.7,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS memory_metadata (
    meta_key TEXT PRIMARY KEY,
    meta_value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Unified FTS5 index across searchable memory entities
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    source_table UNINDEXED,
    source_id UNINDEXED,
    title,
    body,
    tags,
    tokenize = 'porter unicode61'
);
"""


class ZiaDatabase:
    """Thread-safe SQLite + FTS5 database manager."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._is_memory = str(db_path) == ":memory:"
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn

        if not self._is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if not self._is_memory:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        conn.executescript(SCHEMA_SQL)

        if not self._is_memory and os.name == "posix" and self.db_path.exists():
            os.chmod(self.db_path, 0o600)

        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            yield conn
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def integrity_check(self) -> tuple[bool, str]:
        conn = self.connect()
        row = conn.execute("PRAGMA integrity_check;").fetchone()
        msg = str(row[0]) if row else "unknown"
        return (msg.lower() == "ok"), msg

    def fts_supported(self) -> bool:
        conn = self.connect()
        try:
            conn.execute("SELECT count(*) FROM memory_fts;").fetchone()
            return True
        except sqlite3.DatabaseError:
            return False

    def upsert_fts(
        self,
        conn: sqlite3.Connection,
        source_table: str,
        source_id: int,
        title: str,
        body: str,
        tags: str = "",
    ) -> None:
        conn.execute(
            "DELETE FROM memory_fts WHERE source_table = ? AND source_id = ?;",
            (source_table, source_id),
        )
        conn.execute(
            "INSERT INTO memory_fts (source_table, source_id, title, body, tags) VALUES (?, ?, ?, ?, ?);",
            (source_table, source_id, title, body, tags),
        )

    def delete_fts(
        self,
        conn: sqlite3.Connection,
        source_table: str,
        source_id: int,
    ) -> None:
        conn.execute(
            "DELETE FROM memory_fts WHERE source_table = ? AND source_id = ?;",
            (source_table, source_id),
        )

    # --- Metadata & daily usage counters ---

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        conn = self.connect()
        row = conn.execute(
            "SELECT meta_value FROM memory_metadata WHERE meta_key = ?;",
            (key,),
        ).fetchone()
        return str(row["meta_value"]) if row else default

    def set_meta(self, key: str, value: str) -> None:
        now = utc_now_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO memory_metadata (meta_key, meta_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(meta_key) DO UPDATE SET
                    meta_value = excluded.meta_value,
                    updated_at = excluded.updated_at;
                """,
                (key, value, now),
            )

    def get_daily_usage(self, provider: str, date_str: str) -> dict[str, int]:
        raw = self.get_meta(f"usage:{provider}:{date_str}")
        if not raw:
            return {"requests": 0, "tokens": 0}
        try:
            data = json.loads(raw)
            return {
                "requests": int(data.get("requests", 0)),
                "tokens": int(data.get("tokens", 0)),
            }
        except (ValueError, TypeError):
            return {"requests": 0, "tokens": 0}

    def record_daily_usage(self, provider: str, date_str: str, tokens: int) -> dict[str, int]:
        key = f"usage:{provider}:{date_str}"
        now = utc_now_iso()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT meta_value FROM memory_metadata WHERE meta_key = ?;",
                (key,),
            ).fetchone()
            current = {"requests": 0, "tokens": 0}
            if row:
                try:
                    loaded = json.loads(str(row["meta_value"]))
                    current["requests"] = int(loaded.get("requests", 0))
                    current["tokens"] = int(loaded.get("tokens", 0))
                except (ValueError, TypeError):
                    pass
            current["requests"] += 1
            current["tokens"] += max(0, int(tokens))
            conn.execute(
                """
                INSERT INTO memory_metadata (meta_key, meta_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(meta_key) DO UPDATE SET
                    meta_value = excluded.meta_value,
                    updated_at = excluded.updated_at;
                """,
                (key, json.dumps(current), now),
            )
            return current

    # --- Conversations & Messages ---

    def get_or_create_active_conversation(self, title: str = "Primary Session") -> int:
        conn = self.connect()
        row = conn.execute(
            "SELECT id FROM conversations WHERE is_active = 1 ORDER BY id DESC LIMIT 1;"
        ).fetchone()
        if row:
            return int(row["id"])
        now = utc_now_iso()
        with self.transaction() as conn_tx:
            cur = conn_tx.execute(
                "INSERT INTO conversations (title, started_at, updated_at, is_active) VALUES (?, ?, ?, 1);",
                (title, now, now),
            )
            return int(cur.lastrowid)

    def add_message(self, conversation_id: int, role: str, content: str) -> int:
        now = utc_now_iso()
        token_est = max(1, len(content) // 4)
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, token_estimate, created_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (conversation_id, role, content, token_est, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?;",
                (now, conversation_id),
            )
            return int(cur.lastrowid)

    def get_recent_messages(self, conversation_id: int, limit: int = 10) -> list[dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?;
            """,
            (conversation_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def count_messages(self, conversation_id: int) -> int:
        conn = self.connect()
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM messages WHERE conversation_id = ?;",
            (conversation_id,),
        ).fetchone()
        return int(row["cnt"]) if row else 0

    # --- Backup & Restore ---

    def backup_to(self, backup_dir: Path) -> Path:
        if self._is_memory:
            raise ValueError("Cannot file-backup an in-memory database.")
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = backup_dir / f"zia-{stamp}.db"
        src_conn = self.connect()
        dst_conn = sqlite3.connect(str(dest))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
        if os.name == "posix" and dest.exists():
            os.chmod(dest, 0o600)
        return dest

    def restore_from(self, backup_file: Path) -> None:
        if self._is_memory:
            raise ValueError("Cannot restore into an in-memory database.")
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        # Verify backup integrity before replacing live DB
        check_conn = sqlite3.connect(str(backup_file))
        try:
            row = check_conn.execute("PRAGMA integrity_check;").fetchone()
            if not row or str(row[0]).lower() != "ok":
                raise ValueError(f"Backup file failed integrity check: {backup_file}")
        finally:
            check_conn.close()

        self.close()
        shutil.copy2(backup_file, self.db_path)
        self.connect()
