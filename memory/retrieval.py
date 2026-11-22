"""Memory storage, FTS5 search, ranking, duplicate detection, and context building."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from memory.database import ZiaDatabase, utc_now_iso

_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
        "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
        "here", "his", "how", "i", "if", "in", "is", "it", "its", "me", "my",
        "no", "not", "of", "on", "or", "our", "out", "she", "should", "so",
        "some", "tell", "that", "the", "their", "them", "then", "there", "these",
        "they", "this", "to", "up", "us", "was", "we", "were", "what", "when",
        "where", "which", "who", "why", "will", "with", "would", "you", "your",
        "zia", "hey", "please", "today",
    }
)

_SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:password|passcode|pin|secret\s*key|api\s*key|token)\s*(?:is|=|:)\s*\S+", re.IGNORECASE), "credential"),
    (re.compile(r"\b(?:gsk_[A-Za-z0-9_-]{10,}|AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9_-]{16,})\b"), "api_key"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "payment_card"),
]


def detect_sensitive_kind(text: str) -> str | None:
    """Return a label if text appears to contain sensitive credentials/PII."""
    for pattern, label in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def build_fts_query(text: str) -> str:
    """Convert natural language input into a safe FTS5 OR query."""
    tokens = re.findall(r"[A-Za-z0-9_]{2,}", text.lower())
    filtered = [t for t in tokens if t not in _STOP_WORDS]
    if not filtered:
        filtered = tokens
    # Deduplicate preserving order, cap at 12 terms
    seen: set[str] = set()
    unique: list[str] = []
    for tok in filtered:
        if tok not in seen:
            seen.add(tok)
            unique.append(f'"{tok}"')
        if len(unique) >= 12:
            break
    return " OR ".join(unique)


@dataclass(slots=True)
class MemoryMatch:
    source_table: str
    source_id: int
    title: str
    body: str
    importance: float
    bm25_score: float
    combined_score: float
    created_at: str


class MemoryManager:
    """High-level persistent memory API backed by ZiaDatabase + FTS5."""

    def __init__(
        self,
        db: ZiaDatabase,
        duplicate_threshold: float = 0.85,
    ) -> None:
        self.db = db
        self.duplicate_threshold = duplicate_threshold

    # --- Preferences ---

    def set_preference(
        self,
        key: str,
        value: str,
        category: str = "general",
        importance: float = 0.75,
        approved: bool = False,
    ) -> dict[str, Any]:
        sensitive_kind = detect_sensitive_kind(f"{key} {value}")
        if sensitive_kind and not approved:
            return {
                "saved": False,
                "requires_approval": True,
                "reason": f"Detected potentially sensitive data ({sensitive_kind}). Explicit approval required.",
            }

        now = utc_now_iso()
        clean_key = key.strip().lower().replace(" ", "_")
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO personal_preferences (key, value, category, importance, sensitive, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    importance = MAX(personal_preferences.importance, excluded.importance),
                    updated_at = excluded.updated_at;
                """,
                (clean_key, value.strip(), category, importance, int(bool(sensitive_kind)), now, now),
            )
            row = conn.execute(
                "SELECT id FROM personal_preferences WHERE key = ?;",
                (clean_key,),
            ).fetchone()
            pref_id = int(row["id"])
            self.db.upsert_fts(
                conn,
                "personal_preferences",
                pref_id,
                clean_key.replace("_", " "),
                value.strip(),
                f"preference {category}",
            )
        return {"saved": True, "table": "personal_preferences", "id": pref_id, "key": clean_key}

    # --- Facts ---

    def save_fact(
        self,
        fact: str,
        category: str = "general",
        importance: float = 0.65,
        source: str = "conversation",
        approved: bool = False,
    ) -> dict[str, Any]:
        clean_fact = fact.strip()
        if not clean_fact:
            return {"saved": False, "reason": "Empty fact cannot be saved."}

        sensitive_kind = detect_sensitive_kind(clean_fact)
        if sensitive_kind and not approved:
            return {
                "saved": False,
                "requires_approval": True,
                "reason": f"Detected potentially sensitive data ({sensitive_kind}). Explicit approval required.",
            }

        conn = self.db.connect()
        existing = conn.execute(
            "SELECT id, fact, importance FROM important_facts ORDER BY id DESC LIMIT 200;"
        ).fetchall()
        for row in existing:
            if _similarity(clean_fact, str(row["fact"])) >= self.duplicate_threshold:
                fact_id = int(row["id"])
                now = utc_now_iso()
                new_imp = max(float(row["importance"]), importance)
                with self.db.transaction() as tx:
                    tx.execute(
                        """
                        UPDATE important_facts
                        SET fact = ?, importance = ?, updated_at = ?
                        WHERE id = ?;
                        """,
                        (clean_fact if len(clean_fact) >= len(str(row["fact"])) else str(row["fact"]), new_imp, now, fact_id),
                    )
                    self.db.upsert_fts(tx, "important_facts", fact_id, category, clean_fact, f"fact {category}")
                return {
                    "saved": True,
                    "deduplicated": True,
                    "table": "important_facts",
                    "id": fact_id,
                }

        now = utc_now_iso()
        with self.db.transaction() as tx:
            cur = tx.execute(
                """
                INSERT INTO important_facts (fact, category, importance, source, sensitive, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (clean_fact, category, importance, source, int(bool(sensitive_kind)), now, now),
            )
            fact_id = int(cur.lastrowid)
            self.db.upsert_fts(tx, "important_facts", fact_id, category, clean_fact, f"fact {category}")
        return {"saved": True, "deduplicated": False, "table": "important_facts", "id": fact_id}

    # --- Projects & Tasks ---

    def upsert_project(
        self,
        name: str,
        description: str = "",
        status: str = "active",
        priority: int = 2,
        importance: float = 0.85,
    ) -> int:
        now = utc_now_iso()
        clean_name = name.strip()
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO projects (name, description, status, priority, importance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = CASE WHEN excluded.description != '' THEN excluded.description ELSE projects.description END,
                    status = excluded.status,
                    priority = excluded.priority,
                    updated_at = excluded.updated_at;
                """,
                (clean_name, description.strip(), status, priority, importance, now, now),
            )
            row = conn.execute("SELECT id, description FROM projects WHERE name = ?;", (clean_name,)).fetchone()
            pid = int(row["id"])
            self.db.upsert_fts(conn, "projects", pid, clean_name, str(row["description"]), f"project {status}")
            return pid

    def list_projects(self, status: str | None = None) -> list[dict[str, Any]]:
        conn = self.db.connect()
        if status:
            rows = conn.execute(
                "SELECT * FROM projects WHERE status = ? ORDER BY priority ASC, updated_at DESC;",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY priority ASC, updated_at DESC;"
            ).fetchall()
        return [dict(r) for r in rows]

    def add_task(
        self,
        title: str,
        notes: str = "",
        project_name: str | None = None,
        priority: int = 2,
        due_at: str | None = None,
    ) -> dict[str, Any]:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("Task title cannot be empty.")

        project_id: int | None = None
        if project_name:
            project_id = self.upsert_project(project_name)

        now = utc_now_iso()
        with self.db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO tasks (project_id, title, notes, status, priority, due_at, created_at, updated_at)
                VALUES (?, ?, ?, 'open', ?, ?, ?, ?);
                """,
                (project_id, clean_title, notes.strip(), priority, due_at, now, now),
            )
            tid = int(cur.lastrowid)
            self.db.upsert_fts(
                conn,
                "tasks",
                tid,
                clean_title,
                f"{notes.strip()} {project_name or ''}".strip(),
                f"task open p{priority}",
            )
        return {"id": tid, "title": clean_title, "project_id": project_id, "status": "open", "priority": priority}

    def list_tasks(self, status: str = "open") -> list[dict[str, Any]]:
        conn = self.db.connect()
        rows = conn.execute(
            """
            SELECT t.*, p.name AS project_name
            FROM tasks t
            LEFT JOIN projects p ON t.project_id = p.id
            WHERE (? = 'all' OR t.status = ?)
            ORDER BY t.priority ASC, t.due_at IS NULL, t.due_at ASC, t.id DESC;
            """,
            (status, status),
        ).fetchall()
        return [dict(r) for r in rows]

    def complete_task(self, task_id: int | None = None, title_query: str | None = None) -> dict[str, Any]:
        conn = self.db.connect()
        row = None
        if task_id is not None:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?;", (task_id,)).fetchone()
        elif title_query:
            row = conn.execute(
                "SELECT * FROM tasks WHERE status != 'done' AND lower(title) LIKE ? ORDER BY id DESC LIMIT 1;",
                (f"%{title_query.strip().lower()}%",),
            ).fetchone()

        if not row:
            return {"completed": False, "reason": "Matching task not found."}

        tid = int(row["id"])
        now = utc_now_iso()
        with self.db.transaction() as tx:
            tx.execute(
                "UPDATE tasks SET status = 'done', completed_at = ?, updated_at = ? WHERE id = ?;",
                (now, now, tid),
            )
            self.db.upsert_fts(tx, "tasks", tid, str(row["title"]), str(row["notes"]), "task done")
        return {"completed": True, "id": tid, "title": str(row["title"])}

    # --- Events ---

    def add_event(self, title: str, event_at: str, description: str = "", importance: float = 0.65) -> int:
        now = utc_now_iso()
        with self.db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO events (title, description, event_at, importance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (title.strip(), description.strip(), event_at, importance, now, now),
            )
            eid = int(cur.lastrowid)
            self.db.upsert_fts(conn, "events", eid, title.strip(), f"{description.strip()} {event_at}", "event")
            return eid

    # --- Search, Ranking & Context Assembly ---

    def search(self, query: str, limit: int = 8) -> list[MemoryMatch]:
        fts_query = build_fts_query(query)
        conn = self.db.connect()
        rows: list[Any] = []
        if fts_query:
            rows = conn.execute(
                """
                SELECT source_table, source_id, title, body, bm25(memory_fts) AS rank
                FROM memory_fts
                WHERE memory_fts MATCH ?
                ORDER BY rank ASC
                LIMIT ?;
                """,
                (fts_query, limit * 3),
            ).fetchall()

        matches: list[MemoryMatch] = []
        now_dt = datetime.now(timezone.utc)
        for row in rows:
            tbl = str(row["source_table"])
            sid = int(row["source_id"])
            meta = self._fetch_entity_meta(tbl, sid)
            if meta is None:
                continue
            importance, created_at = meta
            # bm25() in SQLite FTS5 returns negative numbers (more negative = better match)
            bm25_raw = float(row["rank"])
            text_relevance = min(1.5, max(0.0, -bm25_raw / 5.0))
            recency_bonus = self._recency_bonus(created_at, now_dt)
            combined = (0.55 * text_relevance) + (0.30 * importance) + (0.15 * recency_bonus)
            matches.append(
                MemoryMatch(
                    source_table=tbl,
                    source_id=sid,
                    title=str(row["title"]),
                    body=str(row["body"]),
                    importance=importance,
                    bm25_score=bm25_raw,
                    combined_score=round(combined, 4),
                    created_at=created_at,
                )
            )

        matches.sort(key=lambda m: m.combined_score, reverse=True)
        return matches[:limit]

    def _fetch_entity_meta(self, table: str, entity_id: int) -> tuple[float, str] | None:
        conn = self.db.connect()
        allowed = {
            "important_facts": ("importance", "updated_at"),
            "personal_preferences": ("importance", "updated_at"),
            "projects": ("importance", "updated_at"),
            "tasks": ("0.75", "updated_at"),
            "events": ("importance", "event_at"),
            "conversation_summaries": ("importance", "created_at"),
        }
        if table not in allowed:
            return None
        imp_col, ts_col = allowed[table]
        row = conn.execute(
            f"SELECT {imp_col} AS imp, {ts_col} AS ts FROM {table} WHERE id = ?;",
            (entity_id,),
        ).fetchone()
        if not row:
            return None
        return float(row["imp"]), str(row["ts"])

    @staticmethod
    def _recency_bonus(iso_ts: str, now_dt: datetime) -> float:
        try:
            dt = datetime.fromisoformat(iso_ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now_dt - dt).total_seconds() / 86400.0)
            return 1.0 / (1.0 + age_days / 14.0)
        except ValueError:
            return 0.3

    def build_context_for_query(
        self,
        query: str,
        max_memories: int = 8,
        max_chars: int = 3500,
    ) -> str:
        """Analyze query, rank FTS5 matches, include active tasks/projects & latest summary."""
        sections: list[str] = []
        used_chars = 0

        matches = self.search(query, limit=max_memories)
        if matches:
            mem_lines = ["Relevant long-term memory:"]
            for m in matches:
                tag = m.source_table.rstrip("s")
                mem_lines.append(f"- [{tag}#{m.source_id}] {m.title}: {m.body}")
            block = "\n".join(mem_lines)
            if len(block) <= max_chars:
                sections.append(block)
                used_chars += len(block)

        # Include top open tasks/projects compactly if budget allows
        open_tasks = self.list_tasks("open")[:5]
        if open_tasks:
            task_lines = ["Open tasks:"]
            for t in open_tasks:
                proj = f" ({t['project_name']})" if t.get("project_name") else ""
                due = f" [due {t['due_at']}]" if t.get("due_at") else ""
                task_lines.append(f"- #{t['id']} {t['title']}{proj}{due}")
            block = "\n".join(task_lines)
            if used_chars + len(block) + 2 <= max_chars:
                sections.append(block)
                used_chars += len(block) + 2

        conn = self.db.connect()
        summary_row = conn.execute(
            "SELECT summary, created_at FROM conversation_summaries ORDER BY id DESC LIMIT 1;"
        ).fetchone()
        if summary_row:
            block = f"Recent conversation summary ({summary_row['created_at'][:10]}):\n{summary_row['summary']}"
            if used_chars + len(block) + 2 <= max_chars:
                sections.append(block)

        return "\n\n".join(sections)

    # --- Inspect, Update, Delete, Export ---

    def list_memories(self, limit: int = 50) -> list[dict[str, Any]]:
        conn = self.db.connect()
        items: list[dict[str, Any]] = []
        for row in conn.execute(
            "SELECT id, fact AS content, category, importance, updated_at FROM important_facts ORDER BY id DESC LIMIT ?;",
            (limit,),
        ).fetchall():
            items.append({"table": "important_facts", **dict(row)})
        for row in conn.execute(
            "SELECT id, key || ': ' || value AS content, category, importance, updated_at FROM personal_preferences ORDER BY id DESC LIMIT ?;",
            (limit,),
        ).fetchall():
            items.append({"table": "personal_preferences", **dict(row)})
        for row in conn.execute(
            "SELECT id, name || ' (' || status || '): ' || description AS content, 'project' AS category, importance, updated_at FROM projects ORDER BY id DESC LIMIT ?;",
            (limit,),
        ).fetchall():
            items.append({"table": "projects", **dict(row)})
        return items[:limit]

    def delete_memory(self, table: str, record_id: int, confirm: bool = False) -> dict[str, Any]:
        allowed = {"important_facts", "personal_preferences", "projects", "tasks", "events", "conversation_summaries"}
        if table not in allowed:
            return {"deleted": False, "reason": f"Unsupported memory table: {table}"}
        if not confirm:
            return {
                "deleted": False,
                "requires_confirmation": True,
                "reason": f"Confirmation required to permanently delete {table}#{record_id}.",
            }
        with self.db.transaction() as conn:
            cur = conn.execute(f"DELETE FROM {table} WHERE id = ?;", (record_id,))
            self.db.delete_fts(conn, table, record_id)
            if cur.rowcount == 0:
                return {"deleted": False, "reason": f"{table}#{record_id} not found."}
        return {"deleted": True, "table": table, "id": record_id}

    def export_all(self, output_path: Path | None = None) -> dict[str, Any]:
        conn = self.db.connect()
        tables = [
            "personal_preferences",
            "important_facts",
            "projects",
            "tasks",
            "events",
            "reminders",
            "conversation_summaries",
        ]
        payload: dict[str, Any] = {"exported_at": utc_now_iso(), "tables": {}}
        for tbl in tables:
            rows = conn.execute(f"SELECT * FROM {tbl} ORDER BY id ASC;").fetchall()
            payload["tables"][tbl] = [dict(r) for r in rows]

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
