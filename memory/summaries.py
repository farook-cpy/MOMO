"""Conversation summarization helpers."""

from __future__ import annotations

from typing import Any

from memory.database import ZiaDatabase, utc_now_iso


def build_extractive_summary(messages: list[dict[str, Any]], max_chars: int = 900) -> str:
    """Create a deterministic compact summary when LLM summarization is offline/unavailable."""
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).capitalize()
        content = " ".join(str(msg.get("content", "")).split())
        if not content:
            continue
        snippet = content[:140] + ("..." if len(content) > 140 else "")
        lines.append(f"{role}: {snippet}")
    joined = " | ".join(lines)
    if len(joined) > max_chars:
        return joined[: max_chars - 3] + "..."
    return joined


def store_summary(
    db: ZiaDatabase,
    conversation_id: int,
    summary_text: str,
    message_count: int,
    importance: float = 0.75,
) -> int:
    now = utc_now_iso()
    with db.transaction() as conn:
        cur = conn.execute(
            """
            INSERT INTO conversation_summaries (conversation_id, summary, message_count, importance, created_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            (conversation_id, summary_text.strip(), message_count, importance, now),
        )
        sid = int(cur.lastrowid)
        db.upsert_fts(
            conn,
            "conversation_summaries",
            sid,
            f"Summary {now[:10]}",
            summary_text.strip(),
            "summary conversation",
        )
        return sid
