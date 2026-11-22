"""Tests for SQLite + FTS5 persistent memory, deduplication, privacy guard, and backup."""

from __future__ import annotations

import json
from pathlib import Path

from memory.database import ZiaDatabase
from memory.extraction import extract_explicit_memories
from memory.retrieval import MemoryManager


EXPECTED_TABLES = {
    "conversations",
    "messages",
    "personal_preferences",
    "important_facts",
    "projects",
    "tasks",
    "events",
    "reminders",
    "conversation_summaries",
    "memory_metadata",
    "memory_fts",
}


def test_schema_has_all_required_tables_and_fts5(test_db: ZiaDatabase) -> None:
    conn = test_db.connect()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view');"
    ).fetchall()
    names = {str(r["name"]) for r in rows}
    assert EXPECTED_TABLES.issubset(names)
    ok, msg = test_db.integrity_check()
    assert ok and msg.lower() == "ok"
    assert test_db.fts_supported()


def test_memory_save_search_ranking_and_context(test_db: ZiaDatabase) -> None:
    mem = MemoryManager(test_db)
    mem.upsert_project("Xperia ZIA", "Headless terminal AI assistant on Termux", priority=1)
    mem.add_task("Run audio feasibility probe on Xperia", project_name="Xperia ZIA", priority=1)
    mem.save_fact("User prefers concise spoken responses without fluff.", category="style", importance=0.9)
    mem.set_preference("wake_word", "Hey Zia")

    matches = mem.search("Xperia audio probe")
    assert matches
    assert any("Xperia" in m.title or "Xperia" in m.body for m in matches)

    # Punctuation / FTS operator characters in natural language must not raise syntax errors
    safe_matches = mem.search("What's my Xperia-ZIA: task?! (today)*")
    assert isinstance(safe_matches, list)

    ctx = mem.build_context_for_query("What should I work on for Xperia today?")
    assert "Xperia" in ctx
    assert "Run audio feasibility probe" in ctx


def test_duplicate_detection_updates_existing_fact(test_db: ZiaDatabase) -> None:
    mem = MemoryManager(test_db, duplicate_threshold=0.80)
    r1 = mem.save_fact("User lives in Toronto and works remotely.", importance=0.6)
    r2 = mem.save_fact("User lives in Toronto and works remotely!", importance=0.9)
    assert r1["saved"] and not r1["deduplicated"]
    assert r2["saved"] and r2["deduplicated"]
    assert r1["id"] == r2["id"]


def test_sensitive_data_blocked_without_approval(test_db: ZiaDatabase) -> None:
    mem = MemoryManager(test_db)
    blocked = mem.save_fact("My password is SuperSecretPassword123")
    assert blocked["saved"] is False
    assert blocked["requires_approval"] is True

    ext = extract_explicit_memories("Remember that my API key is gsk_abcdefghijklmnop1234", mem)
    assert ext and ext[0]["requires_approval"] is True

    allowed = mem.save_fact("My router pin is 9988", approved=True)
    assert allowed["saved"] is True


def test_memory_deletion_requires_confirmation_and_updates_fts(test_db: ZiaDatabase) -> None:
    mem = MemoryManager(test_db)
    saved = mem.save_fact("Temporary calibration note for Xperia speaker.")
    fid = int(saved["id"])

    denied = mem.delete_memory("important_facts", fid, confirm=False)
    assert denied["deleted"] is False
    assert denied["requires_confirmation"] is True

    confirmed = mem.delete_memory("important_facts", fid, confirm=True)
    assert confirmed["deleted"] is True
    assert mem.search("calibration speaker") == []


def test_export_and_backup_restore(test_db: ZiaDatabase, tmp_path: Path) -> None:
    mem = MemoryManager(test_db)
    mem.save_fact("Backup verification fact")

    export_file = tmp_path / "export.json"
    payload = mem.export_all(export_file)
    assert export_file.exists()
    loaded = json.loads(export_file.read_text(encoding="utf-8"))
    assert len(loaded["tables"]["important_facts"]) == 1
    assert payload["tables"]["important_facts"][0]["fact"] == "Backup verification fact"

    backup_file = test_db.backup_to(tmp_path / "backups")
    assert backup_file.exists()

    # Mutate DB after backup, then restore
    mem.save_fact("Post-backup transient fact")
    assert len(mem.list_memories()) == 2
    test_db.restore_from(backup_file)
    assert len(mem.list_memories()) == 1
