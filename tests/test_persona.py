"""Tests for Isabella-inspired persona modes, mood detection, boundaries, voice plan, and proactive manager."""

from __future__ import annotations

from datetime import datetime

from automation.proactive import ProactiveManager, is_in_quiet_hours
from core.persona import (
    ConversationalState,
    PersonalityMode,
    apply_boundary_or_mode_feedback,
    detect_mood,
    plan_voice_delivery,
)
from memory.database import ZiaDatabase
from memory.retrieval import MemoryManager


def test_mood_detection_and_voice_delivery_plan() -> None:
    assert detect_mood("Ugh this bug is so annoying") == "frustrated"
    assert detect_mood("Finally it worked, awesome!") == "excited"
    assert detect_mood("I'm really stressed and exhausted today") == "struggling"
    assert detect_mood("haha just kidding") == "joking"

    state = ConversationalState(mode=PersonalityMode.SUPPORTIVE, detected_mood="struggling")
    slow_plan = plan_voice_delivery("**Take a breath.**", state)
    assert slow_plan.rate < 1.0
    assert slow_plan.spoken_text == "Take a breath."

    playful_state = ConversationalState(mode=PersonalityMode.PLAYFUL, detected_mood="joking")
    fast_plan = plan_voice_delivery("Oh sure, blame the compiler.", playful_state)
    assert fast_plan.rate > 1.0


def test_mode_switches_and_boundary_feedback(test_db: ZiaDatabase) -> None:
    mem = MemoryManager(test_db)
    state = ConversationalState()

    assert "playful" in (apply_boundary_or_mode_feedback("/mode playful", state, mem) or "").lower()
    assert state.mode == PersonalityMode.PLAYFUL

    assert "quiet" in (apply_boundary_or_mode_feedback("be quiet", state, mem) or "").lower()
    assert state.mode == PersonalityMode.QUIET

    reply = apply_boundary_or_mode_feedback("call me Ace", state, mem)
    assert reply == "Got it, Ace."
    assert any("Ace" in m["content"] for m in mem.list_memories())


def test_proactive_quiet_hours_and_daily_cap(test_db: ZiaDatabase) -> None:
    mem = MemoryManager(test_db)
    mem.add_task("Finish Xperia audio probe")
    pm = ProactiveManager(test_db, mem, quiet_start="23:00", quiet_end="07:30", max_per_day=1)

    late = datetime(2026, 11, 22, 23, 30)
    assert is_in_quiet_hours(late, "23:00", "07:30") is True
    assert pm.maybe_build_prompt(PersonalityMode.NORMAL, now_local=late) is None

    daytime = datetime(2026, 11, 22, 14, 0)
    assert pm.maybe_build_prompt(PersonalityMode.QUIET, now_local=daytime) is None
    first = pm.maybe_build_prompt(PersonalityMode.NORMAL, now_local=daytime)
    assert first is not None and "Xperia audio probe" in first
    # Daily cap of 1 reached
    assert pm.maybe_build_prompt(PersonalityMode.NORMAL, now_local=daytime) is None
