"""Isabella-inspired persona engine: modes, mood detection, boundaries, and voice expression."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from memory.retrieval import MemoryManager


class PersonalityMode(str, Enum):
    NORMAL = "normal"
    PLAYFUL = "playful"
    SERIOUS = "serious"
    SUPPORTIVE = "supportive"
    FOCUS = "focus"
    QUIET = "quiet"


MODE_GUIDANCE: dict[PersonalityMode, str] = {
    PersonalityMode.NORMAL: "Natural, friendly, witty, relaxed, and observant.",
    PersonalityMode.PLAYFUL: "Lean into jokes, light teasing, quick comebacks, and dry sarcasm.",
    PersonalityMode.SERIOUS: "Direct, focused, minimal humor, precise and grounded.",
    PersonalityMode.SUPPORTIVE: "Patient, thoughtful, calm, attentive, and considerate without clichés.",
    PersonalityMode.FOCUS: "Minimal chatter; prioritize reminders, tasks, and concise execution.",
    PersonalityMode.QUIET: "Only respond when directly addressed; keep replies brief and non-intrusive.",
}


@dataclass(slots=True)
class ConversationalState:
    mode: PersonalityMode = PersonalityMode.NORMAL
    detected_mood: str = "neutral"  # neutral | frustrated | excited | struggling | joking | technical
    intensity: float = 0.0


@dataclass(slots=True)
class VoiceDeliveryPlan:
    rate: float
    pitch: float
    spoken_text: str


_MOOD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("frustrated", re.compile(r"\b(ugh|annoying|broken|hate|stupid|stuck|wtf|tired of|frustrat)\b", re.IGNORECASE)),
    ("struggling", re.compile(r"\b(stressed|overwhelmed|exhausted|sad|down|burnt out|anxious|rough day)\b", re.IGNORECASE)),
    ("excited", re.compile(r"\b(awesome|amazing|worked|finally|yes!|hell yeah|nailed it|great news)\b", re.IGNORECASE)),
    ("joking", re.compile(r"\b(haha|hehe|lol|lmao|just kidding|jk|roast me)\b", re.IGNORECASE)),
    ("technical", re.compile(r"\b(error|traceback|sqlite|api|latency|bug|compile|adb|termux|python)\b", re.IGNORECASE)),
]


def detect_mood(text: str) -> str:
    """Detect conversational mood cues without assuming permanence across sessions."""
    for mood, pat in _MOOD_PATTERNS:
        if pat.search(text):
            return mood
    return "neutral"


def apply_boundary_or_mode_feedback(
    text: str,
    state: ConversationalState,
    memory: MemoryManager,
) -> str | None:
    """Handle explicit user boundaries, mode switches, and style feedback."""
    lower = text.lower().strip()

    if lower.startswith("/mode "):
        raw = lower[6:].strip()
        try:
            state.mode = PersonalityMode(raw)
            return f"Switched to {state.mode.value} mode."
        except ValueError:
            valid = ", ".join(m.value for m in PersonalityMode)
            return f"Unknown mode '{raw}'. Available: {valid}."

    if re.search(r"\b(be quiet|quiet mode|stop talking)\b", lower):
        state.mode = PersonalityMode.QUIET
        return "Got it. Quiet mode on—I'll only speak when you ask."

    if re.search(r"\b(focus mode|just tasks)\b", lower):
        state.mode = PersonalityMode.FOCUS
        return "Focus mode on. Keeping things minimal."

    if re.search(r"\b(be serious|serious mode|no jokes)\b", lower):
        state.mode = PersonalityMode.SERIOUS
        return "Understood. Switching to serious mode."

    if re.search(r"\b(playful mode|be playful)\b", lower):
        state.mode = PersonalityMode.PLAYFUL
        return "Playful mode on. Don't say I didn't warn you."

    if re.search(r"\b(normal mode|back to normal)\b", lower):
        state.mode = PersonalityMode.NORMAL
        return "Back to normal mode."

    if re.search(r"\b(don't talk like that|stop teasing|too sarcastic)\b", lower):
        state.mode = PersonalityMode.SERIOUS
        memory.set_preference("tone_boundary", text.strip(), category="boundary", importance=0.9)
        return "Fair enough—I'll dial that back and keep it straightforward."

    m_nick = re.match(r"^(?:call me|my nickname is)\s+([a-zA-Z0-9 _-]{2,24})$", text.strip(), re.IGNORECASE)
    if m_nick:
        nick = m_nick.group(1).strip()
        memory.set_preference("nickname", nick, category="relationship", importance=0.85)
        return f"Got it, {nick}."

    return None


def plan_voice_delivery(
    text: str,
    state: ConversationalState,
    base_rate: float = 1.0,
    base_pitch: float = 1.0,
) -> VoiceDeliveryPlan:
    """Compute speaking rate/pitch and clean spoken text from persona state."""
    rate = base_rate
    pitch = base_pitch

    if state.mode in {PersonalityMode.SERIOUS, PersonalityMode.SUPPORTIVE} or state.detected_mood == "struggling":
        rate = max(0.8, base_rate * 0.92)
        pitch = max(0.85, base_pitch * 0.97)
    elif state.mode == PersonalityMode.PLAYFUL or state.detected_mood in {"excited", "joking"}:
        rate = min(1.25, base_rate * 1.08)
        pitch = min(1.15, base_pitch * 1.04)

    # Strip common markdown artifacts for cleaner TTS delivery
    cleaned = re.sub(r"[`*_#]+", "", text).strip()
    return VoiceDeliveryPlan(rate=round(rate, 2), pitch=round(pitch, 2), spoken_text=cleaned)
