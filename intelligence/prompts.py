"""Prompt assembly for ZIA."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def build_system_prompt(
    personality_text: str,
    timezone_name: str,
    memory_context: str = "",
    voice_mode: bool = False,
    mode_guidance: str = "",
    detected_mood: str = "neutral",
) -> str:
    """Assemble compact, context-aware system instructions."""
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = timezone.utc
    now_str = datetime.now(tz).strftime("%A, %Y-%m-%d %H:%M %Z")

    parts = [
        personality_text.strip(),
        f"Current local time: {now_str}",
        (
            "Tool policy: Use the provided typed tools whenever the user asks about time, date, "
            "reminders, tasks, projects, stored memories, battery, network, or uptime. "
            "Never fabricate tool outputs. Require explicit confirmation before deleting memories."
        ),
    ]
    if mode_guidance:
        parts.append(f"Active personality mode: {mode_guidance}")
    if detected_mood and detected_mood != "neutral":
        parts.append(f"Detected user conversational mood: {detected_mood}. Adapt tone naturally.")
    if voice_mode:
        parts.append(
            "Output format: This response will be spoken aloud. Keep it concise, natural, "
            "and avoid markdown formatting, bullet symbols, or code blocks unless asked."
        )
    if memory_context.strip():
        parts.append("--- CONTEXT FROM MEMORY ---\n" + memory_context.strip())

    return "\n\n".join(parts)
