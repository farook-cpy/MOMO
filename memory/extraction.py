"""Lightweight rule-based memory extraction with sensitive-data gating."""

from __future__ import annotations

import re
from typing import Any

from memory.retrieval import MemoryManager, detect_sensitive_kind

_REMEMBER_PATTERNS = [
    re.compile(r"^\s*(?:please\s+)?remember\s+(?:that\s+)?(.+)$", re.IGNORECASE),
    re.compile(r"^\s*note\s+that\s+(.+)$", re.IGNORECASE),
]

_PREF_PATTERN = re.compile(
    r"^\s*(?:my\s+favorite\s+([a-z0-9 _-]+?)\s+is\s+(.+?)|i\s+prefer\s+(.+?))\s*[.!]?$",
    re.IGNORECASE,
)


def extract_explicit_memories(user_text: str, memory: MemoryManager) -> list[dict[str, Any]]:
    """Persist explicit 'remember that ...' facts unless sensitive data requires approval."""
    results: list[dict[str, Any]] = []
    text = user_text.strip()

    for pat in _REMEMBER_PATTERNS:
        m = pat.match(text)
        if m:
            candidate = m.group(1).strip()
            sensitive = detect_sensitive_kind(candidate)
            if sensitive:
                results.append(
                    {
                        "saved": False,
                        "requires_approval": True,
                        "candidate": candidate,
                        "reason": f"Sensitive {sensitive} detected; ask user before storing.",
                    }
                )
            else:
                res = memory.save_fact(candidate, category="explicit", importance=0.8)
                results.append(res)
            return results

    pref_m = _PREF_PATTERN.match(text)
    if pref_m:
        if pref_m.group(1) and pref_m.group(2):
            res = memory.set_preference(pref_m.group(1), pref_m.group(2), category="preference")
            results.append(res)
        elif pref_m.group(3):
            res = memory.save_fact(f"User preference: {pref_m.group(3).strip()}", category="preference", importance=0.72)
            results.append(res)
    return results
