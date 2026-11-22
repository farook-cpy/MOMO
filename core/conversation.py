"""Conversation turn orchestration, tool loop, memory pipeline, and offline fallback."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Awaitable, Callable

from automation.scheduler import ReminderScheduler
from core.config import ZiaConfig
from core.persona import (
    MODE_GUIDANCE,
    ConversationalState,
    apply_boundary_or_mode_feedback,
    detect_mood,
)
from core.tools import ToolRegistry
from intelligence.prompts import build_system_prompt
from intelligence.provider import LLMMessage, LLMRouter, ProviderError
from memory.database import ZiaDatabase
from memory.extraction import extract_explicit_memories
from memory.retrieval import MemoryManager
from memory.summaries import build_extractive_summary, store_summary

logger = logging.getLogger("zia.conversation")


@dataclass(slots=True)
class TurnResult:
    reply: str
    provider: str
    model: str
    offline_fallback: bool = False
    tool_calls_executed: int = 0
    missed_reminders: list[dict] | None = None


class ConversationEngine:
    """Handles text/voice user turns: memory retrieval -> LLM/tools -> persistence."""

    def __init__(
        self,
        config: ZiaConfig,
        db: ZiaDatabase,
        memory: MemoryManager,
        tools: ToolRegistry,
        router: LLMRouter,
        scheduler: ReminderScheduler | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.memory = memory
        self.tools = tools
        self.router = router
        self.scheduler = scheduler
        self.conversation_id = db.get_or_create_active_conversation()
        self._personality = config.load_personality()
        self.persona_state = ConversationalState()

    async def process_turn(
        self,
        user_text: str,
        voice_mode: bool = False,
        on_token: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> TurnResult:
        clean_input = user_text.strip()
        if not clean_input:
            return TurnResult(reply="", provider="none", model="none", offline_fallback=True)

        # Check missed reminders preserved across downtime/restarts
        missed = self.scheduler.pop_missed_announcements() if self.scheduler else []

        # Record user turn
        self.db.add_message(self.conversation_id, "user", clean_input)

        # Update dynamic conversational mood & handle explicit mode/boundary commands
        self.persona_state.detected_mood = detect_mood(clean_input)
        boundary_reply = apply_boundary_or_mode_feedback(clean_input, self.persona_state, self.memory)
        if boundary_reply is not None:
            self.db.add_message(self.conversation_id, "assistant", boundary_reply)
            return TurnResult(reply=boundary_reply, provider="persona", model=self.persona_state.mode.value, offline_fallback=True)

        # Check explicit memory commands ("remember that ...") with sensitive guard
        ext_results = extract_explicit_memories(clean_input, self.memory)
        for res in ext_results:
            if res.get("requires_approval"):
                msg = (
                    "That looks like sensitive information "
                    f"({res.get('reason', 'approval required')}). "
                    "I didn't save it automatically—say explicitly if you still want it stored."
                )
                self.db.add_message(self.conversation_id, "assistant", msg)
                return TurnResult(reply=msg, provider="guard", model="local", offline_fallback=True)

        # 1-5: Retrieve compact memory context + short-term history
        mem_context = self.memory.build_context_for_query(
            clean_input,
            max_memories=self.config.memory.max_retrieved_memories,
            max_chars=self.config.memory.max_context_chars,
        )
        history_rows = self.db.get_recent_messages(
            self.conversation_id,
            limit=self.config.memory.short_term_turns * 2,
        )

        # Try deterministic local command first if offline_mode or no keys configured
        if self.config.offline_mode or (not self.config.groq_api_key and not self.config.gemini_api_key):
            reply = self.handle_offline(clean_input)
            reply = self._prepend_missed(reply, missed)
            self.db.add_message(self.conversation_id, "assistant", reply)
            self._maybe_summarize()
            return TurnResult(
                reply=reply,
                provider="offline",
                model="local-rules",
                offline_fallback=True,
                missed_reminders=missed,
            )

        # 6: Build LLM message list
        sys_prompt = build_system_prompt(
            personality_text=self._personality,
            timezone_name=self.config.timezone,
            memory_context=mem_context,
            voice_mode=voice_mode,
            mode_guidance=MODE_GUIDANCE[self.persona_state.mode],
            detected_mood=self.persona_state.detected_mood,
        )
        llm_messages: list[LLMMessage] = [LLMMessage(role="system", content=sys_prompt)]
        for row in history_rows:
            role = str(row["role"])
            if role in {"user", "assistant"}:
                llm_messages.append(LLMMessage(role=role, content=str(row["content"])))

        tool_schemas = self.tools.schemas()
        executed_tools = 0

        try:
            # Allow up to 4 tool-call rounds before forcing a final answer
            response = await self.router.complete(
                llm_messages,
                tools=tool_schemas,
                on_token=None,  # stream only final text if no tool calls
            )
            while response.tool_calls and executed_tools < 4:
                llm_messages.append(
                    LLMMessage(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls,
                    )
                )
                for tc in response.tool_calls:
                    executed_tools += 1
                    result_json = self.tools.execute_json(tc.name, tc.arguments)
                    llm_messages.append(
                        LLMMessage(
                            role="tool",
                            content=result_json,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )
                response = await self.router.complete(
                    llm_messages,
                    tools=tool_schemas if executed_tools < 4 else None,
                    on_token=on_token,
                )

            final_text = response.text.strip() or "Done."
            final_text = self._prepend_missed(final_text, missed)
            self.db.add_message(self.conversation_id, "assistant", final_text)
            self._maybe_summarize()
            return TurnResult(
                reply=final_text,
                provider=response.provider,
                model=response.model,
                offline_fallback=False,
                tool_calls_executed=executed_tools,
                missed_reminders=missed,
            )

        except ProviderError as exc:
            logger.warning("Cloud LLM unavailable, falling back to offline handler: %s", exc)
            fallback_reply = self.handle_offline(clean_input, cloud_error=str(exc))
            fallback_reply = self._prepend_missed(fallback_reply, missed)
            self.db.add_message(self.conversation_id, "assistant", fallback_reply)
            return TurnResult(
                reply=fallback_reply,
                provider="offline",
                model="local-fallback",
                offline_fallback=True,
                missed_reminders=missed,
            )

    def _prepend_missed(self, reply: str, missed: list[dict]) -> str:
        if not missed:
            return reply
        titles = []
        for r in missed:
            titles.append(r["title"])
            if self.scheduler:
                self.scheduler.reminders.mark_delivered(int(r["id"]))
        note = "By the way, you had missed reminders: " + ", ".join(titles) + "."
        return f"{note}\n\n{reply}"

    def _maybe_summarize(self) -> None:
        count = self.db.count_messages(self.conversation_id)
        trigger = max(8, self.config.memory.summary_trigger_messages)
        if count > 0 and count % trigger == 0:
            recent = self.db.get_recent_messages(self.conversation_id, limit=trigger)
            summary = build_extractive_summary(recent)
            if summary:
                store_summary(self.db, self.conversation_id, summary, len(recent))

    def handle_offline(self, user_text: str, cloud_error: str | None = None) -> str:
        """Deterministic offline command handler for core local utilities."""
        lower = user_text.lower().strip()

        if re.search(r"\b(what time|current time|time is it)\b", lower):
            res = self.tools.execute("get_time")
            return f"It's {res['formatted']}."

        if re.search(r"\b(what date|today'?s date|what day)\b", lower):
            res = self.tools.execute("get_date")
            return f"Today is {res['formatted']}."

        if "battery" in lower:
            res = self.tools.execute("get_battery_status")
            if res.get("percentage") is None:
                return "Battery telemetry isn't exposed on this host."
            return f"Battery is at {res['percentage']}% ({res.get('status', 'unknown')})."

        if "network" in lower or "wifi" in lower or "internet" in lower or "online" in lower:
            res = self.tools.execute("get_network_status")
            state = "online" if res.get("online") else "offline"
            ssid = f" on {res['ssid']}" if res.get("ssid") else ""
            return f"Network is currently {state}{ssid}."

        if "uptime" in lower:
            res = self.tools.execute("get_system_uptime")
            return f"ZIA has been running for {res['process_uptime_seconds']} seconds."

        if re.search(r"\b(list|show|what are my)\s+reminders\b", lower) or lower == "reminders":
            items = self.tools.execute("list_reminders").get("reminders", [])
            if not items:
                return "You have no pending reminders."
            lines = [f"#{r['id']} {r['title']} (due {r['due_at']})" for r in items[:8]]
            return "Pending reminders: " + "; ".join(lines) + "."

        m_rem = re.match(r"^(?:remind me to|set a reminder to)\s+(.+?)\s+(in\s+\d+\s+\w+|at\s+\d{2}:\d{2})$", lower)
        if m_rem:
            title = m_rem.group(1).strip()
            due = m_rem.group(2).replace("at ", "").strip()
            res = self.tools.execute("create_reminder", {"title": title, "due_at": due})
            if res.get("created"):
                return f"Reminder saved: {title} ({res['reminder']['due_at']})."

        if re.search(r"\b(tasks|projects|work on today|todo)\b", lower):
            data = self.tools.execute("list_projects")
            tasks = data.get("open_tasks", [])
            if not tasks:
                return "No open tasks recorded yet."
            lines = [f"#{t['id']} {t['title']}" for t in tasks[:6]]
            return "Open tasks: " + "; ".join(lines) + "."

        m_task = re.match(r"^(?:add task|new task)\s+(.+)$", user_text.strip(), re.IGNORECASE)
        if m_task:
            res = self.tools.execute("add_task", {"title": m_task.group(1).strip()})
            return f"Added task #{res['id']}: {res['title']}."

        m_mem = re.match(r"^(?:search memory|find memory|recall)\s+(.+)$", user_text.strip(), re.IGNORECASE)
        if m_mem:
            matches = self.tools.execute("search_memory", {"query": m_mem.group(1).strip()}).get("matches", [])
            if not matches:
                return "No matching memories found."
            return "Memory matches: " + "; ".join(f"{m['title']}: {m['body']}" for m in matches[:4])

        if cloud_error:
            return (
                "I'm currently unable to reach the cloud LLM providers, so I can only handle "
                "local commands (time, date, reminders, tasks, memory lookup, battery, network, uptime)."
            )
        return (
            "Offline mode active. Ask about time, date, reminders, tasks, memory search, "
            "battery, network, or uptime—or configure GROQ_API_KEY / GEMINI_API_KEY."
        )
