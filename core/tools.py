"""Safe, typed local tools exposed to ZIA's LLM and offline handler."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from automation.reminders import ReminderManager
from core.config import ZiaConfig
from memory.retrieval import MemoryManager
from system.connectivity import get_network_status as sys_network_status
from system.health import get_battery_status as sys_battery_status
from system.health import get_system_uptime as sys_system_uptime


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    destructive: bool = False


class ToolRegistry:
    """Validated, sandboxed registry of local Python tools (never executes shell commands)."""

    def __init__(
        self,
        config: ZiaConfig,
        memory: MemoryManager,
        reminders: ReminderManager,
    ) -> None:
        self.config = config
        self.memory = memory
        self.reminders = reminders
        self._tools: dict[str, ToolSpec] = {}
        self._register_defaults()

    def _local_now(self) -> datetime:
        try:
            tz = ZoneInfo(self.config.timezone)
        except Exception:
            tz = timezone.utc
        return datetime.now(tz)

    def _register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def _register_defaults(self) -> None:
        self._register(
            ToolSpec(
                name="get_time",
                description="Get the current local time.",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda _args: {
                    "time": self._local_now().strftime("%H:%M:%S"),
                    "formatted": self._local_now().strftime("%I:%M %p %Z"),
                    "timezone": self.config.timezone,
                },
            )
        )
        self._register(
            ToolSpec(
                name="get_date",
                description="Get the current local date and day of the week.",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda _args: {
                    "date": self._local_now().strftime("%Y-%m-%d"),
                    "day": self._local_now().strftime("%A"),
                    "formatted": self._local_now().strftime("%A, %B %d, %Y"),
                },
            )
        )
        self._register(
            ToolSpec(
                name="create_reminder",
                description="Create a persistent reminder or recurring routine.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short reminder title"},
                        "due_at": {
                            "type": "string",
                            "description": "ISO timestamp, HH:MM, or relative like 'in 15 minutes'",
                        },
                        "message": {"type": "string", "description": "Spoken reminder message"},
                        "schedule_type": {
                            "type": "string",
                            "enum": [
                                "once",
                                "daily",
                                "weekly",
                                "routine",
                                "deadline",
                                "morning_summary",
                                "evening_summary",
                            ],
                        },
                        "recurrence_rule": {"type": "string", "description": "Optional rule e.g. 'every 4h'"},
                    },
                    "required": ["title", "due_at"],
                },
                handler=self._tool_create_reminder,
            )
        )
        self._register(
            ToolSpec(
                name="list_reminders",
                description="List scheduled reminders.",
                parameters={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["pending", "missed", "triggered", "cancelled", "all"],
                        }
                    },
                    "required": [],
                },
                handler=lambda args: {
                    "reminders": self.reminders.list_reminders(str(args.get("status", "pending")))
                },
            )
        )
        self._register(
            ToolSpec(
                name="cancel_reminder",
                description="Cancel a reminder by ID or title.",
                parameters={
                    "type": "object",
                    "properties": {
                        "reminder_id": {"type": "integer"},
                        "title_query": {"type": "string"},
                    },
                    "required": [],
                },
                handler=self._tool_cancel_reminder,
            )
        )
        self._register(
            ToolSpec(
                name="save_memory",
                description="Save an important fact or personal preference to long-term memory.",
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Fact or preference to remember"},
                        "category": {"type": "string"},
                        "importance": {"type": "number"},
                        "approved": {
                            "type": "boolean",
                            "description": "Set true only if user explicitly approved storing sensitive data",
                        },
                    },
                    "required": ["content"],
                },
                handler=self._tool_save_memory,
            )
        )
        self._register(
            ToolSpec(
                name="search_memory",
                description="Search long-term memory using FTS5.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
                handler=self._tool_search_memory,
            )
        )
        self._register(
            ToolSpec(
                name="delete_memory",
                description="Permanently delete a memory record. Requires confirm=true after user confirmation.",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "enum": [
                                "important_facts",
                                "personal_preferences",
                                "projects",
                                "tasks",
                                "events",
                                "conversation_summaries",
                            ],
                        },
                        "record_id": {"type": "integer"},
                        "confirm": {"type": "boolean"},
                    },
                    "required": ["table", "record_id"],
                },
                handler=self._tool_delete_memory,
                destructive=True,
            )
        )
        self._register(
            ToolSpec(
                name="get_battery_status",
                description="Get current device battery level, charging state, and temperature.",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda _args: sys_battery_status(),
            )
        )
        self._register(
            ToolSpec(
                name="get_network_status",
                description="Get internet connectivity and Wi-Fi status.",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda _args: sys_network_status(),
            )
        )
        self._register(
            ToolSpec(
                name="get_system_uptime",
                description="Get ZIA process and OS uptime.",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda _args: sys_system_uptime(),
            )
        )
        self._register(
            ToolSpec(
                name="list_projects",
                description="List tracked projects and optionally open tasks.",
                parameters={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["active", "paused", "completed", "archived"]}
                    },
                    "required": [],
                },
                handler=lambda args: {
                    "projects": self.memory.list_projects(args.get("status")),
                    "open_tasks": self.memory.list_tasks("open"),
                },
            )
        )
        self._register(
            ToolSpec(
                name="add_task",
                description="Add a new task, optionally linked to a project.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "notes": {"type": "string"},
                        "project_name": {"type": "string"},
                        "priority": {"type": "integer"},
                        "due_at": {"type": "string"},
                    },
                    "required": ["title"],
                },
                handler=self._tool_add_task,
            )
        )
        self._register(
            ToolSpec(
                name="complete_task",
                description="Mark an open task as completed by ID or title.",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer"},
                        "title_query": {"type": "string"},
                    },
                    "required": [],
                },
                handler=self._tool_complete_task,
            )
        )

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = self._tools.get(name)
        if not spec:
            return {"error": f"Unknown tool: {name}"}
        args = arguments or {}
        if not isinstance(args, dict):
            return {"error": "Tool arguments must be an object."}

        required = spec.parameters.get("required") or []
        for req in required:
            if req not in args or args[req] is None or args[req] == "":
                return {"error": f"Missing required parameter: {req}"}

        try:
            return spec.handler(args)
        except Exception as exc:
            return {"error": f"Tool {name} failed: {exc}"}

    def execute_json(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        return json.dumps(self.execute(name, arguments), ensure_ascii=False)

    # --- Handlers ---

    def _tool_create_reminder(self, args: dict[str, Any]) -> dict[str, Any]:
        created = self.reminders.create_reminder(
            title=str(args["title"]),
            due_at=str(args["due_at"]),
            message=str(args.get("message", "")),
            schedule_type=str(args.get("schedule_type", "once")),
            recurrence_rule=str(args.get("recurrence_rule", "")),
        )
        return {"created": True, "reminder": created}

    def _tool_cancel_reminder(self, args: dict[str, Any]) -> dict[str, Any]:
        rid = int(args["reminder_id"]) if args.get("reminder_id") is not None else None
        tq = str(args["title_query"]) if args.get("title_query") else None
        return self.reminders.cancel_reminder(reminder_id=rid, title_query=tq)

    def _tool_save_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.memory.save_fact(
            fact=str(args["content"]),
            category=str(args.get("category", "general")),
            importance=float(args.get("importance", 0.7)),
            approved=bool(args.get("approved", False)),
        )

    def _tool_search_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = max(1, min(20, int(args.get("limit", 6))))
        matches = self.memory.search(str(args["query"]), limit=limit)
        return {
            "matches": [
                {
                    "table": m.source_table,
                    "id": m.source_id,
                    "title": m.title,
                    "body": m.body,
                    "score": m.combined_score,
                }
                for m in matches
            ]
        }

    def _tool_delete_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.memory.delete_memory(
            table=str(args["table"]),
            record_id=int(args["record_id"]),
            confirm=bool(args.get("confirm", False)),
        )

    def _tool_add_task(self, args: dict[str, Any]) -> dict[str, Any]:
        prio = max(1, min(5, int(args.get("priority", 2))))
        return self.memory.add_task(
            title=str(args["title"]),
            notes=str(args.get("notes", "")),
            project_name=str(args["project_name"]) if args.get("project_name") else None,
            priority=prio,
            due_at=str(args["due_at"]) if args.get("due_at") else None,
        )

    def _tool_complete_task(self, args: dict[str, Any]) -> dict[str, Any]:
        tid = int(args["task_id"]) if args.get("task_id") is not None else None
        tq = str(args["title_query"]) if args.get("title_query") else None
        return self.memory.complete_task(task_id=tid, title_query=tq)
