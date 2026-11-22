"""Command-line interface for ZIA."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from audio.microphone import test_microphone_stream
from audio.tts import build_tts
from core.assistant import ZiaAssistant
from core.config import load_config
from memory.database import ZiaDatabase
from memory.retrieval import MemoryManager
from system.health import run_doctor
from system.watchdog import is_pid_running, read_pid_file, stop_process_by_pid


def _cmd_chat(assistant: ZiaAssistant) -> int:
    print(f"{assistant.config.name} terminal chat ready. Type 'exit' or 'quit' to leave.")
    if assistant.config.offline_mode or (
        not assistant.config.groq_api_key and not assistant.config.gemini_api_key
    ):
        print("[note] Running in offline/local mode (no cloud LLM API keys configured).")

    async def _loop() -> None:
        while True:
            try:
                user_input = await asyncio.to_thread(input, "\nYou> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            text = user_input.strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit", "/quit", "/exit"}:
                break
            result = await assistant.ask(text, voice_mode=False)
            tag = f"[{result.provider}:{result.model}]"
            print(f"\nZIA {tag}> {result.reply}")

    try:
        asyncio.run(_loop())
    finally:
        assistant.shutdown()
    return 0


def _cmd_doctor(config_only: bool = False) -> int:
    cfg = load_config()
    db = ZiaDatabase(cfg.db_path)
    db.connect()
    try:
        checks = run_doctor(cfg, db)
    finally:
        db.close()

    failed = 0
    for chk in checks:
        print(f"[{chk.status:<4}] {chk.name:<24} {chk.detail}")
        if chk.status == "FAIL":
            failed += 1
    return 1 if failed else 0


def _cmd_status() -> int:
    cfg = load_config()
    pid = read_pid_file(cfg.system.pid_file)
    running = pid is not None and is_pid_running(pid)
    print(f"Daemon: {'RUNNING (pid=' + str(pid) + ')' if running else 'STOPPED'}")
    print(f"Database: {cfg.db_path} (exists={cfg.db_path.exists()})")
    print(f"Primary LLM: {cfg.llm.primary_provider} ({cfg.llm.groq.model})")
    print(f"Fallback LLM: {cfg.llm.fallback_provider} ({cfg.llm.gemini.model})")
    return 0


def _cmd_logs(lines: int = 50) -> int:
    cfg = load_config()
    log_file = cfg.system.log_dir / "zia.log"
    if not log_file.exists():
        print(f"No log file yet at {log_file}")
        return 0
    all_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in all_lines[-lines:]:
        print(line)
    return 0


def _cmd_start(daemon: bool = False, no_voice: bool = False) -> int:
    cfg = load_config()
    pid = read_pid_file(cfg.system.pid_file)
    if pid is not None and is_pid_running(pid):
        print(f"ZIA is already running (pid={pid}).")
        return 0

    if daemon:
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "main.py"), "run-daemon"]
        if no_voice:
            cmd.append("--no-voice")
        cfg.system.log_dir.mkdir(parents=True, exist_ok=True)
        out_file = (cfg.system.log_dir / "daemon.out").open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=out_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"ZIA background daemon launched (pid={proc.pid}).")
        return 0

    assistant = ZiaAssistant(cfg, console_logs=True)
    asyncio.run(assistant.run_daemon(enable_voice=not no_voice))
    return 0


def _cmd_stop() -> int:
    cfg = load_config()
    if stop_process_by_pid(cfg.system.pid_file):
        print("ZIA daemon stopped.")
        return 0
    print("ZIA daemon is not running.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zia", description="ZIA Terminal-Only AI Companion")
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Start ZIA service")
    p_start.add_argument("--daemon", action="store_true", help="Run detached in background")
    p_start.add_argument("--no-voice", action="store_true", help="Disable voice loop (scheduler + watchdog only)")

    p_run = sub.add_parser("run-daemon", help="Internal foreground daemon entry")
    p_run.add_argument("--no-voice", action="store_true")

    sub.add_parser("stop", help="Stop ZIA background daemon")

    p_restart = sub.add_parser("restart", help="Restart ZIA background daemon")
    p_restart.add_argument("--no-voice", action="store_true")

    sub.add_parser("status", help="Show daemon and configuration status")

    p_logs = sub.add_parser("logs", help="Show recent redacted logs")
    p_logs.add_argument("-n", "--lines", type=int, default=50)

    sub.add_parser("chat", help="Interactive terminal chat with ZIA")
    sub.add_parser("test-mic", help="Test Termux microphone capture")
    sub.add_parser("test-speaker", help="Test TTS speaker playback")
    sub.add_parser("test-wakeword", help="Check wake-word engine readiness")
    sub.add_parser("config", help="Show active non-secret configuration")
    sub.add_parser("doctor", help="Run full system and dependency diagnostics")

    p_mem = sub.add_parser("memory", help="Inspect, search, export, or delete memories")
    mem_sub = p_mem.add_subparsers(dest="mem_action", required=True)
    p_mem_list = mem_sub.add_parser("list")
    p_mem_list.add_argument("--limit", type=int, default=30)
    p_mem_search = mem_sub.add_parser("search")
    p_mem_search.add_argument("query")
    p_mem_export = mem_sub.add_parser("export")
    p_mem_export.add_argument("--out", default="data/memory_export.json")
    p_mem_del = mem_sub.add_parser("delete")
    p_mem_del.add_argument("table")
    p_mem_del.add_argument("record_id", type=int)
    p_mem_del.add_argument("--yes", action="store_true", help="Confirm permanent deletion")

    p_rem = sub.add_parser("reminders", help="Manage scheduled reminders")
    rem_sub = p_rem.add_subparsers(dest="rem_action", required=True)
    p_rem_list = rem_sub.add_parser("list")
    p_rem_list.add_argument("--status", default="pending")
    p_rem_add = rem_sub.add_parser("add")
    p_rem_add.add_argument("title")
    p_rem_add.add_argument("due_at")
    p_rem_add.add_argument("--type", default="once")

    p_backup = sub.add_parser("backup", help="Create verified SQLite backup")
    p_restore = sub.add_parser("restore", help="Restore SQLite database from backup")
    p_restore.add_argument("backup_file")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config()

    if args.command == "chat":
        return _cmd_chat(ZiaAssistant(cfg))
    if args.command == "start":
        return _cmd_start(daemon=args.daemon, no_voice=args.no_voice)
    if args.command == "run-daemon":
        return _cmd_start(daemon=False, no_voice=args.no_voice)
    if args.command == "stop":
        return _cmd_stop()
    if args.command == "restart":
        _cmd_stop()
        return _cmd_start(daemon=True, no_voice=args.no_voice)
    if args.command == "status":
        return _cmd_status()
    if args.command == "logs":
        return _cmd_logs(args.lines)
    if args.command == "doctor":
        return _cmd_doctor()
    if args.command == "config":
        safe = {
            "name": cfg.name,
            "timezone": cfg.timezone,
            "offline_mode": cfg.offline_mode,
            "db_path": str(cfg.db_path),
            "primary_provider": cfg.llm.primary_provider,
            "fallback_provider": cfg.llm.fallback_provider,
            "groq_model": cfg.llm.groq.model,
            "gemini_model": cfg.llm.gemini.model,
            "groq_key_configured": bool(cfg.groq_api_key),
            "gemini_key_configured": bool(cfg.gemini_api_key),
            "mic_backend": cfg.audio.mic_backend,
            "tts_backend": cfg.audio.tts_backend,
            "stt_backend": cfg.audio.stt_backend,
            "wakeword_backend": cfg.audio.wakeword_backend,
        }
        print(json.dumps(safe, indent=2))
        return 0
    if args.command == "test-mic":
        res = asyncio.run(test_microphone_stream(cfg.audio, duration_seconds=3.0))
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 1
    if args.command == "test-speaker":
        tts = build_tts(cfg.audio)
        ok = asyncio.run(tts.speak("Zia speaker test complete."))
        print(f"Speaker test ({tts.__class__.__name__}): {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1
    if args.command == "test-wakeword":
        from audio.wakeword import build_wakeword_detector

        try:
            det = build_wakeword_detector(cfg.audio, cfg.root_dir)
            print(f"Wake-word engine ready: {det.__class__.__name__}")
            return 0
        except Exception as exc:
            print(f"Wake-word unavailable: {exc}")
            return 1
    if args.command == "memory":
        db = ZiaDatabase(cfg.db_path)
        db.connect()
        mem = MemoryManager(db)
        try:
            if args.mem_action == "list":
                print(json.dumps(mem.list_memories(args.limit), indent=2))
            elif args.mem_action == "search":
                matches = mem.search(args.query)
                print(
                    json.dumps(
                        [
                            {
                                "table": m.source_table,
                                "id": m.source_id,
                                "title": m.title,
                                "body": m.body,
                                "score": m.combined_score,
                            }
                            for m in matches
                        ],
                        indent=2,
                    )
                )
            elif args.mem_action == "export":
                out = Path(args.out)
                if not out.is_absolute():
                    out = cfg.root_dir / out
                mem.export_all(out)
                print(f"Exported memory to {out}")
            elif args.mem_action == "delete":
                res = mem.delete_memory(args.table, args.record_id, confirm=args.yes)
                print(json.dumps(res, indent=2))
                return 0 if res.get("deleted") else 1
        finally:
            db.close()
        return 0
    if args.command == "reminders":
        from automation.reminders import ReminderManager

        db = ZiaDatabase(cfg.db_path)
        db.connect()
        rems = ReminderManager(db, default_timezone=cfg.timezone)
        try:
            if args.rem_action == "list":
                print(json.dumps(rems.list_reminders(args.status), indent=2))
            elif args.rem_action == "add":
                created = rems.create_reminder(args.title, args.due_at, schedule_type=args.type)
                print(json.dumps(created, indent=2))
        finally:
            db.close()
        return 0
    if args.command == "backup":
        db = ZiaDatabase(cfg.db_path)
        db.connect()
        try:
            dest = db.backup_to(cfg.backup_dir)
            print(f"Backup created: {dest}")
        finally:
            db.close()
        return 0
    if args.command == "restore":
        db = ZiaDatabase(cfg.db_path)
        try:
            db.restore_from(Path(args.backup_file))
            print(f"Restored database from {args.backup_file}")
        finally:
            db.close()
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
