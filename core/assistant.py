"""Top-level ZIA assistant runtime tying together intelligence, memory, tools, scheduler, and voice."""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from typing import Any

from audio.microphone import PulseAudioMicrophone, write_wav
from audio.stt import build_stt
from audio.tts import BaseTTS, build_tts
from audio.vad import has_sustained_speech
from audio.wakeword import BaseWakeWordDetector, ManualTriggerWakeWord, build_wakeword_detector
from automation.reminders import ReminderManager
from automation.scheduler import ReminderScheduler
from core.config import ZiaConfig, load_config
from core.conversation import ConversationEngine, TurnResult
from core.state_machine import PipelineState, VoiceStateMachine
from core.tools import ToolRegistry
from intelligence.gemini import GeminiProvider
from intelligence.groq import GroqProvider
from intelligence.provider import LLMRouter
from memory.database import ZiaDatabase
from memory.retrieval import MemoryManager
from system.logging_setup import setup_logging
from system.watchdog import (
    WatchdogSupervisor,
    acquire_termux_wakelock,
    release_termux_wakelock,
    remove_pid_file,
    write_pid_file,
)

logger = logging.getLogger("zia.assistant")


class ZiaAssistant:
    """Lifecycle and execution controller for ZIA (CLI, daemon, and voice pipeline)."""

    def __init__(self, config: ZiaConfig | None = None, console_logs: bool = False) -> None:
        self.config = config or load_config()
        setup_logging(
            self.config.system.log_dir,
            level=self.config.system.log_level,
            console=console_logs,
        )
        self.db = ZiaDatabase(self.config.db_path)
        self.db.connect()
        self.memory = MemoryManager(
            self.db,
            duplicate_threshold=self.config.memory.duplicate_similarity_threshold,
        )
        self.reminders = ReminderManager(self.db, default_timezone=self.config.timezone)
        self.tools = ToolRegistry(self.config, self.memory, self.reminders)

        self.providers = {
            "groq": GroqProvider(
                api_key=self.config.groq_api_key,
                endpoint=self.config.llm.groq,
                timeout_seconds=self.config.llm.request_timeout_seconds,
            ),
            "gemini": GeminiProvider(
                api_key=self.config.gemini_api_key,
                endpoint=self.config.llm.gemini,
                timeout_seconds=self.config.llm.request_timeout_seconds,
            ),
        }
        self.router = LLMRouter(self.config, self.db, self.providers)
        self.tts: BaseTTS = build_tts(self.config.audio, root_dir=self.config.root_dir)
        self.scheduler = ReminderScheduler(
            self.reminders,
            poll_interval_seconds=self.config.scheduler.poll_interval_seconds,
            on_due=self._deliver_reminder,
        )
        self.watchdog = WatchdogSupervisor(self.config, self.db)
        self.engine = ConversationEngine(
            config=self.config,
            db=self.db,
            memory=self.memory,
            tools=self.tools,
            router=self.router,
            scheduler=self.scheduler,
        )
        self.state_machine = VoiceStateMachine()
        self._stop_event = asyncio.Event()

    async def _deliver_reminder(self, reminder: dict[str, Any]) -> bool:
        text = f"Reminder: {reminder['message']}"
        logger.info("Delivering reminder #%s: %s", reminder["id"], reminder["title"])
        return await self.tts.speak(text)

    async def ask(self, text: str, voice_mode: bool = False) -> TurnResult:
        return await self.engine.process_turn(text, voice_mode=voice_mode)

    def shutdown(self) -> None:
        self._stop_event.set()
        self.scheduler.stop()
        self.watchdog.stop()
        self.db.close()

    async def run_daemon(self, enable_voice: bool = True) -> None:
        """Run persistent background assistant (scheduler + watchdog + optional voice loop)."""
        acquire_termux_wakelock()
        write_pid_file(self.config.system.pid_file)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_stop)
            except NotImplementedError:
                pass  # Windows event loop fallback

        self.state_machine.reset()
        self.state_machine.transition(PipelineState.INITIALIZE)

        tasks: list[asyncio.Task[Any]] = [
            asyncio.create_task(self.watchdog.supervise_task("scheduler", self.scheduler.run)),
            asyncio.create_task(self.watchdog.supervise_task("watchdog", self.watchdog.run)),
        ]

        if enable_voice:
            tasks.append(
                asyncio.create_task(self.watchdog.supervise_task("voice_loop", self._voice_loop))
            )
        else:
            self.state_machine.transition(PipelineState.WAKE_WORD_LISTENING)

        logger.info("ZIA daemon started (voice=%s).", enable_voice)
        try:
            await self._stop_event.wait()
        finally:
            self.scheduler.stop()
            self.watchdog.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.state_machine.state != PipelineState.STOPPED:
                self.state_machine.transition(PipelineState.STOPPED)
            remove_pid_file(self.config.system.pid_file)
            release_termux_wakelock()
            self.db.close()
            logger.info("ZIA daemon stopped cleanly.")

    def _request_stop(self) -> None:
        logger.info("Received shutdown signal.")
        self._stop_event.set()

    async def _voice_loop(self) -> None:
        """Half-duplex voice pipeline (requires PulseAudio + STT + WakeWord on Xperia)."""
        if self.state_machine.state == PipelineState.INITIALIZE:
            self.state_machine.transition(PipelineState.WAKE_WORD_LISTENING)

        if not PulseAudioMicrophone.is_available():
            logger.warning(
                "PulseAudio (parec) unavailable; voice loop idling in scheduler-only mode."
            )
            await self._stop_event.wait()
            return

        try:
            detector: BaseWakeWordDetector = build_wakeword_detector(
                self.config.audio, self.config.root_dir
            )
        except Exception as exc:
            logger.warning("Wake-word detector unavailable (%s); falling back to manual mode.", exc)
            detector = ManualTriggerWakeWord()

        if isinstance(detector, ManualTriggerWakeWord):
            logger.info("Wake-word backend is 'manual'; voice loop waiting without mic capture.")
            await self._stop_event.wait()
            return

        stt = build_stt(self.config.audio, self.config.root_dir)
        if stt is None:
            logger.warning("No STT engine configured; voice loop idling.")
            await self._stop_event.wait()
            return

        mic = PulseAudioMicrophone(self.config.audio)
        tmp_wav = Path(self.config.root_dir / "data" / "utterance.wav")
        await mic.start()
        try:
            while not self._stop_event.is_set():
                frame = await mic.read_frame()
                if not detector.process_frame(frame):
                    continue

                self.state_machine.transition(PipelineState.WAKE_WORD_DETECTED)
                detector.reset()
                await self._handle_voice_interaction(mic, stt, tmp_wav)
        finally:
            await mic.stop()

    async def _handle_voice_interaction(
        self,
        mic: PulseAudioMicrophone,
        stt: Any,
        tmp_wav: Path,
    ) -> None:
        """Execute a spoken interaction + follow-up window with half-duplex echo guard."""
        in_followup = False
        while not self._stop_event.is_set():
            self.state_machine.transition(PipelineState.RECORD_USER_SPEECH)
            timeout = (
                self.config.audio.followup_timeout_seconds
                if in_followup
                else self.config.audio.max_record_seconds
            )
            pcm = await mic.record_utterance(timeout_seconds=timeout)
            if not pcm:
                self.state_machine.transition(PipelineState.WAKE_WORD_LISTENING)
                return

            # Reject brief room clicks/noise before invoking STT
            if not has_sustained_speech(
                pcm,
                frame_bytes=mic.frame_bytes,
                energy_threshold=self.config.audio.vad_energy_threshold,
                min_speech_frames=2,
            ):
                self.state_machine.transition(PipelineState.WAKE_WORD_LISTENING)
                return

            self.state_machine.transition(PipelineState.SPEECH_TO_TEXT)
            write_wav(
                tmp_wav,
                pcm,
                sample_rate=self.config.audio.sample_rate,
                channels=self.config.audio.channels,
            )
            try:
                transcript = (await stt.transcribe_wav(tmp_wav)).strip()
            except Exception as exc:
                logger.warning("STT failed: %s", exc)
                self.state_machine.transition(PipelineState.WAKE_WORD_LISTENING)
                return

            if not transcript:
                self.state_machine.transition(PipelineState.WAKE_WORD_LISTENING)
                return

            self.state_machine.transition(PipelineState.MEMORY_RETRIEVAL)
            self.state_machine.transition(PipelineState.LLM_PROCESSING)
            turn = await self.ask(transcript, voice_mode=True)

            self.state_machine.transition(PipelineState.TEXT_TO_SPEECH)
            self.state_machine.transition(PipelineState.SPEAKER_OUTPUT)

            # Half-duplex feedback prevention: stop mic during TTS + cooldown
            await mic.stop()
            try:
                await self.tts.speak(turn.reply)
                await asyncio.sleep(self.config.audio.echo_cooldown_ms / 1000.0)
            finally:
                await mic.start()

            self.state_machine.transition(PipelineState.FOLLOW_UP_LISTENING)
            in_followup = True
            # Listen briefly for a follow-up utterance; if silent, loop returns to WAKE_WORD_LISTENING
