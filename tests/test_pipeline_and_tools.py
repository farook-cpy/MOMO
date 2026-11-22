"""Tests for voice state machine, safe local tools, offline fallback, VAD, and CLI."""

from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from audio.vad import EnergyVAD
from cli import main as cli_main
from core.assistant import ZiaAssistant
from core.config import ZiaConfig
from core.state_machine import InvalidStateTransition, PipelineState, VoiceStateMachine


def test_state_machine_transitions() -> None:
    sm = VoiceStateMachine()
    assert sm.state == PipelineState.START
    sm.transition(PipelineState.INITIALIZE)
    sm.transition(PipelineState.WAKE_WORD_LISTENING)
    sm.transition(PipelineState.WAKE_WORD_DETECTED)
    sm.transition(PipelineState.RECORD_USER_SPEECH)
    sm.transition(PipelineState.SPEECH_TO_TEXT)
    sm.transition(PipelineState.MEMORY_RETRIEVAL)
    sm.transition(PipelineState.LLM_PROCESSING)
    sm.transition(PipelineState.TEXT_TO_SPEECH)
    sm.transition(PipelineState.SPEAKER_OUTPUT)
    sm.transition(PipelineState.FOLLOW_UP_LISTENING)
    sm.transition(PipelineState.WAKE_WORD_LISTENING)

    with pytest.raises(InvalidStateTransition):
        sm.transition(PipelineState.SPEAKER_OUTPUT)


def test_local_tools_and_confirmation_guard(tmp_config: ZiaConfig) -> None:
    zia = ZiaAssistant(tmp_config)
    try:
        t = zia.tools.execute("get_time")
        d = zia.tools.execute("get_date")
        assert "formatted" in t and "date" in d

        task = zia.tools.execute("add_task", {"title": "Test Termux mic", "project_name": "ZIA"})
        assert task["status"] == "open"
        done = zia.tools.execute("complete_task", {"task_id": task["id"]})
        assert done["completed"] is True

        saved = zia.tools.execute("save_memory", {"content": "Xperia uses Snapdragon 888"})
        assert saved["saved"] is True

        # Deletion must require confirmation
        denied = zia.tools.execute(
            "delete_memory",
            {"table": "important_facts", "record_id": saved["id"], "confirm": False},
        )
        assert denied["requires_confirmation"] is True

        ok_del = zia.tools.execute(
            "delete_memory",
            {"table": "important_facts", "record_id": saved["id"], "confirm": True},
        )
        assert ok_del["deleted"] is True
    finally:
        zia.shutdown()


@pytest.mark.asyncio
async def test_offline_fallback_and_missed_reminder_reporting(tmp_config: ZiaConfig) -> None:
    tmp_config.offline_mode = True
    zia = ZiaAssistant(tmp_config)
    try:
        past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        rem = zia.reminders.create_reminder("Call Alex", past)
        zia.reminders.mark_missed(int(rem["id"]))

        res = await zia.ask("What time is it?")
        assert res.offline_fallback is True
        assert "Call Alex" in res.reply
        assert ":" in res.reply

        task_res = await zia.ask("add task Verify offline mode")
        assert "Verify offline mode" in task_res.reply
    finally:
        zia.shutdown()


@pytest.mark.asyncio
async def test_llm_tool_calling_loop(tmp_config: ZiaConfig) -> None:
    tmp_config.groq_api_key = "gsk_mock_key_1234567890"
    tmp_config.llm.stream_responses = False
    step = 0

    def mock_groq(_request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "add_task",
                                            "arguments": '{"title": "Test speaker output"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Added that task for you."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_groq))
    zia = ZiaAssistant(tmp_config)
    try:
        zia.providers["groq"]._external_client = client
        res = await zia.ask("Add a task to test speaker output")
        assert res.tool_calls_executed == 1
        assert "Added that task" in res.reply
        assert any(t["title"] == "Test speaker output" for t in zia.memory.list_tasks("open"))
    finally:
        await client.aclose()
        zia.shutdown()


def test_energy_vad_detects_speech_then_silence() -> None:
    import math

    vad = EnergyVAD(
        energy_threshold=300.0,
        silence_timeout_seconds=0.16,
        frame_duration_ms=80,
        min_speech_frames=1,
    )
    # 500 Hz sine wave (speech-band ZCR) vs silence
    loud_samples = [int(2000 * math.sin(2 * math.pi * 500 * i / 16000)) for i in range(1280)]
    loud_frame = struct.pack("<1280h", *loud_samples)
    quiet_frame = struct.pack("<1280h", *([0] * 1280))

    sp, end = vad.process_frame(loud_frame)
    assert sp is True and end is False

    _, end1 = vad.process_frame(quiet_frame)
    assert end1 is False
    _, end2 = vad.process_frame(quiet_frame)
    assert end2 is True


def test_stt_noise_hallucination_filter() -> None:
    from audio.stt import clean_transcript

    assert clean_transcript("[BLANK_AUDIO]") == ""
    assert clean_transcript("(music) Thank you.") == ""
    assert clean_transcript("  What time is it?  ") == "What time is it?"


def test_cli_config_command() -> None:
    assert cli_main(["config"]) == 0
