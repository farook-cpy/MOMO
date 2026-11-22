"""Tests for Groq/Gemini providers, failover, backoff, quota limits, and log redaction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from core.config import ZiaConfig
from intelligence.gemini import GeminiProvider
from intelligence.groq import GroqProvider
from intelligence.provider import LLMMessage, LLMRouter, QuotaExceededError
from memory.database import ZiaDatabase
from system.logging_setup import redact_text, setup_logging


@pytest.mark.asyncio
async def test_groq_primary_switches_to_gemini_on_rate_limit(
    tmp_config: ZiaConfig,
    test_db: ZiaDatabase,
) -> None:
    tmp_config.groq_api_key = "gsk_test_key_1234567890"
    tmp_config.gemini_api_key = "AIzaSyTestKey123456789012345"
    tmp_config.llm.max_retries = 2
    tmp_config.llm.stream_responses = False

    groq_calls = 0

    def groq_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal groq_calls
        groq_calls += 1
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-goog-api-key") == tmp_config.gemini_api_key
        assert "key=" not in str(request.url)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "Fallback Gemini response."}]}}],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 6,
                    "totalTokenCount": 18,
                },
            },
        )

    groq_client = httpx.AsyncClient(transport=httpx.MockTransport(groq_handler))
    gemini_client = httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler))

    try:
        providers = {
            "groq": GroqProvider(tmp_config.groq_api_key, tmp_config.llm.groq, client=groq_client),
            "gemini": GeminiProvider(tmp_config.gemini_api_key, tmp_config.llm.gemini, client=gemini_client),
        }
        router = LLMRouter(tmp_config, test_db, providers)
        res = await router.complete([LLMMessage(role="user", content="Hello")])
        assert groq_calls == 2  # retried twice with backoff
        assert res.provider == "gemini"
        assert res.text == "Fallback Gemini response."
        usage = test_db.get_daily_usage("gemini", router._today_key())
        assert usage["requests"] == 1
        assert usage["tokens"] == 18
    finally:
        await groq_client.aclose()
        await gemini_client.aclose()


@pytest.mark.asyncio
async def test_daily_quota_cap_enforced(tmp_config: ZiaConfig, test_db: ZiaDatabase) -> None:
    tmp_config.groq_api_key = "gsk_test_key_1234567890"
    tmp_config.llm.groq.daily_request_limit = 1
    tmp_config.llm.stream_responses = False

    def groq_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(groq_handler))
    try:
        providers = {"groq": GroqProvider(tmp_config.groq_api_key, tmp_config.llm.groq, client=client)}
        router = LLMRouter(tmp_config, test_db, providers)
        first = await router.complete([LLMMessage(role="user", content="Ping")])
        assert first.text == "Hi"

        with pytest.raises(QuotaExceededError):
            router.check_quota("groq")
    finally:
        await client.aclose()


def test_log_redaction_strips_credentials(tmp_path: Path) -> None:
    raw = "Authorization: Bearer gsk_supersecretkey123456 x-goog-api-key=AIzaSyAbCdEfGhIjKlMnOpQrStUvWx"
    redacted = redact_text(raw)
    assert "gsk_supersecretkey123456" not in redacted
    assert "AIzaSyAbCdEfGhIjKlMnOpQrStUvWx" not in redacted

    logger = setup_logging(tmp_path / "logs", level="INFO")
    logger.info("Leaky key test %s", "gsk_anotherSecretValue12345")
    for h in logger.handlers:
        h.flush()
    log_text = (tmp_path / "logs" / "zia.log").read_text(encoding="utf-8")
    assert "gsk_anotherSecretValue12345" not in log_text
