"""Base LLM provider abstractions, retry logic, quota guard, and failover router."""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from core.config import ZiaConfig
from memory.database import ZiaDatabase

logger = logging.getLogger("zia.llm")


class ProviderError(RuntimeError):
    """Base error for LLM provider failures."""

    def __init__(self, message: str, retryable: bool = False, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class RateLimitError(ProviderError):
    """Raised on HTTP 429 or upstream rate limit."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message, retryable=True, status_code=429)
        self.retry_after = retry_after


class QuotaExceededError(ProviderError):
    """Raised when local daily safety cap is reached."""


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class LLMMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"


class BaseLLMProvider(ABC):
    name: str

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if credentials and model are present."""

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 512,
        stream: bool = False,
        on_token: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> LLMResponse:
        """Execute a completion request."""


class LLMRouter:
    """Routes LLM requests across primary/fallback providers with backoff & daily caps."""

    def __init__(
        self,
        config: ZiaConfig,
        db: ZiaDatabase,
        providers: dict[str, BaseLLMProvider],
    ) -> None:
        self.config = config
        self.db = db
        self.providers = providers

    def _limits_for(self, name: str) -> tuple[int, int]:
        if name == "groq":
            return (
                self.config.llm.groq.daily_request_limit,
                self.config.llm.groq.daily_token_limit,
            )
        if name == "gemini":
            return (
                self.config.llm.gemini.daily_request_limit,
                self.config.llm.gemini.daily_token_limit,
            )
        return (1000, 500_000)

    def _today_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def check_quota(self, provider_name: str) -> None:
        req_limit, tok_limit = self._limits_for(provider_name)
        usage = self.db.get_daily_usage(provider_name, self._today_key())
        if usage["requests"] >= req_limit:
            raise QuotaExceededError(
                f"Daily request cap reached for {provider_name} ({usage['requests']}/{req_limit})."
            )
        if usage["tokens"] >= tok_limit:
            raise QuotaExceededError(
                f"Daily token cap reached for {provider_name} ({usage['tokens']}/{tok_limit})."
            )

    def provider_order(self, preferred: str | None = None) -> list[str]:
        order: list[str] = []
        for name in (preferred, self.config.llm.primary_provider, self.config.llm.fallback_provider):
            if name and name in self.providers and name not in order:
                order.append(name)
        for name in self.providers:
            if name not in order:
                order.append(name)
        return order

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        preferred_provider: str | None = None,
        on_token: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> LLMResponse:
        if self.config.offline_mode:
            raise ProviderError("Assistant is configured in offline mode.", retryable=False)

        errors: list[str] = []
        for name in self.provider_order(preferred_provider):
            provider = self.providers[name]
            if not provider.is_configured():
                errors.append(f"{name}: API key not configured")
                continue

            try:
                self.check_quota(name)
            except QuotaExceededError as exc:
                logger.warning(str(exc))
                errors.append(str(exc))
                continue

            try:
                response = await self._call_with_retry(provider, messages, tools, on_token)
                total = response.total_tokens or (response.prompt_tokens + response.completion_tokens)
                self.db.record_daily_usage(name, self._today_key(), total)
                return response
            except ProviderError as exc:
                logger.warning("Provider %s failed: %s", name, exc)
                errors.append(f"{name}: {exc}")

        raise ProviderError("All LLM providers failed (" + "; ".join(errors) + ")", retryable=True)

    async def _call_with_retry(
        self,
        provider: BaseLLMProvider,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None,
        on_token: Callable[[str], Awaitable[None] | None] | None,
    ) -> LLMResponse:
        max_retries = max(1, self.config.llm.max_retries)
        base_delay = max(0.1, self.config.llm.retry_base_delay_seconds)
        last_exc: ProviderError | None = None

        for attempt in range(max_retries):
            try:
                return await provider.generate(
                    messages=messages,
                    tools=tools,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    stream=self.config.llm.stream_responses and (tools is None or len(tools) == 0),
                    on_token=on_token,
                )
            except ProviderError as exc:
                last_exc = exc
                if not exc.retryable or attempt + 1 >= max_retries:
                    raise
                delay = base_delay * (2**attempt) + random.uniform(0.0, 0.35)
                if isinstance(exc, RateLimitError) and exc.retry_after is not None:
                    delay = max(delay, min(exc.retry_after, 10.0))
                logger.info(
                    "Retrying %s (attempt %d/%d) after %.2fs: %s",
                    provider.name,
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc
