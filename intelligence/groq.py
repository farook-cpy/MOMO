"""Groq OpenAI-compatible async HTTPX provider."""

from __future__ import annotations

import inspect
import json
from typing import Any, Awaitable, Callable

import httpx

from core.config import ProviderEndpointConfig
from intelligence.provider import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    ProviderError,
    RateLimitError,
    ToolCall,
)


class GroqProvider(BaseLLMProvider):
    name = "groq"

    def __init__(
        self,
        api_key: str,
        endpoint: ProviderEndpointConfig,
        timeout_seconds: float = 25.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self._external_client = client

    def is_configured(self) -> bool:
        return bool(self.api_key and self.endpoint.model)

    def _serialize_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "call_0",
                        "content": msg.content,
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
            else:
                out.append({"role": msg.role, "content": msg.content})
        return out

    async def _get_client(self) -> tuple[httpx.AsyncClient, bool]:
        if self._external_client is not None:
            return self._external_client, False
        return httpx.AsyncClient(timeout=self.timeout_seconds), True

    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 512,
        stream: bool = False,
        on_token: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> LLMResponse:
        if not self.is_configured():
            raise ProviderError("Groq API key is missing.", retryable=False)

        url = f"{self.endpoint.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": self._serialize_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
            payload["tool_choice"] = "auto"

        client, should_close = await self._get_client()
        try:
            if stream:
                return await self._generate_stream(client, url, headers, payload, on_token)
            return await self._generate_once(client, url, headers, payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Groq request timed out: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Groq network error: {exc}", retryable=True) from exc
        finally:
            if should_close:
                await client.aclose()

    def _raise_for_status(self, resp: httpx.Response, body_text: str) -> None:
        if resp.status_code == 429:
            retry_after_hdr = resp.headers.get("retry-after")
            retry_after = float(retry_after_hdr) if retry_after_hdr and retry_after_hdr.isdigit() else None
            raise RateLimitError(f"Groq rate limit (429): {body_text[:200]}", retry_after=retry_after)
        if resp.status_code >= 500:
            raise ProviderError(
                f"Groq server error ({resp.status_code}): {body_text[:200]}",
                retryable=True,
                status_code=resp.status_code,
            )
        if resp.status_code >= 400:
            raise ProviderError(
                f"Groq request error ({resp.status_code}): {body_text[:200]}",
                retryable=False,
                status_code=resp.status_code,
            )

    async def _generate_once(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> LLMResponse:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            self._raise_for_status(resp, resp.text)

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("Groq returned empty choices.", retryable=True)

        choice = choices[0]
        msg = choice.get("message") or {}
        tool_calls: list[ToolCall] = []
        for raw_tc in msg.get("tool_calls") or []:
            fn = raw_tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except ValueError:
                parsed_args = {}
            tool_calls.append(
                ToolCall(
                    id=str(raw_tc.get("id") or "call_0"),
                    name=str(fn.get("name") or ""),
                    arguments=parsed_args,
                )
            )

        usage = data.get("usage") or {}
        return LLMResponse(
            text=str(msg.get("content") or ""),
            provider=self.name,
            model=str(data.get("model") or self.endpoint.model),
            tool_calls=tool_calls,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )

    async def _generate_stream(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        on_token: Callable[[str], Awaitable[None] | None] | None,
    ) -> LLMResponse:
        chunks: list[str] = []
        usage: dict[str, Any] = {}
        finish_reason = "stop"

        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                self._raise_for_status(resp, body)

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except ValueError:
                    continue
                if "usage" in event and event["usage"]:
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if choices[0].get("finish_reason"):
                    finish_reason = str(choices[0]["finish_reason"])
                token = delta.get("content")
                if token:
                    chunks.append(token)
                    if on_token is not None:
                        res = on_token(token)
                        if inspect.isawaitable(res):
                            await res

        return LLMResponse(
            text="".join(chunks),
            provider=self.name,
            model=self.endpoint.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            finish_reason=finish_reason,
        )
