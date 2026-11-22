"""Google Gemini REST API async HTTPX provider (header-based API key)."""

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


class GeminiProvider(BaseLLMProvider):
    name = "gemini"

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

    def _serialize_payload(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    system_parts.append(msg.content)
            elif msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif msg.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.name or "tool",
                                    "response": {"result": msg.content},
                                }
                            }
                        ],
                    }
                )

        payload: dict[str, Any] = {
            "contents": contents or [{"role": "user", "parts": [{"text": "Hello"}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": tools}]
        return payload

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
            raise ProviderError("Gemini API key is missing.", retryable=False)

        action = "streamGenerateContent?alt=sse" if stream else "generateContent"
        url = f"{self.endpoint.base_url.rstrip('/')}/models/{self.endpoint.model}:{action}"
        # Keep API key in header rather than URL query string so it never appears in URLs/logs
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = self._serialize_payload(messages, tools, temperature, max_tokens)

        client, should_close = await self._get_client()
        try:
            if stream:
                return await self._generate_stream(client, url, headers, payload, on_token)
            return await self._generate_once(client, url, headers, payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Gemini request timed out: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini network error: {exc}", retryable=True) from exc
        finally:
            if should_close:
                await client.aclose()

    def _raise_for_status(self, resp: httpx.Response, body_text: str) -> None:
        if resp.status_code == 429:
            raise RateLimitError(f"Gemini rate limit (429): {body_text[:200]}")
        if resp.status_code >= 500:
            raise ProviderError(
                f"Gemini server error ({resp.status_code}): {body_text[:200]}",
                retryable=True,
                status_code=resp.status_code,
            )
        if resp.status_code >= 400:
            raise ProviderError(
                f"Gemini request error ({resp.status_code}): {body_text[:200]}",
                retryable=False,
                status_code=resp.status_code,
            )

    @staticmethod
    def _parse_parts(parts: list[dict[str, Any]]) -> tuple[str, list[ToolCall]]:
        texts: list[str] = []
        calls: list[ToolCall] = []
        for idx, part in enumerate(parts):
            if "text" in part and part["text"]:
                texts.append(str(part["text"]))
            elif "functionCall" in part:
                fc = part["functionCall"] or {}
                calls.append(
                    ToolCall(
                        id=f"gemini_call_{idx}",
                        name=str(fc.get("name") or ""),
                        arguments=dict(fc.get("args") or {}),
                    )
                )
        return "".join(texts), calls

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
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError("Gemini returned no candidates.", retryable=True)

        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text, tool_calls = self._parse_parts(parts)
        usage = data.get("usageMetadata") or {}

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.endpoint.model,
            tool_calls=tool_calls,
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            completion_tokens=int(usage.get("candidatesTokenCount") or 0),
            total_tokens=int(usage.get("totalTokenCount") or 0),
            finish_reason=str(cand.get("finishReason") or "STOP"),
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
        finish_reason = "STOP"

        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                self._raise_for_status(resp, body)

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    event = json.loads(data_str)
                except ValueError:
                    continue
                if "usageMetadata" in event:
                    usage = event["usageMetadata"]
                cands = event.get("candidates") or []
                if not cands:
                    continue
                if cands[0].get("finishReason"):
                    finish_reason = str(cands[0]["finishReason"])
                parts = (cands[0].get("content") or {}).get("parts") or []
                text, _ = self._parse_parts(parts)
                if text:
                    chunks.append(text)
                    if on_token is not None:
                        res = on_token(text)
                        if inspect.isawaitable(res):
                            await res

        return LLMResponse(
            text="".join(chunks),
            provider=self.name,
            model=self.endpoint.model,
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            completion_tokens=int(usage.get("candidatesTokenCount") or 0),
            total_tokens=int(usage.get("totalTokenCount") or 0),
            finish_reason=finish_reason,
        )
