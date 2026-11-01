"""Generic arm for any OpenAI-compatible chat-completions endpoint (configured in arms.toml)."""

from __future__ import annotations

import time
from typing import Any

from ..schema import Decision
from .base import (
    SYSTEM_PROMPT,
    ParseError,
    RemoteDecisionModel,
    Transport,
    parse_probability_json,
    probability_schema,
    render_user_prompt,
)


def build_chat_request(model: str, decision: Decision, max_tokens: int = 512) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(decision)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "decision",
                "strict": True,
                "schema": probability_schema(decision),
            },
        },
    }


def parse_chat_response(
    decision: Decision, response: dict[str, Any]
) -> tuple[dict[str, float], int, int]:
    usage = response.get("usage") or {}
    tin = int(usage.get("prompt_tokens") or 0)
    tout = int(usage.get("completion_tokens") or 0)
    choices = response.get("choices") or []
    if not choices:
        raise ParseError("no choices in response")
    content = (choices[0].get("message") or {}).get("content")
    if not content:
        raise ParseError("empty message content")
    return parse_probability_json(decision, content), tin, tout


class ChatCompletionsModel(RemoteDecisionModel):
    def __init__(self, *args: Any, max_tokens: int = 512, **kwargs: Any):
        super().__init__(*args, est_output_tokens=max_tokens, **kwargs)
        self.max_tokens = max_tokens

    def build_request(self, decision: Decision) -> dict[str, Any]:
        return build_chat_request(self.model_id, decision, self.max_tokens)

    def parse_response(
        self, decision: Decision, response: dict[str, Any]
    ) -> tuple[dict[str, float], int, int]:
        return parse_chat_response(decision, response)


def http_transport(base_url: str, api_key: str | None, timeout_s: float = 60.0) -> Transport:
    import httpx

    from ..retry import RetryableError

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    client = httpx.Client(timeout=timeout_s, headers=headers)
    url = base_url.rstrip("/") + "/chat/completions"

    def send(
        request: dict[str, Any], decision: Decision, repeat: int
    ) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        try:
            resp = client.post(url, json=request)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableError(str(exc)) from exc
        latency = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        data: dict[str, Any] = resp.json()
        return data, latency

    return send
