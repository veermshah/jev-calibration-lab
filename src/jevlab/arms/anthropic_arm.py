"""Claude arm: stated probabilities through structured output (output_config json_schema).

Thinking is left at the model default; `effort` is set low because this is a
classification route where latency and cost matter. Thinking tokens are billed as
output tokens and show up in `usage.output_tokens`, so cost accounting includes them.
"""

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


def build_claude_request(
    model: str, decision: Decision, effort: str = "low", max_tokens: int = 2048
) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": render_user_prompt(decision)}],
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": probability_schema(decision)},
        },
    }


def parse_claude_response(
    decision: Decision, response: dict[str, Any]
) -> tuple[dict[str, float], int, int]:
    usage = response.get("usage") or {}
    tin = int(usage.get("input_tokens") or 0)
    tout = int(usage.get("output_tokens") or 0)
    stop = response.get("stop_reason")
    if stop == "refusal":
        raise ParseError("model refused")
    if stop == "max_tokens":
        raise ParseError("hit max_tokens before finishing the JSON")
    text = next(
        (b.get("text", "") for b in response.get("content", []) if b.get("type") == "text"), None
    )
    if not text:
        raise ParseError("no text block in response")
    return parse_probability_json(decision, text), tin, tout


class ClaudeModel(RemoteDecisionModel):
    def __init__(self, *args: Any, effort: str = "low", max_tokens: int = 2048, **kwargs: Any):
        super().__init__(*args, est_output_tokens=max_tokens, **kwargs)
        self.effort = effort
        self.max_tokens = max_tokens

    def build_request(self, decision: Decision) -> dict[str, Any]:
        return build_claude_request(self.model_id, decision, self.effort, self.max_tokens)

    def parse_response(self, decision, response):  # type: ignore[no-untyped-def]
        return parse_claude_response(decision, response)


def sdk_transport(api_key: str | None = None, max_retries: int = 4, timeout_s: float = 60.0) -> Transport:
    """Real transport through the official SDK. The SDK retries 408/409/429/5xx itself."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries, timeout=timeout_s)

    def send(request: dict[str, Any], decision: Decision, repeat: int) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        message = client.messages.create(**request)
        latency = (time.perf_counter() - t0) * 1000.0
        data: dict[str, Any] = message.model_dump(mode="json")
        return data, latency

    return send
