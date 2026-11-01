"""TypeSafe Jev (System One API) arm.

Request and response shapes follow TypeSafe's public API reference
(https://docs.typesafe.ai/api): POST https://api.typesafe.ai/v1/systemone with
`Authorization: Bearer <key>`, a `state`, and named `questions`. Every decision is
sent as one `choice` question whose `criteria` maps choice names to descriptions;
the answer's `probabilities` object is Jev's native per-choice distribution.
"""

from __future__ import annotations

import time
from typing import Any

from ..schema import Decision
from .base import ParseError, RemoteDecisionModel, Transport

QUESTION_KEY = "answer"


def build_jev_request(model: str, decision: Decision) -> dict[str, Any]:
    return {
        "model": model,
        "state": decision.state,
        "questions": {
            QUESTION_KEY: {
                "type": "choice",
                "instructions": decision.question,
                "criteria": dict(decision.choices),
            }
        },
    }


def parse_jev_response(
    decision: Decision, response: dict[str, Any]
) -> tuple[dict[str, float], int, int]:
    answer = (response.get("answers") or {}).get(QUESTION_KEY)
    if not isinstance(answer, dict):
        raise ParseError("response has no answer for the question")
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and probs:
        out = {cid: float(probs.get(cid, 0.0)) for cid in decision.choice_ids}
    elif "choice" in answer and answer["choice"] in decision.choices:
        # Fallback: only the top choice and its confidence; spread the rest evenly.
        conf = float(answer.get("confidence") or 0.0)
        rest = (1.0 - conf) / (len(decision.choices) - 1)
        out = {cid: (conf if cid == answer["choice"] else rest) for cid in decision.choice_ids}
    else:
        raise ParseError("answer has neither probabilities nor a valid choice")
    usage = response.get("usage") or {}
    return out, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)


class JevModel(RemoteDecisionModel):
    def build_request(self, decision: Decision) -> dict[str, Any]:
        return build_jev_request(self.model_id, decision)

    def parse_response(
        self, decision: Decision, response: dict[str, Any]
    ) -> tuple[dict[str, float], int, int]:
        return parse_jev_response(decision, response)


def http_transport(base_url: str, api_key: str, timeout_s: float = 30.0) -> Transport:
    import httpx

    from ..retry import RetryableError

    client = httpx.Client(
        timeout=timeout_s,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )

    def send(
        request: dict[str, Any], decision: Decision, repeat: int
    ) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        try:
            resp = client.post(base_url, json=request)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableError(str(exc)) from exc
        latency = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise RuntimeError(f"TypeSafe API HTTP {resp.status_code}: {resp.text[:500]}")
        data: dict[str, Any] = resp.json()
        return data, latency

    return send
