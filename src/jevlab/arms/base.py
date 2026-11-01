"""The `DecisionModel` interface every arm implements, plus the shared remote-call flow."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..budget import Budget
from ..cache import ResponseCache, request_key
from ..pricing import PriceTable
from ..retry import with_retries
from ..schema import Decision, Prediction, uniform

# A transport sends one provider request and returns (response JSON, latency in ms).
# Real transports ignore `decision` and `repeat`; synthetic ones use them to fake a reply.
Transport = Callable[[dict[str, Any], Decision, int], tuple[dict[str, Any], float]]


class DecisionModel(ABC):
    """An arm: given a typed decision, return per-choice probabilities, latency and cost."""

    name: str
    model_id: str
    synthetic: bool = False
    concurrency: int = 1

    def prepare(self, dataset: str) -> None:  # noqa: B027 - optional hook
        """Called once per dataset before predictions (e.g. to train a local model)."""

    @abstractmethod
    def predict(self, decision: Decision, repeat: int = 0) -> Prediction: ...


class ParseError(ValueError):
    pass


class RemoteDecisionModel(DecisionModel):
    """Shared flow for API-backed arms: cache -> budget check -> call with retries -> parse."""

    def __init__(
        self,
        name: str,
        model_id: str,
        transport: Transport,
        prices: PriceTable,
        cache: ResponseCache,
        budget: Budget,
        *,
        synthetic: bool = False,
        concurrency: int = 4,
        est_output_tokens: int = 1024,
        retries: int = 4,
    ):
        self.name = name
        self.model_id = model_id
        self.transport = transport
        self.prices = prices
        self.cache = cache
        self.budget = budget
        self.synthetic = synthetic
        self.concurrency = concurrency
        self.est_output_tokens = est_output_tokens
        self.retries = retries

    @abstractmethod
    def build_request(self, decision: Decision) -> dict[str, Any]: ...

    @abstractmethod
    def parse_response(
        self, decision: Decision, response: dict[str, Any]
    ) -> tuple[dict[str, float], int, int]:
        """Return (probabilities by choice id, input tokens, output tokens)."""

    def estimate_input_tokens(self, request: dict[str, Any]) -> int:
        # Deliberately pessimistic: ~3 characters per token over the whole request.
        import json

        return len(json.dumps(request)) // 3 + 50

    def predict(self, decision: Decision, repeat: int = 0) -> Prediction:
        request = self.build_request(decision)
        key = request_key(
            self.model_id, {"request": request, "repeat": repeat, "synthetic": self.synthetic}
        )
        hit = self.cache.get(self.model_id, key)
        if hit is not None:
            return self._from_response(decision, hit["response"], hit["latency_ms"], cached=True)

        estimate = self.prices.cost(
            self.model_id, self.estimate_input_tokens(request), self.est_output_tokens
        )
        with self.budget.charge(estimate) as actual:
            t0 = time.perf_counter()
            response, latency_ms = with_retries(
                lambda: self.transport(request, decision, repeat), attempts=self.retries
            )
            if latency_ms <= 0:
                latency_ms = (time.perf_counter() - t0) * 1000.0
            pred = self._from_response(decision, response, latency_ms, cached=False)
            # Under --dry-run this is the projected cost of the same call made for real.
            actual.append(pred.cost_usd)
        self.cache.put(self.model_id, key, {"response": response, "latency_ms": latency_ms})
        return pred

    def _from_response(
        self, decision: Decision, response: dict[str, Any], latency_ms: float, cached: bool
    ) -> Prediction:
        try:
            probs, tin, tout = self.parse_response(decision, response)
            error = None
        except (ParseError, KeyError, TypeError, ValueError) as exc:
            probs, tin, tout = uniform(decision.choice_ids), 0, 0
            usage = response.get("usage") or {}
            tin = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            tout = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            error = f"{type(exc).__name__}: {exc}"
        return Prediction(
            probs=probs,
            latency_ms=float(latency_ms),
            cost_usd=self.prices.cost(self.model_id, tin, tout),
            input_tokens=tin,
            output_tokens=tout,
            cached=cached,
            synthetic=self.synthetic,
            error=error,
        )


# --------------------------------------------------------------------------- prompts shared by LLM arms

SYSTEM_PROMPT = (
    "You are a calibrated decision model. You receive a state, a question about it and a "
    "closed set of choices. Return a probability for every choice: your honest estimate of "
    "how likely it is that careful human annotators chose that label. Probabilities must be "
    "between 0 and 1 and sum to 1. Use the full range: be confident only when the evidence "
    "is clear. Reply with the JSON object only."
)


def render_user_prompt(decision: Decision) -> str:
    choices = "\n".join(f"- {cid}: {desc}" for cid, desc in decision.choices.items())
    return (
        f"<state>\n{decision.state}\n</state>\n\n"
        f"Question: {decision.question}\n\n"
        f"Choices (use these ids as keys):\n{choices}"
    )


def probability_schema(decision: Decision) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "probabilities": {
                "type": "object",
                "properties": {cid: {"type": "number"} for cid in decision.choice_ids},
                "required": decision.choice_ids,
                "additionalProperties": False,
            }
        },
        "required": ["probabilities"],
        "additionalProperties": False,
    }


def parse_probability_json(decision: Decision, text: str) -> dict[str, float]:
    import json

    data = json.loads(text)
    probs = data.get("probabilities", data) if isinstance(data, dict) else None
    if not isinstance(probs, dict):
        raise ParseError("no probabilities object in model output")
    out = {cid: float(probs.get(cid, 0.0)) for cid in decision.choice_ids}
    if sum(max(0.0, v) for v in out.values()) <= 0:
        raise ParseError("all probabilities are zero")
    return out
