"""Synthetic transports for `--dry-run`.

Nothing here measures a real model. Each profile is an invented generator that
produces plausible-looking replies in the exact wire format of the real provider,
so a dry run exercises the whole pipeline (request building, caching, the budget
cap, response parsing, metrics, report) without network access or API keys.

How a synthetic reply is made, per (profile, item, repeat):
1. Draw a latent "true" confidence c for the item from a Beta distribution whose
   mean is the dataset's base difficulty plus the profile's skill offset.
2. The prediction is correct with probability c, so the latent c is calibrated.
3. Repeats add noise to c (and can flip the label), giving run-to-run variance.
4. The stated confidence is a distortion of c: logit(c) * sharpness + bias, then
   optionally rounded the way LLMs tend to round stated probabilities.
5. The remaining mass is spread over the other choices.

Every prediction produced this way is flagged `synthetic=True` all the way to the report.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..schema import Decision
from .base import Transport

# Invented base accuracies, roughly in line with published numbers for each task.
DATASET_BASE = {"sst2": 0.91, "boolq": 0.85, "liar": 0.30, "banking77": 0.88, "sms_spam": 0.97}


@dataclass(frozen=True)
class SyntheticProfile:
    name: str
    skill: float  # added to the dataset base accuracy
    sharpness: float  # >1 means overconfident stated probabilities
    bias: float  # added to the logit of the stated confidence
    repeat_noise: float  # std of logit noise between repeats
    flip: float  # chance a repeat switches label, scaled by (1 - c)
    round_to: float | None  # LLM-style rounding of stated probabilities
    latency_ms: float  # median
    latency_sigma: float  # lognormal sigma
    output_tokens: int


PROFILES = {
    "jev": SyntheticProfile("jev", 0.00, 1.05, 0.0, 0.03, 0.0, None, 160, 0.35, 12),
    "haiku": SyntheticProfile("haiku", -0.03, 1.8, 0.35, 0.30, 0.15, 0.05, 750, 0.30, 140),
    "sonnet": SyntheticProfile("sonnet", 0.04, 1.4, 0.25, 0.18, 0.08, 0.05, 1700, 0.30, 180),
    "generic": SyntheticProfile("generic", -0.05, 1.5, 0.3, 0.25, 0.12, 0.1, 900, 0.40, 90),
}


def _rng(*parts: object) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def synthetic_distribution(
    decision: Decision, profile: SyntheticProfile, repeat: int = 0
) -> tuple[dict[str, float], float]:
    """Return (probabilities by choice id, latency_ms) for one synthetic reply."""
    ids = decision.choice_ids
    k = len(ids)
    floor = 1.0 / k
    item = _rng(profile.name, decision.id)
    base = DATASET_BASE.get(decision.dataset, 0.8) + profile.skill
    mean = min(max((base - floor) / (1 - floor), 0.02), 0.98)
    conc = 3.0
    c = floor + (1 - floor) * float(item.beta(mean * conc, (1 - mean) * conc))
    correct = bool(item.random() < c)
    others = [i for i in ids if i != decision.gold]
    wrong_pick = others[int(item.integers(len(others)))]
    label = decision.gold if correct else wrong_pick

    rep = _rng(profile.name, decision.id, "repeat", repeat)
    if repeat > 0:
        c = _sigmoid(_logit(c) + float(rep.normal(0, profile.repeat_noise)))
        if rep.random() < profile.flip * (1 - c):
            label = others[int(rep.integers(len(others)))] if label == decision.gold else decision.gold

    stated = _sigmoid(_logit(c) * profile.sharpness + profile.bias)
    if profile.round_to:
        stated = round(stated / profile.round_to) * profile.round_to
    stated = min(max(stated, floor + 1e-3), 0.999)

    rest = [i for i in ids if i != label]
    weights = rep.dirichlet(np.ones(len(rest)) * 0.7) if len(rest) > 1 else np.ones(1)
    if label != decision.gold:  # the gold class usually holds most of the leftover mass
        weights = weights * 0.4
        weights[rest.index(decision.gold)] += 0.6
    probs = {label: stated}
    for cid, w in zip(rest, weights, strict=True):
        probs[cid] = float((1 - stated) * w / weights.sum())
    latency = float(profile.latency_ms * math.exp(rep.normal(0, profile.latency_sigma)))
    return {cid: probs[cid] for cid in ids}, latency


def _prompt_tokens(request: dict[str, Any]) -> int:
    return max(1, len(json.dumps(request)) // 4)


def jev_transport(profile: SyntheticProfile) -> Transport:
    def send(request: dict[str, Any], decision: Decision, repeat: int) -> tuple[dict[str, Any], float]:
        probs, latency = synthetic_distribution(decision, profile, repeat)
        top = max(probs, key=lambda c: probs[c])
        return (
            {
                "model": f"{request['model']} (synthetic)",
                "answers": {
                    "answer": {
                        "type": "choice",
                        "choice": top,
                        "confidence": probs[top],
                        "probabilities": probs,
                    }
                },
                "usage": {"input_tokens": _prompt_tokens(request), "output_tokens": profile.output_tokens},
            },
            latency,
        )

    return send


def claude_transport(profile: SyntheticProfile) -> Transport:
    def send(request: dict[str, Any], decision: Decision, repeat: int) -> tuple[dict[str, Any], float]:
        probs, latency = synthetic_distribution(decision, profile, repeat)
        text = json.dumps({"probabilities": {k: round(v, 3) for k, v in probs.items()}})
        return (
            {
                "id": "msg_synthetic",
                "type": "message",
                "role": "assistant",
                "model": request["model"],
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": _prompt_tokens(request), "output_tokens": profile.output_tokens},
            },
            latency,
        )

    return send


def chat_transport(profile: SyntheticProfile) -> Transport:
    def send(request: dict[str, Any], decision: Decision, repeat: int) -> tuple[dict[str, Any], float]:
        probs, latency = synthetic_distribution(decision, profile, repeat)
        text = json.dumps({"probabilities": {k: round(v, 3) for k, v in probs.items()}})
        return (
            {
                "id": "chatcmpl-synthetic",
                "model": request["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": _prompt_tokens(request), "completion_tokens": profile.output_tokens},
            },
            latency,
        )

    return send
