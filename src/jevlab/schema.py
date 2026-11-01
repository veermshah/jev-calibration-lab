"""Core data types shared by datasets, arms, the runner and the metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Decision:
    """One typed decision: a state, a question about it, a closed set of choices and a gold label.

    `choices` maps a short choice id (used as the key in every model's output) to a
    human-readable description. Order is preserved and is the order used for
    probability vectors everywhere in the code base.
    """

    id: str
    dataset: str
    state: str
    question: str
    choices: dict[str, str]
    gold: str

    def __post_init__(self) -> None:
        if self.gold not in self.choices:
            raise ValueError(f"gold label {self.gold!r} not in choices for item {self.id}")
        if len(self.choices) < 2:
            raise ValueError(f"item {self.id} needs at least two choices")

    @property
    def choice_ids(self) -> list[str]:
        return list(self.choices)

    @property
    def gold_index(self) -> int:
        return self.choice_ids.index(self.gold)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Decision:
        return cls(
            id=str(d["id"]),
            dataset=str(d["dataset"]),
            state=str(d["state"]),
            question=str(d["question"]),
            choices=dict(d["choices"]),
            gold=str(d["gold"]),
        )


@dataclass
class Prediction:
    """What an arm returns for one decision."""

    probs: dict[str, float]
    latency_ms: float
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    synthetic: bool = False
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def vector(self, choice_ids: list[str]) -> list[float]:
        """Probability vector aligned with `choice_ids`, renormalised to sum to one."""
        return normalise([float(self.probs.get(c, 0.0)) for c in choice_ids])


def normalise(p: list[float]) -> list[float]:
    """Clip negatives and renormalise. Falls back to uniform if everything is zero."""
    clipped = [max(0.0, float(x)) if x == x else 0.0 for x in p]  # x == x filters NaN
    total = sum(clipped)
    if total <= 0:
        return [1.0 / len(p)] * len(p)
    return [x / total for x in clipped]


def uniform(choice_ids: list[str]) -> dict[str, float]:
    return {c: 1.0 / len(choice_ids) for c in choice_ids}
