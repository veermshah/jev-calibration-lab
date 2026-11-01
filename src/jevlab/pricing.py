"""Per-model pricing, loaded from an editable TOML file (configs/pricing.toml)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    source: str = ""

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


class PriceTable:
    def __init__(self, prices: dict[str, Price]):
        self.prices = prices

    @classmethod
    def load(cls, path: Path) -> PriceTable:
        with path.open("rb") as f:
            raw = tomllib.load(f)
        prices = {
            name: Price(
                float(v.get("input_per_mtok", 0.0)),
                float(v.get("output_per_mtok", 0.0)),
                str(v.get("source", "")),
            )
            for name, v in raw.get("models", {}).items()
        }
        return cls(prices)

    def get(self, model: str) -> Price:
        if model not in self.prices:
            raise KeyError(f"no price for model {model!r}; add it to the pricing file")
        return self.prices[model]

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        return self.get(model).cost(input_tokens, output_tokens)
