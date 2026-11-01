"""Hard spending cap, checked before every paid call."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Tracks spend against a cap. Each call reserves a worst-case estimate first.

    A call is refused if spent + outstanding reservations + its estimate would exceed
    the cap, so concurrent workers can never overshoot it together.
    """

    def __init__(self, cap_usd: float):
        self.cap_usd = float(cap_usd)
        self.spent_usd = 0.0
        self.reserved_usd = 0.0
        self.calls = 0
        self.refused = 0
        self._lock = threading.Lock()

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.spent_usd - self.reserved_usd

    def reserve(self, estimate_usd: float) -> None:
        with self._lock:
            if self.spent_usd + self.reserved_usd + estimate_usd > self.cap_usd + 1e-12:
                self.refused += 1
                raise BudgetExceeded(
                    f"budget cap ${self.cap_usd:.2f} reached "
                    f"(spent ${self.spent_usd:.4f}, next call est. ${estimate_usd:.4f})"
                )
            self.reserved_usd += estimate_usd

    def settle(self, estimate_usd: float, actual_usd: float) -> None:
        with self._lock:
            self.reserved_usd -= estimate_usd
            self.spent_usd += actual_usd
            self.calls += 1

    @contextmanager
    def charge(self, estimate_usd: float) -> Iterator[list[float]]:
        """Reserve `estimate_usd`; the body appends the actual cost to the yielded list."""
        self.reserve(estimate_usd)
        actual: list[float] = []
        try:
            yield actual
        finally:
            # A failed call is charged its estimate: we cannot know what was billed.
            self.settle(estimate_usd, actual[0] if actual else estimate_usd)
