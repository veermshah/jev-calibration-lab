"""Retries with exponential backoff and jitter for transient HTTP failures."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RetryableError(RuntimeError):
    """Raised by transports for failures worth retrying (429, 5xx, timeouts)."""


def with_retries(
    fn: Callable[[], T],
    attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (RetryableError,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on:
            if attempt == attempts:
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            sleep(delay * (0.5 + random.random() / 2))
    raise AssertionError("unreachable")
