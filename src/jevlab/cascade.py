"""Cascade analysis: ask a cheap, fast model first; escalate to a stronger one when unsure.

For each threshold tau the first arm's answer is kept when its top probability is at
least tau; otherwise the item is escalated and the second arm's answer is used. The
escalated item pays for both calls and waits for both (sequential cascade).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import metrics as M


def sweep(
    probs_first: ArrayLike,
    probs_second: ArrayLike,
    y: ArrayLike,
    cost_first: ArrayLike,
    cost_second: ArrayLike,
    latency_first: ArrayLike,
    latency_second: ArrayLike,
    taus: ArrayLike | None = None,
) -> list[dict[str, Any]]:
    p1 = np.asarray(probs_first, dtype=float)
    p2 = np.asarray(probs_second, dtype=float)
    yy = np.asarray(y, dtype=int)
    c1, c2 = np.asarray(cost_first, dtype=float), np.asarray(cost_second, dtype=float)
    l1, l2 = np.asarray(latency_first, dtype=float), np.asarray(latency_second, dtype=float)
    grid = np.linspace(0.0, 1.0, 21) if taus is None else np.asarray(taus, dtype=float)
    # tau slightly above 1 means "always escalate"
    grid = np.unique(np.concatenate([grid, [1.0 + 1e-9]]))
    conf1 = p1.max(axis=1)
    out = []
    for tau in grid:
        esc = conf1 < tau
        final = np.where(esc[:, None], p2, p1)
        out.append(
            {
                "tau": float(min(tau, 1.0)) if tau <= 1.0 else 1.0,
                "always_escalate": bool(tau > 1.0),
                "escalation_rate": float(esc.mean()),
                "accuracy": M.accuracy(final, yy),
                "ece": M.ece(final, yy),
                "cost_per_1k": float((c1 + esc * c2).mean() * 1000.0),
                "mean_latency_ms": float((l1 + esc * l2).mean()),
            }
        )
    return out
