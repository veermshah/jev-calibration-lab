import json
from pathlib import Path

import pytest

from jevlab.budget import Budget
from jevlab.cache import ResponseCache
from jevlab.pricing import PriceTable
from jevlab.schema import Decision

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if any test tries to open a socket."""
    import socket

    def guard(*args, **kwargs):
        raise RuntimeError("network access is not allowed in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)


@pytest.fixture
def prices():
    return PriceTable.load(ROOT / "configs" / "pricing.toml")


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "cache")


@pytest.fixture
def budget():
    return Budget(15.0)


@pytest.fixture
def decision():
    return Decision(
        id="sst2-1",
        dataset="sst2",
        state="a charming and often affecting journey .",
        question="What is the sentiment of this movie-review excerpt?",
        choices={"negative": "Negative.", "positive": "Positive."},
        gold="positive",
    )


def load_fixture(name):
    return json.loads((FIX / name).read_text())
