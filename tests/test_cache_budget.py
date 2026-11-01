import pytest

from jevlab.arms.jev import JevModel
from jevlab.budget import Budget, BudgetExceeded
from jevlab.cache import ResponseCache, request_key
from jevlab.retry import RetryableError, with_retries

from .conftest import load_fixture


def counting_transport():
    calls = {"n": 0}

    def transport(request, decision, repeat):
        calls["n"] += 1
        return load_fixture("jev_response.json"), 50.0

    return transport, calls


def test_request_key_is_order_independent():
    assert request_key("m", {"a": 1, "b": 2}) == request_key("m", {"b": 2, "a": 1})
    assert request_key("m", {"a": 1}) != request_key("other", {"a": 1})


def test_cache_miss_then_hit(decision, prices, cache, budget):
    transport, calls = counting_transport()
    arm = JevModel("jev", "jev-latest", transport, prices, cache, budget)
    first = arm.predict(decision)
    second = arm.predict(decision)
    assert calls["n"] == 1
    assert (cache.misses, cache.hits) == (1, 1)
    assert not first.cached and second.cached
    assert second.probs == first.probs and second.latency_ms == first.latency_ms
    assert budget.calls == 1  # the cached call was free


def test_cache_persists_across_instances(decision, prices, tmp_path, budget):
    transport, calls = counting_transport()
    JevModel("jev", "jev-latest", transport, prices, ResponseCache(tmp_path), budget).predict(
        decision
    )
    JevModel("jev", "jev-latest", transport, prices, ResponseCache(tmp_path), budget).predict(
        decision
    )
    assert calls["n"] == 1


def test_repeats_and_synthetic_get_separate_cache_entries(decision, prices, cache, budget):
    transport, calls = counting_transport()
    arm = JevModel("jev", "jev-latest", transport, prices, cache, budget)
    arm.predict(decision, repeat=0)
    arm.predict(decision, repeat=1)
    JevModel("jev", "jev-latest", transport, prices, cache, budget, synthetic=True).predict(
        decision
    )
    assert calls["n"] == 3


def test_budget_cap_stops_calls(decision, prices, cache):
    transport, calls = counting_transport()
    tiny = Budget(0.0)
    arm = JevModel("jev", "jev-latest", transport, prices, cache, tiny)
    with pytest.raises(BudgetExceeded):
        arm.predict(decision)
    assert calls["n"] == 0
    assert tiny.refused == 1


def test_budget_allows_until_cap_then_refuses():
    b = Budget(1.0)
    with b.charge(0.4) as actual:
        actual.append(0.5)
    with b.charge(0.4) as actual:
        actual.append(0.4)
    assert b.spent_usd == pytest.approx(0.9)
    with pytest.raises(BudgetExceeded), b.charge(0.2):
        pass
    assert b.spent_usd == pytest.approx(0.9)


def test_failed_call_is_charged_its_estimate():
    b = Budget(1.0)
    with pytest.raises(RuntimeError), b.charge(0.3):
        raise RuntimeError("boom")
    assert b.spent_usd == pytest.approx(0.3) and b.reserved_usd == pytest.approx(0)


def test_cache_hit_needs_no_budget(decision, prices, cache):
    transport, calls = counting_transport()
    JevModel("jev", "jev-latest", transport, prices, cache, Budget(1.0)).predict(decision)
    pred = JevModel("jev", "jev-latest", transport, prices, cache, Budget(0.0)).predict(decision)
    assert pred.cached and calls["n"] == 1


def test_retries_with_backoff():
    attempts = {"n": 0}
    sleeps = []

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RetryableError("503")
        return "ok"

    assert with_retries(flaky, attempts=4, base_delay=1.0, sleep=sleeps.append) == "ok"
    assert attempts["n"] == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0] * 0.9


def test_retries_give_up():
    def always():
        raise RetryableError("429")

    with pytest.raises(RetryableError):
        with_retries(always, attempts=2, sleep=lambda s: None)
