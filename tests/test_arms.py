import json

import pytest

from jevlab.arms import ConfigError, build_arms, load_arm_config
from jevlab.arms.anthropic_arm import ClaudeModel, build_claude_request, parse_claude_response
from jevlab.arms.base import ParseError
from jevlab.arms.jev import JevModel, build_jev_request, parse_jev_response
from jevlab.arms.openai_compat import build_chat_request, parse_chat_response
from jevlab.arms.synthetic import PROFILES, synthetic_distribution

from .conftest import ROOT, load_fixture


def test_jev_request_matches_documented_shape(decision):
    req = build_jev_request("jev-latest", decision)
    assert req["model"] == "jev-latest"
    assert req["state"] == decision.state
    q = req["questions"]["answer"]
    assert q["type"] == "choice"
    assert q["instructions"] == decision.question
    assert q["criteria"] == decision.choices


def test_jev_parses_recorded_response(decision):
    probs, tin, tout = parse_jev_response(decision, load_fixture("jev_response.json"))
    assert probs == {"negative": 0.18, "positive": 0.82}
    assert (tin, tout) == (96, 9)


def test_jev_falls_back_to_choice_and_confidence(decision):
    resp = {"answers": {"answer": {"type": "choice", "choice": "negative", "confidence": 0.6}}}
    probs, _, _ = parse_jev_response(decision, resp)
    assert probs == pytest.approx({"negative": 0.6, "positive": 0.4})


def test_jev_missing_answer_raises(decision):
    with pytest.raises(ParseError):
        parse_jev_response(decision, {"answers": {}})


def test_claude_request_uses_structured_output(decision):
    req = build_claude_request("claude-haiku-5-5", decision)
    schema = req["output_config"]["format"]["schema"]
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert schema["properties"]["probabilities"]["required"] == ["negative", "positive"]
    assert schema["additionalProperties"] is False
    assert "temperature" not in req and "thinking" not in req
    assert "a charming" in req["messages"][0]["content"]


def test_claude_parses_recorded_response(decision):
    probs, tin, tout = parse_claude_response(decision, load_fixture("claude_response.json"))
    assert probs == {"negative": 0.3, "positive": 0.7}
    assert (tin, tout) == (210, 48)


def test_claude_refusal_is_an_error(decision):
    resp = load_fixture("claude_response.json") | {"stop_reason": "refusal"}
    with pytest.raises(ParseError):
        parse_claude_response(decision, resp)


def test_chat_roundtrip(decision):
    req = build_chat_request("m", decision)
    assert req["response_format"]["json_schema"]["strict"] is True
    resp = {
        "choices": [{"message": {"content": json.dumps({"probabilities": {"negative": 1, "positive": 3}})}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    probs, tin, tout = parse_chat_response(decision, resp)
    assert probs == {"negative": 1.0, "positive": 3.0}  # renormalised later by Prediction.vector
    assert (tin, tout) == (5, 2)


def _recorded(name):
    calls = []

    def transport(request, decision, repeat):
        calls.append(request)
        return load_fixture(name), 123.0

    return transport, calls


def test_remote_arm_end_to_end_with_recorded_transport(decision, prices, cache, budget):
    transport, calls = _recorded("jev_response.json")
    arm = JevModel("jev", "jev-latest", transport, prices, cache, budget)
    pred = arm.predict(decision)
    assert pred.vector(decision.choice_ids) == pytest.approx([0.18, 0.82])
    assert pred.latency_ms == 123.0 and not pred.cached
    assert pred.cost_usd == pytest.approx(96 * 0.042 / 1e6)
    assert budget.spent_usd == pytest.approx(pred.cost_usd)


def test_unparseable_reply_becomes_uniform_with_error(decision, prices, cache, budget):
    def transport(request, d, repeat):
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "nope"}], "usage": {}}, 5.0

    arm = ClaudeModel("haiku", "claude-haiku-5-5", transport, prices, cache, budget)
    pred = arm.predict(decision)
    assert pred.error and pred.probs == {"negative": 0.5, "positive": 0.5}


def test_synthetic_is_deterministic_and_valid(decision):
    for profile in PROFILES.values():
        a, la = synthetic_distribution(decision, profile, 0)
        b, lb = synthetic_distribution(decision, profile, 0)
        assert a == b and la == lb
        assert sum(a.values()) == pytest.approx(1.0)
        assert all(0 <= v <= 1 for v in a.values())


def test_dry_run_builds_every_configured_arm(prices, cache, budget, tmp_path):
    cfg = load_arm_config(ROOT / "configs" / "arms.toml")
    arms = build_arms(
        list(cfg), cfg, dry_run=True, prices=prices, cache=cache, budget=budget, data_dir=tmp_path
    )
    assert {a.name for a in arms} == set(cfg)
    assert all(a.synthetic for a in arms if a.name != "logreg")


def test_real_run_without_key_is_a_clear_error(prices, cache, budget, tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    cfg = load_arm_config(ROOT / "configs" / "arms.toml")
    with pytest.raises(ConfigError, match="TYPESAFE_API_KEY"):
        build_arms(["jev"], cfg, dry_run=False, prices=prices, cache=cache, budget=budget, data_dir=tmp_path)


def test_unknown_arm(prices, cache, budget, tmp_path):
    with pytest.raises(ConfigError):
        build_arms(["nope"], {}, dry_run=True, prices=prices, cache=cache, budget=budget, data_dir=tmp_path)
