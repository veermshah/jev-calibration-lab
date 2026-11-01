from pathlib import Path

import pytest

from jevlab import datasets as ds
from jevlab.schema import Decision

RAW = Path(__file__).parent / "fixtures" / "raw"


@pytest.mark.parametrize("name", list(ds.DATASETS))
def test_every_adapter_produces_valid_decisions(name):
    spec = ds.DATASETS[name]
    eval_items, train_items = ds.sample_split(spec, RAW / name, n=3, seed=0)
    assert eval_items, name
    for d in eval_items + train_items:
        assert isinstance(d, Decision)
        assert d.dataset == name
        assert d.gold in d.choices
        assert d.state and d.question


def test_sst2_labels():
    items, _ = ds.sample_split(ds.DATASETS["sst2"], RAW / "sst2", n=6, seed=0)
    by_text = {d.state: d.gold for d in items}
    assert by_text["a charming journey ."] == "positive"
    assert by_text["dull and lifeless ."] == "negative"


def test_boolq_question_and_gold():
    items, _ = ds.sample_split(ds.DATASETS["boolq"], RAW / "boolq", n=4, seed=0)
    d = next(i for i in items if "sky" in i.question)
    assert d.question == "Using only the passage, answer the question: Is the sky blue?"
    assert d.gold == "yes"
    assert list(d.choices) == ["yes", "no"]


def test_liar_is_six_way_and_hides_credit_history():
    items, _ = ds.sample_split(ds.DATASETS["liar"], RAW / "liar", n=3, seed=0)
    assert all(len(d.choices) == 6 for d in items)
    moon = next(d for d in items if "cheese" in d.state)
    assert moon.gold == "pants-fire"
    assert "unknown" in moon.state  # empty job field
    assert "counts" not in moon.state.lower()


def test_banking77_filters_to_subset():
    items, train = ds.sample_split(ds.DATASETS["banking77"], RAW / "banking77", n=10, seed=0)
    # change_pin (21) is outside the subset and must be dropped
    assert len(items) == 4
    assert {d.gold for d in items} == {
        "card_arrival",
        "activate_my_card",
        "compromised_card",
        "declined_card_payment",
    }
    assert all(len(d.choices) == 10 for d in items)


def test_sms_train_is_disjoint_complement():
    items, train = ds.sample_split(ds.DATASETS["sms_spam"], RAW / "sms_spam", n=4, seed=1)
    assert len(items) == 4 and len(train) == 6
    assert not {d.id for d in items} & {d.id for d in train}


def test_sampling_is_seeded():
    a, _ = ds.sample_split(ds.DATASETS["sms_spam"], RAW / "sms_spam", n=5, seed=7)
    b, _ = ds.sample_split(ds.DATASETS["sms_spam"], RAW / "sms_spam", n=5, seed=7)
    assert [d.id for d in a] == [d.id for d in b]


def test_jsonl_roundtrip(tmp_path):
    items, _ = ds.sample_split(ds.DATASETS["liar"], RAW / "liar", n=3, seed=0)
    ds.write_jsonl(tmp_path / "x.jsonl", items)
    assert ds.read_jsonl(tmp_path / "x.jsonl") == items


def test_decision_rejects_bad_gold():
    with pytest.raises(ValueError):
        Decision("x", "d", "s", "q", {"a": "", "b": ""}, "c")
