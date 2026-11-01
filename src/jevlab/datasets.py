"""Dataset adapters: public, human-labeled datasets turned into typed decisions.

Every adapter converts a raw row into a `Decision` (state + question + allowed
choices + gold label). `jevlab fetch` downloads the raw files into `data/raw/`
and writes a fixed-seed evaluation sample plus a disjoint training split into
`data/processed/<name>/{eval,train}.jsonl`.

Sources (all labels come from people, not models):
- SST-2: Socher et al. 2013, binary movie-review sentiment (GLUE release).
- BoolQ: Clark et al. 2019, yes/no questions about Wikipedia passages.
- LIAR: Wang 2017, PolitiFact statements with six-way truthfulness ratings.
- banking77: Casanueva et al. 2020, customer-service intents (10-intent card subset).
- SMS Spam Collection: Almeida et al. 2011, ham/spam SMS messages.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import Decision

HF_PARQUET = "https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{config}/{split}/0000.parquet"
LIAR_ZIP = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"

DEFAULT_N = 300
DEFAULT_SEED = 20260915  # Jev launch date, fixed for every sample in this repo


# --------------------------------------------------------------------------- adapters


def sst2_decision(row: pd.Series, item_id: str) -> Decision:
    return Decision(
        id=item_id,
        dataset="sst2",
        state=str(row["sentence"]).strip(),
        question="What is the sentiment of this movie-review excerpt?",
        choices={
            "negative": "The excerpt expresses a negative opinion of the movie.",
            "positive": "The excerpt expresses a positive opinion of the movie.",
        },
        gold="positive" if int(row["label"]) == 1 else "negative",
    )


def boolq_decision(row: pd.Series, item_id: str) -> Decision:
    question = str(row["question"]).strip()
    question = question[0].upper() + question[1:] if question else question
    answer = row["answer"]
    if isinstance(answer, str):
        answer = answer.strip().lower() == "true"
    return Decision(
        id=item_id,
        dataset="boolq",
        state=str(row["passage"]).strip(),
        question=f"Using only the passage, answer the question: {question}?",
        choices={
            "yes": "The passage supports answering yes.",
            "no": "The passage supports answering no.",
        },
        gold="yes" if bool(answer) else "no",
    )


LIAR_LABELS = {
    "pants-fire": "Pants on fire: the statement is false and ridiculous.",
    "false": "False: the statement is not accurate.",
    "barely-true": "Barely true: an element of truth, but ignores critical facts.",
    "half-true": "Half true: partially accurate, leaves out important details.",
    "mostly-true": "Mostly true: accurate but needs clarification or additional information.",
    "true": "True: the statement is accurate and nothing significant is missing.",
}
LIAR_COLUMNS = [
    "json_id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job",
    "state_info",
    "party",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]


def _clean(v: object) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    s = str(v).strip()
    return s or "unknown"


def liar_decision(row: pd.Series, item_id: str) -> Decision:
    # The speaker's historical credit counts are deliberately left out: they leak the label.
    state = (
        f"Statement: {_clean(row['statement'])}\n"
        f"Speaker: {_clean(row['speaker'])} ({_clean(row['job'])}, {_clean(row['party'])})\n"
        f"Context: {_clean(row['context'])}"
    )
    return Decision(
        id=item_id,
        dataset="liar",
        state=state,
        question="How would PolitiFact rate the truthfulness of this statement?",
        choices=dict(LIAR_LABELS),
        gold=str(row["label"]).strip(),
    )


BANKING77_NAMES = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money", "Refund_not_showing_up",
    "request_refund", "reverted_card_payment?", "supported_cards_and_currencies",
    "terminate_account", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds", "verify_top_up",
    "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity",
    "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal",
]  # fmt: skip

# Ten card-related intents that people (and models) confuse with each other.
BANKING77_SUBSET = {
    "activate_my_card": "Wants to activate a card they have received.",
    "card_arrival": "Asks whether or why their card has not arrived yet.",
    "card_delivery_estimate": "Asks how long card delivery takes.",
    "card_not_working": "Their card is not working.",
    "card_payment_not_recognised": "Sees a card payment they do not recognise.",
    "compromised_card": "Worries their card details were compromised or used by someone else.",
    "declined_card_payment": "A card payment was declined.",
    "getting_spare_card": "Wants an additional or spare card.",
    "lost_or_stolen_card": "Their card was lost or stolen.",
    "pending_card_payment": "A card payment is still pending.",
}


def banking77_decision(row: pd.Series, item_id: str) -> Decision:
    label = row["label"]
    name = BANKING77_NAMES[int(label)] if not isinstance(label, str) or label.isdigit() else label
    return Decision(
        id=item_id,
        dataset="banking77",
        state=str(row["text"]).strip(),
        question="Which intent best describes this bank customer's message?",
        choices=dict(BANKING77_SUBSET),
        gold=name,
    )


def sms_spam_decision(row: pd.Series, item_id: str) -> Decision:
    return Decision(
        id=item_id,
        dataset="sms_spam",
        state=str(row["sms"]).strip(),
        question="Is this SMS message spam?",
        choices={
            "ham": "A normal personal or transactional message.",
            "spam": "Unsolicited advertising, a scam, or a premium-rate lure.",
        },
        gold="spam" if int(row["label"]) == 1 else "ham",
    )


# --------------------------------------------------------------------------- raw loaders


def _parquet(raw: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(raw / name)


def _liar_tsv(raw: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(
        raw / name,
        sep="\t",
        header=None,
        names=LIAR_COLUMNS,
        quoting=csv.QUOTE_NONE,
        dtype=str,
        keep_default_na=False,
    )


def _banking_subset(df: pd.DataFrame) -> pd.DataFrame:
    names = df["label"].map(lambda x: BANKING77_NAMES[int(x)])
    return df[names.isin(list(BANKING77_SUBSET))]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    downloads: dict[str, str]  # local file name -> URL
    load_eval: Callable[[Path], pd.DataFrame]
    load_train: Callable[[Path], pd.DataFrame] | None  # None: train = complement of eval sample
    adapt: Callable[[pd.Series, str], Decision]


def _hf(repo: str, config: str, split: str) -> str:
    return HF_PARQUET.format(repo=repo, config=config, split=split)


DATASETS: dict[str, DatasetSpec] = {
    "sst2": DatasetSpec(
        "sst2",
        "SST-2 sentiment (validation)",
        {
            "train.parquet": _hf("stanfordnlp/sst2", "default", "train"),
            "validation.parquet": _hf("stanfordnlp/sst2", "default", "validation"),
        },
        lambda raw: _parquet(raw, "validation.parquet"),
        lambda raw: _parquet(raw, "train.parquet"),
        sst2_decision,
    ),
    "boolq": DatasetSpec(
        "boolq",
        "BoolQ yes/no QA (validation)",
        {
            "train.parquet": _hf("google/boolq", "default", "train"),
            "validation.parquet": _hf("google/boolq", "default", "validation"),
        },
        lambda raw: _parquet(raw, "validation.parquet"),
        lambda raw: _parquet(raw, "train.parquet"),
        boolq_decision,
    ),
    "liar": DatasetSpec(
        "liar",
        "LIAR six-way fact-check (test)",
        {"liar_dataset.zip": LIAR_ZIP},
        lambda raw: _liar_tsv(raw, "test.tsv"),
        lambda raw: _liar_tsv(raw, "train.tsv"),
        liar_decision,
    ),
    "banking77": DatasetSpec(
        "banking77",
        "banking77, 10 card intents (test)",
        {
            "train.parquet": _hf("PolyAI/banking77", "default", "train"),
            "test.parquet": _hf("PolyAI/banking77", "default", "test"),
        },
        lambda raw: _banking_subset(_parquet(raw, "test.parquet")),
        lambda raw: _banking_subset(_parquet(raw, "train.parquet")),
        banking77_decision,
    ),
    "sms_spam": DatasetSpec(
        "sms_spam",
        "SMS Spam Collection",
        {"train.parquet": _hf("ucirvine/sms_spam", "plain_text", "train")},
        lambda raw: _parquet(raw, "train.parquet"),
        None,
        sms_spam_decision,
    ),
}


# --------------------------------------------------------------------------- build / IO


def to_decisions(spec: DatasetSpec, df: pd.DataFrame) -> list[Decision]:
    return [spec.adapt(row, f"{spec.name}-{idx}") for idx, row in df.iterrows()]


def sample_split(
    spec: DatasetSpec, raw: Path, n: int = DEFAULT_N, seed: int = DEFAULT_SEED
) -> tuple[list[Decision], list[Decision]]:
    """Return (eval sample, disjoint train split) for a dataset whose raw files are in `raw`."""
    eval_df = spec.load_eval(raw)
    take = min(n, len(eval_df))
    sample = eval_df.sample(n=take, random_state=seed)
    train_df = eval_df.drop(index=sample.index) if spec.load_train is None else spec.load_train(raw)
    eval_items = to_decisions(spec, sample)
    train_items = to_decisions(spec, train_df)
    eval_ids = {d.id for d in eval_items}
    if spec.load_train is None:
        assert not eval_ids & {d.id for d in train_items}, "train/eval overlap"
    return eval_items, train_items


def write_jsonl(path: Path, items: list[Decision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in items:
            f.write(d.to_json() + "\n")


def read_jsonl(path: Path) -> list[Decision]:
    with path.open(encoding="utf-8") as f:
        return [Decision.from_dict(json.loads(line)) for line in f if line.strip()]


def download(spec: DatasetSpec, raw: Path, timeout: float = 120.0) -> None:
    import httpx

    raw.mkdir(parents=True, exist_ok=True)
    for fname, url in spec.downloads.items():
        dest = raw / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        if fname.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(dest.read_bytes())) as zf:
                for member in zf.namelist():
                    if member.endswith(".tsv"):
                        (raw / Path(member).name).write_bytes(zf.read(member))


def fetch(
    names: list[str],
    data_dir: Path,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    log: Callable[[str], None] = print,
) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    for name in names:
        spec = DATASETS[name]
        raw = data_dir / "raw" / name
        download(spec, raw)
        eval_items, train_items = sample_split(spec, raw, n=n, seed=seed)
        out = data_dir / "processed" / name
        write_jsonl(out / "eval.jsonl", eval_items)
        write_jsonl(out / "train.jsonl", train_items)
        sizes[name] = (len(eval_items), len(train_items))
        log(f"{name:10s} eval={len(eval_items):4d} train={len(train_items):6d}")
    return sizes


def load_eval(name: str, data_dir: Path) -> list[Decision]:
    path = data_dir / "processed" / name / "eval.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `jevlab fetch` first")
    return read_jsonl(path)


def load_train(name: str, data_dir: Path) -> list[Decision]:
    path = data_dir / "processed" / name / "train.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `jevlab fetch` first")
    return read_jsonl(path)
