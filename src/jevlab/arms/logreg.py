"""Classical baseline: TF-IDF features + logistic regression, trained on a disjoint split."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np

from ..datasets import load_train
from ..schema import Decision, Prediction
from .base import DecisionModel


def decision_text(d: Decision) -> str:
    # The question only varies per item for BoolQ, but including it is harmless elsewhere.
    return f"{d.question} || {d.state}"


class LogRegModel(DecisionModel):
    synthetic = False
    concurrency = 1

    def __init__(self, name: str, model_id: str, data_dir: Path, seed: int = 0):
        self.name = name
        self.model_id = model_id
        self.data_dir = data_dir
        self.seed = seed
        self._models: dict[str, object] = {}
        self.train_sizes: dict[str, int] = {}

    def fit(self, dataset: str, train: list[Decision]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.pipeline import make_pipeline

        texts = [decision_text(d) for d in train]
        labels = [d.gold for d in train]
        n_classes = len(set(labels))
        folds = max(2, min(3, min(labels.count(c) for c in set(labels))))
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2 if len(texts) > 50 else 1, sublinear_tf=True),
            LogisticRegressionCV(
                Cs=[0.25, 1.0, 4.0, 16.0],
                cv=folds,
                scoring="neg_log_loss",
                max_iter=3000,
                random_state=self.seed,
            ),
        )
        if n_classes < 2:
            raise ValueError(f"training split for {dataset} has a single class")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # sklearn attribute-layout notice
            pipe.fit(texts, labels)
        self._models[dataset] = pipe
        self.train_sizes[dataset] = len(texts)

    def prepare(self, dataset: str) -> None:
        if dataset not in self._models:
            self.fit(dataset, load_train(dataset, self.data_dir))

    def predict(self, decision: Decision, repeat: int = 0) -> Prediction:
        self.prepare(decision.dataset)
        pipe = self._models[decision.dataset]
        t0 = time.perf_counter()
        proba = pipe.predict_proba([decision_text(decision)])[0]  # type: ignore[attr-defined]
        latency = (time.perf_counter() - t0) * 1000.0
        classes = list(pipe.classes_)  # type: ignore[attr-defined]
        probs = {cid: float(proba[classes.index(cid)]) if cid in classes else 0.0 for cid in decision.choice_ids}
        total = sum(probs.values()) or 1.0
        probs = {k: float(np.clip(v / total, 0, 1)) for k, v in probs.items()}
        return Prediction(probs=probs, latency_ms=latency, cost_usd=0.0)
