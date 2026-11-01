"""On-disk response cache keyed by (model, request hash), so reruns cost nothing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


def request_key(model: str, payload: Any) -> str:
    blob = json.dumps({"model": model, "payload": payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, root: Path | None):
        self.root = root
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def _path(self, model: str, key: str) -> Path:
        assert self.root is not None
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
        return self.root / safe / key[:2] / f"{key}.json"

    def get(self, model: str, key: str) -> dict[str, Any] | None:
        if self.root is None:
            with self._lock:
                self.misses += 1
            return None
        path = self._path(model, key)
        if path.exists():
            with self._lock:
                self.hits += 1
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return data
        with self._lock:
            self.misses += 1
        return None

    def put(self, model: str, key: str, record: dict[str, Any]) -> None:
        if self.root is None:
            return
        path = self._path(model, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        os.replace(tmp, path)  # atomic, safe under concurrent writers
