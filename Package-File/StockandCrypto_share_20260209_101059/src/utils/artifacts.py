from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_json(data: Any) -> str:
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hash_bytes(blob)


def hash_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files(paths: Iterable[str | Path]) -> str:
    h = hashlib.sha256()
    for path in sorted([str(Path(p)) for p in paths]):
        digest = hash_file(path)
        h.update(path.encode("utf-8"))
        h.update(digest.encode("utf-8"))
    return h.hexdigest()


def get_git_commit_short(default: str = "-") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or default
    except Exception:
        return default
