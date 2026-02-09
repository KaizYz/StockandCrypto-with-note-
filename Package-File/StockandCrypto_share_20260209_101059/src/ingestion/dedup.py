from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Tuple

import pandas as pd


_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fff]+")


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def stable_hash(text: Any) -> str:
    norm = str(text or "").strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _token_hash64(token: str) -> int:
    h = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def simhash64(text: Any) -> int:
    norm = normalize_text(text)
    if not norm:
        return 0
    bits = [0] * 64
    for token in norm.split():
        h = _token_hash64(token)
        for i in range(64):
            bits[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i, score in enumerate(bits):
        if score >= 0:
            out |= (1 << i)
    return out


def hamming_distance(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def add_dedup_keys(
    df: pd.DataFrame,
    *,
    title_col: str = "title",
    summary_col: str = "summary",
    url_col: str = "url",
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["dedup_key_url"] = ""
        out["dedup_key_simhash"] = ""
        return out
    out = df.copy()
    out["dedup_key_url"] = out[url_col].fillna("").map(lambda x: stable_hash(str(x).strip()) if str(x).strip() else "")
    merged_text = (
        out[title_col].fillna("").astype(str)
        + " "
        + out[summary_col].fillna("").astype(str)
    )
    out["dedup_key_simhash"] = merged_text.map(lambda x: f"{simhash64(x):016x}")
    return out


def assign_dedup_groups(
    df: pd.DataFrame,
    *,
    simhash_distance: int = 3,
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["dedup_group_id"] = ""
        return out

    out = add_dedup_keys(df)
    out = out.copy()
    out["dedup_group_id"] = ""

    used = {}
    reps: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    group_idx = 0

    if "available_at_utc" in out.columns:
        out["_sort_ts"] = pd.to_datetime(out["available_at_utc"], utc=True, errors="coerce")
    else:
        out["_sort_ts"] = pd.NaT
    out = out.sort_values(["market", "symbol", "_sort_ts"], na_position="last").reset_index(drop=True)

    for i, row in out.iterrows():
        market = str(row.get("market", "")).strip().lower()
        symbol = str(row.get("symbol", "")).strip().upper()
        key = (market, symbol)
        url_key = str(row.get("dedup_key_url", "")).strip()
        simhex = str(row.get("dedup_key_simhash", "")).strip()
        simint = int(simhex, 16) if simhex else 0

        if url_key:
            if url_key in used:
                out.at[i, "dedup_group_id"] = used[url_key]
                continue

        assigned = ""
        for rep_sim, gid in reps.get(key, []):
            if hamming_distance(simint, rep_sim) <= int(simhash_distance):
                assigned = gid
                break

        if not assigned:
            group_idx += 1
            assigned = f"{market}:{symbol}:{group_idx:08d}"
            reps.setdefault(key, []).append((simint, assigned))
        out.at[i, "dedup_group_id"] = assigned
        if url_key:
            used[url_key] = assigned

    out = out.drop(columns=["_sort_ts"], errors="ignore")
    return out
