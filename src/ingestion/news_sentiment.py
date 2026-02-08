from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

from src.ingestion.dedup import assign_dedup_groups
from src.ingestion.symbol_mapper import (
    canonical_symbol,
    collect_symbol_targets,
    map_text_to_symbol_candidates,
    normalize_market,
)
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json, write_csv


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
ALPHA_VANTAGE_API = "https://www.alphavantage.co/query"
NEWSAPI_API = "https://newsapi.org/v2/everything"
FINNHUB_COMPANY_NEWS_API = "https://finnhub.io/api/v1/company-news"
MARKETAUX_API = "https://api.marketaux.com/v1/news/all"
BYBIT_ANNOUNCE_API = "https://api.bybit.com/v5/announcements/index"
SEC_ATOM_API = "https://www.sec.gov/cgi-bin/browse-edgar"

DEFAULT_RSS_FEEDS = [
    "https://www.binance.com/en/support/announcement/rss",
]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _safe_int(value: Any, default: int = 0) -> int:
    out = _safe_float(value)
    if np.isfinite(out):
        return int(round(out))
    return int(default)


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _parse_ts_utc(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    text = _safe_str(value)
    if not text:
        return pd.NaT

    if len(text) == 15 and text[8] == "T" and text.replace("T", "").isdigit():
        try:
            return pd.Timestamp(datetime.strptime(text, "%Y%m%dT%H%M%S"), tz="UTC")
        except Exception:
            pass

    if text.isdigit():
        try:
            iv = int(text)
            if iv > 10_000_000_000:
                return pd.to_datetime(iv, unit="ms", utc=True, errors="coerce")
            return pd.to_datetime(iv, unit="s", utc=True, errors="coerce")
        except Exception:
            pass

    return pd.to_datetime(text, utc=True, errors="coerce")


def _normalize_sentiment_score(value: Any) -> float:
    x = _safe_float(value)
    if not np.isfinite(x):
        return float("nan")
    if 0.0 <= x <= 1.0:
        x = 2.0 * x - 1.0
    return float(np.clip(x, -1.0, 1.0))


def _simple_lexicon_sentiment(text: str) -> tuple[float, float]:
    txt = _safe_str(text).lower()
    if not txt:
        return 0.0, 0.0

    pos_words = [
        "beat",
        "beats",
        "growth",
        "surge",
        "rally",
        "upgrade",
        "strong",
        "bullish",
        "approval",
        "partnership",
        "record high",
        "breakout",
        "利好",
        "增长",
        "上调",
        "突破",
        "看涨",
        "超预期",
        "大涨",
    ]
    neg_words = [
        "miss",
        "downgrade",
        "decline",
        "drop",
        "crash",
        "hack",
        "lawsuit",
        "investigation",
        "bearish",
        "bankruptcy",
        "warning",
        "recession",
        "利空",
        "下调",
        "下跌",
        "看跌",
        "亏损",
        "爆雷",
        "暴跌",
    ]

    pos = sum(1 for w in pos_words if w in txt)
    neg = sum(1 for w in neg_words if w in txt)
    total = pos + neg
    if total <= 0:
        return 0.0, 0.0
    score = (pos - neg) / float(total + 1)
    conf = min(1.0, (abs(pos - neg) + 1.0) / float(total + 2.0))
    return float(np.clip(score, -1.0, 1.0)), float(conf)


def _request_json(
    url: str,
    *,
    params: Dict[str, Any],
    headers: Dict[str, str],
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout_sec)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(backoff_sec * (2**attempt))
    raise RuntimeError(f"request failed: {url} ({last_exc})")


def _request_text(
    url: str,
    *,
    params: Dict[str, Any],
    headers: Dict[str, str],
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout_sec)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(backoff_sec * (2**attempt))
    raise RuntimeError(f"request failed: {url} ({last_exc})")


def _provider_headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }


def _coalesce_summary(value: Any, max_len: int = 300) -> str:
    text = _safe_str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _base_news_row(
    *,
    provider: str,
    market: str,
    symbol: str,
    title: Any,
    summary: Any,
    url: Any,
    published_at_utc: Any,
    ingested_at_utc: pd.Timestamp,
    raw_sentiment: float,
    sentiment_confidence: float,
    relevance: float,
    source_tier: str,
    source_weight: float,
    language: str,
    match_type: str,
    entity_name: str = "",
) -> Dict[str, Any]:
    pub_ts = _parse_ts_utc(published_at_utc)
    ing_ts = pd.to_datetime(ingested_at_utc, utc=True, errors="coerce")
    avail_ts = max(pub_ts, ing_ts) if pd.notna(pub_ts) else ing_ts
    return {
        "published_at_utc": pub_ts,
        "ingested_at_utc": ing_ts,
        "available_at_utc": avail_ts,
        "provider": _safe_str(provider).lower(),
        "market": normalize_market(market),
        "symbol": canonical_symbol(symbol, market),
        "entity_name": _safe_str(entity_name),
        "title": _safe_str(title),
        "summary": _coalesce_summary(summary),
        "url": _safe_str(url),
        "language": _safe_str(language).lower() or "en",
        "raw_sentiment": float(np.clip(_safe_float(raw_sentiment), -1.0, 1.0)),
        "sentiment_confidence": float(np.clip(_safe_float(sentiment_confidence), 0.0, 1.0)),
        "relevance": float(np.clip(_safe_float(relevance), 0.0, 1.0)),
        "source_tier": _safe_str(source_tier) or "tier2",
        "source_weight": float(np.clip(_safe_float(source_weight), 0.0, 5.0)),
        "match_type": _safe_str(match_type) or "regex",
    }


def _alpha_ticker(symbol: str, market: str) -> str:
    mk = normalize_market(market)
    s = canonical_symbol(symbol, mk)
    if mk == "crypto":
        return s
    return s


def _fetch_alpha_vantage_rows(
    *,
    target: Dict[str, str],
    api_key: str,
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
    source_weight: float,
    tier: str,
    limit: int,
) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    market = target["market"]
    symbol = target["symbol"]
    ticker = _alpha_ticker(symbol, market)
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": int(limit),
        "sort": "LATEST",
        "apikey": api_key,
    }
    data = _request_json(
        ALPHA_VANTAGE_API,
        params=params,
        headers=_provider_headers(),
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    feed = data.get("feed", []) if isinstance(data, dict) else []
    now = _utc_now()
    rows: List[Dict[str, Any]] = []
    for item in feed:
        score = _normalize_sentiment_score(item.get("overall_sentiment_score"))
        if not np.isfinite(score):
            score, conf = _simple_lexicon_sentiment(
                f"{item.get('title', '')} {item.get('summary', '')}"
            )
        else:
            conf = max(0.3, min(1.0, abs(score)))

        rel = 0.7
        ticker_list = item.get("ticker_sentiment", [])
        if isinstance(ticker_list, list):
            for t in ticker_list:
                t_sym = _safe_str(t.get("ticker")).upper()
                if t_sym == ticker.upper():
                    rel = _safe_float(t.get("relevance_score"))
                    break

        rows.append(
            _base_news_row(
                provider="alpha_vantage",
                market=market,
                symbol=symbol,
                title=item.get("title"),
                summary=item.get("summary"),
                url=item.get("url"),
                published_at_utc=item.get("time_published"),
                ingested_at_utc=now,
                raw_sentiment=score,
                sentiment_confidence=conf,
                relevance=rel if np.isfinite(rel) else 0.7,
                source_tier=tier,
                source_weight=source_weight,
                language="en",
                match_type="provider_ticker",
                entity_name=target.get("name", ""),
            )
        )
    return rows


def _fetch_newsapi_rows(
    *,
    target: Dict[str, str],
    api_key: str,
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
    source_weight: float,
    tier: str,
    limit: int,
) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    q = target.get("query", "") or target.get("symbol", "")
    now = _utc_now()
    from_ts = (now - pd.Timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "q": q,
        "from": from_ts,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": min(max(int(limit), 10), 100),
        "apiKey": api_key,
    }
    data = _request_json(
        NEWSAPI_API,
        params=params,
        headers=_provider_headers(),
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    rows = []
    for item in data.get("articles", []):
        score, conf = _simple_lexicon_sentiment(
            f"{item.get('title', '')} {item.get('description', '')}"
        )
        rows.append(
            _base_news_row(
                provider="newsapi",
                market=target["market"],
                symbol=target["symbol"],
                title=item.get("title"),
                summary=item.get("description"),
                url=item.get("url"),
                published_at_utc=item.get("publishedAt"),
                ingested_at_utc=now,
                raw_sentiment=score,
                sentiment_confidence=max(conf, 0.35),
                relevance=0.7,
                source_tier=tier,
                source_weight=source_weight,
                language="en",
                match_type="regex",
                entity_name=target.get("name", ""),
            )
        )
    return rows


def _fetch_gdelt_rows(
    *,
    target: Dict[str, str],
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
    source_weight: float,
    tier: str,
    limit: int,
) -> List[Dict[str, Any]]:
    q = target.get("query", "") or target.get("symbol", "")
    now = _utc_now()
    params = {
        "query": q,
        "mode": "ArtList",
        "format": "json",
        "sort": "datedesc",
        "maxrecords": int(limit),
    }
    data = _request_json(
        GDELT_DOC_API,
        params=params,
        headers=_provider_headers(),
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    articles = data.get("articles", []) if isinstance(data, dict) else []
    rows = []
    for item in articles:
        title = item.get("title")
        summary = item.get("seendate", "")
        text = f"{title or ''} {summary or ''}"
        score, conf = _simple_lexicon_sentiment(text)
        rows.append(
            _base_news_row(
                provider="gdelt",
                market=target["market"],
                symbol=target["symbol"],
                title=title,
                summary=item.get("socialimage", ""),
                url=item.get("url"),
                published_at_utc=item.get("seendate"),
                ingested_at_utc=now,
                raw_sentiment=score,
                sentiment_confidence=max(conf, 0.3),
                relevance=0.6,
                source_tier=tier,
                source_weight=source_weight,
                language=str(item.get("language", "en")).lower(),
                match_type="regex",
                entity_name=target.get("name", ""),
            )
        )
    return rows


def _fetch_finnhub_rows(
    *,
    target: Dict[str, str],
    api_key: str,
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
    source_weight: float,
    tier: str,
) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    if normalize_market(target.get("market")) != "us_equity":
        return []
    now = _utc_now()
    params = {
        "symbol": target["symbol"],
        "from": (now - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        "to": now.strftime("%Y-%m-%d"),
        "token": api_key,
    }
    data = _request_json(
        FINNHUB_COMPANY_NEWS_API,
        params=params,
        headers=_provider_headers(),
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    rows = []
    if not isinstance(data, list):
        return rows
    for item in data:
        score, conf = _simple_lexicon_sentiment(
            f"{item.get('headline', '')} {item.get('summary', '')}"
        )
        rows.append(
            _base_news_row(
                provider="finnhub",
                market=target["market"],
                symbol=target["symbol"],
                title=item.get("headline"),
                summary=item.get("summary"),
                url=item.get("url"),
                published_at_utc=item.get("datetime"),
                ingested_at_utc=now,
                raw_sentiment=score,
                sentiment_confidence=max(conf, 0.35),
                relevance=0.8,
                source_tier=tier,
                source_weight=source_weight,
                language="en",
                match_type="provider_ticker",
                entity_name=target.get("name", ""),
            )
        )
    return rows


def _fetch_marketaux_rows(
    *,
    target: Dict[str, str],
    api_key: str,
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
    source_weight: float,
    tier: str,
    limit: int,
) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    params = {
        "symbols": target.get("symbol", ""),
        "language": "en",
        "limit": int(limit),
        "api_token": api_key,
    }
    data = _request_json(
        MARKETAUX_API,
        params=params,
        headers=_provider_headers(),
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    rows = []
    now = _utc_now()
    for item in data.get("data", []):
        score = _normalize_sentiment_score(item.get("sentiment_score"))
        if not np.isfinite(score):
            score, conf = _simple_lexicon_sentiment(
                f"{item.get('title', '')} {item.get('description', '')}"
            )
        else:
            conf = max(0.3, min(1.0, abs(score)))
        rows.append(
            _base_news_row(
                provider="marketaux",
                market=target["market"],
                symbol=target["symbol"],
                title=item.get("title"),
                summary=item.get("description"),
                url=item.get("url"),
                published_at_utc=item.get("published_at"),
                ingested_at_utc=now,
                raw_sentiment=score if np.isfinite(score) else 0.0,
                sentiment_confidence=conf,
                relevance=0.75,
                source_tier=tier,
                source_weight=source_weight,
                language="en",
                match_type="provider_ticker",
                entity_name=target.get("name", ""),
            )
        )
    return rows


def _fetch_rss_items(
    *,
    url: str,
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
) -> List[Dict[str, Any]]:
    xml_text = _request_text(
        url,
        params={},
        headers=_provider_headers(),
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    root = ElementTree.fromstring(xml_text)
    items: List[Dict[str, Any]] = []

    for item in root.findall(".//item"):
        items.append(
            {
                "title": item.findtext("title"),
                "summary": item.findtext("description"),
                "url": item.findtext("link"),
                "published": item.findtext("pubDate"),
                "language": root.findtext(".//language") or "en",
            }
        )

    if not items:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//a:entry", ns):
            link_node = entry.find("a:link", ns)
            href = link_node.get("href") if link_node is not None else ""
            items.append(
                {
                    "title": entry.findtext("a:title", default="", namespaces=ns),
                    "summary": entry.findtext("a:summary", default="", namespaces=ns),
                    "url": href,
                    "published": entry.findtext("a:updated", default="", namespaces=ns),
                    "language": "en",
                }
            )
    return items


def _map_generic_items_to_targets(
    *,
    provider: str,
    items: Iterable[Dict[str, Any]],
    targets: List[Dict[str, str]],
    source_weight: float,
    tier: str,
) -> List[Dict[str, Any]]:
    now = _utc_now()
    rows: List[Dict[str, Any]] = []
    for item in items:
        title = _safe_str(item.get("title"))
        summary = _safe_str(item.get("summary"))
        text = f"{title} {summary}".strip()
        score, conf = _simple_lexicon_sentiment(text)
        candidates = map_text_to_symbol_candidates(
            text,
            targets,
            market=_safe_str(item.get("market")),
        )
        if not candidates:
            continue
        for c in candidates:
            rows.append(
                _base_news_row(
                    provider=provider,
                    market=c.get("market", ""),
                    symbol=c.get("symbol", ""),
                    title=title,
                    summary=summary,
                    url=item.get("url"),
                    published_at_utc=item.get("published"),
                    ingested_at_utc=now,
                    raw_sentiment=score,
                    sentiment_confidence=max(conf, 0.3),
                    relevance=0.55,
                    source_tier=tier,
                    source_weight=source_weight,
                    language=item.get("language", "en"),
                    match_type="regex",
                    entity_name=c.get("name", ""),
                )
            )
    return rows


def _fetch_bybit_announcement_items(
    *,
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
) -> List[Dict[str, Any]]:
    params = {"locale": "en-US", "limit": 50}
    data = _request_json(
        BYBIT_ANNOUNCE_API,
        params=params,
        headers=_provider_headers(),
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    rows = data.get("result", {}).get("list", [])
    out = []
    for item in rows:
        out.append(
            {
                "title": item.get("title"),
                "summary": item.get("description", ""),
                "url": item.get("url"),
                "published": item.get("releaseTime"),
                "language": "en",
            }
        )
    return out


def _fetch_sec_atom_items(
    *,
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
) -> List[Dict[str, Any]]:
    params = {
        "action": "getcurrent",
        "type": "",
        "company": "",
        "dateb": "",
        "owner": "include",
        "start": 0,
        "count": 100,
        "output": "atom",
    }
    xml_text = _request_text(
        SEC_ATOM_API,
        params=params,
        headers={
            **_provider_headers(),
            "From": "research@example.com",
        },
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    root = ElementTree.fromstring(xml_text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: List[Dict[str, Any]] = []
    for entry in root.findall(".//a:entry", ns):
        title = entry.findtext("a:title", default="", namespaces=ns)
        summary = entry.findtext("a:summary", default="", namespaces=ns)
        link_node = entry.find("a:link", ns)
        href = link_node.get("href") if link_node is not None else ""
        updated = entry.findtext("a:updated", default="", namespaces=ns)
        out.append(
            {
                "title": title,
                "summary": summary,
                "url": href,
                "published": updated,
                "language": "en",
                "market": "us_equity",
            }
        )
    return out


def _source_weight(cfg_news: Dict[str, Any], tier: str) -> float:
    weights = cfg_news.get("source_tier_weights", {"tier1": 1.0, "tier2": 0.7, "tier3": 0.4})
    return _safe_float(weights.get(tier, 0.7))


def _provider_enabled(providers: List[str], key: str) -> bool:
    return key.lower() in {str(x).strip().lower() for x in providers}


def _api_key(cfg_news: Dict[str, Any], field: str, fallback_env: str = "") -> str:
    key_cfg = cfg_news.get("api_keys", {})
    env_name = _safe_str(key_cfg.get(field) or fallback_env)
    if not env_name:
        return ""
    return _safe_str(os.getenv(env_name))


def _append_provider_status(
    status: List[Dict[str, Any]],
    *,
    provider: str,
    ok: bool,
    fetched: int,
    message: str = "",
) -> None:
    status.append(
        {
            "provider": provider,
            "ok": bool(ok),
            "rows": int(fetched),
            "message": _safe_str(message),
        }
    )


def _coerce_ts_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
    return out


def _default_news_cfg() -> Dict[str, Any]:
    return {
        "enabled": True,
        "markets": ["crypto", "us_equity", "cn_equity"],
        "providers": [
            "alpha_vantage",
            "newsapi",
            "gdelt",
            "finnhub",
            "marketaux",
            "bybit_announcements",
            "binance_rss",
            "sec_atom",
        ],
        "source_tier_by_provider": {
            "alpha_vantage": "tier1",
            "newsapi": "tier2",
            "gdelt": "tier3",
            "finnhub": "tier1",
            "marketaux": "tier2",
            "bybit_announcements": "tier1",
            "binance_rss": "tier1",
            "sec_atom": "tier1",
        },
        "source_tier_weights": {"tier1": 1.0, "tier2": 0.7, "tier3": 0.4},
        "api_keys": {
            "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
            "newsapi": "NEWSAPI_API_KEY",
            "finnhub": "FINNHUB_API_KEY",
            "marketaux": "MARKETAUX_API_KEY",
        },
        "timeout_sec": 15,
        "max_retries": 2,
        "backoff_sec": 1.2,
        "provider_limit_per_symbol": 20,
        "retention_days": 45,
        "max_rows_total": 200_000,
        "dedup_simhash_distance": 3,
        "rss_feeds": DEFAULT_RSS_FEEDS,
        "max_symbols_per_market": {"crypto": 20, "us_equity": 20, "cn_equity": 20},
    }


def _merge_news_cfg(cfg_news: Dict[str, Any]) -> Dict[str, Any]:
    return {**_default_news_cfg(), **(cfg_news or {})}


def _resolve_output_paths(cfg: Dict[str, Any], cfg_news: Dict[str, Any]) -> Dict[str, Path]:
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    processed_dir = Path(paths_cfg.get("processed_data_dir", "data/processed"))
    out_root = Path(cfg_news.get("output_dir", processed_dir / "news"))
    out_root = ensure_dir(out_root)
    return {
        "root": out_root,
        "raw_csv": Path(cfg_news.get("raw_output_csv", out_root / "news_raw.csv")),
        "raw_parquet": Path(cfg_news.get("raw_output_parquet", out_root / "news_raw.parquet")),
        "status_json": Path(cfg_news.get("status_output_json", out_root / "news_ingestion_status.json")),
        "status_csv": Path(cfg_news.get("status_output_csv", out_root / "news_ingestion_status.csv")),
    }


def _load_existing_news(path_csv: Path) -> pd.DataFrame:
    if not path_csv.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path_csv)
    except Exception:
        return pd.DataFrame()
    return _coerce_ts_cols(
        df,
        cols=["published_at_utc", "ingested_at_utc", "available_at_utc"],
    )


def _required_news_cols() -> List[str]:
    return [
        "published_at_utc",
        "ingested_at_utc",
        "available_at_utc",
        "provider",
        "market",
        "symbol",
        "entity_name",
        "title",
        "summary",
        "url",
        "language",
        "raw_sentiment",
        "sentiment_confidence",
        "relevance",
        "source_tier",
        "source_weight",
        "match_type",
        "dedup_key_url",
        "dedup_key_simhash",
        "dedup_group_id",
    ]


def _prepare_news_df(rows: List[Dict[str, Any]], dedup_distance: int) -> pd.DataFrame:
    if not rows:
        out = pd.DataFrame(columns=_required_news_cols())
        return out
    df = pd.DataFrame(rows)
    for c in ["provider", "market", "symbol", "title", "summary", "url", "language", "source_tier", "match_type"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    for c in ["raw_sentiment", "sentiment_confidence", "relevance", "source_weight"]:
        if c not in df.columns:
            df[c] = float("nan")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["published_at_utc", "ingested_at_utc", "available_at_utc"]:
        if c not in df.columns:
            df[c] = pd.NaT
    df = _coerce_ts_cols(df, cols=["published_at_utc", "ingested_at_utc", "available_at_utc"])
    df = assign_dedup_groups(df, simhash_distance=int(dedup_distance))
    req = _required_news_cols()
    for c in req:
        if c not in df.columns:
            df[c] = ""
    df = df[req].copy()
    return df


def _merge_news(
    *,
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    dedup_distance: int,
    retention_days: int,
    max_rows_total: int,
) -> pd.DataFrame:
    if existing_df.empty and new_df.empty:
        return pd.DataFrame(columns=_required_news_cols())
    base = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df.copy()
    if base.empty:
        return pd.DataFrame(columns=_required_news_cols())
    base = _coerce_ts_cols(base, cols=["published_at_utc", "ingested_at_utc", "available_at_utc"])
    base = assign_dedup_groups(base, simhash_distance=int(dedup_distance))

    sort_cols: List[str] = []
    for c in ["available_at_utc", "ingested_at_utc", "source_weight", "relevance"]:
        if c in base.columns:
            sort_cols.append(c)
    if sort_cols:
        base = base.sort_values(sort_cols)

    dedup_key_cols = [c for c in ["market", "symbol", "dedup_group_id"] if c in base.columns]
    if dedup_key_cols:
        base = base.drop_duplicates(subset=dedup_key_cols, keep="last")

    if int(retention_days) > 0 and "available_at_utc" in base.columns:
        cutoff = _utc_now() - pd.Timedelta(days=int(retention_days))
        ts = pd.to_datetime(base["available_at_utc"], utc=True, errors="coerce")
        base = base[ts >= cutoff].copy()

    if int(max_rows_total) > 0 and len(base) > int(max_rows_total):
        if "available_at_utc" in base.columns:
            base = base.sort_values("available_at_utc").tail(int(max_rows_total)).copy()
        else:
            base = base.tail(int(max_rows_total)).copy()

    req = _required_news_cols()
    for c in req:
        if c not in base.columns:
            base[c] = ""
    return base[req].reset_index(drop=True)


def _write_news_outputs(
    *,
    merged_df: pd.DataFrame,
    status_rows: List[Dict[str, Any]],
    paths: Dict[str, Path],
    status_payload: Dict[str, Any],
) -> None:
    raw_csv = paths["raw_csv"]
    raw_parquet = paths["raw_parquet"]
    status_json = paths["status_json"]
    status_csv = paths["status_csv"]

    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_parquet.parent.mkdir(parents=True, exist_ok=True)
    status_json.parent.mkdir(parents=True, exist_ok=True)
    status_csv.parent.mkdir(parents=True, exist_ok=True)

    write_csv(merged_df, raw_csv)
    try:
        merged_df.to_parquet(raw_parquet, index=False)
        status_payload["raw_parquet_written"] = True
    except Exception as exc:
        status_payload["raw_parquet_written"] = False
        status_payload["raw_parquet_error"] = f"{type(exc).__name__}: {exc}"

    save_json(status_payload, status_json)
    if status_rows:
        write_csv(pd.DataFrame(status_rows), status_csv)
    else:
        write_csv(pd.DataFrame(columns=["provider", "ok", "rows", "message"]), status_csv)


def _run_alpha_vantage(
    *,
    targets: List[Dict[str, str]],
    cfg_news: Dict[str, Any],
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
    limit: int,
) -> List[Dict[str, Any]]:
    api_key = _api_key(cfg_news, "alpha_vantage", "ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return []
    tier = _safe_str(cfg_news.get("source_tier_by_provider", {}).get("alpha_vantage", "tier1"))
    weight = _source_weight(cfg_news, tier)
    rows: List[Dict[str, Any]] = []
    for tgt in targets:
        rows.extend(
            _fetch_alpha_vantage_rows(
                target=tgt,
                api_key=api_key,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                backoff_sec=backoff_sec,
                source_weight=weight,
                tier=tier,
                limit=limit,
            )
        )
    return rows


def _run_newsapi(
    *,
    targets: List[Dict[str, str]],
    cfg_news: Dict[str, Any],
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
    limit: int,
) -> List[Dict[str, Any]]:
    api_key = _api_key(cfg_news, "newsapi", "NEWSAPI_API_KEY")
    if not api_key:
        return []
    tier = _safe_str(cfg_news.get("source_tier_by_provider", {}).get("newsapi", "tier2"))
    weight = _source_weight(cfg_news, tier)
    rows: List[Dict[str, Any]] = []
    for tgt in targets:
        rows.extend(
            _fetch_newsapi_rows(
                target=tgt,
                api_key=api_key,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                backoff_sec=backoff_sec,
                source_weight=weight,
                tier=tier,
                limit=limit,
            )
        )
    return rows


def _run_gdelt(
    *,
    targets: List[Dict[str, str]],
    cfg_news: Dict[str, Any],
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
    limit: int,
) -> List[Dict[str, Any]]:
    tier = _safe_str(cfg_news.get("source_tier_by_provider", {}).get("gdelt", "tier3"))
    weight = _source_weight(cfg_news, tier)
    rows: List[Dict[str, Any]] = []
    for tgt in targets:
        rows.extend(
            _fetch_gdelt_rows(
                target=tgt,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                backoff_sec=backoff_sec,
                source_weight=weight,
                tier=tier,
                limit=limit,
            )
        )
    return rows


def _run_finnhub(
    *,
    targets: List[Dict[str, str]],
    cfg_news: Dict[str, Any],
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
) -> List[Dict[str, Any]]:
    api_key = _api_key(cfg_news, "finnhub", "FINNHUB_API_KEY")
    if not api_key:
        return []
    tier = _safe_str(cfg_news.get("source_tier_by_provider", {}).get("finnhub", "tier1"))
    weight = _source_weight(cfg_news, tier)
    rows: List[Dict[str, Any]] = []
    for tgt in targets:
        rows.extend(
            _fetch_finnhub_rows(
                target=tgt,
                api_key=api_key,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                backoff_sec=backoff_sec,
                source_weight=weight,
                tier=tier,
            )
        )
    return rows


def _run_marketaux(
    *,
    targets: List[Dict[str, str]],
    cfg_news: Dict[str, Any],
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
    limit: int,
) -> List[Dict[str, Any]]:
    api_key = _api_key(cfg_news, "marketaux", "MARKETAUX_API_KEY")
    if not api_key:
        return []
    tier = _safe_str(cfg_news.get("source_tier_by_provider", {}).get("marketaux", "tier2"))
    weight = _source_weight(cfg_news, tier)
    rows: List[Dict[str, Any]] = []
    for tgt in targets:
        rows.extend(
            _fetch_marketaux_rows(
                target=tgt,
                api_key=api_key,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                backoff_sec=backoff_sec,
                source_weight=weight,
                tier=tier,
                limit=limit,
            )
        )
    return rows


def _run_bybit_announcements(
    *,
    targets: List[Dict[str, str]],
    cfg_news: Dict[str, Any],
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
) -> List[Dict[str, Any]]:
    tier = _safe_str(cfg_news.get("source_tier_by_provider", {}).get("bybit_announcements", "tier1"))
    weight = _source_weight(cfg_news, tier)
    items = _fetch_bybit_announcement_items(
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    for item in items:
        item["market"] = "crypto"
    return _map_generic_items_to_targets(
        provider="bybit_announcements",
        items=items,
        targets=targets,
        source_weight=weight,
        tier=tier,
    )


def _run_binance_rss(
    *,
    targets: List[Dict[str, str]],
    cfg_news: Dict[str, Any],
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
) -> List[Dict[str, Any]]:
    tier = _safe_str(cfg_news.get("source_tier_by_provider", {}).get("binance_rss", "tier1"))
    weight = _source_weight(cfg_news, tier)
    feeds = cfg_news.get("rss_feeds", DEFAULT_RSS_FEEDS)
    rows: List[Dict[str, Any]] = []
    for feed in feeds:
        items = _fetch_rss_items(
            url=str(feed),
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
        )
        for item in items:
            item["market"] = "crypto"
        rows.extend(
            _map_generic_items_to_targets(
                provider="binance_rss",
                items=items,
                targets=targets,
                source_weight=weight,
                tier=tier,
            )
        )
    return rows


def _run_sec_atom(
    *,
    targets: List[Dict[str, str]],
    cfg_news: Dict[str, Any],
    timeout_sec: int,
    max_retries: int,
    backoff_sec: float,
) -> List[Dict[str, Any]]:
    tier = _safe_str(cfg_news.get("source_tier_by_provider", {}).get("sec_atom", "tier1"))
    weight = _source_weight(cfg_news, tier)
    items = _fetch_sec_atom_items(
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    return _map_generic_items_to_targets(
        provider="sec_atom",
        items=items,
        targets=targets,
        source_weight=weight,
        tier=tier,
    )


def run_news_sentiment(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    cfg_news = _merge_news_cfg(cfg.get("news", {}) if isinstance(cfg, dict) else {})
    if not bool(cfg_news.get("enabled", True)):
        payload = {
            "enabled": False,
            "message": "news ingestion disabled by config",
            "generated_at_utc": _utc_now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        print("[WARN] news ingestion disabled.")
        return payload

    timeout_sec = int(cfg_news.get("timeout_sec", 15))
    max_retries = int(cfg_news.get("max_retries", 2))
    backoff_sec = float(cfg_news.get("backoff_sec", 1.2))
    per_symbol_limit = int(cfg_news.get("provider_limit_per_symbol", 20))
    dedup_distance = int(cfg_news.get("dedup_simhash_distance", 3))
    retention_days = int(cfg_news.get("retention_days", 45))
    max_rows_total = int(cfg_news.get("max_rows_total", 200_000))

    out_paths = _resolve_output_paths(cfg, cfg_news)
    processed_dir = cfg.get("paths", {}).get("processed_data_dir", "data/processed")
    targets = collect_symbol_targets(cfg, processed_dir=str(processed_dir))
    enabled_markets = {normalize_market(x) for x in cfg_news.get("markets", ["crypto", "us_equity", "cn_equity"])}
    targets = [t for t in targets if normalize_market(t.get("market")) in enabled_markets]

    providers = [str(x).strip().lower() for x in cfg_news.get("providers", []) if str(x).strip()]
    if not providers:
        providers = _default_news_cfg()["providers"]

    provider_status: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    started = _utc_now()

    runners = {
        "alpha_vantage": lambda: _run_alpha_vantage(
            targets=targets,
            cfg_news=cfg_news,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
            limit=per_symbol_limit,
        ),
        "newsapi": lambda: _run_newsapi(
            targets=targets,
            cfg_news=cfg_news,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
            limit=per_symbol_limit,
        ),
        "gdelt": lambda: _run_gdelt(
            targets=targets,
            cfg_news=cfg_news,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
            limit=per_symbol_limit,
        ),
        "finnhub": lambda: _run_finnhub(
            targets=targets,
            cfg_news=cfg_news,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
        ),
        "marketaux": lambda: _run_marketaux(
            targets=targets,
            cfg_news=cfg_news,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
            limit=per_symbol_limit,
        ),
        "bybit_announcements": lambda: _run_bybit_announcements(
            targets=targets,
            cfg_news=cfg_news,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
        ),
        "binance_rss": lambda: _run_binance_rss(
            targets=targets,
            cfg_news=cfg_news,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
        ),
        "sec_atom": lambda: _run_sec_atom(
            targets=targets,
            cfg_news=cfg_news,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
        ),
    }

    for provider in providers:
        if provider not in runners:
            _append_provider_status(
                provider_status,
                provider=provider,
                ok=False,
                fetched=0,
                message="provider_not_supported",
            )
            continue
        try:
            rows = runners[provider]()
            all_rows.extend(rows)
            _append_provider_status(
                provider_status,
                provider=provider,
                ok=True,
                fetched=len(rows),
                message="ok",
            )
        except Exception as exc:
            _append_provider_status(
                provider_status,
                provider=provider,
                ok=False,
                fetched=0,
                message=f"{type(exc).__name__}: {exc}",
            )

    new_df = _prepare_news_df(all_rows, dedup_distance=dedup_distance)
    existing_df = _load_existing_news(out_paths["raw_csv"])
    merged_df = _merge_news(
        existing_df=existing_df,
        new_df=new_df,
        dedup_distance=dedup_distance,
        retention_days=retention_days,
        max_rows_total=max_rows_total,
    )

    market_counts = (
        merged_df.groupby("market").size().to_dict()
        if not merged_df.empty and "market" in merged_df.columns
        else {}
    )
    symbol_counts = (
        merged_df.groupby(["market", "symbol"]).size().reset_index(name="rows")
        if not merged_df.empty and {"market", "symbol"}.issubset(set(merged_df.columns))
        else pd.DataFrame(columns=["market", "symbol", "rows"])
    )
    top_symbols = symbol_counts.sort_values("rows", ascending=False).head(20).to_dict(orient="records")

    ended = _utc_now()
    status_payload: Dict[str, Any] = {
        "enabled": True,
        "config_path": config_path,
        "started_at_utc": started.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "finished_at_utc": ended.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "duration_sec": _safe_float((ended - started).total_seconds()),
        "targets_count": int(len(targets)),
        "targets_by_market": pd.Series([t.get("market", "") for t in targets]).value_counts().to_dict(),
        "providers_requested": providers,
        "providers": provider_status,
        "rows_fetched_this_run": int(len(new_df)),
        "rows_total_after_merge": int(len(merged_df)),
        "rows_by_market": market_counts,
        "top_symbols": top_symbols,
        "output_raw_csv": str(out_paths["raw_csv"]),
        "output_raw_parquet": str(out_paths["raw_parquet"]),
    }

    _write_news_outputs(
        merged_df=merged_df,
        status_rows=provider_status,
        paths=out_paths,
        status_payload=status_payload,
    )

    print(
        "[OK] news sentiment ingestion done: "
        f"new_rows={len(new_df)} total_rows={len(merged_df)} -> {out_paths['raw_csv']}"
    )
    return status_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest real-time news sentiment from open/freemium sources.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_news_sentiment(config_path=args.config)


if __name__ == "__main__":
    main()
