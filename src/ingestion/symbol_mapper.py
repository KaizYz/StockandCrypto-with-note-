from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


_CRYPTO_ALIAS: Dict[str, List[str]] = {
    "BTC": ["bitcoin", "btc", "xbt"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "BNB": ["bnb", "binance coin"],
    "XRP": ["xrp", "ripple"],
    "DOGE": ["dogecoin", "doge"],
    "ADA": ["cardano", "ada"],
}


def normalize_market(value: Any) -> str:
    key = str(value or "").strip().lower()
    aliases = {
        "cn_a": "cn_equity",
        "a_share": "cn_equity",
        "a_shares": "cn_equity",
        "ashares": "cn_equity",
        "us": "us_equity",
        "equity_us": "us_equity",
        "stock_us": "us_equity",
    }
    return aliases.get(key, key)


def canonical_symbol(symbol: Any, market: Any) -> str:
    mk = normalize_market(market)
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""

    if mk == "crypto":
        out = raw.replace("-", "").replace("_", "").replace("/", "")
        if out.endswith("USDT"):
            out = out[: -len("USDT")]
        if out.endswith("USD"):
            out = out[: -len("USD")]
        return out

    if mk == "cn_equity":
        if "." in raw:
            raw = raw.split(".")[0]
        return raw

    if mk == "us_equity":
        return raw.replace("-", ".")

    return raw


def symbol_aliases(symbol: Any, market: Any, name: Any = "") -> List[str]:
    mk = normalize_market(market)
    sym = canonical_symbol(symbol, mk)
    aliases: List[str] = []
    if sym:
        aliases.append(sym.lower())
    if name:
        aliases.append(str(name).strip().lower())

    if mk == "crypto":
        aliases.extend(_CRYPTO_ALIAS.get(sym, []))
        if sym:
            aliases.append(f"{sym.lower()}usdt")

    out: List[str] = []
    seen = set()
    for item in aliases:
        token = str(item or "").strip().lower()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def symbol_to_query(symbol: Any, market: Any, name: Any = "") -> str:
    aliases = symbol_aliases(symbol=symbol, market=market, name=name)
    if not aliases:
        return ""
    if len(aliases) == 1:
        return aliases[0]
    return " OR ".join(f'"{x}"' if " " in x else x for x in aliases[:6])


def collect_symbol_targets(
    cfg: Dict[str, Any],
    *,
    processed_dir: str = "data/processed",
) -> List[Dict[str, str]]:
    news_cfg = cfg.get("news", {}) if isinstance(cfg, dict) else {}
    max_by_market = news_cfg.get(
        "max_symbols_per_market",
        {"crypto": 15, "us_equity": 20, "cn_equity": 20},
    )
    enabled_markets = {
        normalize_market(m)
        for m in news_cfg.get("markets", ["crypto", "us_equity"])
    }
    ranked_path = Path(processed_dir) / "tracking" / "ranked_universe.csv"
    rows: List[Dict[str, str]] = []
    if ranked_path.exists():
        ranked = pd.read_csv(ranked_path)
        ranked = ranked.copy()
        ranked["market"] = ranked.get("market", "").map(normalize_market)
        for mk in sorted(enabled_markets):
            sub = ranked[ranked["market"] == mk].copy()
            max_n = int(max_by_market.get(mk, 0))
            if max_n > 0:
                sub = sub.head(max_n)
            for _, r in sub.iterrows():
                sym = str(r.get("symbol", "") or r.get("snapshot_symbol", "")).strip()
                name = str(r.get("name", "") or r.get("display", "")).strip()
                if not sym:
                    continue
                rows.append(
                    {
                        "market": mk,
                        "symbol": canonical_symbol(sym, mk),
                        "name": name,
                        "query": symbol_to_query(sym, mk, name),
                    }
                )

    if not rows:
        forecast_symbols = (
            cfg.get("forecast_config", {})
            .get("symbols", {})
            .get("default", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        )
        if "crypto" in enabled_markets:
            for sym in forecast_symbols[: int(max_by_market.get("crypto", 15))]:
                rows.append(
                    {
                        "market": "crypto",
                        "symbol": canonical_symbol(sym, "crypto"),
                        "name": str(sym).upper(),
                        "query": symbol_to_query(sym, "crypto", sym),
                    }
                )

        for inst in cfg.get("market_snapshot", {}).get("instruments", []):
            mk = normalize_market(inst.get("market"))
            if mk not in enabled_markets:
                continue
            rows.append(
                {
                    "market": mk,
                    "symbol": canonical_symbol(inst.get("symbol", ""), mk),
                    "name": str(inst.get("name", "")),
                    "query": symbol_to_query(
                        inst.get("symbol", ""),
                        mk,
                        inst.get("name", ""),
                    ),
                }
            )

    out: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for row in rows:
        mk = normalize_market(row.get("market"))
        sym = canonical_symbol(row.get("symbol"), mk)
        if not mk or not sym:
            continue
        key = (mk, sym)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "market": mk,
                "symbol": sym,
                "name": str(row.get("name", "")).strip(),
                "query": str(row.get("query", "")).strip(),
            }
        )
    return out


def map_text_to_symbol_candidates(
    text: Any,
    targets: Iterable[Dict[str, str]],
    *,
    market: str = "",
) -> List[Dict[str, str]]:
    blob = str(text or "").strip().lower()
    if not blob:
        return []
    mk = normalize_market(market)
    matched: List[Dict[str, str]] = []
    for tgt in targets:
        tgt_mk = normalize_market(tgt.get("market"))
        if mk and tgt_mk != mk:
            continue
        aliases = symbol_aliases(
            tgt.get("symbol", ""),
            tgt_mk,
            tgt.get("name", ""),
        )
        if any(a and a in blob for a in aliases):
            matched.append(
                {
                    "market": tgt_mk,
                    "symbol": canonical_symbol(tgt.get("symbol"), tgt_mk),
                    "name": str(tgt.get("name", "")),
                }
            )
    return matched
