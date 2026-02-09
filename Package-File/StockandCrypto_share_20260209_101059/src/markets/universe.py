from __future__ import annotations

from io import StringIO
from typing import Dict

import pandas as pd
import requests

try:
    import akshare as ak
except Exception:
    ak = None


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"

# Common stablecoins to exclude from "top market cap" crypto selection.
STABLECOIN_IDS = {
    "tether",
    "usd-coin",
    "dai",
    "true-usd",
    "first-digital-usd",
    "usdd",
    "pax-dollar",
    "paypal-usd",
    "frax",
    "liquity-usd",
    "gemini-dollar",
    "s-usde",
    "ethena-usde",
    "binance-bridged-usdt-bnb-smart-chain",
    "binance-peg-busd",
    "terrausd",
    "usdk",
    "fei-usd",
    "euro-coin",
}

STABLECOIN_SYMBOLS = {
    "usdt",
    "usdc",
    "dai",
    "fdusd",
    "tusd",
    "usdd",
    "usdp",
    "pyusd",
    "gusd",
    "frax",
    "lusd",
    "usde",
    "susde",
    "busd",
    "ust",
    "ustc",
}


def _cn_code_to_yahoo_symbol(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith("6"):
        return f"{code}.SS"
    return f"{code}.SZ"


def get_crypto_universe() -> pd.DataFrame:
    rows = [
        {
            "symbol": "BTC",
            "name": "Bitcoin",
            "display": "BTC / Bitcoin",
            "snapshot_symbol": "BTCUSDT",
            "provider": "binance",
        },
        {
            "symbol": "ETH",
            "name": "Ethereum",
            "display": "ETH / Ethereum",
            "snapshot_symbol": "ETHUSDT",
            "provider": "binance",
        },
        {
            "symbol": "SOL",
            "name": "Solana",
            "display": "SOL / Solana",
            "snapshot_symbol": "SOLUSDT",
            "provider": "binance",
        },
    ]
    return pd.DataFrame(rows)


def _is_stablecoin(row: Dict[str, str]) -> bool:
    coin_id = str(row.get("id", "")).lower()
    symbol = str(row.get("symbol", "")).lower()
    name = str(row.get("name", "")).lower()
    if coin_id in STABLECOIN_IDS:
        return True
    if symbol in STABLECOIN_SYMBOLS:
        return True
    # Conservative heuristic for uncatalogued stables.
    if "stable" in name and "coin" in name:
        return True
    if "usd" in symbol and len(symbol) <= 6 and symbol != "busd":
        return True
    return False


def get_crypto_top100_ex_stable() -> pd.DataFrame:
    kept: list[Dict[str, str]] = []
    page = 1
    while len(kept) < 100 and page <= 4:
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": page,
            "sparkline": "false",
            "price_change_percentage": "24h",
        }
        r = requests.get(COINGECKO_MARKETS_URL, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        for row in rows:
            if _is_stablecoin(row):
                continue
            kept.append(
                {
                    "symbol": str(row.get("symbol", "")).upper(),
                    "name": str(row.get("name", "")),
                    "display": f"{str(row.get('symbol', '')).upper()} / {str(row.get('name', ''))}",
                    # CoinGecko provider uses coin id as query symbol.
                    "snapshot_symbol": str(row.get("id", "")),
                    "provider": "coingecko",
                    "market_cap_rank": int(row.get("market_cap_rank") or 999999),
                }
            )
            if len(kept) >= 100:
                break
        page += 1

    df = pd.DataFrame(kept)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["snapshot_symbol"]).sort_values("market_cap_rank").head(100)
    return df.reset_index(drop=True)


def get_cn_universe(index_code: str) -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("akshare is not installed; cannot fetch A-share constituents.")
    df = ak.index_stock_cons_csindex(symbol=index_code)
    code_col = "成分券代码"
    name_col = "成分券名称"
    if code_col not in df.columns or name_col not in df.columns:
        raise RuntimeError("Unexpected A-share constituent schema from akshare.")

    out = pd.DataFrame(
        {
            "symbol": df[code_col].astype(str).str.zfill(6).map(_cn_code_to_yahoo_symbol),
            "code": df[code_col].astype(str).str.zfill(6),
            "name": df[name_col].astype(str),
        }
    )
    out["display"] = out["code"] + " " + out["name"]
    out = out.drop_duplicates(subset=["symbol"]).sort_values("display").reset_index(drop=True)
    return out


def _read_html_with_headers(url: str) -> list[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def get_us_universe(pool: str) -> pd.DataFrame:
    key = pool.lower()
    if key == "dow30":
        tables = _read_html_with_headers("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
        table = tables[2]  # validated by local probe
        symbol_col, name_col = "Symbol", "Company"
    elif key == "nasdaq100":
        tables = _read_html_with_headers("https://en.wikipedia.org/wiki/Nasdaq-100")
        table = tables[4]  # validated by local probe
        symbol_col, name_col = "Ticker", "Company"
    elif key == "sp500":
        tables = _read_html_with_headers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        table = tables[0]
        symbol_col, name_col = "Symbol", "Security"
    else:
        raise ValueError(f"Unsupported US pool: {pool}")

    out = pd.DataFrame(
        {
            "symbol": table[symbol_col].astype(str).str.replace(".", "-", regex=False),
            "name": table[name_col].astype(str),
        }
    )
    out["display"] = out["symbol"] + " " + out["name"]
    out = out.drop_duplicates(subset=["symbol"]).sort_values("symbol").reset_index(drop=True)
    return out


def get_universe_catalog() -> Dict[str, Dict[str, str]]:
    return {
        "crypto": {
            "top3": "Crypto Top 3 (BTC/ETH/SOL)",
            "top100_ex_stable": "Crypto 市值前100（剔除稳定币）",
        },
        "cn_equity": {
            "sse_composite": "上证指数成分股 (000001)",
            "csi300": "沪深300成分股 (000300)",
        },
        "us_equity": {
            "dow30": "道琼斯30成分股",
            "nasdaq100": "纳斯达克100成分股",
            "sp500": "标普500成分股",
        },
    }


def load_universe(market: str, pool_key: str) -> pd.DataFrame:
    if market == "crypto":
        if pool_key == "top3":
            return get_crypto_universe()
        if pool_key == "top100_ex_stable":
            return get_crypto_top100_ex_stable()
        raise ValueError(f"Unsupported crypto pool: {pool_key}")
    if market == "cn_equity":
        if pool_key == "sse_composite":
            return get_cn_universe("000001")
        if pool_key == "csi300":
            return get_cn_universe("000300")
        raise ValueError(f"Unsupported CN pool: {pool_key}")
    if market == "us_equity":
        return get_us_universe(pool_key)
    raise ValueError(f"Unsupported market: {market}")
