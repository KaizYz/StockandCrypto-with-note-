from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from src.ingestion.update_data import fetch_binance_klines
from src.utils.config import load_config
from src.utils.io import save_json, write_csv


YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_SUMMARY_URL = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"
EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
COINGECKO_SIMPLE_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

BINANCE_SYMBOL_TO_COINGECKO_ID = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
}


@dataclass
class SnapshotResult:
    instrument_id: str
    name: str
    market: str
    symbol: str
    provider: str
    timezone: str
    horizon_label: str
    current_price: float
    predicted_price: float
    predicted_change_pct: float
    predicted_change_abs: float
    expected_date_market: str
    expected_date_utc: str
    model_base_price: float
    q10_change_pct: float
    q50_change_pct: float
    q90_change_pct: float
    price_source: str
    prediction_method: str
    size_factor: float
    value_factor: float
    growth_factor: float
    momentum_factor: float
    reversal_factor: float
    low_vol_factor: float
    size_factor_source: str
    value_factor_source: str
    growth_factor_source: str
    market_cap_usd: float
    history_bars: float
    history_missing_rate: float
    generated_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "name": self.name,
            "market": self.market,
            "symbol": self.symbol,
            "provider": self.provider,
            "timezone": self.timezone,
            "horizon_label": self.horizon_label,
            "current_price": self.current_price,
            "predicted_price": self.predicted_price,
            "predicted_change_pct": self.predicted_change_pct,
            "predicted_change_abs": self.predicted_change_abs,
            "expected_date_market": self.expected_date_market,
            "expected_date_utc": self.expected_date_utc,
            "model_base_price": self.model_base_price,
            "q10_change_pct": self.q10_change_pct,
            "q50_change_pct": self.q50_change_pct,
            "q90_change_pct": self.q90_change_pct,
            "price_source": self.price_source,
            "prediction_method": self.prediction_method,
            "size_factor": self.size_factor,
            "value_factor": self.value_factor,
            "growth_factor": self.growth_factor,
            "momentum_factor": self.momentum_factor,
            "reversal_factor": self.reversal_factor,
            "low_vol_factor": self.low_vol_factor,
            "size_factor_source": self.size_factor_source,
            "value_factor_source": self.value_factor_source,
            "growth_factor_source": self.growth_factor_source,
            "market_cap_usd": self.market_cap_usd,
            "history_bars": self.history_bars,
            "history_missing_rate": self.history_missing_rate,
            "generated_at_utc": self.generated_at_utc,
        }


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _binance_to_coingecko_id(symbol: str) -> str:
    return BINANCE_SYMBOL_TO_COINGECKO_ID.get(symbol.upper(), "")


def _business_days_add(ts: pd.Timestamp, n_days: int) -> pd.Timestamp:
    out = ts
    added = 0
    while added < n_days:
        out = out + pd.Timedelta(days=1)
        if out.weekday() < 5:
            added += 1
    return out


def _calc_expected_date(
    now_utc: pd.Timestamp,
    timezone: str,
    market: str,
    horizon_unit: str,
    horizon_steps: int,
) -> Tuple[str, str]:
    if horizon_unit == "hour":
        exp_utc = now_utc + pd.Timedelta(hours=horizon_steps)
    else:
        if market in {"us_equity", "cn_equity"}:
            local = now_utc.tz_convert(timezone)
            exp_local = _business_days_add(local, horizon_steps)
            exp_utc = exp_local.tz_convert("UTC")
        else:
            exp_utc = now_utc + pd.Timedelta(days=horizon_steps)
    exp_market = exp_utc.tz_convert(timezone)
    return (
        exp_market.strftime("%Y-%m-%d %H:%M %z"),
        exp_utc.strftime("%Y-%m-%d %H:%M UTC"),
    )


def _fetch_binance_live_price(symbol: str, endpoints: List[str]) -> Tuple[float, str]:
    last_exc: Exception | None = None
    for endpoint in endpoints:
        url = f"{endpoint.rstrip('/')}/api/v3/ticker/price"
        try:
            r = requests.get(url, params={"symbol": symbol}, timeout=10)
            r.raise_for_status()
            data = r.json()
            return float(data["price"]), f"binance_ticker:{endpoint}"
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to fetch binance ticker for {symbol}: {last_exc}")


def _fetch_binance_daily_closes(
    symbol: str, endpoints: List[str], lookback_days: int
) -> Tuple[pd.Series, str]:
    end_utc = _now_utc().floor("h")
    start_utc = end_utc - pd.Timedelta(days=lookback_days)
    last_exc: Exception | None = None
    for endpoint in endpoints:
        try:
            df = fetch_binance_klines(
                symbol=symbol,
                interval="1d",
                start_ms=int(start_utc.timestamp() * 1000),
                end_ms=int(end_utc.timestamp() * 1000),
                base_url=endpoint,
            )
            if df.empty:
                raise RuntimeError("empty klines")
            close = pd.to_numeric(df["close"], errors="coerce").dropna().reset_index(drop=True)
            if len(close) < 30:
                raise RuntimeError("insufficient close bars")
            return close, f"binance_klines:{endpoint}"
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to fetch binance klines for {symbol}: {last_exc}")


def _fetch_coingecko_live_price(coin_id: str) -> Tuple[float, str]:
    params = {"ids": coin_id, "vs_currencies": "usd"}
    r = requests.get(COINGECKO_SIMPLE_URL, params=params, headers=DEFAULT_HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    if coin_id not in data or "usd" not in data[coin_id]:
        raise RuntimeError(f"CoinGecko simple price missing for {coin_id}")
    return float(data[coin_id]["usd"]), "coingecko_simple_price"


def _fetch_coingecko_daily_closes(coin_id: str, lookback_days: int) -> Tuple[pd.Series, str]:
    days = max(30, int(lookback_days))
    url = COINGECKO_MARKET_CHART_URL.format(coin_id=coin_id)
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    prices = data.get("prices", [])
    if not prices:
        raise RuntimeError(f"CoinGecko prices empty for {coin_id}")
    close = pd.Series([float(x[1]) for x in prices], dtype="float64").dropna().reset_index(drop=True)
    if len(close) < 30:
        raise RuntimeError(f"CoinGecko insufficient daily prices for {coin_id}")
    return close, "coingecko_market_chart"


def _fetch_coingecko_market_row(coin_id: str) -> Dict[str, Any]:
    params = {
        "vs_currency": "usd",
        "ids": coin_id,
        "order": "market_cap_desc",
        "per_page": 1,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "24h,7d,30d",
    }
    r = requests.get(COINGECKO_MARKETS_URL, params=params, headers=DEFAULT_HEADERS, timeout=20)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return {}
    return rows[0]


def _fetch_yahoo_live_price(symbol: str) -> Tuple[float, str]:
    r = requests.get(
        YAHOO_QUOTE_URL, params={"symbols": symbol}, headers=DEFAULT_HEADERS, timeout=12
    )
    r.raise_for_status()
    data = r.json()
    result = data.get("quoteResponse", {}).get("result", [])
    if not result:
        raise RuntimeError(f"Yahoo quote empty for {symbol}")
    price = result[0].get("regularMarketPrice")
    if price is None:
        raise RuntimeError(f"Yahoo quote missing regularMarketPrice for {symbol}")
    return float(price), "yahoo_quote"


def _extract_yahoo_fundamentals(row: Dict[str, Any]) -> Dict[str, float]:
    trailing_pe = _safe_float(row.get("trailingPE"))
    forward_pe = _safe_float(row.get("forwardPE"))
    price_to_book = _safe_float(row.get("priceToBook"))
    earnings_growth = _safe_float(row.get("earningsQuarterlyGrowth"))
    revenue_growth = _safe_float(row.get("revenueGrowth"))
    if not np.isfinite(earnings_growth):
        eps_trailing = _safe_float(row.get("epsTrailingTwelveMonths"))
        eps_forward = _safe_float(row.get("epsForward"))
        if np.isfinite(eps_trailing) and abs(eps_trailing) > 1e-9 and np.isfinite(eps_forward):
            earnings_growth = eps_forward / eps_trailing - 1.0
    return {
        "market_cap_usd": _safe_float(row.get("marketCap")),
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "price_to_book": price_to_book,
        "earnings_growth": earnings_growth,
        "revenue_growth": revenue_growth,
    }


def _summary_raw(payload: Dict[str, Any], module: str, field: str) -> float:
    val = payload.get(module, {}).get(field)
    if isinstance(val, dict):
        return _safe_float(val.get("raw"))
    return _safe_float(val)


def _fetch_yahoo_fundamentals_summary(symbol: str) -> Dict[str, float]:
    params = {"modules": "defaultKeyStatistics,financialData,summaryDetail"}
    url = YAHOO_SUMMARY_URL.format(symbol=symbol)
    r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    result = data.get("quoteSummary", {}).get("result", [])
    if not result:
        return {}
    payload = result[0]
    market_cap = _summary_raw(payload, "summaryDetail", "marketCap")
    if not np.isfinite(market_cap):
        market_cap = _summary_raw(payload, "defaultKeyStatistics", "marketCap")
    trailing_pe = _summary_raw(payload, "summaryDetail", "trailingPE")
    forward_pe = _summary_raw(payload, "summaryDetail", "forwardPE")
    if not np.isfinite(forward_pe):
        forward_pe = _summary_raw(payload, "defaultKeyStatistics", "forwardPE")
    price_to_book = _summary_raw(payload, "defaultKeyStatistics", "priceToBook")
    earnings_growth = _summary_raw(payload, "financialData", "earningsGrowth")
    revenue_growth = _summary_raw(payload, "financialData", "revenueGrowth")
    return {
        "market_cap_usd": market_cap,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "price_to_book": price_to_book,
        "earnings_growth": earnings_growth,
        "revenue_growth": revenue_growth,
    }


def _symbol_to_eastmoney_secid(symbol: str, market: str) -> str:
    if market == "cn_equity":
        code = symbol.split(".")[0]
        if str(code).startswith("6"):
            return f"1.{code}"
        return f"0.{code}"
    if market == "us_equity":
        ticker = symbol.replace("-", ".")
        return f"105.{ticker}"
    raise ValueError(f"Unsupported market for Eastmoney: {market}")


def _fetch_eastmoney_live_bundle(symbol: str, market: str) -> Tuple[float, str, Dict[str, float]]:
    secid = _symbol_to_eastmoney_secid(symbol, market)
    params = {"secid": secid, "fields": "f43,f57,f58,f116,f117,f162,f167"}
    r = requests.get(EASTMONEY_QUOTE_URL, params=params, headers=DEFAULT_HEADERS, timeout=12)
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data")
    if not data or data.get("f43") is None:
        raise RuntimeError(f"Eastmoney quote missing for secid={secid}")
    raw = float(data["f43"])
    if market == "cn_equity":
        price = raw / 100.0
    else:
        price = raw / 1000.0
    trailing_pe = _safe_float(data.get("f162"))
    if np.isfinite(trailing_pe) and trailing_pe > 0:
        trailing_pe = trailing_pe / 100.0
    price_to_book = _safe_float(data.get("f167"))
    if np.isfinite(price_to_book) and price_to_book > 0:
        price_to_book = price_to_book / 100.0
    fundamentals = {
        "market_cap_usd": _safe_float(data.get("f116")),
        "trailing_pe": trailing_pe,
        "forward_pe": float("nan"),
        "price_to_book": price_to_book,
        "earnings_growth": float("nan"),
        "revenue_growth": float("nan"),
    }
    return float(price), f"eastmoney_quote:{secid}", fundamentals


def _fetch_eastmoney_live_price(symbol: str, market: str) -> Tuple[float, str]:
    price, source, _ = _fetch_eastmoney_live_bundle(symbol, market)
    return price, source


def _fetch_yahoo_quote_bulk(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    if not symbols:
        return {}
    r = requests.get(
        YAHOO_QUOTE_URL,
        params={"symbols": ",".join(symbols)},
        headers=DEFAULT_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    result = data.get("quoteResponse", {}).get("result", [])
    out: Dict[str, Dict[str, Any]] = {}
    for row in result:
        symbol = row.get("symbol")
        if symbol:
            out[str(symbol)] = row
    return out


def _fetch_yahoo_daily_closes(symbol: str, lookback_days: int) -> Tuple[pd.Series, str]:
    now = _now_utc()
    period2 = int(now.timestamp())
    period1 = int((now - pd.Timedelta(days=lookback_days)).timestamp())
    url = YAHOO_CHART_URL.format(symbol=symbol)
    params = {
        "interval": "1d",
        "period1": period1,
        "period2": period2,
        "includePrePost": "false",
        "events": "div,splits",
    }
    r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    result = data.get("chart", {}).get("result", [])
    if not result:
        raise RuntimeError(f"Yahoo chart empty for {symbol}")
    quote = result[0].get("indicators", {}).get("quote", [])
    if not quote:
        raise RuntimeError(f"Yahoo chart quote missing for {symbol}")
    close = pd.Series(quote[0].get("close", []), dtype="float64").dropna().reset_index(drop=True)
    if len(close) < 30:
        raise RuntimeError(f"Insufficient close bars from Yahoo for {symbol}")
    return close, "yahoo_chart"


def _symbol_to_stooq(symbol: str, market: str) -> str:
    base = symbol.split(".")[0].lower()
    if market == "us_equity":
        return f"{base}.us"
    if market == "cn_equity":
        return f"{base}.cn"
    return base


def _fetch_stooq_daily_closes(symbol: str, market: str) -> Tuple[pd.Series, str]:
    stooq_symbol = _symbol_to_stooq(symbol, market)
    params = {"s": stooq_symbol, "i": "d"}
    r = requests.get(STOOQ_DAILY_URL, params=params, headers=DEFAULT_HEADERS, timeout=15)
    r.raise_for_status()
    text = r.text
    if "No data" in text or "Date,Open,High,Low,Close,Volume" not in text:
        raise RuntimeError(f"Stooq no data for {stooq_symbol}")
    df = pd.read_csv(pd.io.common.StringIO(text))
    if "Close" not in df.columns:
        raise RuntimeError(f"Stooq close missing for {stooq_symbol}")
    close = pd.to_numeric(df["Close"], errors="coerce").dropna().reset_index(drop=True)
    if len(close) < 30:
        raise RuntimeError(f"Stooq insufficient close bars for {stooq_symbol}")
    return close, f"stooq_chart:{stooq_symbol}"


def _predict_change_from_history(closes: pd.Series, horizon_steps: int) -> Dict[str, float]:
    ret = closes.pct_change().dropna()
    if ret.empty:
        return {"q10": -0.01, "q50": 0.0, "q90": 0.01}

    short = float(ret.tail(5).mean()) if len(ret) >= 5 else float(ret.mean())
    medium = float(ret.tail(20).mean()) if len(ret) >= 20 else float(ret.mean())
    vol = float(ret.tail(20).std(ddof=0)) if len(ret) >= 20 else float(ret.std(ddof=0) or 0.01)
    if not np.isfinite(vol) or vol <= 0:
        vol = 0.01

    one_step = 0.65 * short + 0.35 * medium
    clamp = 3.0 * vol
    one_step = float(np.clip(one_step, -clamp, clamp))
    q50 = float((1.0 + one_step) ** horizon_steps - 1.0)
    sigma_h = vol * np.sqrt(max(horizon_steps, 1))
    q10 = float(q50 - 1.28 * sigma_h)
    q90 = float(q50 + 1.28 * sigma_h)
    return {"q10": q10, "q50": q50, "q90": q90}


def _compute_factor_values(
    closes: pd.Series,
    fundamentals: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    fundamentals = fundamentals or {}
    close = pd.to_numeric(closes, errors="coerce").dropna().reset_index(drop=True)
    ret = close.pct_change().dropna()
    if close.empty:
        return {
            "size_factor": float("nan"),
            "value_factor": float("nan"),
            "growth_factor": float("nan"),
            "momentum_factor": float("nan"),
            "reversal_factor": float("nan"),
            "low_vol_factor": float("nan"),
            "size_factor_source": "na",
            "value_factor_source": "na",
            "growth_factor_source": "na",
            "market_cap_usd": float("nan"),
        }

    # Market behavior factors (time-series proxies).
    momentum = _safe_float(close.pct_change(20).iloc[-1]) if len(close) >= 21 else _safe_float(ret.mean())
    reversal = (
        -_safe_float(close.pct_change(5).iloc[-1]) if len(close) >= 6 else -_safe_float(ret.tail(3).mean())
    )
    low_vol = -_safe_float(ret.tail(20).std(ddof=0)) if len(ret) >= 5 else float("nan")

    # Risk factors with fundamental-first fallback to price-based proxy.
    market_cap_usd = _safe_float(fundamentals.get("market_cap_usd"))
    if np.isfinite(market_cap_usd) and market_cap_usd > 0:
        size_factor = float(np.log(market_cap_usd))
        size_source = "market_cap_log"
    else:
        size_factor = float(np.log(max(_safe_float(close.iloc[-1]), 1e-8)))
        size_source = "proxy_log_price"

    trailing_pe = _safe_float(fundamentals.get("trailing_pe"))
    forward_pe = _safe_float(fundamentals.get("forward_pe"))
    price_to_book = _safe_float(fundamentals.get("price_to_book"))
    if np.isfinite(trailing_pe) and trailing_pe > 0:
        value_factor = 1.0 / trailing_pe
        value_source = "earnings_yield_1_over_trailing_pe"
    elif np.isfinite(forward_pe) and forward_pe > 0:
        value_factor = 1.0 / forward_pe
        value_source = "earnings_yield_1_over_forward_pe"
    elif np.isfinite(price_to_book) and price_to_book > 0:
        value_factor = 1.0 / price_to_book
        value_source = "book_to_price_1_over_pb"
    else:
        lookback = close.tail(252) if len(close) >= 30 else close
        peak = _safe_float(lookback.max())
        last = _safe_float(close.iloc[-1])
        value_factor = (peak - last) / peak if np.isfinite(peak) and peak > 0 else float("nan")
        value_source = "proxy_drawdown_to_1y_high"

    earnings_growth = _safe_float(fundamentals.get("earnings_growth"))
    revenue_growth = _safe_float(fundamentals.get("revenue_growth"))
    if np.isfinite(earnings_growth):
        growth_factor = earnings_growth
        growth_source = "earnings_growth"
    elif np.isfinite(revenue_growth):
        growth_factor = revenue_growth
        growth_source = "revenue_growth"
    else:
        growth_factor = _safe_float(close.pct_change(90).iloc[-1]) if len(close) >= 91 else _safe_float(ret.mean())
        growth_source = "proxy_return_90d"

    return {
        "size_factor": _safe_float(size_factor),
        "value_factor": _safe_float(value_factor),
        "growth_factor": _safe_float(growth_factor),
        "momentum_factor": _safe_float(momentum),
        "reversal_factor": _safe_float(reversal),
        "low_vol_factor": _safe_float(low_vol),
        "size_factor_source": size_source,
        "value_factor_source": value_source,
        "growth_factor_source": growth_source,
        "market_cap_usd": market_cap_usd,
    }


def _build_snapshot_rows(instruments: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    binance_endpoints = cfg.get("data", {}).get(
        "binance_endpoints", ["https://api.binance.com", "https://api.binance.us"]
    )
    if bool(cfg.get("data", {}).get("disable_binance_us", False)):
        filtered = [ep for ep in binance_endpoints if "binance.us" not in str(ep).lower()]
        if filtered:
            binance_endpoints = filtered
    now_utc = _now_utc()
    rows: List[Dict[str, Any]] = []
    yahoo_symbols = [
        str(inst.get("symbol", ""))
        for inst in instruments
        if str(inst.get("provider", "")) == "yahoo" and str(inst.get("symbol", ""))
    ]
    yahoo_quote_map: Dict[str, Dict[str, Any]] = {}
    yahoo_live_map: Dict[str, float] = {}
    try:
        yahoo_quote_map = _fetch_yahoo_quote_bulk(yahoo_symbols)
        yahoo_live_map = {
            symbol: _safe_float(row.get("regularMarketPrice"))
            for symbol, row in yahoo_quote_map.items()
            if np.isfinite(_safe_float(row.get("regularMarketPrice")))
        }
    except Exception:
        yahoo_quote_map = {}
        yahoo_live_map = {}

    for inst in instruments:
        inst_id = str(inst.get("id", "unknown"))
        name = str(inst.get("name", inst_id))
        market = str(inst.get("market", "unknown"))
        symbol = str(inst.get("symbol", ""))
        provider = str(inst.get("provider", ""))
        timezone = str(inst.get("timezone", "UTC"))
        horizon_unit = str(inst.get("horizon_unit", "day"))
        horizon_steps = int(inst.get("horizon_steps", 1))
        lookback_days = int(inst.get("history_lookback_days", 1825))
        horizon_label = f"{horizon_steps}{'h' if horizon_unit == 'hour' else 'd'}"

        try:
            fundamentals: Dict[str, float] = {}
            if provider == "binance":
                live_price, price_source = _fetch_binance_live_price(symbol, binance_endpoints)
                closes, _ = _fetch_binance_daily_closes(
                    symbol=symbol,
                    endpoints=binance_endpoints,
                    lookback_days=lookback_days,
                )
                coin_id = _binance_to_coingecko_id(symbol)
                if coin_id:
                    cg_row = _fetch_coingecko_market_row(coin_id)
                    fundamentals = {
                        "market_cap_usd": _safe_float(cg_row.get("market_cap")),
                        "trailing_pe": float("nan"),
                        "forward_pe": float("nan"),
                        "price_to_book": float("nan"),
                        "earnings_growth": float("nan"),
                        "revenue_growth": _safe_float(
                            cg_row.get("price_change_percentage_30d_in_currency")
                        )
                        / 100.0,
                    }
            elif provider == "coingecko":
                # For coingecko provider, `symbol` is coin id (e.g., "bitcoin", "ethereum").
                live_price, price_source = _fetch_coingecko_live_price(symbol)
                closes, _ = _fetch_coingecko_daily_closes(symbol, lookback_days=lookback_days)
                cg_row = _fetch_coingecko_market_row(symbol)
                fundamentals = {
                    "market_cap_usd": _safe_float(cg_row.get("market_cap")),
                    "trailing_pe": float("nan"),
                    "forward_pe": float("nan"),
                    "price_to_book": float("nan"),
                    "earnings_growth": float("nan"),
                    "revenue_growth": _safe_float(cg_row.get("price_change_percentage_30d_in_currency"))
                    / 100.0,
                }
            elif provider == "yahoo":
                price_source = "yahoo_quote"
                closes_source = "yahoo_chart"
                try:
                    closes, closes_source = _fetch_yahoo_daily_closes(symbol, lookback_days=lookback_days)
                except Exception:
                    closes, closes_source = _fetch_stooq_daily_closes(symbol, market=market)
                fundamentals = _extract_yahoo_fundamentals(yahoo_quote_map.get(symbol, {}))
                if not np.isfinite(_safe_float(fundamentals.get("market_cap_usd"))):
                    try:
                        summary_fund = _fetch_yahoo_fundamentals_summary(symbol)
                        for k, v in summary_fund.items():
                            if np.isfinite(_safe_float(v)):
                                fundamentals[k] = _safe_float(v)
                    except Exception:
                        pass

                if market in {"cn_equity", "us_equity"}:
                    try:
                        live_price, price_source, eastmoney_fund = _fetch_eastmoney_live_bundle(
                            symbol, market=market
                        )
                        for k, v in eastmoney_fund.items():
                            if not np.isfinite(_safe_float(fundamentals.get(k))) and np.isfinite(_safe_float(v)):
                                fundamentals[k] = _safe_float(v)
                    except Exception:
                        if symbol in yahoo_live_map:
                            live_price = float(yahoo_live_map[symbol])
                            price_source = "yahoo_quote"
                        else:
                            live_price = float(closes.iloc[-1])
                            price_source = closes_source.replace("chart", "latest")
                else:
                    if symbol in yahoo_live_map:
                        live_price = float(yahoo_live_map[symbol])
                    else:
                        # Yahoo quote may be rate-limited; fallback to latest available close.
                        live_price = float(closes.iloc[-1])
                        price_source = closes_source.replace("chart", "latest")
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            quant = _predict_change_from_history(closes, horizon_steps=horizon_steps)
            factors = _compute_factor_values(closes, fundamentals=fundamentals)
            q10, q50, q90 = quant["q10"], quant["q50"], quant["q90"]
            pred_price = float(live_price * (1.0 + q50))
            delta_abs = float(pred_price - live_price)
            history_bars = float(len(closes))
            if market in {"cn_equity", "us_equity"}:
                expected_bars = float(max(int(lookback_days * 0.72), 1))
            else:
                expected_bars = float(max(int(lookback_days * 0.95), 1))
            history_missing_rate = float(max(0.0, 1.0 - history_bars / expected_bars))
            exp_market, exp_utc = _calc_expected_date(
                now_utc=now_utc,
                timezone=timezone,
                market=market,
                horizon_unit=horizon_unit,
                horizon_steps=horizon_steps,
            )

            result = SnapshotResult(
                instrument_id=inst_id,
                name=name,
                market=market,
                symbol=symbol,
                provider=provider,
                timezone=timezone,
                horizon_label=horizon_label,
                current_price=float(live_price),
                predicted_price=pred_price,
                predicted_change_pct=q50,
                predicted_change_abs=delta_abs,
                expected_date_market=exp_market,
                expected_date_utc=exp_utc,
                model_base_price=float(live_price),
                q10_change_pct=q10,
                q50_change_pct=q50,
                q90_change_pct=q90,
                price_source=price_source,
                prediction_method="baseline_momentum_quantile",
                size_factor=float(factors.get("size_factor", np.nan)),
                value_factor=float(factors.get("value_factor", np.nan)),
                growth_factor=float(factors.get("growth_factor", np.nan)),
                momentum_factor=float(factors.get("momentum_factor", np.nan)),
                reversal_factor=float(factors.get("reversal_factor", np.nan)),
                low_vol_factor=float(factors.get("low_vol_factor", np.nan)),
                size_factor_source=str(factors.get("size_factor_source", "na")),
                value_factor_source=str(factors.get("value_factor_source", "na")),
                growth_factor_source=str(factors.get("growth_factor_source", "na")),
                market_cap_usd=float(factors.get("market_cap_usd", np.nan)),
                history_bars=history_bars,
                history_missing_rate=history_missing_rate,
                generated_at_utc=now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            )
            rows.append(result.to_dict())
        except Exception as exc:
            rows.append(
                {
                    "instrument_id": inst_id,
                    "name": name,
                    "market": market,
                    "symbol": symbol,
                    "provider": provider,
                    "timezone": timezone,
                    "horizon_label": horizon_label,
                    "current_price": np.nan,
                    "predicted_price": np.nan,
                    "predicted_change_pct": np.nan,
                    "predicted_change_abs": np.nan,
                    "expected_date_market": "-",
                    "expected_date_utc": "-",
                    "model_base_price": np.nan,
                    "q10_change_pct": np.nan,
                    "q50_change_pct": np.nan,
                    "q90_change_pct": np.nan,
                    "size_factor": np.nan,
                    "value_factor": np.nan,
                    "growth_factor": np.nan,
                    "momentum_factor": np.nan,
                    "reversal_factor": np.nan,
                    "low_vol_factor": np.nan,
                    "size_factor_source": "na",
                    "value_factor_source": "na",
                    "growth_factor_source": "na",
                    "market_cap_usd": np.nan,
                    "history_bars": np.nan,
                    "history_missing_rate": np.nan,
                    "price_source": f"error:{type(exc).__name__}",
                    "prediction_method": "baseline_momentum_quantile",
                    "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "error_message": str(exc),
                }
            )

    return rows


def build_market_snapshot_from_instruments(
    instruments: List[Dict[str, Any]], config_path: str = "configs/config.yaml"
) -> pd.DataFrame:
    cfg = load_config(config_path)
    if not instruments:
        return pd.DataFrame()
    rows = _build_snapshot_rows(instruments, cfg)
    return pd.DataFrame(rows)


def build_market_snapshot(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)
    market_cfg = cfg.get("market_snapshot", {})
    if not market_cfg.get("enabled", True):
        return pd.DataFrame()
    instruments = market_cfg.get("instruments", [])
    rows = _build_snapshot_rows(instruments, cfg)
    return pd.DataFrame(rows)


def run_market_snapshot(config_path: str) -> None:
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    market_cfg = cfg.get("market_snapshot", {})
    processed_dir = Path(paths_cfg.get("processed_data_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = build_market_snapshot(config_path)
    if df.empty:
        print("[WARN] market_snapshot is disabled or returned empty.")
        return

    out_csv = Path(market_cfg.get("output_csv", processed_dir / "market_snapshot.csv"))
    out_json = Path(market_cfg.get("output_json", processed_dir / "market_snapshot.json"))
    write_csv(df, out_csv)
    save_json({"rows": df.to_dict(orient="records")}, out_json)
    print(f"[OK] Saved market snapshot -> {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build live market snapshot (crypto + A-share + US).")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_market_snapshot(args.config)


if __name__ == "__main__":
    main()
