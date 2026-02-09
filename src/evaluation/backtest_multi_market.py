from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from src.ingestion.update_data import fetch_binance_klines
from src.markets.universe import load_universe
from src.models.policy import apply_policy_frame, get_policy_config
from src.utils.config import load_config, save_yaml
from src.utils.io import save_json, write_csv


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
COINGECKO_MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _annual_factor_for_market(market: str) -> float:
    if str(market) == "crypto":
        return 365.0
    return 252.0


def _compute_supertrend_direction(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    multiplier: float = 3.0,
) -> pd.Series:
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    a = pd.to_numeric(atr, errors="coerce")
    hl2 = (h + l) / 2.0
    upper_basic = hl2 + multiplier * a
    lower_basic = hl2 - multiplier * a

    final_upper = upper_basic.copy()
    final_lower = lower_basic.copy()
    direction = pd.Series(index=c.index, dtype=float)

    for i in range(len(c)):
        if i == 0:
            direction.iloc[i] = 1.0
            continue
        pu = final_upper.iloc[i - 1]
        pl = final_lower.iloc[i - 1]
        pc = c.iloc[i - 1]
        cu = upper_basic.iloc[i]
        cl = lower_basic.iloc[i]
        if np.isfinite(cu):
            final_upper.iloc[i] = cu if (not np.isfinite(pu) or cu < pu or pc > pu) else pu
        else:
            final_upper.iloc[i] = pu
        if np.isfinite(cl):
            final_lower.iloc[i] = cl if (not np.isfinite(pl) or cl > pl or pc < pl) else pl
        else:
            final_lower.iloc[i] = pl

        prev_dir = direction.iloc[i - 1]
        if not np.isfinite(prev_dir):
            prev_dir = 1.0
        if prev_dir > 0:
            direction.iloc[i] = -1.0 if c.iloc[i] < final_lower.iloc[i] else 1.0
        else:
            direction.iloc[i] = 1.0 if c.iloc[i] > final_upper.iloc[i] else -1.0

    return direction.ffill().fillna(1.0)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=max(int(span), 1), adjust=False).mean()


def _atr_from_bars(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df.get("high"), errors="coerce")
    low = pd.to_numeric(df.get("low"), errors="coerce")
    close = pd.to_numeric(df.get("close"), errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(max(int(period), 1)).mean()


def _atr_wilder_from_bars(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df.get("high"), errors="coerce")
    low = pd.to_numeric(df.get("low"), errors="coerce")
    close = pd.to_numeric(df.get("close"), errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    p = max(int(period), 1)
    atr = tr.ewm(alpha=1.0 / p, adjust=False, min_periods=p).mean()
    return atr


def _compute_pine_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    factor: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    h = pd.to_numeric(high, errors="coerce").to_numpy(dtype=float)
    l = pd.to_numeric(low, errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(atr, errors="coerce").to_numpy(dtype=float)
    n = len(c)

    final_upper = np.full(n, np.nan, dtype=float)
    final_lower = np.full(n, np.nan, dtype=float)
    supertrend = np.full(n, np.nan, dtype=float)
    direction = np.full(n, np.nan, dtype=float)

    for i in range(n):
        src = (h[i] + l[i]) / 2.0 if np.isfinite(h[i]) and np.isfinite(l[i]) else np.nan
        upper_band = src + factor * a[i] if np.isfinite(src) and np.isfinite(a[i]) else np.nan
        lower_band = src - factor * a[i] if np.isfinite(src) and np.isfinite(a[i]) else np.nan

        prev_upper = 0.0
        prev_lower = 0.0
        prev_super = np.nan
        prev_close = np.nan
        if i > 0:
            prev_upper = final_upper[i - 1] if np.isfinite(final_upper[i - 1]) else 0.0
            prev_lower = final_lower[i - 1] if np.isfinite(final_lower[i - 1]) else 0.0
            prev_super = supertrend[i - 1]
            prev_close = c[i - 1]

        if np.isfinite(lower_band):
            keep_lower = (lower_band > prev_lower) or (np.isfinite(prev_close) and prev_close < prev_lower)
            final_lower[i] = lower_band if keep_lower else prev_lower
        else:
            final_lower[i] = prev_lower if i > 0 else np.nan

        if np.isfinite(upper_band):
            keep_upper = (upper_band < prev_upper) or (np.isfinite(prev_close) and prev_close > prev_upper)
            final_upper[i] = upper_band if keep_upper else prev_upper
        else:
            final_upper[i] = prev_upper if i > 0 else np.nan

        prev_atr = a[i - 1] if i > 0 else np.nan
        if not np.isfinite(prev_atr):
            direction[i] = 1.0
        elif np.isfinite(prev_super) and np.isfinite(prev_upper) and prev_super == prev_upper:
            direction[i] = -1.0 if (np.isfinite(c[i]) and np.isfinite(final_upper[i]) and c[i] > final_upper[i]) else 1.0
        else:
            direction[i] = 1.0 if (np.isfinite(c[i]) and np.isfinite(final_lower[i]) and c[i] < final_lower[i]) else -1.0

        supertrend[i] = final_lower[i] if direction[i] == -1.0 else final_upper[i]

    return pd.Series(supertrend, index=close.index, dtype=float), pd.Series(direction, index=close.index, dtype=float)


def _compute_lux_machine_supertrend(
    df: pd.DataFrame,
    *,
    atr_len: int = 10,
    factor: float = 3.0,
    training_data_period: int = 100,
    highvol: float = 0.75,
    midvol: float = 0.50,
    lowvol: float = 0.25,
    max_iter: int = 50,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index.copy())
    atr = _atr_wilder_from_bars(df, period=atr_len)
    n = len(df)

    assigned_atr = np.full(n, np.nan, dtype=float)
    cluster_idx = np.full(n, np.nan, dtype=float)
    centroid_high = np.full(n, np.nan, dtype=float)
    centroid_mid = np.full(n, np.nan, dtype=float)
    centroid_low = np.full(n, np.nan, dtype=float)

    lookback = max(int(training_data_period), 3)
    hv_guess = float(np.clip(highvol, 0.0, 1.0))
    mv_guess = float(np.clip(midvol, 0.0, 1.0))
    lv_guess = float(np.clip(lowvol, 0.0, 1.0))
    max_steps = max(int(max_iter), 1)
    atr_np = pd.to_numeric(atr, errors="coerce").to_numpy(dtype=float)

    for i in range(n):
        v = atr_np[i]
        if not np.isfinite(v) or i < (lookback - 1):
            continue
        window = atr_np[i - lookback + 1 : i + 1]
        window = window[np.isfinite(window)]
        if window.size < 3:
            assigned_atr[i] = v
            continue

        upper = float(np.nanmax(window))
        lower = float(np.nanmin(window))
        spread = upper - lower
        c = np.array(
            [
                lower + spread * hv_guess,
                lower + spread * mv_guess,
                lower + spread * lv_guess,
            ],
            dtype=float,
        )
        if not np.isfinite(c).all():
            med = float(np.nanmedian(window))
            c = np.array([med, med, med], dtype=float)

        for _ in range(max_steps):
            d = np.abs(window[:, None] - c[None, :])
            cls = np.argmin(d, axis=1)
            new_c = c.copy()
            for k in range(3):
                pts = window[cls == k]
                if pts.size > 0:
                    new_c[k] = float(np.nanmean(pts))
            if np.allclose(new_c, c, atol=1e-10, rtol=0.0, equal_nan=True):
                c = new_c
                break
            c = new_c

        d_curr = np.abs(c - v)
        k_curr = int(np.argmin(d_curr))
        assigned_atr[i] = c[k_curr]
        cluster_idx[i] = float(k_curr)
        centroid_high[i] = c[0]
        centroid_mid[i] = c[1]
        centroid_low[i] = c[2]

    st, direction = _compute_pine_supertrend(
        pd.to_numeric(df.get("high"), errors="coerce"),
        pd.to_numeric(df.get("low"), errors="coerce"),
        pd.to_numeric(df.get("close"), errors="coerce"),
        pd.Series(assigned_atr, index=df.index, dtype=float),
        factor=float(factor),
    )
    out["lux_ms_atr"] = atr
    out["lux_ms_assigned_atr"] = pd.Series(assigned_atr, index=df.index, dtype=float)
    out["lux_ms_cluster"] = pd.Series(cluster_idx, index=df.index, dtype=float)
    out["lux_ms_centroid_high"] = pd.Series(centroid_high, index=df.index, dtype=float)
    out["lux_ms_centroid_mid"] = pd.Series(centroid_mid, index=df.index, dtype=float)
    out["lux_ms_centroid_low"] = pd.Series(centroid_low, index=df.index, dtype=float)
    out["lux_ms_supertrend"] = st
    out["lux_ms_direction"] = direction
    # Pine direction convention: -1 bullish, +1 bearish.
    out["lux_ms_bull"] = direction < 0
    out["lux_ms_bear"] = direction > 0
    return out


def _join_reason_tokens(tokens: List[str]) -> str:
    out = [str(t).strip() for t in tokens if str(t).strip()]
    return ";".join(out) if out else "signal_neutral"


def _build_reason_for_row(row: pd.Series, action: str) -> str:
    action_key = str(action or "Flat")
    long_tokens: List[str] = []
    short_tokens: List[str] = []

    if bool(row.get("ema_cross_up", False)):
        long_tokens.append("ema_bull_cross")
    if bool(row.get("ema_cross_down", False)):
        short_tokens.append("ema_bear_cross")
    if bool(row.get("macd_cross_up", False)):
        long_tokens.append("macd_golden_cross")
    if bool(row.get("macd_cross_down", False)):
        short_tokens.append("macd_dead_cross")
    if bool(row.get("supertrend_bull", False)):
        long_tokens.append("supertrend_bullish")
    if bool(row.get("supertrend_bear", False)):
        short_tokens.append("supertrend_bearish")
    if bool(row.get("lux_ms_bull", False)):
        long_tokens.append("lux_ms_bullish")
    if bool(row.get("lux_ms_bear", False)):
        short_tokens.append("lux_ms_bearish")
    if bool(row.get("bos_up", False)):
        long_tokens.append("bos_up")
    if bool(row.get("bos_down", False)):
        short_tokens.append("bos_down")
    if bool(row.get("choch_bull", False)):
        long_tokens.append("choch_bull")
    if bool(row.get("choch_bear", False)):
        short_tokens.append("choch_bear")
    if bool(row.get("volume_surge", False)):
        long_tokens.append("volume_surge")
        short_tokens.append("volume_surge")

    if action_key == "Long":
        return _join_reason_tokens(long_tokens + [str(row.get("policy_reason", "long_signal"))])
    if action_key == "Short":
        return _join_reason_tokens(short_tokens + [str(row.get("policy_reason", "short_signal"))])

    mixed = []
    if bool(row.get("ema_trend_bull", False)):
        mixed.append("ema_trend_up")
    if bool(row.get("ema_trend_bear", False)):
        mixed.append("ema_trend_down")
    return _join_reason_tokens(mixed + [str(row.get("policy_reason", "flat"))])


def _symbol_to_stooq(symbol: str, market: str) -> str:
    base = symbol.split(".")[0].lower()
    if market == "us_equity":
        return f"{base}.us"
    if market == "cn_equity":
        return f"{base}.cn"
    return base


def _fetch_binance_daily_bars(
    symbol: str,
    lookback_days: int,
    endpoints: List[str],
) -> Tuple[pd.DataFrame, str]:
    end_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = end_utc - timedelta(days=max(lookback_days, 30))
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    last_exc: Exception | None = None
    for endpoint in endpoints:
        try:
            df = fetch_binance_klines(
                symbol=symbol,
                interval="1d",
                start_ms=start_ms,
                end_ms=end_ms,
                base_url=endpoint,
            )
            if df.empty:
                raise RuntimeError("empty klines")
            out = df.copy()
            out["timestamp_utc"] = pd.to_datetime(out["open_time_ms"], unit="ms", utc=True)
            out = out.sort_values("timestamp_utc")
            out = out[["timestamp_utc", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
            return out, f"binance_klines:{endpoint}"
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to fetch binance bars for {symbol}: {last_exc}")


def _fetch_coingecko_daily_bars(coin_id: str, lookback_days: int) -> Tuple[pd.DataFrame, str]:
    days = max(30, int(lookback_days))
    url = COINGECKO_MARKET_CHART_URL.format(coin_id=coin_id)
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=25)
    r.raise_for_status()
    data = r.json()
    prices = data.get("prices", [])
    vols = data.get("total_volumes", [])
    if not prices:
        raise RuntimeError(f"coingecko prices empty: {coin_id}")
    vol_map = {int(v[0]): _safe_float(v[1]) for v in vols}
    rows: List[Dict[str, Any]] = []
    prev_close = float(prices[0][1])
    for item in prices:
        ts = pd.to_datetime(int(item[0]), unit="ms", utc=True).floor("D")
        close = _safe_float(item[1])
        open_ = prev_close
        high = max(open_, close)
        low = min(open_, close)
        volume = vol_map.get(int(item[0]), float("nan"))
        rows.append(
            {
                "timestamp_utc": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        prev_close = close
    out = pd.DataFrame(rows).dropna(subset=["close"]).sort_values("timestamp_utc").reset_index(drop=True)
    return out, "coingecko_market_chart"


def _fetch_yahoo_daily_bars(symbol: str, lookback_days: int) -> Tuple[pd.DataFrame, str]:
    now = datetime.now(timezone.utc)
    period2 = int(now.timestamp())
    period1 = int((now - timedelta(days=max(lookback_days, 30))).timestamp())
    url = YAHOO_CHART_URL.format(symbol=symbol)
    params = {
        "interval": "1d",
        "period1": period1,
        "period2": period2,
        "includePrePost": "false",
        "events": "div,splits",
    }
    r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    result = data.get("chart", {}).get("result", [])
    if not result:
        raise RuntimeError(f"Yahoo chart empty for {symbol}")
    item = result[0]
    ts = item.get("timestamp", [])
    quote = item.get("indicators", {}).get("quote", [{}])[0]
    if not ts:
        raise RuntimeError(f"Yahoo timestamps empty for {symbol}")
    out = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(ts, unit="s", utc=True),
            "open": pd.to_numeric(quote.get("open", []), errors="coerce"),
            "high": pd.to_numeric(quote.get("high", []), errors="coerce"),
            "low": pd.to_numeric(quote.get("low", []), errors="coerce"),
            "close": pd.to_numeric(quote.get("close", []), errors="coerce"),
            "volume": pd.to_numeric(quote.get("volume", []), errors="coerce"),
        }
    )
    out = out.dropna(subset=["close"]).sort_values("timestamp_utc").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"Yahoo bar data empty after cleaning for {symbol}")
    out["open"] = out["open"].fillna(out["close"].shift(1)).fillna(out["close"])
    out["high"] = out["high"].fillna(np.maximum(out["open"], out["close"]))
    out["low"] = out["low"].fillna(np.minimum(out["open"], out["close"]))
    out["volume"] = out["volume"].fillna(0.0)
    return out, "yahoo_chart"


def _fetch_stooq_daily_bars(symbol: str, market: str) -> Tuple[pd.DataFrame, str]:
    stooq_symbol = _symbol_to_stooq(symbol, market)
    params = {"s": stooq_symbol, "i": "d"}
    r = requests.get(STOOQ_DAILY_URL, params=params, headers=DEFAULT_HEADERS, timeout=20)
    r.raise_for_status()
    text = r.text
    if "No data" in text or "Date,Open,High,Low,Close,Volume" not in text:
        raise RuntimeError(f"Stooq no data: {stooq_symbol}")
    df = pd.read_csv(StringIO(text))
    if "Date" not in df.columns:
        raise RuntimeError(f"Stooq Date missing: {stooq_symbol}")
    out = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(df["Date"], utc=True, errors="coerce"),
            "open": pd.to_numeric(df.get("Open"), errors="coerce"),
            "high": pd.to_numeric(df.get("High"), errors="coerce"),
            "low": pd.to_numeric(df.get("Low"), errors="coerce"),
            "close": pd.to_numeric(df.get("Close"), errors="coerce"),
            "volume": pd.to_numeric(df.get("Volume"), errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp_utc", "close"]).sort_values("timestamp_utc").reset_index(drop=True)
    out["open"] = out["open"].fillna(out["close"].shift(1)).fillna(out["close"])
    out["high"] = out["high"].fillna(np.maximum(out["open"], out["close"]))
    out["low"] = out["low"].fillna(np.minimum(out["open"], out["close"]))
    out["volume"] = out["volume"].fillna(0.0)
    return out, f"stooq:{stooq_symbol}"


def _fetch_daily_bars(
    *,
    market: str,
    symbol: str,
    provider: str,
    lookback_days: int,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, str]:
    if market == "crypto":
        if provider == "coingecko":
            return _fetch_coingecko_daily_bars(symbol, lookback_days=lookback_days)
        endpoints = cfg.get("data", {}).get(
            "binance_endpoints",
            ["https://api.binance.com", "https://api.binance.us"],
        )
        if bool(cfg.get("data", {}).get("disable_binance_us", False)):
            filtered = [ep for ep in endpoints if "binance.us" not in str(ep).lower()]
            if filtered:
                endpoints = filtered
        return _fetch_binance_daily_bars(symbol, lookback_days=lookback_days, endpoints=endpoints)

    try:
        return _fetch_yahoo_daily_bars(symbol, lookback_days=lookback_days)
    except Exception:
        return _fetch_stooq_daily_bars(symbol, market=market)


def _build_universe_for_backtest(cfg: Dict[str, Any]) -> pd.DataFrame:
    tracking_cfg = cfg.get("tracking", {})
    bt_cfg = cfg.get("backtest_multi_market", {})
    selected_universes = tracking_cfg.get(
        "universes",
        {
            "crypto": ["top100_ex_stable"],
            "cn_equity": ["sse_composite", "csi300"],
            "us_equity": ["dow30", "nasdaq100", "sp500"],
        },
    )
    max_per_market_cfg = bt_cfg.get(
        "max_symbols_per_market",
        {"crypto": 8, "cn_equity": 8, "us_equity": 8},
    )

    rows: List[Dict[str, Any]] = []
    # Ensure crypto top3 symbols are always included so dashboard-selected BTC/ETH/SOL have coverage.
    if "crypto" in selected_universes:
        pools = [str(p) for p in selected_universes.get("crypto", [])]
        if "top3" not in pools:
            pools = ["top3"] + pools
            selected_universes = dict(selected_universes)
            selected_universes["crypto"] = pools

    for market, pools in selected_universes.items():
        for pool_key in pools:
            try:
                uni = load_universe(market, pool_key)
            except Exception as exc:
                print(f"[WARN] universe load failed {market}/{pool_key}: {exc}")
                continue
            uni = uni.copy()
            if market == "crypto":
                if "provider" not in uni.columns:
                    uni["provider"] = "binance"
                if "snapshot_symbol" not in uni.columns:
                    uni["snapshot_symbol"] = uni["symbol"].astype(str)
                uni["symbol_bt"] = uni["snapshot_symbol"].astype(str)
            else:
                uni["provider"] = uni.get("provider", "yahoo")
                uni["symbol_bt"] = uni["symbol"].astype(str)
            for _, row in uni.iterrows():
                rows.append(
                    {
                        "market": market,
                        "pool_key": pool_key,
                        "name": str(row.get("name", row.get("display", row.get("symbol_bt", "")))),
                        "symbol": str(row.get("symbol_bt", "")),
                        "provider": str(row.get("provider", "yahoo")),
                        "fallback_symbol": (
                            f"{str(row.get('symbol', '')).upper()}USDT"
                            if market == "crypto" and str(row.get("provider", "binance")) == "coingecko"
                            else str(row.get("symbol_bt", ""))
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["market", "symbol"]).reset_index(drop=True)
    frames: List[pd.DataFrame] = []
    for market, sub in out.groupby("market", dropna=False):
        max_n = int(max_per_market_cfg.get(str(market), 8))
        if max_n > 0:
            sub = sub.head(max_n).copy()
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_model_like_signals(df: pd.DataFrame, cfg: Dict[str, Any], market: str) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    out["open"] = pd.to_numeric(out.get("open"), errors="coerce")
    out["high"] = pd.to_numeric(out.get("high"), errors="coerce")
    out["low"] = pd.to_numeric(out.get("low"), errors="coerce")
    out["close"] = pd.to_numeric(out.get("close"), errors="coerce")
    out["volume"] = pd.to_numeric(out.get("volume"), errors="coerce")
    out["ret_1"] = out["close"].pct_change()
    out["mom_5"] = out["ret_1"].rolling(5).mean()
    out["mom_20"] = out["ret_1"].rolling(20).mean()
    out["vol_20"] = out["ret_1"].rolling(20).std(ddof=0)
    out["vol_20"] = out["vol_20"].replace(0.0, np.nan)

    out["ema_fast_20"] = _ema(out["close"], 20)
    out["ema_slow_55"] = _ema(out["close"], 55)
    out["ema_trend_bull"] = out["ema_fast_20"] >= out["ema_slow_55"]
    out["ema_trend_bear"] = out["ema_fast_20"] < out["ema_slow_55"]
    out["ema_cross_up"] = (out["ema_fast_20"] > out["ema_slow_55"]) & (
        out["ema_fast_20"].shift(1) <= out["ema_slow_55"].shift(1)
    )
    out["ema_cross_down"] = (out["ema_fast_20"] < out["ema_slow_55"]) & (
        out["ema_fast_20"].shift(1) >= out["ema_slow_55"].shift(1)
    )

    out["macd_line"] = _ema(out["close"], 12) - _ema(out["close"], 26)
    out["macd_signal"] = _ema(out["macd_line"], 9)
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]
    out["macd_cross_up"] = (out["macd_line"] > out["macd_signal"]) & (
        out["macd_line"].shift(1) <= out["macd_signal"].shift(1)
    )
    out["macd_cross_down"] = (out["macd_line"] < out["macd_signal"]) & (
        out["macd_line"].shift(1) >= out["macd_signal"].shift(1)
    )

    out["atr_14"] = _atr_from_bars(out, period=14)
    out["atr_14_pct"] = out["atr_14"] / out["close"].replace(0.0, np.nan)
    out["volume_ma_20"] = out["volume"].rolling(20).mean()
    out["volume_surge"] = out["volume"] > (1.5 * out["volume_ma_20"])
    out["supertrend_direction"] = _compute_supertrend_direction(
        out["high"], out["low"], out["close"], out["atr_14"], multiplier=3.0
    )
    out["supertrend_bull"] = out["supertrend_direction"] > 0
    out["supertrend_bear"] = out["supertrend_direction"] < 0
    lux_ms = _compute_lux_machine_supertrend(
        out,
        atr_len=10,
        factor=3.0,
        training_data_period=100,
        highvol=0.75,
        midvol=0.50,
        lowvol=0.25,
    )
    for col in lux_ms.columns:
        out[col] = lux_ms[col]

    out["swing_high_20"] = out["high"].shift(1).rolling(20).max()
    out["swing_low_20"] = out["low"].shift(1).rolling(20).min()
    out["bos_up"] = out["close"] > out["swing_high_20"]
    out["bos_down"] = out["close"] < out["swing_low_20"]
    out["choch_bull"] = out["ema_trend_bear"].shift(1).eq(True) & out["bos_up"]
    out["choch_bear"] = out["ema_trend_bull"].shift(1).eq(True) & out["bos_down"]

    one_step = 0.65 * out["mom_5"].fillna(0.0) + 0.35 * out["mom_20"].fillna(0.0)
    clamp = (3.0 * out["vol_20"].fillna(out["vol_20"].median())).fillna(0.05)
    out["q50_change_pct"] = np.clip(one_step, -clamp, clamp)
    sigma = out["vol_20"].fillna(out["vol_20"].median()).fillna(0.01)
    out["q10_change_pct"] = out["q50_change_pct"] - 1.28 * sigma
    out["q90_change_pct"] = out["q50_change_pct"] + 1.28 * sigma
    out["volatility_score"] = out["q90_change_pct"] - out["q10_change_pct"]

    ret_thr = _safe_float(
        cfg.get("policy", {}).get("thresholds", {}).get(
            "ret_threshold",
            cfg.get("forecast_config", {}).get("trend_labels", {}).get("ret_threshold", 0.002),
        )
    )
    if not np.isfinite(ret_thr) or ret_thr <= 0:
        ret_thr = 0.002
    scale = max(ret_thr, 1e-4) * 8.0
    out["p_up"] = 0.5 + 0.5 * np.tanh(out["q50_change_pct"].fillna(0.0) / scale)
    out["p_down"] = 1.0 - out["p_up"]

    v = pd.to_numeric(out["volatility_score"], errors="coerce")
    vmin = _safe_float(v.min())
    vmax = _safe_float(v.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) <= 1e-12:
        vol_norm = pd.Series([0.5] * len(out), index=out.index, dtype=float)
    else:
        vol_norm = ((v - vmin) / (vmax - vmin)).clip(0.0, 1.0)
    conf_prob = (2.0 * (out["p_up"] - 0.5).abs()).clip(0.0, 1.0)
    conf_width = (1.0 - vol_norm).clip(0.0, 1.0)
    out["confidence_score"] = (100.0 * (0.6 * conf_prob + 0.4 * conf_width)).clip(0.0, 100.0)

    q50 = _safe_float(v.quantile(0.50))
    q75 = _safe_float(v.quantile(0.75))
    q90 = _safe_float(v.quantile(0.90))
    if not np.isfinite(q50):
        q50 = 0.01
    if not np.isfinite(q75):
        q75 = q50 * 1.5
    if not np.isfinite(q90):
        q90 = q75 * 1.5
    out["risk_level"] = np.where(
        v < q50,
        "low",
        np.where(v < q75, "medium", np.where(v < q90, "high", "extreme")),
    )

    out["market"] = market
    out["market_type"] = "spot" if market == "crypto" else "cash"
    out["current_price"] = out["close"]
    out = apply_policy_frame(
        out,
        cfg,
        market_col="market",
        market_type_col="market_type",
        p_up_col="p_up",
        q10_col="q10_change_pct",
        q50_col="q50_change_pct",
        q90_col="q90_change_pct",
        volatility_col="volatility_score",
        confidence_col="confidence_score",
        current_price_col="current_price",
        risk_level_col="risk_level",
    )

    stop_pct = (1.2 * pd.to_numeric(out["atr_14_pct"], errors="coerce")).clip(0.005, 0.08)
    stop_pct = stop_pct.fillna(0.015)
    conf = pd.to_numeric(out.get("confidence_score"), errors="coerce").fillna(50.0)
    rr = np.where(conf >= 70.0, 2.2, np.where(conf >= 50.0, 1.8, 1.5))
    out["trade_stop_loss_pct"] = stop_pct
    out["trade_take_profit_pct"] = stop_pct * rr
    out["trade_rr_ratio"] = pd.Series(rr, index=out.index, dtype=float).fillna(1.8)
    out["trade_signal"] = out.get("policy_action", pd.Series(["Flat"] * len(out), index=out.index)).astype(str)
    out["trade_trend_context"] = np.where(
        (out["ema_trend_bull"] & out["supertrend_bull"]),
        "bullish",
        np.where((out["ema_trend_bear"] & out["supertrend_bear"]), "bearish", "mixed"),
    )
    out["trade_reason_tokens"] = [
        _build_reason_for_row(row, action=row.get("trade_signal", "Flat")) for _, row in out.iterrows()
    ]
    out["trade_reason_detail"] = out["trade_reason_tokens"]

    close_px = pd.to_numeric(out["close"], errors="coerce")
    action = out["trade_signal"].astype(str)
    out["trade_stop_loss_price"] = np.where(
        action.eq("Long"),
        close_px * (1.0 - out["trade_stop_loss_pct"]),
        np.where(action.eq("Short"), close_px * (1.0 + out["trade_stop_loss_pct"]), np.nan),
    )
    out["trade_take_profit_price"] = np.where(
        action.eq("Long"),
        close_px * (1.0 + out["trade_take_profit_pct"]),
        np.where(action.eq("Short"), close_px * (1.0 - out["trade_take_profit_pct"]), np.nan),
    )

    score = (
        out["ema_trend_bull"].astype(int)
        + out["macd_cross_up"].astype(int)
        + out["supertrend_bull"].astype(int)
        + out["lux_ms_bull"].astype(int)
        + out["bos_up"].astype(int)
        + out["choch_bull"].astype(int)
        - out["ema_trend_bear"].astype(int)
        - out["macd_cross_down"].astype(int)
        - out["supertrend_bear"].astype(int)
        - out["lux_ms_bear"].astype(int)
        - out["bos_down"].astype(int)
        - out["choch_bear"].astype(int)
    )
    out["trade_support_score"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    return out


def _signal_from_baseline(
    frame: pd.DataFrame,
    strategy: str,
    *,
    allow_short: bool,
) -> pd.Series:
    if strategy == "policy":
        return pd.to_numeric(frame.get("policy_signed_position"), errors="coerce").fillna(0.0)
    if strategy == "buy_hold":
        return pd.Series([1.0] * len(frame), index=frame.index, dtype=float)
    if strategy == "ma_crossover":
        fast = frame["close"].rolling(10).mean()
        slow = frame["close"].rolling(30).mean()
        if allow_short:
            return np.where(fast > slow, 1.0, np.where(fast < slow, -1.0, 0.0))
        return np.where(fast > slow, 1.0, 0.0)
    if strategy == "naive_prev_bar":
        ret = frame["close"].pct_change().fillna(0.0)
        if allow_short:
            return np.sign(ret)
        return (ret > 0).astype(float)
    if strategy == "lux_machine_supertrend":
        lux_dir = pd.to_numeric(frame.get("lux_ms_direction"), errors="coerce")
        if allow_short:
            # Pine convention: dir < 0 bullish, dir > 0 bearish.
            return np.where(lux_dir < 0, 1.0, np.where(lux_dir > 0, -1.0, 0.0))
        return np.where(lux_dir < 0, 1.0, 0.0)
    raise ValueError(f"Unsupported strategy: {strategy}")


def _run_strategy(
    frame: pd.DataFrame,
    *,
    strategy: str,
    allow_short: bool,
    delay_bars: int,
    fee_bps: float,
    slippage_bps: float,
    impact_lambda_bps: float,
    impact_beta: float,
    risk_exit_enabled: bool,
    fallback_stop_loss_pct: float,
    fallback_take_profit_pct: float,
) -> pd.DataFrame:
    out = frame.copy()
    raw_signal = pd.Series(_signal_from_baseline(out, strategy, allow_short=allow_short), index=out.index)
    out["signal"] = pd.to_numeric(raw_signal, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    lag = max(int(delay_bars), 0)
    out["position"] = out["signal"].shift(lag).fillna(0.0)
    for col in [
        "trade_signal",
        "trade_reason_tokens",
        "trade_reason_detail",
        "trade_trend_context",
        "trade_stop_loss_pct",
        "trade_take_profit_pct",
        "trade_rr_ratio",
        "trade_stop_loss_price",
        "trade_take_profit_price",
        "policy_reason",
        "trade_support_score",
    ]:
        if col in out.columns:
            out[f"{col}_entry"] = out[col].shift(1).fillna(out[col])

    open_px = pd.to_numeric(out["open"], errors="coerce")
    high_px = pd.to_numeric(out["high"], errors="coerce")
    low_px = pd.to_numeric(out["low"], errors="coerce")
    close_px = pd.to_numeric(out["close"], errors="coerce")
    raw_bar_ret = (close_px / open_px) - 1.0
    raw_bar_ret = raw_bar_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["bar_ret"] = raw_bar_ret

    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["impact_bps"] = impact_lambda_bps * np.power(out["turnover"], impact_beta)
    out["cost_pct"] = out["turnover"] * ((fee_bps + slippage_bps) / 10000.0) + (out["impact_bps"] / 10000.0)

    side = np.sign(pd.to_numeric(out["position"], errors="coerce").fillna(0.0))
    size = pd.to_numeric(out["position"], errors="coerce").abs().fillna(0.0)
    long_ret = raw_bar_ret.copy()
    short_ret = (-raw_bar_ret).copy()
    exit_reason = np.array(["time_exit"] * len(out), dtype=object)

    if risk_exit_enabled:
        sl_raw = out.get("trade_stop_loss_pct_entry")
        tp_raw = out.get("trade_take_profit_pct_entry")
        if isinstance(sl_raw, pd.Series):
            sl_pct = pd.to_numeric(sl_raw, errors="coerce")
        else:
            sl_pct = pd.Series(np.nan, index=out.index, dtype=float)
        if isinstance(tp_raw, pd.Series):
            tp_pct = pd.to_numeric(tp_raw, errors="coerce")
        else:
            tp_pct = pd.Series(np.nan, index=out.index, dtype=float)
        sl_pct = sl_pct.fillna(float(fallback_stop_loss_pct)).clip(lower=0.001, upper=0.20)
        tp_pct = tp_pct.fillna(float(fallback_take_profit_pct)).clip(lower=0.001, upper=0.40)

        long_hit_stop = ((open_px - low_px) / open_px) >= sl_pct
        long_hit_take = ((high_px - open_px) / open_px) >= tp_pct
        short_hit_stop = ((high_px - open_px) / open_px) >= sl_pct
        short_hit_take = ((open_px - low_px) / open_px) >= tp_pct

        long_ret = np.where(long_hit_stop, -sl_pct, np.where(long_hit_take, tp_pct, long_ret))
        short_ret = np.where(short_hit_stop, -sl_pct, np.where(short_hit_take, tp_pct, short_ret))

        is_long = side > 0
        is_short = side < 0
        exit_reason = np.where(is_long & long_hit_stop, "stop_loss", exit_reason)
        exit_reason = np.where(is_long & (~long_hit_stop) & long_hit_take, "take_profit", exit_reason)
        exit_reason = np.where(is_short & short_hit_stop, "stop_loss", exit_reason)
        exit_reason = np.where(is_short & (~short_hit_stop) & short_hit_take, "take_profit", exit_reason)
        exit_reason = np.where(side == 0, "flat", exit_reason)
    else:
        exit_reason = np.where(side == 0, "flat", exit_reason)

    side_ret = np.where(side > 0, long_ret, np.where(side < 0, short_ret, 0.0))
    out["bar_exit_reason"] = pd.Series(exit_reason, index=out.index)
    out["bar_ret_effective"] = pd.Series(side_ret, index=out.index, dtype=float).fillna(0.0)
    out["gross_ret_effective"] = size * out["bar_ret_effective"]
    out["strategy_ret"] = out["gross_ret_effective"] - out["cost_pct"]
    out["equity"] = (1.0 + out["strategy_ret"]).cumprod()
    out["strategy"] = strategy
    return out


def _compute_metrics(frame: pd.DataFrame, market: str) -> Dict[str, float]:
    if frame.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "volatility": 0.0,
            "win_rate": 0.0,
            "avg_win_loss_ratio": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "exposure": 0.0,
            "trades_count": 0,
            "turnover_avg": 0.0,
        }

    ret = pd.to_numeric(frame["strategy_ret"], errors="coerce").fillna(0.0)
    eq = pd.to_numeric(frame["equity"], errors="coerce").ffill().fillna(1.0)
    total_return = float(eq.iloc[-1] - 1.0)
    annual_factor = _annual_factor_for_market(market)
    n = max(len(frame), 1)
    cagr = float((1.0 + total_return) ** (annual_factor / n) - 1.0) if total_return > -1 else -1.0
    mu = float(ret.mean())
    sd = float(ret.std(ddof=0))
    neg_sd = float(ret[ret < 0].std(ddof=0)) if (ret < 0).any() else 0.0
    sharpe = float((mu / sd) * np.sqrt(annual_factor)) if sd > 0 else 0.0
    sortino = float((mu / neg_sd) * np.sqrt(annual_factor)) if neg_sd > 0 else 0.0
    vol = float(sd * np.sqrt(annual_factor))
    exposure = float((frame["position"].abs() > 1e-9).mean())
    trades_count = int((frame["turnover"] > 1e-9).sum())
    turnover_avg = float(frame["turnover"].mean())

    pnl = ret[frame["position"].abs() > 1e-9]
    if pnl.empty:
        win_rate = 0.0
        avg_win_loss_ratio = 0.0
        profit_factor = 0.0
        expectancy = 0.0
    else:
        win = pnl[pnl > 0]
        loss = pnl[pnl < 0]
        win_rate = float((pnl > 0).mean())
        avg_win = float(win.mean()) if not win.empty else 0.0
        avg_loss = float(abs(loss.mean())) if not loss.empty else 0.0
        avg_win_loss_ratio = float(avg_win / avg_loss) if avg_loss > 0 else 0.0
        gross_profit = float(win.sum()) if not win.empty else 0.0
        gross_loss = float(abs(loss.sum())) if not loss.empty else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0
        expectancy = float(win_rate * avg_win - (1.0 - win_rate) * avg_loss)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": _max_drawdown(eq),
        "sharpe": sharpe,
        "sortino": sortino,
        "volatility": vol,
        "win_rate": win_rate,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "exposure": exposure,
        "trades_count": trades_count,
        "turnover_avg": turnover_avg,
    }


def _extract_trade_log(
    frame: pd.DataFrame,
    *,
    market: str,
    symbol: str,
    strategy: str,
    fold: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    active = work[work["position"].abs() > 1e-9].copy()
    if active.empty:
        return pd.DataFrame()
    active = active.reset_index(drop=True)
    active["trade_id"] = np.arange(1, len(active) + 1)
    active["side"] = np.where(active["position"] > 0, "long", "short")
    active["entry_time"] = pd.to_datetime(active["timestamp_utc"], utc=True, errors="coerce")
    active["exit_time"] = active["entry_time"] + pd.Timedelta(days=1)
    active["entry_price"] = active["open"]
    active["exit_price"] = active["close"]
    active["size"] = active["position"].abs()
    active["gross_pnl_pct"] = pd.to_numeric(active.get("gross_ret_effective"), errors="coerce").fillna(
        active["position"] * active["bar_ret"]
    )
    active["cost_pct"] = active["cost_pct"]
    active["net_pnl_pct"] = active["strategy_ret"]
    active["holding_period"] = 1
    if "policy_reason_entry" in active.columns:
        active["reason"] = active["policy_reason_entry"].fillna("baseline")
    elif "policy_reason" in active.columns:
        active["reason"] = active["policy_reason"].shift(1).fillna(active["policy_reason"]).fillna("baseline")
    else:
        active["reason"] = "baseline"

    active["entry_signal"] = active.get("trade_signal_entry", pd.Series(["Flat"] * len(active)))
    active["entry_signal_reason"] = active.get(
        "trade_reason_tokens_entry",
        active.get("trade_reason_tokens", pd.Series(["signal_neutral"] * len(active))),
    )
    active["entry_trend_context"] = active.get(
        "trade_trend_context_entry",
        active.get("trade_trend_context", pd.Series(["mixed"] * len(active))),
    )
    support_raw = active.get("trade_support_score_entry", active.get("trade_support_score"))
    if isinstance(support_raw, pd.Series):
        active["entry_support_score"] = pd.to_numeric(support_raw, errors="coerce").fillna(0.0)
    else:
        active["entry_support_score"] = 0.0
    active["stop_loss_price"] = pd.to_numeric(
        active.get("trade_stop_loss_price_entry", active.get("trade_stop_loss_price")),
        errors="coerce",
    )
    active["take_profit_price"] = pd.to_numeric(
        active.get("trade_take_profit_price_entry", active.get("trade_take_profit_price")),
        errors="coerce",
    )
    active["stop_loss_pct"] = pd.to_numeric(
        active.get("trade_stop_loss_pct_entry", active.get("trade_stop_loss_pct")),
        errors="coerce",
    )
    active["take_profit_pct"] = pd.to_numeric(
        active.get("trade_take_profit_pct_entry", active.get("trade_take_profit_pct")),
        errors="coerce",
    )
    active["rr_ratio"] = pd.to_numeric(
        active.get("trade_rr_ratio_entry", active.get("trade_rr_ratio")),
        errors="coerce",
    )
    active["exit_reason"] = active.get("bar_exit_reason", pd.Series(["time_exit"] * len(active)))
    cols = [
        "trade_id",
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "size",
        "gross_pnl_pct",
        "cost_pct",
        "net_pnl_pct",
        "holding_period",
        "reason",
        "entry_signal",
        "entry_signal_reason",
        "entry_trend_context",
        "entry_support_score",
        "stop_loss_price",
        "take_profit_price",
        "stop_loss_pct",
        "take_profit_pct",
        "rr_ratio",
        "exit_reason",
    ]
    out = active[cols].copy()
    out["market"] = market
    out["symbol"] = symbol
    out["strategy"] = strategy
    out["fold"] = fold
    out["entry_time"] = out["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    out["exit_time"] = out["exit_time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return out[
        [
            "market",
            "symbol",
            "strategy",
            "fold",
            "trade_id",
            "entry_time",
            "exit_time",
            "side",
            "entry_price",
            "exit_price",
            "size",
            "gross_pnl_pct",
            "cost_pct",
            "net_pnl_pct",
            "holding_period",
            "reason",
            "entry_signal",
            "entry_signal_reason",
            "entry_trend_context",
            "entry_support_score",
            "stop_loss_price",
            "take_profit_price",
            "stop_loss_pct",
            "take_profit_pct",
            "rr_ratio",
            "exit_reason",
        ]
    ]


def _build_fold_ranges(
    n: int,
    *,
    min_train_size: int,
    test_size: int,
    max_folds: int,
) -> List[Tuple[int, int, int]]:
    ranges: List[Tuple[int, int, int]] = []
    for i in range(max_folds):
        train_end = min_train_size + i * test_size
        test_start = train_end
        test_end = test_start + test_size
        if test_end > n:
            break
        ranges.append((i, test_start, test_end))
    return ranges


def _normalize_strategy_list(strategies: List[str]) -> List[str]:
    out = [str(x) for x in strategies]
    if "policy" not in out:
        out = ["policy"] + out
    if "lux_machine_supertrend" not in out:
        out.append("lux_machine_supertrend")
    return out


def _run_backtest_on_modeled(
    *,
    modeled: pd.DataFrame,
    market: str,
    symbol: str,
    strategies: List[str],
    min_train_size: int,
    test_size: int,
    max_folds: int,
    delay_bars: int,
    fee_bps: float,
    slippage_bps: float,
    impact_lambda_bps: float,
    impact_beta: float,
    risk_exit_enabled: bool,
    fallback_stop_loss_pct: float,
    fallback_take_profit_pct: float,
) -> Dict[str, pd.DataFrame]:
    metrics_rows: List[Dict[str, Any]] = []
    equity_rows: List[pd.DataFrame] = []
    trade_rows: List[pd.DataFrame] = []

    fold_ranges = _build_fold_ranges(
        len(modeled),
        min_train_size=min_train_size,
        test_size=test_size,
        max_folds=max_folds,
    )
    allow_short_default = market in {"crypto", "us_equity"}
    for fold, test_start, test_end in fold_ranges:
        test_slice_start = max(0, test_start - 1)
        base = modeled.iloc[test_slice_start:test_end].copy().reset_index(drop=True)
        if base.empty:
            continue
        for strategy in strategies:
            sim = _run_strategy(
                base,
                strategy=str(strategy),
                allow_short=allow_short_default,
                delay_bars=delay_bars,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                impact_lambda_bps=impact_lambda_bps,
                impact_beta=impact_beta,
                risk_exit_enabled=risk_exit_enabled,
                fallback_stop_loss_pct=fallback_stop_loss_pct,
                fallback_take_profit_pct=fallback_take_profit_pct,
            )
            sim_eval = sim.iloc[1:].copy().reset_index(drop=True)
            if sim_eval.empty:
                continue
            m = _compute_metrics(sim_eval, market=market)
            m.update(
                {
                    "market": market,
                    "symbol": symbol,
                    "strategy": str(strategy),
                    "fold": int(fold),
                    "period_start": str(pd.to_datetime(sim_eval["timestamp_utc"].min(), utc=True)),
                    "period_end": str(pd.to_datetime(sim_eval["timestamp_utc"].max(), utc=True)),
                }
            )
            metrics_rows.append(m)

            eq = sim_eval[["timestamp_utc", "equity", "strategy_ret", "position", "turnover"]].copy()
            eq["market"] = market
            eq["symbol"] = symbol
            eq["strategy"] = str(strategy)
            eq["fold"] = int(fold)
            equity_rows.append(eq)

            trades = _extract_trade_log(
                sim_eval,
                market=market,
                symbol=symbol,
                strategy=str(strategy),
                fold=int(fold),
            )
            if not trades.empty:
                trade_rows.append(trades)

    metrics_df = pd.DataFrame(metrics_rows)
    equity_df = pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame()
    trades_df = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    latest_signal = modeled.tail(1).copy()
    if not latest_signal.empty:
        latest_signal = latest_signal.assign(market=market, symbol=symbol)
    if metrics_df.empty:
        return {
            "metrics_by_fold": pd.DataFrame(),
            "metrics_summary": pd.DataFrame(),
            "compare_baselines": pd.DataFrame(),
            "equity": equity_df,
            "trades": trades_df,
            "latest_signal": latest_signal,
        }

    summary = (
        metrics_df.groupby(["market", "symbol", "strategy"], as_index=False)[
            [
                "total_return",
                "cagr",
                "max_drawdown",
                "sharpe",
                "sortino",
                "volatility",
                "win_rate",
                "avg_win_loss_ratio",
                "profit_factor",
                "expectancy",
                "exposure",
                "trades_count",
                "turnover_avg",
            ]
        ]
        .mean()
    )
    policy_summary = summary[summary["strategy"] == "policy"].copy()
    baseline_summary = summary[summary["strategy"] != "policy"].copy()
    compare_rows: List[Dict[str, Any]] = []
    for _, prow in policy_summary.iterrows():
        p_ret = _safe_float(prow["total_return"])
        p_sharpe = _safe_float(prow["sharpe"])
        p_dd = _safe_float(prow["max_drawdown"])
        subset = baseline_summary[
            (baseline_summary["market"] == prow["market"]) & (baseline_summary["symbol"] == prow["symbol"])
        ]
        for _, brow in subset.iterrows():
            compare_rows.append(
                {
                    "market": str(prow["market"]),
                    "symbol": str(prow["symbol"]),
                    "baseline": brow["strategy"],
                    "policy_total_return": p_ret,
                    "baseline_total_return": _safe_float(brow["total_return"]),
                    "delta_total_return": p_ret - _safe_float(brow["total_return"]),
                    "policy_sharpe": p_sharpe,
                    "baseline_sharpe": _safe_float(brow["sharpe"]),
                    "delta_sharpe": p_sharpe - _safe_float(brow["sharpe"]),
                    "policy_max_drawdown": p_dd,
                    "baseline_max_drawdown": _safe_float(brow["max_drawdown"]),
                    "delta_max_drawdown": p_dd - _safe_float(brow["max_drawdown"]),
                }
            )
    compare_df = pd.DataFrame(compare_rows)
    return {
        "metrics_by_fold": metrics_df,
        "metrics_summary": summary,
        "compare_baselines": compare_df,
        "equity": equity_df,
        "trades": trades_df,
        "latest_signal": latest_signal,
    }


def run_single_symbol_backtest(
    *,
    config_path: str,
    market: str,
    symbol: str,
    provider: str | None = None,
    fallback_symbol: str | None = None,
    lookback_days: int | None = None,
    fee_bps: float | None = None,
    slippage_bps: float | None = None,
    delay_bars: int | None = None,
) -> Dict[str, pd.DataFrame]:
    cfg = load_config(config_path)
    bt_cfg = cfg.get("backtest_multi_market", {})
    lookback_days_cfg = bt_cfg.get(
        "lookback_days",
        {"crypto": 1825, "cn_equity": 1825, "us_equity": 1825},
    )
    min_train_size = int(bt_cfg.get("min_train_size", 180))
    test_size = int(bt_cfg.get("test_size", 60))
    max_folds = int(bt_cfg.get("max_folds", 4))
    strategies = _normalize_strategy_list(
        bt_cfg.get("strategies", ["policy", "buy_hold", "ma_crossover", "naive_prev_bar", "lux_machine_supertrend"])
    )
    policy_cfg = get_policy_config(cfg)
    exec_cfg = policy_cfg.get("execution", {})
    fee_bps_cfg = _safe_float(exec_cfg.get("fee_bps", 10.0))
    slippage_bps_cfg = _safe_float(exec_cfg.get("slippage_bps", 10.0))
    delay_cfg = int(bt_cfg.get("delay_bars", 1))
    fee_bps_val = _safe_float(fee_bps) if fee_bps is not None else fee_bps_cfg
    slippage_bps_val = _safe_float(slippage_bps) if slippage_bps is not None else slippage_bps_cfg
    delay_bars_val = int(delay_bars) if delay_bars is not None else delay_cfg
    impact_cfg = bt_cfg.get("impact", {})
    impact_lambda_bps = _safe_float(impact_cfg.get("lambda_bps", 1.0))
    impact_beta = _safe_float(impact_cfg.get("beta", 0.5))
    if not np.isfinite(impact_beta) or impact_beta <= 0:
        impact_beta = 0.5
    risk_exits_cfg = bt_cfg.get("risk_exits", {})
    risk_exit_enabled = bool(risk_exits_cfg.get("enabled", True))
    fallback_stop_loss_pct = _safe_float(risk_exits_cfg.get("stop_loss_pct", 0.015))
    fallback_take_profit_pct = _safe_float(risk_exits_cfg.get("take_profit_pct", 0.03))
    if not np.isfinite(fallback_stop_loss_pct) or fallback_stop_loss_pct <= 0:
        fallback_stop_loss_pct = 0.015
    if not np.isfinite(fallback_take_profit_pct) or fallback_take_profit_pct <= 0:
        fallback_take_profit_pct = 0.03

    mk = str(market)
    sym = str(symbol)
    prov = str(provider or ("binance" if mk == "crypto" and sym.upper().endswith("USDT") else "yahoo"))
    lb_days = int(lookback_days if lookback_days is not None else lookback_days_cfg.get(mk, 1825))

    try:
        bars, _ = _fetch_daily_bars(
            market=mk,
            symbol=sym,
            provider=prov,
            lookback_days=lb_days,
            cfg=cfg,
        )
    except Exception as exc:
        if mk == "crypto":
            fs = str(fallback_symbol or "").upper().strip()
            if not fs and prov == "coingecko":
                sym_guess = str(symbol).upper().strip()
                if sym_guess.isalpha() and len(sym_guess) <= 6:
                    fs = f"{sym_guess}USDT"
            if fs.endswith("USDT"):
                bars, _ = _fetch_daily_bars(
                    market=mk,
                    symbol=fs,
                    provider="binance",
                    lookback_days=lb_days,
                    cfg=cfg,
                )
                sym = fs
            else:
                raise exc
        else:
            raise exc
    bars = bars.dropna(subset=["timestamp_utc", "open", "close"]).sort_values("timestamp_utc").reset_index(drop=True)
    if len(bars) < (min_train_size + test_size):
        raise RuntimeError(
            f"insufficient_history_for_backtest: bars={len(bars)}, required>={min_train_size + test_size}"
        )
    modeled = _build_model_like_signals(bars, cfg=cfg, market=mk)
    return _run_backtest_on_modeled(
        modeled=modeled,
        market=mk,
        symbol=sym,
        strategies=[str(x) for x in strategies],
        min_train_size=min_train_size,
        test_size=test_size,
        max_folds=max_folds,
        delay_bars=delay_bars_val,
        fee_bps=fee_bps_val,
        slippage_bps=slippage_bps_val,
        impact_lambda_bps=impact_lambda_bps,
        impact_beta=impact_beta,
        risk_exit_enabled=risk_exit_enabled,
        fallback_stop_loss_pct=fallback_stop_loss_pct,
        fallback_take_profit_pct=fallback_take_profit_pct,
    )


def run_multi_market_backtest(config_path: str) -> None:
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    bt_cfg = cfg.get("backtest_multi_market", {})
    if not bool(bt_cfg.get("enabled", True)):
        print("[WARN] backtest_multi_market is disabled in config.")
        return
    processed_dir = Path(paths_cfg.get("processed_data_dir", "data/processed"))
    out_dir = processed_dir / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    lookback_days_cfg = bt_cfg.get(
        "lookback_days",
        {"crypto": 1825, "cn_equity": 1825, "us_equity": 1825},
    )
    min_train_size = int(bt_cfg.get("min_train_size", 180))
    test_size = int(bt_cfg.get("test_size", 60))
    max_folds = int(bt_cfg.get("max_folds", 4))
    strategies = _normalize_strategy_list(
        bt_cfg.get("strategies", ["policy", "buy_hold", "ma_crossover", "naive_prev_bar", "lux_machine_supertrend"])
    )

    policy_cfg = get_policy_config(cfg)
    exec_cfg = policy_cfg.get("execution", {})
    fee_bps = _safe_float(exec_cfg.get("fee_bps", 10.0))
    slippage_bps = _safe_float(exec_cfg.get("slippage_bps", 10.0))
    delay_bars = int(bt_cfg.get("delay_bars", 1))
    impact_cfg = bt_cfg.get("impact", {})
    impact_lambda_bps = _safe_float(impact_cfg.get("lambda_bps", 1.0))
    impact_beta = _safe_float(impact_cfg.get("beta", 0.5))
    if not np.isfinite(impact_beta) or impact_beta <= 0:
        impact_beta = 0.5
    risk_exits_cfg = bt_cfg.get("risk_exits", {})
    risk_exit_enabled = bool(risk_exits_cfg.get("enabled", True))
    fallback_stop_loss_pct = _safe_float(risk_exits_cfg.get("stop_loss_pct", 0.015))
    fallback_take_profit_pct = _safe_float(risk_exits_cfg.get("take_profit_pct", 0.03))
    if not np.isfinite(fallback_stop_loss_pct) or fallback_stop_loss_pct <= 0:
        fallback_stop_loss_pct = 0.015
    if not np.isfinite(fallback_take_profit_pct) or fallback_take_profit_pct <= 0:
        fallback_take_profit_pct = 0.03

    cost_cfg = cfg.get("cost_stress", {}) if isinstance(cfg, dict) else {}
    cost_stress_enabled = bool(cost_cfg.get("enabled", True))
    fee_grid = [float(x) for x in cost_cfg.get("fee_bps_grid", [2, 5, 10])]
    slippage_grid = [float(x) for x in cost_cfg.get("slippage_bps_grid", [1, 3, 8])]
    delay_grid = [int(x) for x in cost_cfg.get("delay_bars_grid", [0, 1, 2])]
    stress_strategy = str(cost_cfg.get("strategy", "policy"))

    universe = _build_universe_for_backtest(cfg)
    if universe.empty:
        print("[WARN] No instruments available for multi-market backtest.")
        return
    save_json({"rows": universe.to_dict(orient="records")}, out_dir / "universe_snapshot.json")

    metrics_rows: List[Dict[str, Any]] = []
    equity_rows: List[pd.DataFrame] = []
    trade_rows: List[pd.DataFrame] = []
    latest_signal_rows: List[pd.DataFrame] = []
    integrity_rows: List[Dict[str, Any]] = []
    cost_stress_rows: List[Dict[str, Any]] = []

    for _, inst in universe.iterrows():
        market = str(inst.get("market", ""))
        symbol = str(inst.get("symbol", ""))
        provider = str(inst.get("provider", "yahoo"))
        fallback_symbol = str(inst.get("fallback_symbol", ""))
        lookback_days = int(lookback_days_cfg.get(market, 1825))
        try:
            bars, data_source = _fetch_daily_bars(
                market=market,
                symbol=symbol,
                provider=provider,
                lookback_days=lookback_days,
                cfg=cfg,
            )
        except Exception as exc:
            # Crypto fallback: if CoinGecko blocked/rate-limited, try Binance spot symbol.
            if market == "crypto" and fallback_symbol and fallback_symbol.upper().endswith("USDT"):
                try:
                    bars, data_source = _fetch_daily_bars(
                        market=market,
                        symbol=fallback_symbol.upper(),
                        provider="binance",
                        lookback_days=lookback_days,
                        cfg=cfg,
                    )
                    symbol = fallback_symbol.upper()
                except Exception:
                    integrity_rows.append(
                        {
                            "market": market,
                            "symbol": symbol,
                            "provider": provider,
                            "status": "fetch_failed",
                            "message": str(exc),
                        }
                    )
                    continue
            else:
                integrity_rows.append(
                    {
                        "market": market,
                        "symbol": symbol,
                        "provider": provider,
                        "status": "fetch_failed",
                        "message": str(exc),
                    }
                )
                continue

        bars = bars.dropna(subset=["timestamp_utc", "open", "close"]).sort_values("timestamp_utc").reset_index(drop=True)
        history_bars = int(len(bars))
        missing_rate = float(
            bars[["open", "high", "low", "close", "volume"]].isna().any(axis=1).mean()
        )
        integrity_rows.append(
            {
                "market": market,
                "symbol": symbol,
                "provider": provider,
                "status": "ok",
                "history_bars": history_bars,
                "missing_rate": missing_rate,
                "data_source": data_source,
            }
        )

        if history_bars < (min_train_size + test_size):
            continue

        modeled = _build_model_like_signals(bars, cfg=cfg, market=market)
        result = _run_backtest_on_modeled(
            modeled=modeled,
            market=market,
            symbol=symbol,
            strategies=[str(x) for x in strategies],
            min_train_size=min_train_size,
            test_size=test_size,
            max_folds=max_folds,
            delay_bars=delay_bars,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            impact_lambda_bps=impact_lambda_bps,
            impact_beta=impact_beta,
            risk_exit_enabled=risk_exit_enabled,
            fallback_stop_loss_pct=fallback_stop_loss_pct,
            fallback_take_profit_pct=fallback_take_profit_pct,
        )
        m_df = result.get("metrics_by_fold", pd.DataFrame())
        if not m_df.empty:
            metrics_rows.extend(m_df.to_dict(orient="records"))
        eq_df = result.get("equity", pd.DataFrame())
        if not eq_df.empty:
            equity_rows.append(eq_df)
        tr_df = result.get("trades", pd.DataFrame())
        if not tr_df.empty:
            trade_rows.append(tr_df)
        sig_df = result.get("latest_signal", pd.DataFrame())
        if not sig_df.empty:
            latest_signal_rows.append(sig_df)

        if cost_stress_enabled:
            for fee_i, slip_i, delay_i in product(fee_grid, slippage_grid, delay_grid):
                stress = _run_backtest_on_modeled(
                    modeled=modeled,
                    market=market,
                    symbol=symbol,
                    strategies=[stress_strategy],
                    min_train_size=min_train_size,
                    test_size=test_size,
                    max_folds=max_folds,
                    delay_bars=int(delay_i),
                    fee_bps=float(fee_i),
                    slippage_bps=float(slip_i),
                    impact_lambda_bps=impact_lambda_bps,
                    impact_beta=impact_beta,
                    risk_exit_enabled=risk_exit_enabled,
                    fallback_stop_loss_pct=fallback_stop_loss_pct,
                    fallback_take_profit_pct=fallback_take_profit_pct,
                )
                ssum = stress.get("metrics_summary", pd.DataFrame())
                if ssum.empty:
                    continue
                row = ssum.iloc[0].to_dict()
                row.update(
                    {
                        "fee_bps": float(fee_i),
                        "slippage_bps": float(slip_i),
                        "delay_bars": int(delay_i),
                    }
                )
                cost_stress_rows.append(row)

    if not metrics_rows:
        print("[WARN] Multi-market backtest produced no metrics.")
        return

    metrics_df = pd.DataFrame(metrics_rows)
    equity_df = pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame()
    trades_df = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    latest_signals_df = pd.concat(latest_signal_rows, ignore_index=True) if latest_signal_rows else pd.DataFrame()
    integrity_df = pd.DataFrame(integrity_rows)

    write_csv(metrics_df, out_dir / "metrics_by_fold.csv")
    write_csv(equity_df, out_dir / "equity.csv")
    write_csv(trades_df, out_dir / "trades.csv")
    write_csv(latest_signals_df, out_dir / "latest_signals.csv")
    write_csv(integrity_df, out_dir / "survivorship_coverage.csv")
    save_json({"rows": integrity_rows}, out_dir / "data_integrity_checks.json")

    summary = (
        metrics_df.groupby(["market", "symbol", "strategy"], as_index=False)[
            [
                "total_return",
                "cagr",
                "max_drawdown",
                "sharpe",
                "sortino",
                "volatility",
                "win_rate",
                "avg_win_loss_ratio",
                "profit_factor",
                "expectancy",
                "exposure",
                "trades_count",
                "turnover_avg",
            ]
        ]
        .mean()
    )
    write_csv(summary, out_dir / "metrics_summary.csv")

    policy_summary = summary[summary["strategy"] == "policy"].copy()
    baseline_summary = summary[summary["strategy"] != "policy"].copy()
    compare_rows: List[Dict[str, Any]] = []
    for _, prow in policy_summary.iterrows():
        market = str(prow["market"])
        symbol = str(prow["symbol"])
        p_ret = _safe_float(prow["total_return"])
        p_sharpe = _safe_float(prow["sharpe"])
        p_dd = _safe_float(prow["max_drawdown"])
        subset = baseline_summary[
            (baseline_summary["market"] == market) & (baseline_summary["symbol"] == symbol)
        ]
        for _, brow in subset.iterrows():
            compare_rows.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "baseline": brow["strategy"],
                    "policy_total_return": p_ret,
                    "baseline_total_return": _safe_float(brow["total_return"]),
                    "delta_total_return": p_ret - _safe_float(brow["total_return"]),
                    "policy_sharpe": p_sharpe,
                    "baseline_sharpe": _safe_float(brow["sharpe"]),
                    "delta_sharpe": p_sharpe - _safe_float(brow["sharpe"]),
                    "policy_max_drawdown": p_dd,
                    "baseline_max_drawdown": _safe_float(brow["max_drawdown"]),
                    "delta_max_drawdown": p_dd - _safe_float(brow["max_drawdown"]),
                }
            )
    compare_df = pd.DataFrame(compare_rows)
    write_csv(compare_df, out_dir / "compare_baselines.csv")
    write_csv(pd.DataFrame(cost_stress_rows), out_dir / "cost_stress_matrix.csv")

    metrics_payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config_path": config_path,
        "symbols_count": int(len(universe)),
        "folds_count": int(metrics_df["fold"].nunique()) if "fold" in metrics_df.columns else 0,
        "summary_by_market_strategy": (
            summary.groupby(["market", "strategy"], as_index=False)[
                ["total_return", "sharpe", "max_drawdown", "win_rate", "profit_factor"]
            ]
            .mean()
            .to_dict(orient="records")
        ),
    }
    save_json(metrics_payload, out_dir / "metrics.json")
    save_yaml({"backtest_multi_market": bt_cfg, "policy": policy_cfg}, out_dir / "config_snapshot.yaml")

    print(f"[OK] Multi-market backtest saved -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-market walk-forward backtest (Crypto/A-share/US).")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_multi_market_backtest(args.config)


if __name__ == "__main__":
    main()
