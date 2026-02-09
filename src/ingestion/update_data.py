from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

from src.preprocessing.quality import build_quality_report
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json, write_csv

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def _interval_to_pandas_freq(interval: str) -> str:
    mapping = {"1h": "h", "1d": "D"}
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[interval]


def _interval_to_millis(interval: str) -> int:
    mapping = {"1h": 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000}
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[interval]


def fetch_binance_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    timeout_sec: int = 20,
    base_url: str = "https://api.binance.com",
) -> pd.DataFrame:
    klines_url = f"{base_url.rstrip('/')}/api/v3/klines"
    rows: List[List[str]] = []
    cur = start_ms
    step = _interval_to_millis(interval)

    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(klines_url, params=params, timeout=timeout_sec)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_open_ms = int(batch[-1][0])
        next_cur = last_open_ms + step
        if next_cur <= cur:
            break
        cur = next_cur
        if len(batch) < 1000:
            break

    if not rows:
        return pd.DataFrame(
            columns=[
                "open_time_ms",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df = df[["open_time_ms", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_yahoo_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    timeout_sec: int = 20,
) -> pd.DataFrame:
    interval_map = {"1h": "60m", "1d": "1d"}
    if interval not in interval_map:
        raise ValueError(f"Unsupported Yahoo interval: {interval}")

    url = YAHOO_CHART_URL.format(symbol=symbol)
    params = {
        "interval": interval_map[interval],
        "period1": int(start_ms // 1000),
        "period2": int(end_ms // 1000),
        "includePrePost": "false",
        "events": "div,splits",
    }
    r = requests.get(
        url,
        params=params,
        timeout=timeout_sec,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    r.raise_for_status()
    payload = r.json()
    result = payload.get("chart", {}).get("result", [])
    if not result:
        return pd.DataFrame(columns=["open_time_ms", "open", "high", "low", "close", "volume"])
    item = result[0]
    timestamps = item.get("timestamp", []) or []
    quote_list = item.get("indicators", {}).get("quote", []) or []
    if not timestamps or not quote_list:
        return pd.DataFrame(columns=["open_time_ms", "open", "high", "low", "close", "volume"])

    q = quote_list[0]
    n = min(
        len(timestamps),
        len(q.get("open", []) or []),
        len(q.get("high", []) or []),
        len(q.get("low", []) or []),
        len(q.get("close", []) or []),
        len(q.get("volume", []) or []),
    )
    if n <= 0:
        return pd.DataFrame(columns=["open_time_ms", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        {
            "open_time_ms": (pd.Series(timestamps[:n], dtype="int64") * 1000).astype("int64"),
            "open": pd.to_numeric(pd.Series(q.get("open", [])[:n]), errors="coerce"),
            "high": pd.to_numeric(pd.Series(q.get("high", [])[:n]), errors="coerce"),
            "low": pd.to_numeric(pd.Series(q.get("low", [])[:n]), errors="coerce"),
            "close": pd.to_numeric(pd.Series(q.get("close", [])[:n]), errors="coerce"),
            "volume": pd.to_numeric(pd.Series(q.get("volume", [])[:n]), errors="coerce"),
        }
    )
    return df.dropna(subset=["open_time_ms"]).reset_index(drop=True)


def generate_synthetic_ohlcv(
    interval: str, start_utc: datetime, end_utc: datetime, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    freq = _interval_to_pandas_freq(interval)
    idx = pd.date_range(start_utc, end_utc, freq=freq, inclusive="left", tz="UTC")
    if len(idx) == 0:
        return pd.DataFrame(
            columns=[
                "open_time_ms",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
    rets = rng.normal(0.0001 if interval == "1h" else 0.0005, 0.01, len(idx))
    close = 40000.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0001, 0.01, len(idx)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0001, 0.01, len(idx)))
    volume = rng.lognormal(mean=12.0 if interval == "1h" else 14.0, sigma=0.6, size=len(idx))

    df = pd.DataFrame(
        {
            "open_time_ms": (idx.view("int64") // 10**6).astype(np.int64),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return df


def finalize_ohlcv_df(df: pd.DataFrame, interval: str, market_tz: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "timestamp_market",
                "market_tz",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "missing_flag",
            ]
        )

    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["open_time_ms"], unit="ms", utc=True)
    out = out.drop(columns=["open_time_ms"])
    out = out.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])

    # Build a complete time index to expose missing bars explicitly.
    freq = _interval_to_pandas_freq(interval)
    full_idx = pd.date_range(
        out["timestamp_utc"].min(),
        out["timestamp_utc"].max(),
        freq=freq,
        tz="UTC",
    )
    out = out.set_index("timestamp_utc").reindex(full_idx).rename_axis("timestamp_utc").reset_index()
    out["missing_flag"] = out[["open", "high", "low", "close", "volume"]].isna().any(axis=1).astype(int)
    out["timestamp_market"] = out["timestamp_utc"].dt.tz_convert(market_tz)
    out["market_tz"] = market_tz
    out["timestamp_utc"] = out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["timestamp_market"] = out["timestamp_market"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return out[
        [
            "timestamp_utc",
            "timestamp_market",
            "market_tz",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "missing_flag",
        ]
    ]


def run_update(config_path: str) -> None:
    cfg = load_config(config_path)
    data_cfg: Dict = cfg.get("data", {})
    paths_cfg: Dict = cfg.get("paths", {})
    source = data_cfg.get("source", "binance")
    symbol = data_cfg.get("symbol", "BTCUSDT")
    binance_endpoints = data_cfg.get(
        "binance_endpoints",
        ["https://api.binance.com", "https://api.binance.us"],
    )
    if bool(data_cfg.get("disable_binance_us", False)):
        filtered = [ep for ep in binance_endpoints if "binance.us" not in str(ep).lower()]
        if filtered:
            binance_endpoints = filtered
    raw_dir = ensure_dir(paths_cfg.get("raw_data_dir", "data/raw"))
    processed_dir = ensure_dir(paths_cfg.get("processed_data_dir", "data/processed"))
    seed = int(cfg.get("project", {}).get("seed", 42))

    branches = data_cfg.get("branches", {})
    summary: Dict[str, Dict] = {}

    for branch_name, branch_cfg in branches.items():
        interval = str(branch_cfg.get("interval", "1h"))
        lookback_days = int(branch_cfg.get("lookback_days", 365))
        market_tz = str(branch_cfg.get("market_tz", "Asia/Shanghai"))

        end_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_utc = end_utc - timedelta(days=lookback_days)

        df_raw = pd.DataFrame()
        used_source = source
        fetched_ok = False
        if source == "binance":
            last_exc = None
            for endpoint in binance_endpoints:
                try:
                    df_raw = fetch_binance_klines(
                        symbol=symbol,
                        interval=interval,
                        start_ms=int(start_utc.timestamp() * 1000),
                        end_ms=int(end_utc.timestamp() * 1000),
                        base_url=endpoint,
                    )
                    if df_raw.empty:
                        raise RuntimeError(f"Empty dataset from {endpoint}")
                    used_source = f"binance:{endpoint}"
                    fetched_ok = True
                    break
                except Exception as exc:
                    last_exc = exc
                    continue
            if not fetched_ok:
                used_source = "synthetic_fallback"
                print(
                    f"[WARN] Binance fetch failed for {branch_name}: {last_exc}. Falling back to synthetic data."
                )
        elif source == "yahoo":
            try:
                df_raw = fetch_yahoo_klines(
                    symbol=str(symbol),
                    interval=interval,
                    start_ms=int(start_utc.timestamp() * 1000),
                    end_ms=int(end_utc.timestamp() * 1000),
                )
                if df_raw.empty:
                    raise RuntimeError("empty dataset from yahoo")
                used_source = "yahoo_chart"
                fetched_ok = True
            except Exception as exc:
                used_source = "synthetic_fallback"
                print(
                    f"[WARN] Yahoo fetch failed for {branch_name}: {exc}. Falling back to synthetic data."
                )

        if not fetched_ok:
            df_raw = generate_synthetic_ohlcv(interval, start_utc, end_utc, seed=seed)

        out_df = finalize_ohlcv_df(df_raw, interval=interval, market_tz=market_tz)
        out_path = raw_dir / f"{symbol.lower()}_{branch_name}.csv"
        write_csv(out_df, out_path)

        report = build_quality_report(out_df.rename(columns={"timestamp_utc": "timestamp_utc"}))
        report["source_used"] = used_source
        report["branch"] = branch_name
        report["interval"] = interval
        report_path = processed_dir / f"data_quality_report_{branch_name}.json"
        save_json(report, report_path)
        summary[branch_name] = report
        print(f"[OK] Saved {branch_name} raw data -> {out_path}")
        print(f"[OK] Saved quality report -> {report_path}")

    save_json(summary, processed_dir / "data_quality_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update raw OHLCV data.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()
    run_update(args.config)


if __name__ == "__main__":
    main()
