from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from src.ingestion.update_data import fetch_binance_klines
from src.models.policy import apply_policy_frame
from src.utils.config import load_config
from src.utils.io import write_csv


BJ_TZ = "Asia/Shanghai"

SESSION_ORDER = ["asia", "europe", "us"]
SESSION_HOURS_LABEL = {
    "asia": "08:00-15:59",
    "europe": "16:00-23:59",
    "us": "00:00-07:59",
}
SESSION_NAME_CN = {
    "asia": "亚盘",
    "europe": "欧盘",
    "us": "美盘",
}

BYBIT_BASE_URL = "https://api.bybit.com"
BINANCE_SPOT_ENDPOINTS = ["https://api.binance.com", "https://api.binance.us"]
BINANCE_PERP_ENDPOINTS = ["https://fapi.binance.com"]


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _safe_float(x: Any) -> float:
    try:
        out = float(x)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").upper().strip()
    if not s:
        return "BTCUSDT"
    if s.endswith("USDT"):
        return s
    if "/" in s:
        base = s.split("/")[0]
        return f"{base}USDT"
    if len(s) <= 6:
        return f"{s}USDT"
    return s


def _interval_to_millis(interval: str) -> int:
    mapping = {"1h": 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000}
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[interval]


def _session_from_hour(hour_bj: int) -> str:
    h = int(hour_bj)
    if 8 <= h <= 15:
        return "asia"
    if 16 <= h <= 23:
        return "europe"
    return "us"


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = v.notna() & w.gt(0)
    if not mask.any():
        out = v.mean()
        return float(out) if pd.notna(out) else float("nan")
    return float((v[mask] * w[mask]).sum() / w[mask].sum())


def _fetch_binance_live_price(
    symbol: str,
    market_type: str,
    spot_endpoints: List[str],
    perp_endpoints: List[str],
) -> Tuple[float, str]:
    last_exc: Exception | None = None
    if market_type == "perp":
        for endpoint in perp_endpoints:
            try:
                url = f"{endpoint.rstrip('/')}/fapi/v1/ticker/price"
                r = requests.get(url, params={"symbol": symbol}, timeout=10)
                r.raise_for_status()
                data = r.json()
                return float(data["price"]), f"binance_perp_ticker:{endpoint}"
            except Exception as exc:
                last_exc = exc
                continue
    for endpoint in spot_endpoints:
        try:
            url = f"{endpoint.rstrip('/')}/api/v3/ticker/price"
            r = requests.get(url, params={"symbol": symbol}, timeout=10)
            r.raise_for_status()
            data = r.json()
            return float(data["price"]), f"binance_spot_ticker:{endpoint}"
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to fetch Binance ticker for {symbol}: {last_exc}")


def _fetch_bybit_live_price(symbol: str, market_type: str) -> Tuple[float, str]:
    category = "linear" if market_type == "perp" else "spot"
    url = f"{BYBIT_BASE_URL}/v5/market/tickers"
    params = {"category": category, "symbol": symbol}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    rows = data.get("result", {}).get("list", [])
    if not rows:
        raise RuntimeError(f"Bybit ticker empty for {symbol}/{category}")
    return float(rows[0]["lastPrice"]), f"bybit_{category}_ticker"


def _fetch_binance_klines_with_market(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    market_type: str,
    spot_endpoints: List[str],
    perp_endpoints: List[str],
) -> Tuple[pd.DataFrame, str]:
    last_exc: Exception | None = None
    if market_type == "perp":
        step = _interval_to_millis(interval)
        for endpoint in perp_endpoints:
            rows: List[List[Any]] = []
            cur = start_ms
            try:
                while cur < end_ms:
                    url = f"{endpoint.rstrip('/')}/fapi/v1/klines"
                    params = {
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": cur,
                        "endTime": end_ms,
                        "limit": 1000,
                    }
                    r = requests.get(url, params=params, timeout=20)
                    r.raise_for_status()
                    batch = r.json()
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
                    raise RuntimeError("empty klines")
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
                return df, f"binance_perp_klines:{endpoint}"
            except Exception as exc:
                last_exc = exc
                continue

    for endpoint in spot_endpoints:
        try:
            df = fetch_binance_klines(
                symbol=symbol,
                interval=interval,
                start_ms=start_ms,
                end_ms=end_ms,
                base_url=endpoint,
            )
            if df.empty:
                raise RuntimeError("empty klines")
            return df, f"binance_spot_klines:{endpoint}"
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to fetch Binance klines for {symbol}: {last_exc}")


def _fetch_bybit_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    market_type: str,
) -> Tuple[pd.DataFrame, str]:
    category = "linear" if market_type == "perp" else "spot"
    interval_code = "60" if interval == "1h" else "D"
    step = _interval_to_millis(interval)
    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    rows: List[List[Any]] = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval_code,
            "start": cur,
            "end": end_ms,
            "limit": 1000,
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
        batch = payload.get("result", {}).get("list", [])
        if not batch:
            break
        batch_sorted = sorted(batch, key=lambda x: int(x[0]))
        rows.extend(batch_sorted)
        last_open_ms = int(batch_sorted[-1][0])
        next_cur = last_open_ms + step
        if next_cur <= cur:
            break
        cur = next_cur
        if len(batch_sorted) < 1000:
            break

    if not rows:
        raise RuntimeError(f"Bybit klines empty for {symbol}/{category}/{interval_code}")

    df = pd.DataFrame(rows, columns=["open_time_ms", "open", "high", "low", "close", "volume", "turnover"])
    df = df[["open_time_ms", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, f"bybit_{category}_klines"


def _fetch_history_ohlcv(
    symbol: str,
    exchange: str,
    market_type: str,
    interval: str,
    lookback_days: int,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, str]:
    end_utc = _now_utc().floor("h")
    start_utc = end_utc - pd.Timedelta(days=int(lookback_days))
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)

    fc = cfg.get("forecast_config", {})
    source_cfg = fc.get("data_source", {})
    spot_endpoints = source_cfg.get("binance_spot_endpoints", BINANCE_SPOT_ENDPOINTS)
    perp_endpoints = source_cfg.get("binance_perp_endpoints", BINANCE_PERP_ENDPOINTS)

    if exchange == "binance":
        df, source = _fetch_binance_klines_with_market(
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            market_type=market_type,
            spot_endpoints=spot_endpoints,
            perp_endpoints=perp_endpoints,
        )
    elif exchange == "bybit":
        df, source = _fetch_bybit_klines(
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            market_type=market_type,
        )
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")

    if df.empty:
        raise RuntimeError(f"Empty history for {exchange}/{market_type}/{symbol}/{interval}")

    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["open_time_ms"], unit="ms", utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    out = out.drop_duplicates(subset=["timestamp_utc"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["close"]).reset_index(drop=True)
    return out, source


def _fetch_live_price(
    symbol: str,
    exchange: str,
    market_type: str,
    cfg: Dict[str, Any],
) -> Tuple[float, str]:
    fc = cfg.get("forecast_config", {})
    source_cfg = fc.get("data_source", {})
    spot_endpoints = source_cfg.get("binance_spot_endpoints", BINANCE_SPOT_ENDPOINTS)
    perp_endpoints = source_cfg.get("binance_perp_endpoints", BINANCE_PERP_ENDPOINTS)
    if exchange == "binance":
        return _fetch_binance_live_price(symbol, market_type, spot_endpoints, perp_endpoints)
    if exchange == "bybit":
        return _fetch_bybit_live_price(symbol, market_type)
    raise ValueError(f"Unsupported exchange: {exchange}")


def _build_hourly_profile_from_seasonality(hourly_hist: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    h = int(horizon_hours)
    work = hourly_hist[["timestamp_utc", "close"]].copy()
    work["ret_h"] = work["close"].shift(-h) / work["close"] - 1.0
    work = work.dropna(subset=["ret_h"]).copy()
    work["hour_bj"] = work["timestamp_utc"].dt.tz_convert(BJ_TZ).dt.hour

    global_ret = work["ret_h"]
    g_p_up = float((global_ret > 0).mean()) if len(global_ret) > 0 else 0.5
    g_q10 = float(global_ret.quantile(0.1)) if len(global_ret) > 0 else -0.01
    g_q50 = float(global_ret.quantile(0.5)) if len(global_ret) > 0 else 0.0
    g_q90 = float(global_ret.quantile(0.9)) if len(global_ret) > 0 else 0.01

    grouped = (
        work.groupby("hour_bj")["ret_h"]
        .agg(
            p_up=lambda x: float((x > 0).mean()),
            q10=lambda x: float(x.quantile(0.1)),
            q50=lambda x: float(x.quantile(0.5)),
            q90=lambda x: float(x.quantile(0.9)),
            sample_size="count",
        )
        .reset_index()
    )

    idx_df = pd.DataFrame({"hour_bj": list(range(24))})
    out = idx_df.merge(grouped, on="hour_bj", how="left")
    out["p_up"] = out["p_up"].fillna(g_p_up)
    out["q10"] = out["q10"].fillna(g_q10)
    out["q50"] = out["q50"].fillna(g_q50)
    out["q90"] = out["q90"].fillna(g_q90)
    out["sample_size"] = out["sample_size"].fillna(0).astype(int)
    return out


def _build_hourly_profile_from_model(processed_dir: Path, horizon_hours: int) -> pd.DataFrame:
    path = processed_dir / "predictions_hourly.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    p_col = f"dir_h{int(horizon_hours)}_p_up"
    q10_col = f"ret_h{int(horizon_hours)}_q0.1"
    q50_col = f"ret_h{int(horizon_hours)}_q0.5"
    q90_col = f"ret_h{int(horizon_hours)}_q0.9"
    cols = {"timestamp_market", p_col, q10_col, q50_col, q90_col}
    if not cols.issubset(df.columns):
        return pd.DataFrame()

    work = df[list(cols)].copy()
    work["timestamp_market"] = pd.to_datetime(work["timestamp_market"], errors="coerce")
    work = work.dropna(subset=["timestamp_market"])
    work["hour_bj"] = work["timestamp_market"].dt.hour
    grouped = (
        work.groupby("hour_bj")
        .agg(
            p_up=(p_col, "mean"),
            q10=(q10_col, "mean"),
            q50=(q50_col, "mean"),
            q90=(q90_col, "mean"),
            sample_size=(p_col, "count"),
        )
        .reset_index()
    )
    idx_df = pd.DataFrame({"hour_bj": list(range(24))})
    out = idx_df.merge(grouped, on="hour_bj", how="left")
    return out


def _build_daily_rows_from_seasonality(
    daily_hist: pd.DataFrame,
    lookforward_days: int,
    now_bj: pd.Timestamp,
) -> pd.DataFrame:
    work = daily_hist[["timestamp_utc", "close"]].copy()
    work["ret_1d"] = work["close"].shift(-1) / work["close"] - 1.0
    work = work.dropna(subset=["ret_1d"]).copy()
    work["date_bj"] = work["timestamp_utc"].dt.tz_convert(BJ_TZ).dt.normalize()
    work["day_of_week"] = work["date_bj"].dt.day_name()

    grouped = (
        work.groupby("day_of_week")["ret_1d"]
        .agg(
            p_up=lambda x: float((x > 0).mean()),
            q10=lambda x: float(x.quantile(0.1)),
            q50=lambda x: float(x.quantile(0.5)),
            q90=lambda x: float(x.quantile(0.9)),
            sample_size="count",
        )
        .reset_index()
    )
    grouped_map = {row["day_of_week"]: row for row in grouped.to_dict("records")}
    global_ret = work["ret_1d"]
    fallback = {
        "p_up": float((global_ret > 0).mean()) if len(global_ret) > 0 else 0.5,
        "q10": float(global_ret.quantile(0.1)) if len(global_ret) > 0 else -0.02,
        "q50": float(global_ret.quantile(0.5)) if len(global_ret) > 0 else 0.0,
        "q90": float(global_ret.quantile(0.9)) if len(global_ret) > 0 else 0.02,
        "sample_size": 0,
    }

    rows: List[Dict[str, Any]] = []
    for i in range(1, int(lookforward_days) + 1):
        d = (now_bj.normalize() + pd.Timedelta(days=i)).tz_localize(None)
        dow = pd.Timestamp(d).day_name()
        stats = grouped_map.get(dow, fallback)
        rows.append(
            {
                "day_index": i,
                "date_bj": pd.Timestamp(d),
                "day_of_week": dow,
                "p_up": _safe_float(stats["p_up"]),
                "q10": _safe_float(stats["q10"]),
                "q50": _safe_float(stats["q50"]),
                "q90": _safe_float(stats["q90"]),
                "sample_size": int(stats.get("sample_size", 0)),
                "start_window_top1": "W?",
            }
        )
    return pd.DataFrame(rows)


def _build_daily_rows_from_model(
    processed_dir: Path,
    lookforward_days: int,
    now_bj: pd.Timestamp,
) -> pd.DataFrame:
    path = processed_dir / "predictions_daily.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    cols = {
        "dir_h1_p_up",
        "ret_h1_q0.1",
        "ret_h1_q0.5",
        "ret_h1_q0.9",
        "start_window_name",
    }
    if not cols.issubset(df.columns):
        return pd.DataFrame()
    latest = df.iloc[-1]
    p_up_1 = _safe_float(latest.get("dir_h1_p_up"))
    q10_1 = _safe_float(latest.get("ret_h1_q0.1"))
    q50_1 = _safe_float(latest.get("ret_h1_q0.5"))
    q90_1 = _safe_float(latest.get("ret_h1_q0.9"))
    if not (np.isfinite(p_up_1) and np.isfinite(q10_1) and np.isfinite(q50_1) and np.isfinite(q90_1)):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    start_w = str(latest.get("start_window_name", "W?"))
    for i in range(1, int(lookforward_days) + 1):
        d = (now_bj.normalize() + pd.Timedelta(days=i)).tz_localize(None)
        decay = float(np.exp(-0.12 * (i - 1)))
        p_up_i = 0.5 + (p_up_1 - 0.5) * decay
        rows.append(
            {
                "day_index": i,
                "date_bj": pd.Timestamp(d),
                "day_of_week": pd.Timestamp(d).day_name(),
                "p_up": _safe_float(p_up_i),
                "q10": _safe_float((1.0 + q10_1) ** i - 1.0),
                "q50": _safe_float((1.0 + q50_1) ** i - 1.0),
                "q90": _safe_float((1.0 + q90_1) ** i - 1.0),
                "sample_size": int(len(df)),
                "start_window_top1": start_w,
            }
        )
    return pd.DataFrame(rows)


def _apply_labels_and_prices(
    df: pd.DataFrame,
    current_price: float,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    if df.empty:
        return df
    fc = cfg.get("forecast_config", {})
    trend_cfg = fc.get("trend_labels", {})
    p_bull = float(trend_cfg.get("p_bull", 0.55))
    p_bear = float(trend_cfg.get("p_bear", 0.45))
    ret_thr = float(trend_cfg.get("ret_threshold", 0.002))
    conf_cfg = fc.get("confidence_score", {})
    w_prob = float(conf_cfg.get("w_prob", 0.6))
    w_width = float(conf_cfg.get("w_width", 0.4))
    w_recent = float(conf_cfg.get("w_recent_perf", 0.0))
    _ = w_recent  # reserved for future use

    out = df.copy()
    out["p_up"] = pd.to_numeric(out["p_up"], errors="coerce").clip(0.0, 1.0)
    out["p_down"] = 1.0 - out["p_up"]
    out["q10_change_pct"] = pd.to_numeric(out["q10"], errors="coerce")
    out["q50_change_pct"] = pd.to_numeric(out["q50"], errors="coerce")
    out["q90_change_pct"] = pd.to_numeric(out["q90"], errors="coerce")
    out["volatility_score"] = out["q90_change_pct"] - out["q10_change_pct"]

    out["target_price_q10"] = current_price * (1.0 + out["q10_change_pct"])
    out["target_price_q50"] = current_price * (1.0 + out["q50_change_pct"])
    out["target_price_q90"] = current_price * (1.0 + out["q90_change_pct"])

    vol = out["volatility_score"].astype(float)
    if vol.notna().sum() > 0:
        vmin = float(vol.min())
        vmax = float(vol.max())
    else:
        vmin, vmax = 0.0, 1.0
    if vmax - vmin <= 1e-12:
        vol_norm = pd.Series([0.5] * len(out), index=out.index, dtype=float)
    else:
        vol_norm = ((vol - vmin) / (vmax - vmin)).clip(0.0, 1.0)

    conf_prob = (2.0 * (out["p_up"] - 0.5).abs()).clip(0.0, 1.0)
    conf_width = (1.0 - vol_norm).clip(0.0, 1.0)
    out["confidence_score"] = (100.0 * (w_prob * conf_prob + w_width * conf_width)).clip(0.0, 100.0)

    trend = np.where(
        (out["p_up"] >= p_bull) & (out["q50_change_pct"] >= ret_thr),
        "bullish",
        np.where(
            (out["p_up"] <= p_bear) & (out["q50_change_pct"] <= -ret_thr),
            "bearish",
            "sideways",
        ),
    )
    out["trend_label"] = trend

    q50, q75, q90 = (
        float(vol.quantile(0.50)),
        float(vol.quantile(0.75)),
        float(vol.quantile(0.90)),
    )
    risk = np.where(
        vol < q50,
        "low",
        np.where(vol < q75, "medium", np.where(vol < q90, "high", "extreme")),
    )
    out["risk_level"] = risk
    return out


def _build_session_blocks(hourly_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if hourly_df.empty:
        return hourly_df
    fc = cfg.get("forecast_config", {})
    agg_cfg = fc.get("session_aggregation", {})
    method = str(agg_cfg.get("method", "weighted_mean")).lower()
    weight_field = str(agg_cfg.get("weight_field", "confidence_score"))

    work = hourly_df.copy()
    work["session_name"] = work["hour_bj"].map(_session_from_hour)

    rows: List[Dict[str, Any]] = []
    for sess in SESSION_ORDER:
        sub = work[work["session_name"] == sess].copy()
        if sub.empty:
            continue
        if method == "weighted_mean" and weight_field in sub.columns:
            p_up = _weighted_mean(sub["p_up"], sub[weight_field])
            q10 = _weighted_mean(sub["q10_change_pct"], sub[weight_field])
            q50 = _weighted_mean(sub["q50_change_pct"], sub[weight_field])
            q90 = _weighted_mean(sub["q90_change_pct"], sub[weight_field])
            vol = _weighted_mean(sub["volatility_score"], sub[weight_field])
            conf = _weighted_mean(sub["confidence_score"], sub[weight_field])
        else:
            p_up = float(sub["p_up"].mean())
            q10 = float(sub["q10_change_pct"].mean())
            q50 = float(sub["q50_change_pct"].mean())
            q90 = float(sub["q90_change_pct"].mean())
            vol = float(sub["volatility_score"].mean())
            conf = float(sub["confidence_score"].mean())

        rows.append(
            {
                "session_name": sess,
                "session_name_cn": SESSION_NAME_CN[sess],
                "session_hours": SESSION_HOURS_LABEL[sess],
                "p_up": p_up,
                "p_down": 1.0 - p_up,
                "q10_change_pct": q10,
                "q50_change_pct": q50,
                "q90_change_pct": q90,
                "volatility_score": vol,
                "confidence_score": conf,
                "sample_size": int(sub["sample_size"].sum()) if "sample_size" in sub.columns else int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def _attach_common_columns(
    df: pd.DataFrame,
    *,
    forecast_id: str,
    symbol: str,
    exchange: str,
    exchange_actual: str,
    market_type: str,
    mode_requested: str,
    mode_actual: str,
    horizon_label: str,
    current_price: float,
    forecast_generated_at_bj: str,
    data_updated_at_bj: str,
    model_version: str,
    data_source_actual: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["forecast_id"] = forecast_id
    out["symbol"] = symbol
    out["exchange"] = exchange
    out["exchange_actual"] = exchange_actual
    out["market_type"] = market_type
    out["mode_requested"] = mode_requested
    out["mode"] = mode_actual
    out["horizon"] = horizon_label
    out["current_price"] = current_price
    out["forecast_generated_at_bj"] = forecast_generated_at_bj
    out["data_updated_at_bj"] = data_updated_at_bj
    out["model_version"] = model_version
    out["data_source_actual"] = data_source_actual
    return out


@dataclass
class SessionForecastBundle:
    hourly: pd.DataFrame
    blocks: pd.DataFrame
    daily: pd.DataFrame
    metadata: Dict[str, Any]


def build_session_forecast_bundle(
    *,
    symbol: str,
    exchange: str = "binance",
    market_type: str = "perp",
    mode: str = "forecast",
    horizon_hours: int = 4,
    lookforward_days: int = 14,
    config_path: str = "configs/config.yaml",
) -> SessionForecastBundle:
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    processed_dir = Path(paths_cfg.get("processed_data_dir", "data/processed"))

    symbol_norm = _normalize_symbol(symbol)
    exchange_norm = str(exchange).lower()
    market_type_norm = str(market_type).lower()
    mode_requested = str(mode).lower()
    if mode_requested not in {"forecast", "seasonality"}:
        mode_requested = "forecast"

    seasonality_days = int(
        cfg.get("forecast_config", {}).get("seasonality", {}).get("window_days", 180)
    )

    exchange_actual = exchange_norm
    fallback_note = ""
    try:
        hourly_hist, history_source_hourly = _fetch_history_ohlcv(
            symbol=symbol_norm,
            exchange=exchange_actual,
            market_type=market_type_norm,
            interval="1h",
            lookback_days=max(60, seasonality_days),
            cfg=cfg,
        )
        daily_hist, history_source_daily = _fetch_history_ohlcv(
            symbol=symbol_norm,
            exchange=exchange_actual,
            market_type=market_type_norm,
            interval="1d",
            lookback_days=max(120, seasonality_days * 2),
            cfg=cfg,
        )
        current_price, ticker_source = _fetch_live_price(
            symbol=symbol_norm,
            exchange=exchange_actual,
            market_type=market_type_norm,
            cfg=cfg,
        )
    except Exception as primary_exc:
        if exchange_norm != "binance":
            exchange_actual = "binance"
            fallback_note = f"fallback_from_{exchange_norm}:{type(primary_exc).__name__}"
            hourly_hist, history_source_hourly = _fetch_history_ohlcv(
                symbol=symbol_norm,
                exchange=exchange_actual,
                market_type=market_type_norm,
                interval="1h",
                lookback_days=max(60, seasonality_days),
                cfg=cfg,
            )
            daily_hist, history_source_daily = _fetch_history_ohlcv(
                symbol=symbol_norm,
                exchange=exchange_actual,
                market_type=market_type_norm,
                interval="1d",
                lookback_days=max(120, seasonality_days * 2),
                cfg=cfg,
            )
            current_price, ticker_source = _fetch_live_price(
                symbol=symbol_norm,
                exchange=exchange_actual,
                market_type=market_type_norm,
                cfg=cfg,
            )
        else:
            raise

    now_bj = _now_utc().tz_convert(BJ_TZ)
    generated_bj = now_bj.strftime("%Y-%m-%d %H:%M:%S%z")
    data_updated_bj = (
        hourly_hist["timestamp_utc"].max().tz_convert(BJ_TZ).strftime("%Y-%m-%d %H:%M:%S%z")
    )

    model_symbol = _normalize_symbol(cfg.get("data", {}).get("symbol", "BTCUSDT"))
    allow_model_forecast = (
        mode_requested == "forecast" and exchange_actual == "binance" and symbol_norm == model_symbol
    )

    mode_actual = mode_requested
    model_version = "seasonality_v1"
    forecast_source_note = "seasonality"

    if allow_model_forecast:
        latest_versions_path = Path(paths_cfg.get("models_dir", "data/models")) / "latest_versions.json"
        if latest_versions_path.exists():
            try:
                versions = pd.read_json(latest_versions_path, typ="series")
                model_version = f"hourly={versions.get('hourly','-')}|daily={versions.get('daily','-')}"
            except Exception:
                model_version = "forecast_model_latest"
        else:
            model_version = "forecast_model_latest"
    elif mode_requested == "forecast":
        mode_actual = "seasonality_fallback"

    hourly_profile = pd.DataFrame()
    if allow_model_forecast:
        hourly_profile = _build_hourly_profile_from_model(
            processed_dir=processed_dir,
            horizon_hours=horizon_hours,
        )
        if not hourly_profile.empty:
            forecast_source_note = "model_hourly_profile"
    if hourly_profile.empty:
        hourly_profile = _build_hourly_profile_from_seasonality(
            hourly_hist=hourly_hist,
            horizon_hours=horizon_hours,
        )
        if mode_requested == "forecast":
            mode_actual = "seasonality_fallback"
        forecast_source_note = "seasonality_hourly_profile"

    hourly = _apply_labels_and_prices(hourly_profile, current_price=current_price, cfg=cfg)
    hourly["hour_label"] = hourly["hour_bj"].map(lambda x: f"{int(x):02d}:00")
    hourly["session_name"] = hourly["hour_bj"].map(_session_from_hour)
    hourly["session_name_cn"] = hourly["session_name"].map(SESSION_NAME_CN)

    daily = pd.DataFrame()
    if allow_model_forecast:
        daily = _build_daily_rows_from_model(
            processed_dir=processed_dir,
            lookforward_days=lookforward_days,
            now_bj=now_bj,
        )
        if daily.empty:
            mode_actual = "seasonality_fallback"
    if daily.empty:
        daily = _build_daily_rows_from_seasonality(
            daily_hist=daily_hist,
            lookforward_days=lookforward_days,
            now_bj=now_bj,
        )
    daily = _apply_labels_and_prices(daily, current_price=current_price, cfg=cfg)
    daily["date_bj"] = pd.to_datetime(daily["date_bj"], errors="coerce").dt.strftime("%Y-%m-%d")

    blocks_input = hourly.rename(
        columns={
            "q10_change_pct": "q10_change_pct",
            "q50_change_pct": "q50_change_pct",
            "q90_change_pct": "q90_change_pct",
        }
    )
    blocks = _build_session_blocks(blocks_input, cfg=cfg)
    if not blocks.empty:
        blocks = blocks.rename(
            columns={
                "q10_change_pct": "q10",
                "q50_change_pct": "q50",
                "q90_change_pct": "q90",
            }
        )
        blocks = _apply_labels_and_prices(blocks, current_price=current_price, cfg=cfg)

    if "session_name_cn" not in blocks.columns and "session_name" in blocks.columns:
        blocks["session_name_cn"] = blocks["session_name"].map(SESSION_NAME_CN)
    if "session_hours" not in blocks.columns and "session_name" in blocks.columns:
        blocks["session_hours"] = blocks["session_name"].map(SESSION_HOURS_LABEL)

    forecast_id = (
        f"{symbol_norm}|{exchange_norm}|{market_type_norm}|{mode_requested}|h{int(horizon_hours)}|"
        f"{now_bj.strftime('%Y%m%d%H%M%S')}"
    )
    data_source_actual = (
        f"ticker={ticker_source};hourly={history_source_hourly};daily={history_source_daily};profile={forecast_source_note}"
    )
    if fallback_note:
        data_source_actual = f"{data_source_actual};{fallback_note}"

    hourly = _attach_common_columns(
        hourly,
        forecast_id=forecast_id,
        symbol=symbol_norm,
        exchange=exchange_norm,
        exchange_actual=exchange_actual,
        market_type=market_type_norm,
        mode_requested=mode_requested,
        mode_actual=mode_actual,
        horizon_label=f"{int(horizon_hours)}h",
        current_price=current_price,
        forecast_generated_at_bj=generated_bj,
        data_updated_at_bj=data_updated_bj,
        model_version=model_version,
        data_source_actual=data_source_actual,
    )
    blocks = _attach_common_columns(
        blocks,
        forecast_id=forecast_id,
        symbol=symbol_norm,
        exchange=exchange_norm,
        exchange_actual=exchange_actual,
        market_type=market_type_norm,
        mode_requested=mode_requested,
        mode_actual=mode_actual,
        horizon_label=f"{int(horizon_hours)}h",
        current_price=current_price,
        forecast_generated_at_bj=generated_bj,
        data_updated_at_bj=data_updated_bj,
        model_version=model_version,
        data_source_actual=data_source_actual,
    )
    daily = _attach_common_columns(
        daily,
        forecast_id=forecast_id,
        symbol=symbol_norm,
        exchange=exchange_norm,
        exchange_actual=exchange_actual,
        market_type=market_type_norm,
        mode_requested=mode_requested,
        mode_actual=mode_actual,
        horizon_label="1d",
        current_price=current_price,
        forecast_generated_at_bj=generated_bj,
        data_updated_at_bj=data_updated_bj,
        model_version=model_version,
        data_source_actual=data_source_actual,
    )

    # Add policy-layer signals for direct buy/sell/flat research output.
    hourly = hourly.copy()
    blocks = blocks.copy()
    daily = daily.copy()
    hourly["market"] = "crypto"
    blocks["market"] = "crypto"
    daily["market"] = "crypto"
    hourly = apply_policy_frame(
        hourly,
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
    blocks = apply_policy_frame(
        blocks,
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
    daily = apply_policy_frame(
        daily,
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

    policy_cols = [
        "policy_action",
        "policy_position_size",
        "policy_signed_position",
        "policy_expected_edge_pct",
        "policy_expected_edge_abs",
        "policy_reason",
        "policy_allow_short",
        "policy_cost_pct",
        "policy_uncertainty_width",
        "policy_p_up_used",
        "policy_p_up_source",
    ]

    hourly_cols = [
        "forecast_id",
        "symbol",
        "exchange",
        "exchange_actual",
        "market_type",
        "mode_requested",
        "mode",
        "horizon",
        "hour_bj",
        "hour_label",
        "session_name",
        "session_name_cn",
        "p_up",
        "p_down",
        "q10_change_pct",
        "q50_change_pct",
        "q90_change_pct",
        "volatility_score",
        "target_price_q10",
        "target_price_q50",
        "target_price_q90",
        "trend_label",
        "risk_level",
        "confidence_score",
        "sample_size",
        "current_price",
        "forecast_generated_at_bj",
        "data_updated_at_bj",
        "model_version",
        "data_source_actual",
        *policy_cols,
    ]
    hourly = hourly[[c for c in hourly_cols if c in hourly.columns]]

    block_cols = [
        "forecast_id",
        "symbol",
        "exchange",
        "exchange_actual",
        "market_type",
        "mode_requested",
        "mode",
        "horizon",
        "session_name",
        "session_name_cn",
        "session_hours",
        "p_up",
        "p_down",
        "q10_change_pct",
        "q50_change_pct",
        "q90_change_pct",
        "volatility_score",
        "target_price_q10",
        "target_price_q50",
        "target_price_q90",
        "trend_label",
        "risk_level",
        "confidence_score",
        "sample_size",
        "current_price",
        "forecast_generated_at_bj",
        "data_updated_at_bj",
        "model_version",
        "data_source_actual",
        *policy_cols,
    ]
    blocks = blocks[[c for c in block_cols if c in blocks.columns]]

    daily_cols = [
        "forecast_id",
        "symbol",
        "exchange",
        "exchange_actual",
        "market_type",
        "mode_requested",
        "mode",
        "horizon",
        "day_index",
        "date_bj",
        "day_of_week",
        "p_up",
        "p_down",
        "q10_change_pct",
        "q50_change_pct",
        "q90_change_pct",
        "volatility_score",
        "target_price_q10",
        "target_price_q50",
        "target_price_q90",
        "trend_label",
        "risk_level",
        "confidence_score",
        "sample_size",
        "start_window_top1",
        "current_price",
        "forecast_generated_at_bj",
        "data_updated_at_bj",
        "model_version",
        "data_source_actual",
        *policy_cols,
    ]
    daily = daily[[c for c in daily_cols if c in daily.columns]]

    metadata = {
        "forecast_id": forecast_id,
        "symbol": symbol_norm,
        "exchange": exchange_norm,
        "exchange_actual": exchange_actual,
        "market_type": market_type_norm,
        "mode_requested": mode_requested,
        "mode_actual": mode_actual,
        "horizon_hours": int(horizon_hours),
        "lookforward_days": int(lookforward_days),
        "current_price": current_price,
        "forecast_generated_at_bj": generated_bj,
        "data_updated_at_bj": data_updated_bj,
        "model_version": model_version,
        "data_source_actual": data_source_actual,
    }
    return SessionForecastBundle(hourly=hourly, blocks=blocks, daily=daily, metadata=metadata)


def run_session_forecast(config_path: str) -> None:
    cfg = load_config(config_path)
    fc = cfg.get("forecast_config", {})
    paths_cfg = cfg.get("paths", {})
    processed_dir = Path(paths_cfg.get("processed_data_dir", "data/processed"))

    symbols = fc.get("symbols", {}).get("default", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    source_cfg = fc.get("data_source", {})
    exchanges = source_cfg.get("exchanges", ["binance", "bybit"])
    market_types = source_cfg.get("market_types", ["perp", "spot"])
    horizon_hours = int(fc.get("hourly", {}).get("horizon_hours", 4))
    lookforward_days = int(fc.get("daily", {}).get("lookforward_days", 14))
    modes = fc.get("build_modes", ["forecast", "seasonality"])

    hourly_all: List[pd.DataFrame] = []
    blocks_all: List[pd.DataFrame] = []
    daily_all: List[pd.DataFrame] = []

    for symbol in symbols:
        for exchange in exchanges:
            for market_type in market_types:
                for mode in modes:
                    try:
                        bundle = build_session_forecast_bundle(
                            symbol=str(symbol),
                            exchange=str(exchange),
                            market_type=str(market_type),
                            mode=str(mode),
                            horizon_hours=horizon_hours,
                            lookforward_days=lookforward_days,
                            config_path=config_path,
                        )
                        hourly_all.append(bundle.hourly)
                        blocks_all.append(bundle.blocks)
                        daily_all.append(bundle.daily)
                        print(
                            f"[OK] session_forecast {symbol}/{exchange}/{market_type}/{mode}"
                            f" -> mode_actual={bundle.metadata.get('mode_actual')}"
                        )
                    except Exception as exc:
                        print(
                            f"[WARN] session_forecast failed for {symbol}/{exchange}/{market_type}/{mode}: {exc}"
                        )

    hourly_df = pd.concat(hourly_all, ignore_index=True) if hourly_all else pd.DataFrame()
    blocks_df = pd.concat(blocks_all, ignore_index=True) if blocks_all else pd.DataFrame()
    daily_df = pd.concat(daily_all, ignore_index=True) if daily_all else pd.DataFrame()

    write_csv(hourly_df, processed_dir / "session_forecast_hourly.csv")
    write_csv(blocks_df, processed_dir / "session_forecast_blocks.csv")
    write_csv(daily_df, processed_dir / "session_forecast_daily.csv")
    print(f"[OK] Saved session forecasts -> {processed_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build crypto session forecast outputs.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_session_forecast(args.config)


if __name__ == "__main__":
    main()
