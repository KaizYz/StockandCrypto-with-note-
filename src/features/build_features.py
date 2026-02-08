from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.features.news_features import merge_news_features_asof
from src.ingestion.symbol_mapper import canonical_symbol, normalize_market
from src.utils.config import load_config
from src.utils.io import ensure_dir, write_csv


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.fillna(0.0)).cumsum()


def _winsorize_series(s: pd.Series, low_q: float, high_q: float) -> pd.Series:
    lo = s.quantile(low_q)
    hi = s.quantile(high_q)
    return s.clip(lower=lo, upper=hi)


def _build_single_branch_features(
    df: pd.DataFrame, branch_cfg: Dict, global_cfg: Dict
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("timestamp_utc").reset_index(drop=True)

    # Parse timestamps for feature extraction.
    ts_market = pd.to_datetime(out["timestamp_market"], utc=False, errors="coerce")
    out["timestamp_market_dt"] = ts_market

    out["return_1"] = out["close"].pct_change()
    out["log_return_1"] = np.log(out["close"] / out["close"].shift(1))

    lag_returns: List[int] = branch_cfg.get("lag_returns", [])
    for lag in lag_returns:
        out[f"return_lag_{lag}"] = out["close"].pct_change(lag)

    rolling_returns: List[int] = branch_cfg.get("rolling_mean_return", [])
    for w in rolling_returns:
        out[f"return_roll_mean_{w}"] = out["return_1"].rolling(w).mean()

    ema_windows: Iterable[int] = global_cfg.get("features", {}).get(
        "ema_windows", [8, 20, 55, 144, 233]
    )
    for w in ema_windows:
        out[f"ema_{w}"] = _ema(out["close"], w)
        out[f"close_to_ema_{w}"] = out["close"] / out[f"ema_{w}"] - 1.0

    macd_fast, macd_slow, macd_signal = global_cfg.get("features", {}).get(
        "macd_windows", [12, 26, 9]
    )
    out["macd_line"] = _ema(out["close"], macd_fast) - _ema(out["close"], macd_slow)
    out["macd_signal"] = _ema(out["macd_line"], macd_signal)
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]

    # RSI(14)
    delta = out["close"].diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-12)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    vol_windows: List[int] = branch_cfg.get("rolling_vol_windows", [])
    for w in vol_windows:
        out[f"ret_std_{w}"] = out["return_1"].rolling(w).std()

    out["atr_14"] = _atr(out, period=14)
    bb_window = int(global_cfg.get("features", {}).get("bollinger_window", 20))
    bb_std = float(global_cfg.get("features", {}).get("bollinger_std", 2.0))
    ma = out["close"].rolling(bb_window).mean()
    sd = out["close"].rolling(bb_window).std()
    upper = ma + bb_std * sd
    lower = ma - bb_std * sd
    out["bb_width"] = (upper - lower) / (ma + 1e-12)

    out["volume_change_1"] = out["volume"].pct_change()
    vol_ma_windows: List[int] = branch_cfg.get("volume_ma_windows", [])
    for w in vol_ma_windows:
        out[f"volume_ma_{w}"] = out["volume"].rolling(w).mean()
    out["obv"] = _obv(out["close"], out["volume"])

    # Cross-asset quant-style factor proxies for MVP:
    # size/value/growth + momentum/reversal/low-volatility.
    dollar_volume = (out["close"].abs() * out["volume"].abs()).replace(0, np.nan)
    out["factor_size_proxy"] = np.log(dollar_volume.rolling(20).mean())
    rolling_high_252 = out["close"].rolling(252).max()
    out["factor_value_proxy"] = (rolling_high_252 - out["close"]) / (rolling_high_252 + 1e-12)
    out["factor_growth_proxy"] = out["close"].pct_change(90)
    out["factor_momentum_20"] = out["close"].pct_change(20)
    out["factor_reversal_5"] = -out["close"].pct_change(5)
    out["factor_low_vol_20"] = -out["return_1"].rolling(20).std()

    if branch_cfg.get("time_features", "hourly") == "hourly":
        out["hour_of_day"] = out["timestamp_market_dt"].dt.hour
    out["day_of_week"] = out["timestamp_market_dt"].dt.dayofweek
    out["month"] = out["timestamp_market_dt"].dt.month

    # Outlier flag on 1-step return (for model robustness).
    ret_mu = out["return_1"].mean()
    ret_std = out["return_1"].std(ddof=0)
    if pd.isna(ret_std) or ret_std == 0:
        out["outlier_flag"] = 0
    else:
        out["outlier_flag"] = ((out["return_1"] - ret_mu).abs() > 5 * ret_std).astype(int)

    winsor_cfg = global_cfg.get("features", {}).get("winsorize", {})
    if winsor_cfg.get("enabled", True):
        low_q = float(winsor_cfg.get("low_q", 0.01))
        high_q = float(winsor_cfg.get("high_q", 0.99))
        for col in out.select_dtypes(include=[np.number]).columns:
            if col in {"open", "high", "low", "close", "volume"}:
                continue
            out[col] = _winsorize_series(out[col], low_q=low_q, high_q=high_q)

    # Keep timestamp + raw + features. Drop helper DT column.
    out = out.drop(columns=["timestamp_market_dt"])
    return out


def run_build_features(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    news_cfg = cfg.get("news", {}) if isinstance(cfg, dict) else {}

    raw_dir = Path(paths.get("raw_data_dir", "data/raw"))
    out_dir = ensure_dir(paths.get("processed_data_dir", "data/processed"))
    symbol = data_cfg.get("symbol", "BTCUSDT").lower()
    symbol_canon = canonical_symbol(symbol, "crypto")
    market_default = normalize_market(data_cfg.get("market", "crypto"))
    branches = data_cfg.get("branches", {})

    for branch_name, branch_cfg in branches.items():
        raw_path = raw_dir / f"{symbol}_{branch_name}.csv"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw data not found for branch={branch_name}: {raw_path}. Run ingestion first."
            )
        df = pd.read_csv(raw_path)
        numeric_cols = ["open", "high", "low", "close", "volume", "missing_flag"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        feat_df = _build_single_branch_features(df, branch_cfg=branch_cfg, global_cfg=cfg)
        # Merge news features with no-leakage via backward asof on available_at_utc-derived grid.
        if bool(news_cfg.get("enabled", True)):
            feat_df["market"] = market_default
            feat_df["symbol"] = symbol_canon
            try:
                feat_df = merge_news_features_asof(
                    feat_df,
                    ts_col="timestamp_utc",
                    market_col="market",
                    symbol_col="symbol",
                    processed_dir=str(out_dir),
                )
            except Exception as exc:
                print(f"[WARN] merge_news_features_asof failed for {branch_name}: {exc}")
            default_news_values = {
                "news_score_30m": 0.0,
                "news_score_120m": 0.0,
                "news_score_1440m": 0.0,
                "news_count_30m": 0.0,
                "news_count_120m": 0.0,
                "news_count_1440m": 0.0,
                "news_burst_zscore": 0.0,
                "news_pos_neg_ratio": 1.0,
                "news_conflict_score": 0.0,
                "news_event_risk": 0.0,
                "news_gate_pass": 1.0,
            }
            for col, default_value in default_news_values.items():
                if col in feat_df.columns:
                    feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce").fillna(default_value)
            if "news_risk_level" in feat_df.columns:
                feat_df["news_risk_level"] = feat_df["news_risk_level"].fillna("low")
        out_path = out_dir / f"features_{branch_name}.csv"
        write_csv(feat_df, out_path)
        print(f"[OK] Saved features -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build features for hourly/daily branches.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_build_features(args.config)


if __name__ == "__main__":
    main()
