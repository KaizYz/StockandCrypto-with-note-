from __future__ import annotations

from typing import Dict

import pandas as pd


def build_quality_report(df: pd.DataFrame, ts_col: str = "timestamp_utc") -> Dict[str, float | int | str]:
    if df.empty:
        return {
            "rows": 0,
            "latest_timestamp": "",
            "missing_ratio": 0.0,
            "duplicate_ratio": 0.0,
            "outlier_ratio_close_return_5sigma": 0.0,
        }

    rows = len(df)
    missing_ratio = float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]))
    duplicate_ratio = float(df.duplicated(subset=[ts_col]).mean())

    close_ret = df["close"].pct_change()
    mu = close_ret.mean()
    sigma = close_ret.std(ddof=0)
    if sigma == 0 or pd.isna(sigma):
        outlier_ratio = 0.0
    else:
        outlier_ratio = float(((close_ret - mu).abs() > 5 * sigma).mean())

    latest = str(df[ts_col].iloc[-1])
    return {
        "rows": rows,
        "latest_timestamp": latest,
        "missing_ratio": missing_ratio,
        "duplicate_ratio": duplicate_ratio,
        "outlier_ratio_close_return_5sigma": outlier_ratio,
    }

