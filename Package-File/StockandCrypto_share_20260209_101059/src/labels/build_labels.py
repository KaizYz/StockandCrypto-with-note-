from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json, write_csv


def _compute_threshold(abs_ret: pd.Series, q: float, recent_n: int | None = None) -> float:
    s = abs_ret.dropna()
    if recent_n is not None and recent_n > 0 and len(s) > recent_n:
        s = s.iloc[-recent_n:]
    if s.empty:
        return 0.0
    return float(s.quantile(q))


def _map_k_to_window(k: int, window_ends: List[int]) -> int:
    if k <= window_ends[0]:
        return 1
    if k <= window_ends[1]:
        return 2
    return 3


def _build_start_window_label(close: np.ndarray, thr: float, window_ends: List[int]) -> np.ndarray:
    max_k = window_ends[-1]
    n = len(close)
    label = np.full(n, np.nan)
    for i in range(n):
        if i + max_k >= n:
            # Not enough future bars to label.
            continue
        base = close[i]
        if not np.isfinite(base) or base == 0:
            continue
        event_k = None
        for k in range(1, max_k + 1):
            fut = close[i + k]
            if not np.isfinite(fut):
                event_k = None
                break
            ret = fut / base - 1.0
            if abs(ret) >= thr:
                event_k = k
                break
        if event_k is None:
            label[i] = 0  # W0 / no_start
        else:
            label[i] = _map_k_to_window(event_k, window_ends)
    return label


def _build_high_low_targets(df: pd.DataFrame, h: int) -> Tuple[pd.Series, pd.Series]:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    # Rolling over future window by shifting first.
    future_max = high.shift(-1).rolling(window=h, min_periods=h).max().shift(-(h - 1))
    future_min = low.shift(-1).rolling(window=h, min_periods=h).min().shift(-(h - 1))
    y_high = (future_max - close) / close
    y_low = (future_min - close) / close
    return y_high, y_low


def run_build_labels(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    label_cfg = cfg.get("labels", {})
    symbol = data_cfg.get("symbol", "BTCUSDT").lower()

    raw_dir = Path(paths.get("raw_data_dir", "data/raw"))
    out_dir = ensure_dir(paths.get("processed_data_dir", "data/processed"))
    branches: Dict = data_cfg.get("branches", {})
    q = float(label_cfg.get("start_threshold_quantile", 0.8))

    threshold_meta: Dict[str, Dict[str, float | int]] = {}
    for branch_name, branch_cfg in branches.items():
        raw_path = raw_dir / f"{symbol}_{branch_name}.csv"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw data not found for branch={branch_name}: {raw_path}. Run ingestion first."
            )
        df = pd.read_csv(raw_path).sort_values("timestamp_utc").reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume", "missing_flag"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        out = df[["timestamp_utc"]].copy()
        close = df["close"]
        one_step_abs_ret = close.pct_change().abs()

        if branch_name == "hourly":
            recent_n = int(label_cfg.get("threshold_recent_bars_hourly", 24 * 365))
            windows = label_cfg.get("start_windows", {}).get("hourly", [1, 2, 4])
        else:
            recent_n = int(label_cfg.get("threshold_recent_bars_daily", 365 * 5))
            windows = label_cfg.get("start_windows", {}).get("daily", [1, 3, 7])
        windows = [int(w) for w in windows]

        thr = _compute_threshold(one_step_abs_ret, q=q, recent_n=recent_n)
        start_label = _build_start_window_label(close.to_numpy(), thr=thr, window_ends=windows)
        out["y_start_window"] = start_label
        out["y_start_window_name"] = out["y_start_window"].map(
            {0.0: "W0", 1.0: "W1", 2.0: "W2", 3.0: "W3"}
        )

        horizons: List[int] = [int(h) for h in branch_cfg.get("horizons", [1, 2, 4])]
        for h in horizons:
            ret = close.shift(-h) / close - 1.0
            out[f"y_ret_h{h}"] = ret
            out[f"y_dir_h{h}"] = (ret > 0).astype(float)
            if label_cfg.get("include_high_low_targets", False):
                y_high, y_low = _build_high_low_targets(df, h)
                out[f"y_high_h{h}"] = y_high
                out[f"y_low_h{h}"] = y_low

        out_path = out_dir / f"labels_{branch_name}.csv"
        write_csv(out, out_path)
        threshold_meta[branch_name] = {
            "start_threshold_quantile": q,
            "start_threshold_value": thr,
            "windows": windows,
        }
        print(f"[OK] Saved labels -> {out_path}")

    save_json(threshold_meta, out_dir / "label_thresholds.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labels for all branches.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_build_labels(args.config)


if __name__ == "__main__":
    main()

