from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json, write_csv


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _default_gate_cfg() -> Dict[str, Any]:
    return {
        "enabled": True,
        "fail_fast": True,
        "max_missing_rate": 0.05,
        "max_duplicate_ratio": 0.0,
        "require_monotonic_timestamp": True,
        "allow_future_timestamps": False,
        "require_ohlc_validity": True,
        "min_history_bars": {"hourly": 1800, "daily": 500},
    }


def _compute_branch_integrity(
    *,
    df: pd.DataFrame,
    branch: str,
    interval: str,
    gate_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    work = df.copy()
    if "timestamp_utc" not in work.columns:
        raise ValueError(f"{branch}: missing column timestamp_utc")
    work["timestamp_utc_dt"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")

    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    rows = int(len(work))
    history_bars = int(work["close"].notna().sum())
    missing_ratio = float(
        work[required_cols + ["timestamp_utc_dt"]].isna().sum().sum()
        / max(rows * (len(required_cols) + 1), 1)
    )
    duplicate_ratio = float(work["timestamp_utc_dt"].duplicated().mean()) if rows > 0 else 0.0

    monotonic = bool(work["timestamp_utc_dt"].dropna().is_monotonic_increasing)
    future_count = int((work["timestamp_utc_dt"] > pd.Timestamp.now(tz="UTC")).sum())
    invalid_ts_count = int(work["timestamp_utc_dt"].isna().sum())

    ohlc_invalid = (
        (work["low"] > work["high"])
        | (work["open"] < work["low"])
        | (work["open"] > work["high"])
        | (work["close"] < work["low"])
        | (work["close"] > work["high"])
        | (work["volume"] < 0)
    )
    ohlc_invalid_count = int(ohlc_invalid.fillna(False).sum())

    min_history_cfg = gate_cfg.get("min_history_bars", {})
    min_history_req = int(min_history_cfg.get(branch, min_history_cfg.get(interval, 180)))
    max_missing_rate = float(gate_cfg.get("max_missing_rate", 0.05))
    max_duplicate_ratio = float(gate_cfg.get("max_duplicate_ratio", 0.0))
    require_monotonic = bool(gate_cfg.get("require_monotonic_timestamp", True))
    allow_future = bool(gate_cfg.get("allow_future_timestamps", False))
    require_ohlc_validity = bool(gate_cfg.get("require_ohlc_validity", True))

    checks = {
        "history_bars_ok": history_bars >= min_history_req,
        "missing_ratio_ok": missing_ratio <= max_missing_rate,
        "duplicate_ratio_ok": duplicate_ratio <= max_duplicate_ratio,
        "timestamp_parse_ok": invalid_ts_count == 0,
        "timestamp_monotonic_ok": (monotonic if require_monotonic else True),
        "future_timestamp_ok": (future_count == 0 if not allow_future else True),
        "ohlc_validity_ok": (ohlc_invalid_count == 0 if require_ohlc_validity else True),
    }
    passed = bool(all(checks.values()))
    return {
        "branch": branch,
        "interval": interval,
        "rows": rows,
        "history_bars": history_bars,
        "min_history_required": min_history_req,
        "missing_ratio": missing_ratio,
        "max_missing_allowed": max_missing_rate,
        "duplicate_ratio": duplicate_ratio,
        "max_duplicate_allowed": max_duplicate_ratio,
        "timestamp_monotonic": monotonic,
        "invalid_timestamp_count": invalid_ts_count,
        "future_timestamp_count": future_count,
        "ohlc_invalid_count": ohlc_invalid_count,
        "checks": checks,
        "pass": passed,
    }


def run_validate_data(config_path: str) -> None:
    cfg = load_config(config_path)
    gate_cfg = _default_gate_cfg()
    gate_cfg = {**gate_cfg, **(cfg.get("data_gate", {}) or {})}
    enabled = bool(gate_cfg.get("enabled", True))
    if not enabled:
        print("[WARN] data_gate disabled; skip validation.")
        return

    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    raw_dir = Path(paths_cfg.get("raw_data_dir", "data/raw"))
    processed_dir = ensure_dir(paths_cfg.get("processed_data_dir", "data/processed"))
    symbol = str(data_cfg.get("symbol", "BTCUSDT")).lower()

    results: List[Dict[str, Any]] = []
    for branch, branch_cfg in (data_cfg.get("branches", {}) or {}).items():
        interval = str(branch_cfg.get("interval", branch))
        raw_path = raw_dir / f"{symbol}_{branch}.csv"
        if not raw_path.exists():
            results.append(
                {
                    "branch": branch,
                    "interval": interval,
                    "pass": False,
                    "error": f"raw_file_missing:{raw_path}",
                }
            )
            continue
        try:
            df = pd.read_csv(raw_path)
            integrity = _compute_branch_integrity(
                df=df,
                branch=str(branch),
                interval=str(interval),
                gate_cfg=gate_cfg,
            )
            integrity["raw_path"] = str(raw_path)
            results.append(integrity)
        except Exception as exc:
            results.append(
                {
                    "branch": branch,
                    "interval": interval,
                    "pass": False,
                    "error": f"{type(exc).__name__}:{exc}",
                    "raw_path": str(raw_path),
                }
            )

    passed_count = int(sum(1 for row in results if bool(row.get("pass", False))))
    total = int(len(results))
    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config_path": config_path,
        "summary": {
            "total_branches": total,
            "passed_branches": passed_count,
            "failed_branches": total - passed_count,
            "pass_rate": float(passed_count / total) if total > 0 else 0.0,
        },
        "rows": results,
    }

    save_json(payload, Path(processed_dir) / "data_integrity_checks.json")
    write_csv(pd.json_normalize(results), Path(processed_dir) / "data_integrity_checks.csv")
    print(f"[OK] saved data integrity checks -> {Path(processed_dir) / 'data_integrity_checks.json'}")

    fail_fast = bool(gate_cfg.get("fail_fast", True))
    if fail_fast and passed_count < total:
        failed = [r.get("branch", "?") for r in results if not bool(r.get("pass", False))]
        raise RuntimeError(f"Data Gate failed for branches: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw data integrity (Data Gate).")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_validate_data(args.config)


if __name__ == "__main__":
    main()
