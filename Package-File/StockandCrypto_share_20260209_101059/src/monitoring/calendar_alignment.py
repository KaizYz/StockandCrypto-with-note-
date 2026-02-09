from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.io import save_json, write_csv


def _safe_bool(value: Any) -> bool:
    try:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        text = str(value).strip().lower()
        return text in {"1", "true", "yes", "on"}
    except Exception:
        return False


def _record_check(
    rows: List[Dict[str, Any]],
    *,
    scope: str,
    check_name: str,
    passed: bool,
    required: bool = True,
    detail: str = "",
    metric_value: float | int | str | None = None,
) -> None:
    rows.append(
        {
            "scope": scope,
            "check_name": check_name,
            "passed": bool(passed),
            "required": bool(required),
            "detail": str(detail),
            "metric_value": metric_value,
        }
    )


def _check_raw_branch(
    *,
    raw_path: Path,
    interval: str,
    is_crypto_continuous: bool,
    rows: List[Dict[str, Any]],
) -> None:
    scope = f"raw:{raw_path.name}"
    if not raw_path.exists():
        _record_check(rows, scope=scope, check_name="file_exists", passed=False, required=True, detail="missing raw file")
        return

    try:
        df = pd.read_csv(raw_path)
    except Exception as exc:
        _record_check(rows, scope=scope, check_name="read_csv", passed=False, required=True, detail=f"{type(exc).__name__}:{exc}")
        return
    if "timestamp_utc" not in df.columns:
        _record_check(rows, scope=scope, check_name="timestamp_column", passed=False, required=True, detail="timestamp_utc missing")
        return

    ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    _record_check(
        rows,
        scope=scope,
        check_name="timestamp_parse",
        passed=bool(ts.notna().all()),
        required=True,
        detail=f"invalid_count={int(ts.isna().sum())}",
        metric_value=int(ts.isna().sum()),
    )
    ts_ok = ts.dropna().sort_values()
    if ts_ok.empty:
        return

    monotonic = bool(ts_ok.is_monotonic_increasing)
    dup = int(ts_ok.duplicated().sum())
    _record_check(rows, scope=scope, check_name="timestamp_monotonic", passed=monotonic, required=True, detail="")
    _record_check(rows, scope=scope, check_name="timestamp_no_duplicates", passed=(dup == 0), required=True, detail=f"duplicates={dup}", metric_value=dup)

    interval_key = str(interval).lower()
    if not is_crypto_continuous:
        _record_check(rows, scope=scope, check_name="continuous_coverage", passed=True, required=False, detail="not_applicable_non_crypto")
        return
    if interval_key in {"1h", "hourly", "h"}:
        rng = pd.date_range(ts_ok.min().floor("h"), ts_ok.max().floor("h"), freq="h", tz="UTC")
    elif interval_key in {"1d", "daily", "d"}:
        rng = pd.date_range(ts_ok.min().floor("D"), ts_ok.max().floor("D"), freq="D", tz="UTC")
    else:
        _record_check(rows, scope=scope, check_name="continuous_coverage", passed=True, required=False, detail=f"unsupported_interval={interval}")
        return
    actual = ts_ok.dt.floor(rng.freqstr).nunique() if len(rng) > 0 else 0
    expected = len(rng)
    coverage = float(actual / expected) if expected > 0 else 0.0
    _record_check(
        rows,
        scope=scope,
        check_name="continuous_coverage",
        passed=coverage >= 0.99,
        required=True,
        detail=f"coverage={coverage:.4f}, actual={actual}, expected={expected}",
        metric_value=coverage,
    )


def _check_backtest_equity(processed_dir: Path, rows: List[Dict[str, Any]]) -> None:
    eq_path = processed_dir / "backtest" / "equity.csv"
    scope = "backtest:equity"
    if not eq_path.exists():
        _record_check(rows, scope=scope, check_name="equity_exists", passed=False, required=False, detail="missing backtest/equity.csv")
        return
    try:
        eq = pd.read_csv(eq_path)
    except Exception as exc:
        _record_check(rows, scope=scope, check_name="equity_read", passed=False, required=False, detail=f"{type(exc).__name__}:{exc}")
        return
    if eq.empty or "timestamp_utc" not in eq.columns:
        _record_check(rows, scope=scope, check_name="equity_columns", passed=False, required=False, detail="empty or timestamp_utc missing")
        return

    eq["timestamp_utc_dt"] = pd.to_datetime(eq["timestamp_utc"], utc=True, errors="coerce")
    _record_check(
        rows,
        scope=scope,
        check_name="equity_timestamp_parse",
        passed=bool(eq["timestamp_utc_dt"].notna().all()),
        required=False,
        detail=f"invalid_count={int(eq['timestamp_utc_dt'].isna().sum())}",
    )
    us = eq[eq["market"].astype(str) == "us_equity"].copy() if "market" in eq.columns else pd.DataFrame()
    if not us.empty:
        us["ny_time"] = us["timestamp_utc_dt"].dt.tz_convert("America/New_York")
        dup_local = int(us["ny_time"].duplicated().sum())
        _record_check(
            rows,
            scope=scope,
            check_name="us_dst_local_duplicate",
            passed=(dup_local == 0),
            required=False,
            detail=f"local_duplicates={dup_local}",
            metric_value=dup_local,
        )
    else:
        _record_check(rows, scope=scope, check_name="us_dst_local_duplicate", passed=True, required=False, detail="not_applicable_no_us_data")

    cn = eq[eq["market"].astype(str) == "cn_equity"].copy() if "market" in eq.columns else pd.DataFrame()
    if not cn.empty:
        cn["bj_time"] = cn["timestamp_utc_dt"].dt.tz_convert("Asia/Shanghai")
        weekend = int((cn["bj_time"].dt.weekday >= 5).sum())
        _record_check(
            rows,
            scope=scope,
            check_name="cn_weekend_bar",
            passed=(weekend == 0),
            required=False,
            detail=f"weekend_rows={weekend}",
            metric_value=weekend,
        )
    else:
        _record_check(rows, scope=scope, check_name="cn_weekend_bar", passed=True, required=False, detail="not_applicable_no_cn_data")


def run_calendar_alignment(config_path: str) -> None:
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    raw_dir = Path(paths_cfg.get("raw_data_dir", "data/raw"))
    processed_dir = Path(paths_cfg.get("processed_data_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    symbol = str(data_cfg.get("symbol", "BTCUSDT")).lower()
    is_crypto = str(data_cfg.get("symbol", "")).upper().endswith("USDT")
    branches = data_cfg.get("branches", {}) or {}

    rows: List[Dict[str, Any]] = []
    for branch, branch_cfg in branches.items():
        interval = str((branch_cfg or {}).get("interval", branch))
        raw_path = raw_dir / f"{symbol}_{branch}.csv"
        _check_raw_branch(
            raw_path=raw_path,
            interval=interval,
            is_crypto_continuous=is_crypto,
            rows=rows,
        )

    _check_backtest_equity(processed_dir, rows)

    report_df = pd.DataFrame(rows)
    required_df = report_df[report_df["required"].map(_safe_bool)] if not report_df.empty else pd.DataFrame()
    required_fail = int((~required_df["passed"].map(_safe_bool)).sum()) if not required_df.empty else 0
    all_pass = bool(required_fail == 0)

    write_csv(report_df, processed_dir / "calendar_alignment_report.csv")
    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config_path": config_path,
        "all_pass": all_pass,
        "required_checks": int(len(required_df)),
        "required_failed": required_fail,
        "rows": report_df.to_dict(orient="records"),
    }
    save_json(payload, processed_dir / "calendar_alignment_report.json")
    print(f"[OK] saved calendar alignment report -> {processed_dir / 'calendar_alignment_report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calendar/timezone alignment checks.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_calendar_alignment(args.config)


if __name__ == "__main__":
    main()
