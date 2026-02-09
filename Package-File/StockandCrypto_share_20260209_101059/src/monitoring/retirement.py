from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.io import save_json, write_csv


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def run_retirement_check(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    backtest_dir = processed_dir / "backtest"
    retire_cfg = (cfg.get("monitoring", {}) or {}).get("retirement", {})
    consecutive_non_positive_limit = int(retire_cfg.get("consecutive_non_positive_limit", 3))
    allow_red_alerts = int(retire_cfg.get("allow_red_alerts", 0))

    metrics_path = backtest_dir / "metrics_by_fold.csv"
    drift_path = processed_dir / "drift_monitor_daily.csv"
    integrity_path = processed_dir / "data_integrity_checks.csv"

    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    drift = pd.read_csv(drift_path) if drift_path.exists() else pd.DataFrame()
    integrity = pd.read_csv(integrity_path) if integrity_path.exists() else pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    if not metrics.empty:
        policy = metrics[metrics["strategy"].astype(str) == "policy"].copy()
        if not policy.empty:
            for (market, symbol), grp in policy.groupby(["market", "symbol"]):
                grp = grp.sort_values("fold")
                ret = pd.to_numeric(grp.get("total_return"), errors="coerce").fillna(0.0)
                non_pos = (ret <= 0).astype(int)
                consec = 0
                max_consec = 0
                for v in non_pos.tolist():
                    if v == 1:
                        consec += 1
                        max_consec = max(max_consec, consec)
                    else:
                        consec = 0
                red_alerts = 0
                if not drift.empty:
                    red_alerts = int(
                        (
                            (drift["market"].astype(str) == str(market))
                            & (drift["symbol"].astype(str) == str(symbol))
                            & (drift["alert_level"].astype(str) == "red")
                        ).sum()
                    )
                integrity_fail = 0
                if not integrity.empty and {"branch", "pass"}.issubset(integrity.columns):
                    integrity_fail = int((~integrity["pass"].astype(bool)).sum())
                retire = bool(
                    (max_consec >= consecutive_non_positive_limit)
                    or (red_alerts > allow_red_alerts)
                    or (integrity_fail > 0)
                )
                reasons: List[str] = []
                if max_consec >= consecutive_non_positive_limit:
                    reasons.append("consecutive_non_positive_edge")
                if red_alerts > allow_red_alerts:
                    reasons.append("red_drift_alerts")
                if integrity_fail > 0:
                    reasons.append("data_gate_failed")
                rows.append(
                    {
                        "market": market,
                        "symbol": symbol,
                        "max_consecutive_non_positive_folds": int(max_consec),
                        "red_alerts": int(red_alerts),
                        "integrity_fail_count": int(integrity_fail),
                        "retire": retire,
                        "reason": ";".join(reasons) if reasons else "healthy",
                    }
                )

    status_df = pd.DataFrame(rows)
    if not status_df.empty:
        status_df = status_df.sort_values(["retire", "market", "symbol"], ascending=[False, True, True])
    write_csv(status_df, processed_dir / "model_status.csv")

    retired = status_df[status_df["retire"] == True] if not status_df.empty else pd.DataFrame()  # noqa: E712
    report_lines = [
        "# Retirement Report",
        "",
        f"- Generated at: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Retired symbols: {int(len(retired))}",
        "",
    ]
    if retired.empty:
        report_lines.append("- No model retirement triggered.")
    else:
        report_lines.append(retired.to_markdown(index=False))
    (processed_dir / "retirement_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    save_json(
        {
            "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
            "rows": status_df.to_dict(orient="records"),
        },
        processed_dir / "model_status.json",
    )
    print(f"[OK] saved retirement status -> {processed_dir / 'model_status.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model retirement rules.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_retirement_check(args.config)


if __name__ == "__main__":
    main()
