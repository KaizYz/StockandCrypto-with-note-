from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.models.policy import apply_policy_frame, summarize_policy_actions
from src.utils.config import load_config
from src.utils.io import save_json, write_csv


def _load_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _prepare_tracking_frame(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "market" not in out.columns:
        out["market"] = "unknown"
    if "market_type" not in out.columns:
        out["market_type"] = np.where(out["market"].astype(str).eq("crypto"), "spot", "cash")

    if "q50_change_pct" not in out.columns:
        if "predicted_change_pct" in out.columns:
            out["q50_change_pct"] = pd.to_numeric(out["predicted_change_pct"], errors="coerce")
        else:
            out["q50_change_pct"] = float("nan")
    if "q10_change_pct" not in out.columns:
        out["q10_change_pct"] = out["q50_change_pct"] - out["q50_change_pct"].abs().clip(lower=0.005)
    if "q90_change_pct" not in out.columns:
        out["q90_change_pct"] = out["q50_change_pct"] + out["q50_change_pct"].abs().clip(lower=0.005)
    if "volatility_score" not in out.columns:
        out["volatility_score"] = out["q90_change_pct"] - out["q10_change_pct"]
    if "confidence_score" not in out.columns:
        if "total_score" in out.columns:
            out["confidence_score"] = pd.to_numeric(out["total_score"], errors="coerce")
        else:
            out["confidence_score"] = 50.0
    if "p_up" not in out.columns:
        ret_thr = _safe_float(
            cfg.get("policy", {}).get("thresholds", {}).get(
                "ret_threshold",
                cfg.get("forecast_config", {}).get("trend_labels", {}).get("ret_threshold", 0.002),
            )
        )
        if not np.isfinite(ret_thr) or ret_thr <= 0:
            ret_thr = 0.002
        scale = max(ret_thr, 1e-4) * 8.0
        out["p_up"] = 0.5 + 0.5 * np.tanh(pd.to_numeric(out["q50_change_pct"], errors="coerce").fillna(0.0) / scale)
    return out


def run_generate_policy_signals(config_path: str) -> None:
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    processed_dir = Path(paths_cfg.get("processed_data_dir", "data/processed"))
    tracking_dir = processed_dir / "tracking"

    outputs: Dict[str, Dict[str, Any]] = {}

    hourly_path = processed_dir / "session_forecast_hourly.csv"
    hourly = _load_csv_optional(hourly_path)
    if not hourly.empty:
        hourly = hourly.copy()
        hourly["market"] = "crypto"
        if "market_type" not in hourly.columns:
            hourly["market_type"] = "perp"
        hourly_sig = apply_policy_frame(
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
        out_path = processed_dir / "policy_signals_hourly.csv"
        write_csv(hourly_sig, out_path)
        outputs["policy_signals_hourly.csv"] = {
            "rows": int(len(hourly_sig)),
            "summary": summarize_policy_actions(
                hourly_sig,
                group_cols=["symbol", "market_type"],
            ).to_dict(orient="records"),
        }

    daily_path = processed_dir / "session_forecast_daily.csv"
    daily = _load_csv_optional(daily_path)
    if not daily.empty:
        daily = daily.copy()
        daily["market"] = "crypto"
        if "market_type" not in daily.columns:
            daily["market_type"] = "perp"
        daily_sig = apply_policy_frame(
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
        out_path = processed_dir / "policy_signals_daily.csv"
        write_csv(daily_sig, out_path)
        outputs["policy_signals_daily.csv"] = {
            "rows": int(len(daily_sig)),
            "summary": summarize_policy_actions(
                daily_sig,
                group_cols=["symbol", "market_type"],
            ).to_dict(orient="records"),
        }

    tracking_snapshot_path = tracking_dir / "tracking_snapshot.csv"
    tracking_snapshot = _load_csv_optional(tracking_snapshot_path)
    if not tracking_snapshot.empty:
        tracking_for_policy = _prepare_tracking_frame(tracking_snapshot, cfg)
        tracking_sig = apply_policy_frame(
            tracking_for_policy,
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
        out_path = tracking_dir / "policy_signals_multi_market.csv"
        write_csv(tracking_sig, out_path)
        outputs["tracking/policy_signals_multi_market.csv"] = {
            "rows": int(len(tracking_sig)),
            "summary": summarize_policy_actions(
                tracking_sig,
                group_cols=["market"],
            ).to_dict(orient="records"),
        }

    summary_payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config_path": config_path,
        "outputs": outputs,
    }
    save_json(summary_payload, processed_dir / "policy_signals_summary.json")
    print(f"[OK] policy signals generated -> {processed_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate policy-layer buy/sell/flat signals.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_generate_policy_signals(args.config)


if __name__ == "__main__":
    main()

