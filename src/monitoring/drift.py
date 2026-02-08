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


def _compute_psi(base: pd.Series, recent: pd.Series, bins: int = 10) -> float:
    b = pd.to_numeric(base, errors="coerce").dropna().to_numpy()
    r = pd.to_numeric(recent, errors="coerce").dropna().to_numpy()
    if len(b) < 20 or len(r) < 20:
        return float("nan")
    edges = np.quantile(b, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    b_hist, _ = np.histogram(b, bins=edges)
    r_hist, _ = np.histogram(r, bins=edges)
    b_pct = np.clip(b_hist / max(len(b), 1), 1e-6, None)
    r_pct = np.clip(r_hist / max(len(r), 1), 1e-6, None)
    psi = np.sum((r_pct - b_pct) * np.log(r_pct / b_pct))
    return float(psi)


def _ks_distance(base: pd.Series, recent: pd.Series) -> float:
    b = np.sort(pd.to_numeric(base, errors="coerce").dropna().to_numpy())
    r = np.sort(pd.to_numeric(recent, errors="coerce").dropna().to_numpy())
    if len(b) < 20 or len(r) < 20:
        return float("nan")
    grid = np.unique(np.concatenate([b, r]))
    b_cdf = np.searchsorted(b, grid, side="right") / len(b)
    r_cdf = np.searchsorted(r, grid, side="right") / len(r)
    return float(np.max(np.abs(b_cdf - r_cdf)))


def _alert_level(score: float, cfg: Dict[str, Any]) -> str:
    yellow = float(cfg.get("yellow", 0.10))
    orange = float(cfg.get("orange", 0.25))
    red = float(cfg.get("red", 0.40))
    if not np.isfinite(score):
        return "yellow"
    if score >= red:
        return "red"
    if score >= orange:
        return "orange"
    if score >= yellow:
        return "yellow"
    return "green"


def run_drift_monitor(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    backtest_dir = processed_dir / "backtest"
    monitor_cfg = (cfg.get("monitoring", {}) or {}).get("drift", {})

    rows: List[Dict[str, Any]] = []

    # 1) Metric drift from backtest folds.
    metrics_path = backtest_dir / "metrics_by_fold.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if not metrics.empty and {"market", "symbol", "strategy", "fold"}.issubset(metrics.columns):
            metrics["fold"] = pd.to_numeric(metrics["fold"], errors="coerce")
            metrics = metrics.dropna(subset=["fold"])
            metrics["fold"] = metrics["fold"].astype(int)
            numeric_cols = [
                c
                for c in ["total_return", "sharpe", "max_drawdown", "win_rate", "profit_factor"]
                if c in metrics.columns
            ]
            for (market, symbol, strategy), grp in metrics.groupby(["market", "symbol", "strategy"]):
                grp = grp.sort_values("fold")
                recent_n = max(1, int(np.ceil(len(grp) * 0.25)))
                recent = grp.tail(recent_n)
                base = grp.iloc[: max(len(grp) - recent_n, 1)]
                if base.empty:
                    base = grp
                drift_score = 0.0
                reasons: List[str] = []
                for col in numeric_cols:
                    b = _safe_float(base[col].mean())
                    r = _safe_float(recent[col].mean())
                    if not np.isfinite(b) or not np.isfinite(r):
                        continue
                    if col == "max_drawdown":
                        # closer to zero is better
                        delta = abs(r) - abs(b)
                        if delta > 0:
                            reasons.append(f"{col}_worse")
                            drift_score += min(delta / max(abs(b), 1e-6), 1.0)
                    else:
                        delta = b - r
                        if delta > 0:
                            reasons.append(f"{col}_worse")
                            drift_score += min(delta / max(abs(b), 1e-6), 1.0)
                drift_score = drift_score / max(len(numeric_cols), 1)
                level = _alert_level(drift_score, monitor_cfg.get("metric_alerts", {}))
                rows.append(
                    {
                        "scope": "metric",
                        "market": market,
                        "symbol": symbol,
                        "strategy": strategy,
                        "drift_score": drift_score,
                        "alert_level": level,
                        "reason": ";".join(sorted(set(reasons))) if reasons else "stable",
                    }
                )

    # 2) Data drift from feature distributions.
    for feat_path in sorted(processed_dir.glob("features_*.csv")):
        branch = feat_path.stem.replace("features_", "")
        feat = pd.read_csv(feat_path)
        if len(feat) < 200:
            continue
        numeric_cols = [
            c
            for c in feat.columns
            if pd.api.types.is_numeric_dtype(feat[c]) and c not in {"missing_flag"}
        ]
        if not numeric_cols:
            continue
        n = len(feat)
        split = int(n * 0.8)
        base = feat.iloc[:split]
        recent = feat.iloc[split:]
        psi_vals: List[float] = []
        ks_vals: List[float] = []
        for col in numeric_cols:
            psi = _compute_psi(base[col], recent[col], bins=10)
            ks = _ks_distance(base[col], recent[col])
            if np.isfinite(psi):
                psi_vals.append(psi)
            if np.isfinite(ks):
                ks_vals.append(ks)
        psi_mean = float(np.mean(psi_vals)) if psi_vals else float("nan")
        ks_mean = float(np.mean(ks_vals)) if ks_vals else float("nan")
        score = np.nanmax([psi_mean, ks_mean]) if (np.isfinite(psi_mean) or np.isfinite(ks_mean)) else float("nan")
        level = _alert_level(score, monitor_cfg.get("data_alerts", {}))
        rows.append(
            {
                "scope": "data",
                "market": branch,
                "symbol": branch,
                "strategy": "feature_distribution",
                "drift_score": score,
                "psi_mean": psi_mean,
                "ks_mean": ks_mean,
                "alert_level": level,
                "reason": "feature_distribution_shift",
            }
        )

    monitor_df = pd.DataFrame(rows)
    if not monitor_df.empty:
        monitor_df = monitor_df.sort_values(["alert_level", "drift_score"], ascending=[False, False]).reset_index(drop=True)
    write_csv(monitor_df, processed_dir / "drift_monitor_daily.csv")

    alerts = monitor_df[monitor_df["alert_level"].isin(["orange", "red"])].copy() if not monitor_df.empty else pd.DataFrame()
    log_path = processed_dir / "drift_alerts.log"
    if alerts.empty:
        text = f"{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')} | no_orange_red_alerts\n"
    else:
        lines = []
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
        for _, row in alerts.iterrows():
            lines.append(
                f"{ts} | {row.get('alert_level')} | {row.get('scope')} | {row.get('market')}/{row.get('symbol')} | "
                f"score={row.get('drift_score')} | {row.get('reason')}"
            )
        text = "\n".join(lines) + "\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)

    save_json(
        {
            "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
            "alerts_count": int(len(alerts)),
            "rows": monitor_df.to_dict(orient="records"),
        },
        processed_dir / "drift_monitor_daily.json",
    )
    print(f"[OK] saved drift monitor -> {processed_dir / 'drift_monitor_daily.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute daily drift monitor and alerts.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_drift_monitor(args.config)


if __name__ == "__main__":
    main()
