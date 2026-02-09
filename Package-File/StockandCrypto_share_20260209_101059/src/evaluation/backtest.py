from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.policy import apply_policy_frame
from src.utils.config import load_config
from src.utils.io import save_json, write_csv


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _annualization_factor(branch: str) -> float:
    if branch == "hourly":
        return 24.0 * 365.0
    return 365.0


def _run_single_backtest(
    df: pd.DataFrame,
    p_up_col: str,
    ret_col: str,
    up_thr: float,
    down_thr: float,
    fee_bps: float,
    slip_bps: float,
    annual_factor: float,
    q10_col: str | None = None,
    q50_col: str | None = None,
    q90_col: str | None = None,
    cfg: Dict | None = None,
) -> Dict[str, float]:
    use_cols = [p_up_col, ret_col]
    for c in [q10_col, q50_col, q90_col]:
        if c and c in df.columns:
            use_cols.append(c)
    out = df[use_cols].dropna(subset=[ret_col]).copy()
    if out.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "num_trades": 0,
        }

    use_policy = bool(cfg is not None and q50_col and q50_col in out.columns)
    if use_policy:
        policy_input = pd.DataFrame(
            {
                "market": "crypto",
                "market_type": "perp",
                "p_up": pd.to_numeric(out[p_up_col], errors="coerce"),
                "q50_change_pct": pd.to_numeric(out[q50_col], errors="coerce"),
                "q10_change_pct": (
                    pd.to_numeric(out[q10_col], errors="coerce")
                    if q10_col and q10_col in out.columns
                    else float("nan")
                ),
                "q90_change_pct": (
                    pd.to_numeric(out[q90_col], errors="coerce")
                    if q90_col and q90_col in out.columns
                    else float("nan")
                ),
                "volatility_score": (
                    pd.to_numeric(out[q90_col], errors="coerce") - pd.to_numeric(out[q10_col], errors="coerce")
                    if q10_col and q90_col and q10_col in out.columns and q90_col in out.columns
                    else pd.to_numeric(out[q50_col], errors="coerce").abs() * 2.0
                ),
                "confidence_score": 50.0,
                "current_price": 1.0,
                "risk_level": "medium",
            }
        )
        policy_out = apply_policy_frame(policy_input, cfg)
        out["signal_raw"] = np.sign(pd.to_numeric(policy_out["policy_signed_position"], errors="coerce")).fillna(0.0)
        out["position"] = pd.to_numeric(policy_out["policy_signed_position"], errors="coerce").shift(1).fillna(0.0)
    else:
        out["signal_raw"] = np.where(out[p_up_col] > up_thr, 1, np.where(out[p_up_col] < down_thr, -1, 0))
        # t signal -> t+1 execution to avoid look-ahead.
        out["position"] = out["signal_raw"].shift(1).fillna(0.0)

    out["turnover"] = out["position"].diff().abs().fillna(0)
    cost = (fee_bps + slip_bps) / 10000.0
    out["strategy_ret"] = out["position"] * out[ret_col] - out["turnover"] * cost
    out["equity"] = (1.0 + out["strategy_ret"]).cumprod()

    total_ret = float(out["equity"].iloc[-1] - 1.0)
    mu = out["strategy_ret"].mean()
    sd = out["strategy_ret"].std(ddof=0)
    sharpe = float((mu / sd) * np.sqrt(len(out))) if sd and sd > 0 else 0.0
    win_rate = float((out["strategy_ret"] > 0).mean())

    pos_sum = out.loc[out["strategy_ret"] > 0, "strategy_ret"].sum()
    neg_sum = -out.loc[out["strategy_ret"] < 0, "strategy_ret"].sum()
    profit_factor = float(pos_sum / neg_sum) if neg_sum > 0 else 0.0

    return {
        "total_return": total_ret,
        "annualized_return": float((1.0 + total_ret) ** (annual_factor / max(len(out), 1)) - 1.0),
        "max_drawdown": _max_drawdown(out["equity"]),
        "sharpe": sharpe,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": int((out["turnover"] > 0).sum()),
    }


def run_backtest(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    bt_cfg = cfg.get("backtest", {})
    up_thr = float(bt_cfg.get("p_up_long", 0.55))
    down_thr = float(bt_cfg.get("p_up_short", 0.45))
    fee_bps = float(bt_cfg.get("fee_bps", 10))
    slip_bps = float(bt_cfg.get("slippage_bps", 10))

    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    rows: List[Dict] = []

    for branch_name, branch_cfg in data_cfg.get("branches", {}).items():
        pred_path = processed_dir / f"predictions_{branch_name}.csv"
        label_path = processed_dir / f"labels_{branch_name}.csv"
        if not pred_path.exists() or not label_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        labels = pd.read_csv(label_path)
        merged = pred.merge(labels[["timestamp_utc"] + [c for c in labels.columns if c.startswith("y_ret_h")]], on="timestamp_utc", how="inner")

        for h in [int(x) for x in branch_cfg.get("horizons", [1, 2, 4])]:
            p_col = f"dir_h{h}_p_up"
            r_col = f"y_ret_h{h}"
            if p_col not in merged.columns or r_col not in merged.columns:
                continue
            m = _run_single_backtest(
                merged,
                p_up_col=p_col,
                ret_col=r_col,
                up_thr=up_thr,
                down_thr=down_thr,
                fee_bps=fee_bps,
                slip_bps=slip_bps,
                annual_factor=_annualization_factor(branch_name),
                q10_col=f"ret_h{h}_q0.1",
                q50_col=f"ret_h{h}_q0.5",
                q90_col=f"ret_h{h}_q0.9",
                cfg=cfg,
            )
            m.update({"branch": branch_name, "horizon": h})
            rows.append(m)

    if not rows:
        print("[WARN] No backtest results generated.")
        return
    df = pd.DataFrame(rows)
    write_csv(df, processed_dir / "backtest_metrics.csv")
    save_json({"rows": rows}, processed_dir / "backtest_metrics.json")
    print(f"[OK] Saved backtest metrics -> {processed_dir / 'backtest_metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight paper backtest with costs and delay.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_backtest(args.config)


if __name__ == "__main__":
    main()
