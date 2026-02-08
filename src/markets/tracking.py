from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.features.news_features import merge_latest_news_features
from src.markets.snapshot import build_market_snapshot_from_instruments
from src.markets.universe import get_universe_catalog, load_universe
from src.models.policy import apply_policy_frame
from src.utils.config import load_config
from src.utils.io import save_json, write_csv


DEFAULT_UNIVERSES: Dict[str, List[str]] = {
    "crypto": ["top100_ex_stable"],
    "cn_equity": ["sse_composite", "csi300"],
    "us_equity": ["dow30", "nasdaq100", "sp500"],
}

DEFAULT_MAX_SYMBOLS_PER_POOL: Dict[str, int] = {
    "crypto": 100,
    "cn_equity": 80,
    "us_equity": 120,
}

DEFAULT_MARKET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "crypto": {
        "timezone": "Asia/Shanghai",
        "horizon_unit": "hour",
        "horizon_steps": 4,
        "history_lookback_days": 365,
    },
    "cn_equity": {
        "timezone": "Asia/Shanghai",
        "horizon_unit": "day",
        "horizon_steps": 3,
        "history_lookback_days": 730,
    },
    "us_equity": {
        "timezone": "America/New_York",
        "horizon_unit": "day",
        "horizon_steps": 3,
        "history_lookback_days": 730,
    },
}


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _to_bool_series(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def _normalize_universe_df(df: pd.DataFrame, market: str, pool_key: str) -> pd.DataFrame:
    out = df.copy()
    out["market"] = market
    out["pool_key"] = pool_key

    if "provider" not in out.columns:
        if market == "crypto":
            out["provider"] = "binance"
        else:
            out["provider"] = "yahoo"
    out["provider"] = out["provider"].astype(str)

    if "snapshot_symbol" not in out.columns:
        out["snapshot_symbol"] = out["symbol"].astype(str)
    out["snapshot_symbol"] = out["snapshot_symbol"].astype(str)

    if "display" not in out.columns:
        if "name" in out.columns and "symbol" in out.columns:
            out["display"] = out["symbol"].astype(str) + " " + out["name"].astype(str)
        else:
            out["display"] = out.get("symbol", "").astype(str)

    # Unified instrument id for joining snapshot output.
    out["instrument_id"] = (
        out["market"].astype(str) + ":" + out["snapshot_symbol"].astype(str).str.lower()
    )
    return out


def _load_all_universes(cfg: Dict[str, Any]) -> pd.DataFrame:
    tracking_cfg = cfg.get("tracking", {})
    selected_universes = tracking_cfg.get("universes", DEFAULT_UNIVERSES)
    max_symbols_cfg = tracking_cfg.get("max_symbols_per_pool", DEFAULT_MAX_SYMBOLS_PER_POOL)

    frames: List[pd.DataFrame] = []
    for market, pools in selected_universes.items():
        for pool_key in pools:
            try:
                uni = load_universe(market, pool_key)
            except Exception as exc:
                # CoinGecko may return 429 for large universe pulls; fallback to top3 for availability.
                if market == "crypto" and pool_key == "top100_ex_stable":
                    print(
                        f"[WARN] load_universe failed for crypto/top100_ex_stable ({type(exc).__name__}); fallback to top3."
                    )
                    uni = load_universe("crypto", "top3")
                else:
                    print(f"[WARN] load_universe failed for {market}/{pool_key}: {exc}")
                    continue
            max_n = int(max_symbols_cfg.get(market, DEFAULT_MAX_SYMBOLS_PER_POOL.get(market, 100)))
            if max_n > 0:
                uni = uni.head(max_n).reset_index(drop=True)
            frames.append(_normalize_universe_df(uni, market=market, pool_key=pool_key))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["market", "snapshot_symbol"]).reset_index(drop=True)
    return out


def _build_snapshot_instruments(meta: pd.DataFrame, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracking_cfg = cfg.get("tracking", {})
    market_defaults = tracking_cfg.get("market_defaults", DEFAULT_MARKET_DEFAULTS)
    instruments: List[Dict[str, Any]] = []

    for _, row in meta.iterrows():
        market = str(row.get("market", ""))
        defaults = market_defaults.get(market, DEFAULT_MARKET_DEFAULTS.get(market, {}))
        timezone = str(defaults.get("timezone", "UTC"))
        horizon_unit = str(defaults.get("horizon_unit", "day"))
        horizon_steps = int(defaults.get("horizon_steps", 1))
        lookback = int(defaults.get("history_lookback_days", 365))
        instruments.append(
            {
                "id": str(row["instrument_id"]),
                "name": str(row.get("name", row.get("display", row.get("snapshot_symbol", "")))),
                "market": market,
                "symbol": str(row.get("snapshot_symbol", "")),
                "provider": str(row.get("provider", "yahoo")),
                "timezone": timezone,
                "horizon_unit": horizon_unit,
                "horizon_steps": horizon_steps,
                "history_lookback_days": lookback,
            }
        )
    return instruments


def _build_snapshot_parallel(
    instruments: List[Dict[str, Any]], config_path: str, max_workers: int
) -> pd.DataFrame:
    if not instruments:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []

    def _worker(inst: Dict[str, Any]) -> Dict[str, Any]:
        df = build_market_snapshot_from_instruments([inst], config_path=config_path)
        if df.empty:
            return {
                "instrument_id": str(inst.get("id", "")),
                "current_price": np.nan,
                "predicted_price": np.nan,
                "predicted_change_pct": np.nan,
                "price_source": "error:empty_snapshot",
            }
        return df.iloc[0].to_dict()

    workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_worker, inst): inst for inst in instruments}
        for future in as_completed(future_map):
            inst = future_map[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append(
                    {
                        "instrument_id": str(inst.get("id", "")),
                        "current_price": np.nan,
                        "predicted_price": np.nan,
                        "predicted_change_pct": np.nan,
                        "price_source": f"error:{type(exc).__name__}",
                        "error_message": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def _score_and_state(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    tracking_cfg = cfg.get("tracking", {})
    hard_cfg = tracking_cfg.get("hard_filters", {})
    action_cfg = tracking_cfg.get("action_rules", {})

    min_history_bars = int(hard_cfg.get("min_history_bars", 180))
    max_missing_rate = float(hard_cfg.get("max_missing_rate", 0.05))
    min_market_cap = float(hard_cfg.get("min_market_cap_usd", 1_000_000_000.0))
    min_pred_change = float(action_cfg.get("min_predicted_change_pct", 0.0))
    min_factor_support = int(action_cfg.get("min_factor_support_count", 2))

    out = df.copy()
    out["prediction_available"] = out["current_price"].map(np.isfinite) & out["predicted_price"].map(np.isfinite)
    out["risk_factors_available"] = (
        out["size_factor"].map(np.isfinite)
        & out["value_factor"].map(np.isfinite)
        & out["growth_factor"].map(np.isfinite)
    )
    out["behavior_factors_available"] = (
        out["momentum_factor"].map(np.isfinite)
        & out["reversal_factor"].map(np.isfinite)
        & out["low_vol_factor"].map(np.isfinite)
    )
    out["factors_available"] = _to_bool_series(out["risk_factors_available"] | out["behavior_factors_available"])

    out["history_ok"] = out["history_bars"].map(lambda x: _safe_float(x) >= min_history_bars)
    out["missing_ok"] = out["history_missing_rate"].map(
        lambda x: np.isfinite(_safe_float(x)) and _safe_float(x) <= max_missing_rate
    )
    out["liquidity_ok"] = out["market_cap_usd"].map(
        lambda x: (_safe_float(x) >= min_market_cap) if np.isfinite(_safe_float(x)) else False
    )
    # For instruments without market cap, allow fallback via valid prediction + factors.
    out.loc[~out["market_cap_usd"].map(np.isfinite), "liquidity_ok"] = (
        out["prediction_available"] & out["behavior_factors_available"]
    )
    out["hard_filter_pass"] = _to_bool_series(
        out["prediction_available"] & out["history_ok"] & out["missing_ok"] & out["liquidity_ok"]
    )

    cap_rank = (
        out.groupby("market")["market_cap_usd"]
        .rank(method="average", pct=True)
        .fillna(0.5)
        .clip(0.0, 1.0)
    )
    out["liquidity_score"] = 30.0 * cap_rank

    missing_component = (
        (1.0 - (out["history_missing_rate"].fillna(1.0) / max(max_missing_rate, 1e-9)).clip(0.0, 1.0))
        * 15.0
    )
    pred_component = out["prediction_available"].astype(float) * 10.0
    factor_component = out["factors_available"].astype(float) * 5.0
    out["data_quality_score"] = (missing_component + pred_component + factor_component).clip(0.0, 30.0)

    history_ratio = (out["history_bars"].fillna(0.0) / max(min_history_bars, 1)).clip(0.0, 1.0)
    out["history_score"] = 20.0 * history_ratio

    support_hourly = (out["market"] == "crypto").astype(float)
    support_daily = 1.0
    coverage = (
        support_daily * 8.0
        + support_hourly * 4.0
        + out["risk_factors_available"].astype(float) * 4.0
        + out["behavior_factors_available"].astype(float) * 4.0
    )
    out["coverage_score"] = coverage.clip(0.0, 20.0)

    out["total_score_raw"] = (
        out["liquidity_score"] + out["data_quality_score"] + out["history_score"] + out["coverage_score"]
    )
    out["total_score"] = out["total_score_raw"] - (~out["hard_filter_pass"]).astype(float) * 20.0
    out["total_score"] = out["total_score"].clip(0.0, 100.0)

    factor_support_count = (
        (out["momentum_factor"].fillna(-1e9) > 0).astype(int)
        + (out["growth_factor"].fillna(-1e9) > 0).astype(int)
        + (out["value_factor"].fillna(-1e9) > 0).astype(int)
    )
    out["factor_support_count"] = factor_support_count
    out["signal_pass"] = (
        out["predicted_change_pct"].fillna(-1e9) >= min_pred_change
    ) & (out["factor_support_count"] >= min_factor_support)

    out["status"] = np.where(
        out["hard_filter_pass"] & out["signal_pass"],
        "Active",
        np.where(out["hard_filter_pass"], "Watch", "Retired"),
    )
    out["recommended_action"] = np.where(
        out["status"] == "Active",
        "Keep/Open",
        np.where(out["status"] == "Watch", "Monitor/Reduce", "Remove"),
    )

    alerts: List[List[str]] = []
    for _, row in out.iterrows():
        row_alerts: List[str] = []
        if not bool(row.get("prediction_available", False)):
            row_alerts.append("prediction_unavailable")
        if _safe_float(row.get("predicted_change_pct")) <= 0:
            row_alerts.append("predicted_change_non_positive")
        if not bool(row.get("history_ok", False)):
            row_alerts.append("history_too_short")
        if not bool(row.get("missing_ok", False)):
            row_alerts.append("missing_rate_too_high")
        if not bool(row.get("liquidity_ok", False)):
            row_alerts.append("liquidity_insufficient")
        if not bool(row.get("risk_factors_available", False)):
            row_alerts.append("risk_factor_missing")
        if not bool(row.get("behavior_factors_available", False)):
            row_alerts.append("behavior_factor_missing")
        alerts.append(row_alerts)
    out["alerts"] = [";".join(x) if x else "" for x in alerts]
    return out


def _build_data_quality_report(df: pd.DataFrame) -> str:
    total = len(df)
    if total == 0:
        return "# Data Quality Report\n\nNo symbols were processed."
    prediction_ok = int(df["prediction_available"].sum())
    hard_pass = int(df["hard_filter_pass"].sum())
    factors_ok = int(df["factors_available"].sum())
    status_counts = df["status"].value_counts(dropna=False).to_dict()
    top_alerts = (
        df["alerts"]
        .fillna("")
        .str.split(";")
        .explode()
        .loc[lambda s: s != ""]
        .value_counts()
        .head(10)
        .to_dict()
    )
    lines = [
        "# Data Quality Report",
        "",
        f"- Total symbols processed: **{total}**",
        f"- Prediction available: **{prediction_ok}** ({prediction_ok / total:.1%})",
        f"- Hard filter pass: **{hard_pass}** ({hard_pass / total:.1%})",
        f"- Factor available: **{factors_ok}** ({factors_ok / total:.1%})",
        "",
        "## Status Distribution",
    ]
    for k, v in status_counts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Top Alerts")
    if top_alerts:
        for k, v in top_alerts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def run_tracking(config_path: str) -> None:
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    tracking_cfg = cfg.get("tracking", {})
    if not tracking_cfg.get("enabled", True):
        print("[WARN] tracking is disabled.")
        return

    processed_dir = Path(paths_cfg.get("processed_data_dir", "data/processed"))
    out_dir = Path(tracking_cfg.get("output_dir", processed_dir / "tracking"))
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = get_universe_catalog()
    meta = _load_all_universes(cfg)
    if meta.empty:
        print("[WARN] universe list is empty.")
        return

    instruments = _build_snapshot_instruments(meta, cfg)
    workers = int(tracking_cfg.get("snapshot_workers", 8))
    snapshot = _build_snapshot_parallel(instruments, config_path=config_path, max_workers=workers)
    merged = meta.merge(snapshot, how="left", on="instrument_id", suffixes=("", "_snap"))
    scored = _score_and_state(merged, cfg)
    scored = merge_latest_news_features(
        scored,
        market_col="market",
        symbol_col="symbol",
        processed_dir=str(processed_dir),
    )
    news_defaults = {
        "news_score_30m": 0.0,
        "news_score_120m": 0.0,
        "news_score_1440m": 0.0,
        "news_count_30m": 0.0,
        "news_burst_zscore": 0.0,
        "news_pos_neg_ratio": 1.0,
        "news_conflict_score": 0.0,
        "news_event_risk": 0.0,
        "news_gate_pass": 1.0,
        "news_risk_level": "low",
        "news_reason_codes": "",
    }
    for col, default in news_defaults.items():
        if col not in scored.columns:
            scored[col] = default

    # Policy layer for unified buy/sell/flat outputs across markets.
    scored = scored.copy()
    scored["market_type"] = np.where(scored["market"].astype(str).eq("crypto"), "spot", "cash")
    if "q50_change_pct" not in scored.columns and "predicted_change_pct" in scored.columns:
        scored["q50_change_pct"] = pd.to_numeric(scored["predicted_change_pct"], errors="coerce")
    if "q10_change_pct" not in scored.columns:
        q50 = pd.to_numeric(scored.get("q50_change_pct"), errors="coerce")
        scored["q10_change_pct"] = q50 - q50.abs().clip(lower=0.005)
    if "q90_change_pct" not in scored.columns:
        q50 = pd.to_numeric(scored.get("q50_change_pct"), errors="coerce")
        scored["q90_change_pct"] = q50 + q50.abs().clip(lower=0.005)
    if "volatility_score" not in scored.columns:
        scored["volatility_score"] = (
            pd.to_numeric(scored["q90_change_pct"], errors="coerce")
            - pd.to_numeric(scored["q10_change_pct"], errors="coerce")
        )
    if "confidence_score" not in scored.columns:
        scored["confidence_score"] = pd.to_numeric(scored.get("total_score"), errors="coerce")
    if "p_up" not in scored.columns:
        q50 = pd.to_numeric(scored.get("q50_change_pct"), errors="coerce").fillna(0.0)
        scored["p_up"] = 0.5 + 0.5 * np.tanh(q50 / 0.016)
    scored = apply_policy_frame(
        scored,
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

    # Universe JSON outputs requested by the plan.
    for market, filename in [
        ("crypto", "universe_crypto.json"),
        ("cn_equity", "universe_ashares.json"),
        ("us_equity", "universe_us.json"),
    ]:
        rows = meta[meta["market"] == market].copy()
        rows = rows.drop(columns=[c for c in rows.columns if c.endswith("_snap")], errors="ignore")
        save_json(
            {
                "market": market,
                "pools": tracking_cfg.get("universes", DEFAULT_UNIVERSES).get(market, []),
                "count": int(len(rows)),
                "rows": rows.to_dict(orient="records"),
                "labels": catalog.get(market, {}),
            },
            out_dir / filename,
        )

    coverage_cols = [
        "market",
        "pool_key",
        "instrument_id",
        "symbol",
        "snapshot_symbol",
        "name",
        "provider",
        "prediction_available",
        "risk_factors_available",
        "behavior_factors_available",
        "history_bars",
        "history_missing_rate",
        "hard_filter_pass",
        "alerts",
    ]
    ranked_cols = [
        "market",
        "pool_key",
        "instrument_id",
        "name",
        "display",
        "symbol",
        "snapshot_symbol",
        "provider",
        "status",
        "recommended_action",
        "alerts",
        "current_price",
        "predicted_price",
        "predicted_change_pct",
        "policy_action",
        "policy_position_size",
        "policy_signed_position",
        "policy_expected_edge_pct",
        "policy_expected_edge_abs",
        "policy_reason",
        "expected_date_market",
        "total_score",
        "liquidity_score",
        "data_quality_score",
        "history_score",
        "coverage_score",
        "factor_support_count",
        "price_source",
        "news_score_30m",
        "news_score_120m",
        "news_score_1440m",
        "news_count_30m",
        "news_burst_zscore",
        "news_risk_level",
        "news_event_risk",
        "news_gate_pass",
        "news_reason_codes",
    ]

    coverage_df = scored[[c for c in coverage_cols if c in scored.columns]].copy()
    ranked_df = scored[[c for c in ranked_cols if c in scored.columns]].copy()
    ranked_df = ranked_df.sort_values(["status", "total_score"], ascending=[True, False]).reset_index(drop=True)

    actions_df = scored[
        [
            "market",
            "instrument_id",
            "name",
            "status",
            "recommended_action",
            "alerts",
            "predicted_change_pct",
            "policy_action",
            "policy_position_size",
            "policy_signed_position",
            "policy_expected_edge_pct",
            "policy_expected_edge_abs",
            "policy_reason",
            "news_score_30m",
            "news_score_120m",
            "news_score_1440m",
            "news_count_30m",
            "news_burst_zscore",
            "news_risk_level",
            "news_event_risk",
            "news_gate_pass",
            "news_reason_codes",
            "factor_support_count",
            "hard_filter_pass",
        ]
    ].copy()

    write_csv(coverage_df, out_dir / "coverage_matrix.csv")
    write_csv(ranked_df, out_dir / "ranked_universe.csv")
    write_csv(actions_df, out_dir / "tracking_actions.csv")
    write_csv(scored, out_dir / "tracking_snapshot.csv")
    (out_dir / "data_quality_report.md").write_text(
        _build_data_quality_report(scored), encoding="utf-8"
    )
    print(f"[OK] Saved tracking outputs -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build universe ranking + tracking outputs for Crypto/A-share/US."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_tracking(args.config)


if __name__ == "__main__":
    main()
