from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pandas as pd


DEFAULT_POLICY_CONFIG: Dict[str, Any] = {
    "thresholds": {
        "p_bull": 0.55,
        "p_bear": 0.45,
        "ret_threshold": 0.002,
    },
    "execution": {
        "fee_bps": 2.0,
        "slippage_bps": 5.0,
        "max_position": 1.0,
        "min_position": 0.01,
    },
    "sizing": {
        "uncertainty_scale": 12.0,
        "fallback_half_width_pct": 0.01,
        "confidence_power": 1.0,
    },
    "risk_scale": {
        "by_level": {
            "low": 1.0,
            "medium": 0.8,
            "high": 0.6,
            "extreme": 0.4,
        },
        "by_volatility": {
            "enabled": False,
            "bands": [0.02, 0.05, 0.10],
            "scales": [1.0, 0.8, 0.6, 0.4],
        },
    },
    "market_rules": {
        "crypto": {
            "allow_short_perp": True,
            "allow_short_spot": False,
            "allow_short_default": True,
        },
        "cn_equity": {"allow_short": False},
        "us_equity": {"allow_short": True},
        "default": {"allow_short": False},
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


def _merge_nested_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_nested_dict(out[key], value)
        else:
            out[key] = value
    return out


def _normalize_confidence_score(raw_value: float, confidence_power: float) -> float:
    val = _safe_float(raw_value)
    if not np.isfinite(val):
        return 0.5
    if val > 1.0 and val <= 100.0:
        val = val / 100.0
    val = float(np.clip(val, 0.0, 1.0))
    return float(np.power(val, max(confidence_power, 1e-9)))


def _infer_p_up_from_q50(q50_change_pct: float, ret_threshold: float) -> float:
    q50 = _safe_float(q50_change_pct)
    if not np.isfinite(q50):
        return 0.5
    scale = max(ret_threshold, 1e-4) * 8.0
    return float(0.5 + 0.5 * np.tanh(q50 / scale))


def _risk_scale_from_inputs(
    volatility_score: float,
    risk_level: str,
    policy_cfg: Dict[str, Any],
) -> float:
    risk_cfg = policy_cfg.get("risk_scale", {})
    by_level = risk_cfg.get("by_level", {})
    level_key = str(risk_level or "").lower().strip()
    if level_key in by_level:
        return float(np.clip(_safe_float(by_level[level_key]), 0.0, 1.0))

    by_vol = risk_cfg.get("by_volatility", {})
    if bool(by_vol.get("enabled", False)):
        vol = _safe_float(volatility_score)
        if np.isfinite(vol):
            bands = list(by_vol.get("bands", [0.02, 0.05, 0.10]))
            scales = list(by_vol.get("scales", [1.0, 0.8, 0.6, 0.4]))
            bands = [_safe_float(x) for x in bands]
            scales = [_safe_float(x) for x in scales]
            bands = [x for x in bands if np.isfinite(x)]
            scales = [x for x in scales if np.isfinite(x)]
            if len(scales) >= len(bands) + 1 and bands:
                if vol < bands[0]:
                    return float(np.clip(scales[0], 0.0, 1.0))
                for i in range(1, len(bands)):
                    if vol < bands[i]:
                        return float(np.clip(scales[i], 0.0, 1.0))
                return float(np.clip(scales[len(bands)], 0.0, 1.0))

    return 1.0


def _normalize_market(market: str | None) -> str:
    key = str(market or "").lower().strip()
    aliases = {
        "ashares": "cn_equity",
        "cn_a": "cn_equity",
        "a_share": "cn_equity",
        "a_shares": "cn_equity",
        "us": "us_equity",
        "us_stock": "us_equity",
        "stocks_us": "us_equity",
    }
    return aliases.get(key, key or "default")


def _allow_short_for_market(market: str, market_type: str, policy_cfg: Dict[str, Any]) -> bool:
    rules = policy_cfg.get("market_rules", {})
    mkt = _normalize_market(market)
    mkt_rule = rules.get(mkt, {})
    default_rule = rules.get("default", {})
    mt = str(market_type or "").lower().strip()

    if mkt == "crypto":
        if mt == "spot":
            return bool(mkt_rule.get("allow_short_spot", False))
        if mt == "perp":
            return bool(mkt_rule.get("allow_short_perp", True))
        return bool(mkt_rule.get("allow_short_default", True))

    if "allow_short" in mkt_rule:
        return bool(mkt_rule.get("allow_short"))
    return bool(default_rule.get("allow_short", False))


def get_policy_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    policy_cfg = cfg.get("policy", {})
    # Backward compatible defaults from forecast_config / backtest.
    forecast_cfg = cfg.get("forecast_config", {})
    trend_cfg = forecast_cfg.get("trend_labels", {})
    backtest_cfg = cfg.get("backtest", {})

    fallback_cfg = deepcopy(DEFAULT_POLICY_CONFIG)
    fallback_cfg["thresholds"]["p_bull"] = _safe_float(
        trend_cfg.get("p_bull", fallback_cfg["thresholds"]["p_bull"])
    )
    fallback_cfg["thresholds"]["p_bear"] = _safe_float(
        trend_cfg.get("p_bear", fallback_cfg["thresholds"]["p_bear"])
    )
    fallback_cfg["thresholds"]["ret_threshold"] = _safe_float(
        trend_cfg.get("ret_threshold", fallback_cfg["thresholds"]["ret_threshold"])
    )
    fallback_cfg["execution"]["fee_bps"] = _safe_float(
        backtest_cfg.get("fee_bps", fallback_cfg["execution"]["fee_bps"])
    )
    fallback_cfg["execution"]["slippage_bps"] = _safe_float(
        backtest_cfg.get("slippage_bps", fallback_cfg["execution"]["slippage_bps"])
    )

    merged = _merge_nested_dict(fallback_cfg, policy_cfg if isinstance(policy_cfg, dict) else {})
    return merged


def policy_from_forecast(
    *,
    p_up: float,
    q10_change_pct: float,
    q50_change_pct: float,
    q90_change_pct: float,
    volatility_score: float,
    confidence_score: float,
    current_price: float,
    market: str,
    market_type: str,
    policy_cfg: Dict[str, Any],
    risk_level: str = "",
) -> Dict[str, Any]:
    thresholds = policy_cfg.get("thresholds", {})
    execution = policy_cfg.get("execution", {})
    sizing = policy_cfg.get("sizing", {})

    p_bull = _safe_float(thresholds.get("p_bull"))
    p_bear = _safe_float(thresholds.get("p_bear"))
    ret_thr = _safe_float(thresholds.get("ret_threshold"))
    if not np.isfinite(p_bull):
        p_bull = 0.55
    if not np.isfinite(p_bear):
        p_bear = 0.45
    if not np.isfinite(ret_thr):
        ret_thr = 0.002

    fee_bps = _safe_float(execution.get("fee_bps"))
    slippage_bps = _safe_float(execution.get("slippage_bps"))
    max_position = _safe_float(execution.get("max_position"))
    min_position = _safe_float(execution.get("min_position"))
    if not np.isfinite(max_position):
        max_position = 1.0
    if not np.isfinite(min_position):
        min_position = 0.01
    cost_pct = (max(fee_bps, 0.0) + max(slippage_bps, 0.0)) / 10000.0

    uncertainty_scale = _safe_float(sizing.get("uncertainty_scale"))
    fallback_half_width = _safe_float(sizing.get("fallback_half_width_pct"))
    confidence_power = _safe_float(sizing.get("confidence_power"))
    if not np.isfinite(uncertainty_scale):
        uncertainty_scale = 12.0
    if not np.isfinite(fallback_half_width):
        fallback_half_width = 0.01
    if not np.isfinite(confidence_power):
        confidence_power = 1.0

    q50 = _safe_float(q50_change_pct)
    q10 = _safe_float(q10_change_pct)
    q90 = _safe_float(q90_change_pct)
    if np.isfinite(q50) and (not np.isfinite(q10) or not np.isfinite(q90)):
        q10 = q50 - fallback_half_width
        q90 = q50 + fallback_half_width
    uncertainty_width = max(0.0, _safe_float(q90 - q10))

    p_up_used = _safe_float(p_up)
    p_up_source = "model"
    if not np.isfinite(p_up_used):
        p_up_used = _infer_p_up_from_q50(q50, ret_thr)
        p_up_source = "proxy_q50"

    allow_short = _allow_short_for_market(market=market, market_type=market_type, policy_cfg=policy_cfg)
    confidence_used = _normalize_confidence_score(confidence_score, confidence_power=confidence_power)
    risk_scale = _risk_scale_from_inputs(
        volatility_score=volatility_score,
        risk_level=risk_level,
        policy_cfg=policy_cfg,
    )
    prob_strength = float(np.clip(2.0 * abs(p_up_used - 0.5), 0.0, 1.0))
    uncertainty_scale_factor = float(1.0 / (1.0 + uncertainty_scale * uncertainty_width))

    long_edge = _safe_float(q50 - cost_pct)
    short_edge = _safe_float((-q50) - cost_pct)

    base_action = "Flat"
    reason = "threshold_not_met"
    if np.isfinite(q50):
        if p_up_used >= p_bull and long_edge >= ret_thr:
            base_action = "Long"
            reason = "long_signal"
        elif p_up_used <= p_bear:
            if not allow_short:
                base_action = "Flat"
                reason = "short_disallowed"
            elif short_edge >= ret_thr:
                base_action = "Short"
                reason = "short_signal"
            else:
                base_action = "Flat"
                reason = "short_edge_below_threshold"
        else:
            reason = "probability_neutral"
    else:
        reason = "insufficient_inputs"

    position_size = prob_strength * uncertainty_scale_factor * risk_scale * confidence_used
    if not np.isfinite(position_size):
        position_size = 0.0
    position_size = float(np.clip(position_size, 0.0, max_position))

    if base_action == "Flat" or position_size < min_position:
        if base_action != "Flat" and position_size < min_position:
            reason = "position_below_minimum"
        action = "Flat"
        position_size = 0.0
        signed_position = 0.0
    elif base_action == "Long":
        action = "Long"
        signed_position = position_size
    else:
        action = "Short"
        signed_position = -position_size

    if action == "Long":
        expected_edge_pct = long_edge
    elif action == "Short":
        expected_edge_pct = short_edge
    else:
        expected_edge_pct = long_edge
        if allow_short and np.isfinite(short_edge):
            expected_edge_pct = max(long_edge, short_edge)
    expected_edge_pct = 0.0 if not np.isfinite(expected_edge_pct) else float(expected_edge_pct)

    current = _safe_float(current_price)
    expected_edge_abs = (
        float(current * expected_edge_pct) if np.isfinite(current) and np.isfinite(expected_edge_pct) else float("nan")
    )

    return {
        "policy_action": action,
        "policy_position_size": float(position_size),
        "policy_signed_position": float(signed_position),
        "policy_expected_edge_pct": expected_edge_pct,
        "policy_expected_edge_abs": expected_edge_abs,
        "policy_cost_pct": float(cost_pct),
        "policy_uncertainty_width": float(uncertainty_width),
        "policy_p_up_used": float(p_up_used),
        "policy_prob_strength": float(prob_strength),
        "policy_confidence_used": float(confidence_used),
        "policy_risk_scale": float(risk_scale),
        "policy_allow_short": bool(allow_short),
        "policy_reason": reason,
        "policy_p_up_source": p_up_source,
    }


def apply_policy_frame(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    market_col: str = "market",
    market_type_col: str = "market_type",
    p_up_col: str = "p_up",
    q10_col: str = "q10_change_pct",
    q50_col: str = "q50_change_pct",
    q90_col: str = "q90_change_pct",
    volatility_col: str = "volatility_score",
    confidence_col: str = "confidence_score",
    current_price_col: str = "current_price",
    risk_level_col: str = "risk_level",
) -> pd.DataFrame:
    if df.empty:
        return df

    policy_cfg = get_policy_config(cfg)
    out = df.copy()

    for idx, row in out.iterrows():
        market = str(row.get(market_col, "default"))
        market_type = str(row.get(market_type_col, "spot"))
        p_up = _safe_float(row.get(p_up_col))
        q10 = _safe_float(row.get(q10_col))
        q50 = _safe_float(row.get(q50_col))
        q90 = _safe_float(row.get(q90_col))
        vol = _safe_float(row.get(volatility_col))
        if not np.isfinite(vol) and np.isfinite(q90) and np.isfinite(q10):
            vol = float(max(q90 - q10, 0.0))
        conf = _safe_float(row.get(confidence_col))
        current = _safe_float(row.get(current_price_col))
        risk_level = str(row.get(risk_level_col, ""))

        sig = policy_from_forecast(
            p_up=p_up,
            q10_change_pct=q10,
            q50_change_pct=q50,
            q90_change_pct=q90,
            volatility_score=vol,
            confidence_score=conf,
            current_price=current,
            market=market,
            market_type=market_type,
            policy_cfg=policy_cfg,
            risk_level=risk_level,
        )
        for key, value in sig.items():
            out.at[idx, key] = value

    return out


def summarize_policy_actions(
    df: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
    action_col: str = "policy_action",
    position_col: str = "policy_position_size",
    edge_col: str = "policy_expected_edge_pct",
) -> pd.DataFrame:
    if df.empty or action_col not in df.columns:
        return pd.DataFrame()
    groups = group_cols or []
    use_cols = [c for c in [action_col, position_col, edge_col] if c in df.columns]
    work = df[[c for c in groups + use_cols if c in df.columns]].copy()
    if not groups:
        work["_all"] = "all"
        groups = ["_all"]
    summary = (
        work.groupby(groups + [action_col], dropna=False)
        .agg(
            count=(action_col, "size"),
            avg_position=(position_col, "mean") if position_col in work.columns else (action_col, "size"),
            avg_expected_edge=(edge_col, "mean") if edge_col in work.columns else (action_col, "size"),
        )
        .reset_index()
    )
    if "_all" in summary.columns:
        summary = summary.drop(columns=["_all"], errors="ignore")
    return summary

