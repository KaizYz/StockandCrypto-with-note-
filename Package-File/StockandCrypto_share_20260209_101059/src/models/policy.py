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
    "news_gate": {
        "enabled": True,
        "negative_score_2h": -0.25,
        "positive_score_2h": 0.25,
        "burst_zscore": 1.5,
        "score_30m_extreme": 0.40,
        "count_30m_extreme": 3,
        "risk_scale_by_level": {
            "low": 1.0,
            "medium": 0.85,
            "high": 0.70,
        },
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


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _parse_reason_codes(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    out: list[str] = []
    for token in text.split(";"):
        t = str(token or "").strip()
        if t and t not in out:
            out.append(t)
    return out


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
    news_score_30m: float = float("nan"),
    news_score_2h: float = float("nan"),
    news_score_24h: float = float("nan"),
    news_burst_zscore: float = float("nan"),
    news_count_30m: float = float("nan"),
    news_event_risk: bool = False,
    news_risk_level: str = "",
    news_gate_pass: bool = True,
    news_reason_codes: str = "",
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

    news_cfg = policy_cfg.get("news_gate", {}) if isinstance(policy_cfg, dict) else {}
    news_enabled = bool(news_cfg.get("enabled", True))
    news_score_30m_v = _safe_float(news_score_30m)
    news_score_2h_v = _safe_float(news_score_2h)
    news_score_24h_v = _safe_float(news_score_24h)
    news_burst_v = _safe_float(news_burst_zscore)
    news_count_30m_v = _safe_float(news_count_30m)
    news_event_risk_v = _safe_bool(news_event_risk, default=False)
    news_risk_level_v = str(news_risk_level or "").lower().strip()
    if news_risk_level_v not in {"low", "medium", "high"}:
        news_risk_level_v = "high" if news_event_risk_v else "low"
    news_reason_list = _parse_reason_codes(news_reason_codes)

    neg_2h_thr = _safe_float(news_cfg.get("negative_score_2h", -0.25))
    pos_2h_thr = _safe_float(news_cfg.get("positive_score_2h", 0.25))
    burst_thr = _safe_float(news_cfg.get("burst_zscore", 1.5))
    score_30_thr = _safe_float(news_cfg.get("score_30m_extreme", 0.40))
    count_30_thr = _safe_float(news_cfg.get("count_30m_extreme", 3))
    if not np.isfinite(neg_2h_thr):
        neg_2h_thr = -0.25
    if not np.isfinite(pos_2h_thr):
        pos_2h_thr = 0.25
    if not np.isfinite(burst_thr):
        burst_thr = 1.5
    if not np.isfinite(score_30_thr):
        score_30_thr = 0.40
    if not np.isfinite(count_30_thr):
        count_30_thr = 3.0

    if news_enabled:
        local_event = False
        if np.isfinite(news_score_30m_v) and np.isfinite(news_count_30m_v):
            if abs(news_score_30m_v) >= score_30_thr and news_count_30m_v >= count_30_thr:
                local_event = True
                news_reason_list.append("NEWS_EVENT_EXTREME")
        if np.isfinite(news_score_2h_v) and np.isfinite(news_burst_v):
            if news_score_2h_v <= neg_2h_thr and news_burst_v >= burst_thr:
                local_event = True
                news_reason_list.append("NEG_BURST")
            if news_score_2h_v >= pos_2h_thr and news_burst_v >= burst_thr:
                local_event = True
                news_reason_list.append("POS_BURST")
        news_event_risk_v = bool(news_event_risk_v or local_event)
        news_gate_pass_v = bool(_safe_bool(news_gate_pass, default=True) and (not news_event_risk_v))
    else:
        news_gate_pass_v = True

    news_scale_cfg = news_cfg.get("risk_scale_by_level", {})
    news_risk_scale = _safe_float(news_scale_cfg.get(news_risk_level_v, 1.0))
    if not np.isfinite(news_risk_scale):
        news_risk_scale = 1.0
    risk_scale = float(np.clip(risk_scale * news_risk_scale, 0.0, 1.0))
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

    if news_enabled and (not news_gate_pass_v):
        base_action = "Flat"
        if "news_gate_block" not in news_reason_list:
            news_reason_list.append("news_gate_block")
        reason = "news_gate_block"

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
        "news_gate_enabled": bool(news_enabled),
        "news_gate_pass": bool(news_gate_pass_v),
        "news_event_risk": bool(news_event_risk_v),
        "news_risk_level": str(news_risk_level_v),
        "news_score_30m": news_score_30m_v,
        "news_score_2h": news_score_2h_v,
        "news_score_24h": news_score_24h_v,
        "news_burst_zscore": news_burst_v,
        "news_count_30m": news_count_30m_v,
        "news_reason_codes": ";".join(dict.fromkeys(news_reason_list)),
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
    news_score_30m_col: str = "news_score_30m",
    news_score_2h_col: str = "news_score_120m",
    news_score_24h_col: str = "news_score_1440m",
    news_burst_col: str = "news_burst_zscore",
    news_count_30m_col: str = "news_count_30m",
    news_event_risk_col: str = "news_event_risk",
    news_risk_level_col: str = "news_risk_level",
    news_gate_pass_col: str = "news_gate_pass",
    news_reason_codes_col: str = "news_reason_codes",
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
        news_score_30m = _safe_float(row.get(news_score_30m_col))
        news_score_2h = _safe_float(row.get(news_score_2h_col))
        news_score_24h = _safe_float(row.get(news_score_24h_col))
        news_burst = _safe_float(row.get(news_burst_col))
        news_count_30m = _safe_float(row.get(news_count_30m_col))
        news_event_risk = _safe_bool(row.get(news_event_risk_col), default=False)
        news_risk_level = str(row.get(news_risk_level_col, ""))
        news_gate_pass = _safe_bool(row.get(news_gate_pass_col), default=True)
        news_reason_codes = str(row.get(news_reason_codes_col, ""))

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
            news_score_30m=news_score_30m,
            news_score_2h=news_score_2h,
            news_score_24h=news_score_24h,
            news_burst_zscore=news_burst,
            news_count_30m=news_count_30m,
            news_event_risk=news_event_risk,
            news_risk_level=news_risk_level,
            news_gate_pass=news_gate_pass,
            news_reason_codes=news_reason_codes,
        )
        for key, value in sig.items():
            if key not in out.columns:
                out[key] = np.nan
            if isinstance(value, (str, bool, np.bool_)) and out[key].dtype != object:
                out[key] = out[key].astype(object)
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
