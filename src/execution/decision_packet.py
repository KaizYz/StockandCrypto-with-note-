from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _safe_text(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _safe_int(value: Any, default: int = 0) -> int:
    out = _safe_float(value)
    if np.isfinite(out):
        return int(round(out))
    return int(default)


def _to_action_normalized(action_raw: Any) -> str:
    key = _safe_text(action_raw).upper()
    if key in {"LONG", "SHORT", "WAIT"}:
        return key
    if key == "LONG":
        return "LONG"
    if key == "SHORT":
        return "SHORT"
    if key in {"FLAT", "HOLD", "OBSERVE"}:
        return "WAIT"
    return "WAIT"


def _config_hash(cfg: Dict[str, Any]) -> str:
    try:
        blob = json.dumps(cfg or {}, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]
    except Exception:
        return "-"


def _market_profile_key(market: str, timeframe: str) -> str:
    m = _safe_text(market).lower()
    t = _safe_text(timeframe).lower()
    if m == "crypto" and t:
        return f"{m}_{t}"
    return m


def _resolve_execution_assumptions(cfg: Dict[str, Any], *, market: str, timeframe: str) -> Dict[str, Any]:
    policy_cfg = (cfg.get("policy", {}) if isinstance(cfg, dict) else {}) or {}
    policy_exec_cfg = (policy_cfg.get("execution", {}) if isinstance(policy_cfg, dict) else {}) or {}
    bt_cfg = (cfg.get("backtest_multi_market", {}) if isinstance(cfg, dict) else {}) or {}
    bt_impact_cfg = (bt_cfg.get("impact", {}) if isinstance(bt_cfg, dict) else {}) or {}
    reporting_cfg = (cfg.get("reporting", {}) if isinstance(cfg, dict) else {}) or {}
    market_profiles = (policy_exec_cfg.get("market_cost_profiles", {}) if isinstance(policy_exec_cfg, dict) else {}) or {}

    profile_key = _market_profile_key(market, timeframe)
    profile = market_profiles.get(profile_key, {})
    if not isinstance(profile, dict) or not profile:
        profile = market_profiles.get(_safe_text(market).lower(), {})
    if not isinstance(profile, dict):
        profile = {}

    fee_bps = _safe_float(profile.get("fee_bps", policy_exec_cfg.get("fee_bps", 0.0)))
    slippage_bps = _safe_float(profile.get("slippage_bps", policy_exec_cfg.get("slippage_bps", 0.0)))
    impact_bps = _safe_float(
        profile.get(
            "impact_bps",
            policy_exec_cfg.get("impact_bps", bt_impact_cfg.get("lambda_bps", 0.0)),
        )
    )
    delay_bars = _safe_int(profile.get("delay_bars", bt_cfg.get("delay_bars", 1)), default=1)
    fill_price_mode = _safe_text(profile.get("fill_price_mode", reporting_cfg.get("fill_price_mode", "next_open"))) or "next_open"
    cost_profile = _safe_text(profile.get("cost_profile", reporting_cfg.get("cost_profile", "dynamic_v1"))) or "dynamic_v1"
    return {
        "profile_key": profile_key,
        "fee_bps": max(fee_bps, 0.0),
        "slippage_bps": max(slippage_bps, 0.0),
        "impact_bps": max(impact_bps, 0.0),
        "delay_bars": max(delay_bars, 0),
        "fill_price_mode": fill_price_mode,
        "cost_profile": cost_profile,
    }


def _build_reason_codes(
    *,
    row: pd.Series,
    action: str,
    trade_plan: Dict[str, Any],
    confidence_min: float,
) -> List[str]:
    reasons: List[str] = []
    policy_reason = _safe_text(row.get("policy_reason"))
    if policy_reason:
        reasons.append(f"policy:{policy_reason}")

    pred = _safe_float(row.get("predicted_change_pct"))
    if not np.isfinite(pred):
        reasons.append("data:prediction_unavailable")
    risk_level = _safe_text(row.get("risk_level")).lower()
    if risk_level == "extreme":
        reasons.append("risk:extreme")
    conf = _safe_float(row.get("confidence_score"))
    if np.isfinite(conf) and conf < confidence_min:
        reasons.append("model:low_confidence")

    market = _safe_text(row.get("market")).lower()
    market_type = _safe_text(row.get("market_type")).lower()
    allow_short = bool(row.get("policy_allow_short", True))
    if action == "SHORT":
        if market == "cn_equity":
            reasons.append("market:short_disallowed_cn")
        if market == "crypto" and market_type == "spot":
            reasons.append("market:short_disallowed_crypto_spot")
        if not allow_short:
            reasons.append("market:short_disallowed")

    alerts = _safe_text(row.get("alerts"))
    if alerts:
        for item in alerts.split(";"):
            token = _safe_text(item)
            if token:
                reasons.append(f"alert:{token}")

    action_reason = _safe_text(trade_plan.get("action_reason"))
    if action_reason:
        reasons.append(f"decision:{action_reason}")
    if action == "WAIT":
        reasons.append("execution:wait")
    return list(dict.fromkeys(reasons))


def _derive_gate_fields(
    *,
    action: str,
    reasons: List[str],
    trade_plan: Dict[str, Any],
    row: pd.Series,
) -> Dict[str, Any]:
    blocking_prefixes = ("data:", "risk:", "market:", "gate:")
    blocked_reason = [r for r in reasons if any(r.startswith(p) for p in blocking_prefixes)]
    gate_status = "BLOCKED" if blocked_reason else "PASS"
    health_grade_raw = _safe_text(trade_plan.get("model_health", row.get("model_health", ""))).lower()
    if health_grade_raw in {"good", "healthy", "green", "high"}:
        health_grade = "good"
    elif health_grade_raw in {"poor", "bad", "red", "low"}:
        health_grade = "poor"
    elif health_grade_raw in {"medium", "mid", "yellow"}:
        health_grade = "medium"
    else:
        health_grade = "medium"
    risk_budget_left = _safe_float(trade_plan.get("risk_budget_left", row.get("risk_budget_left", 1.0)))
    if not np.isfinite(risk_budget_left):
        risk_budget_left = 1.0
    if action == "WAIT" and not blocked_reason:
        blocked_reason = ["execution:wait"]
    return {
        "gate_status": gate_status,
        "blocked_reason": blocked_reason,
        "health_grade": health_grade,
        "risk_budget_left": float(risk_budget_left),
    }


def build_decision_packet(
    *,
    row: pd.Series,
    trade_plan: Dict[str, Any] | None,
    cfg: Dict[str, Any] | None,
    git_commit: str = "-",
) -> Dict[str, Any]:
    cfg = cfg or {}
    trade_plan = trade_plan or {}
    action = _to_action_normalized(trade_plan.get("action", row.get("policy_action", "WAIT")))
    if action not in {"LONG", "SHORT", "WAIT"}:
        action = "WAIT"

    market = _safe_text(row.get("market")) or "unknown"
    symbol = _safe_text(row.get("symbol")) or _safe_text(row.get("snapshot_symbol"))
    instrument_id = _safe_text(row.get("instrument_id")) or symbol.lower()
    horizon = _safe_text(row.get("horizon_label")) or "4h"
    timeframe = _safe_text(row.get("market_type"))
    if not timeframe:
        timeframe = "perp" if market == "crypto" and horizon.endswith("h") else "spot"

    confidence_min = float(
        (cfg.get("decision", {}) if isinstance(cfg, dict) else {}).get("confidence_min", 60.0)
    )
    reasons = _build_reason_codes(
        row=row,
        action=action,
        trade_plan=trade_plan,
        confidence_min=confidence_min,
    )
    blocking_prefixes = {
        "data:prediction_unavailable",
        "market:short_disallowed_cn",
        "market:short_disallowed_crypto_spot",
        "market:short_disallowed",
        "risk:extreme",
    }
    if any(r in blocking_prefixes for r in reasons):
        action = "WAIT"
        if "execution:forced_wait_by_failsafe" not in reasons:
            reasons.append("execution:forced_wait_by_failsafe")

    # Global Go/No-Go gate: if training readiness is NO-GO, force WAIT.
    try:
        go_live_path = Path("data/processed/go_live_decision.json")
        if go_live_path.exists():
            payload = json.loads(go_live_path.read_text(encoding="utf-8"))
            all_pass = bool(payload.get("go_live", {}).get("all_pass", True))
            if not all_pass:
                action = "WAIT"
                if "gate:no_go" not in reasons:
                    reasons.append("gate:no_go")
    except Exception:
        pass

    entry = _safe_float(trade_plan.get("entry", row.get("current_price")))
    sl = _safe_float(trade_plan.get("stop_loss"))
    tp1 = _safe_float(trade_plan.get("take_profit", row.get("predicted_price")))
    tp2 = _safe_float(trade_plan.get("take_profit_2"))
    rr = _safe_float(trade_plan.get("rr"))
    p_up = _safe_float(trade_plan.get("p_up", row.get("p_up")))
    p_down = _safe_float(trade_plan.get("p_down", row.get("p_down")))
    q10 = _safe_float(trade_plan.get("q10", row.get("q10_change_pct")))
    q50 = _safe_float(trade_plan.get("q50", row.get("q50_change_pct", row.get("predicted_change_pct"))))
    q90 = _safe_float(trade_plan.get("q90", row.get("q90_change_pct")))
    conf = _safe_float(trade_plan.get("confidence_score", row.get("confidence_score")))
    edge = _safe_float(
        trade_plan.get("edge_long" if action == "LONG" else "edge_short", row.get("policy_expected_edge_pct"))
    )
    edge_risk = _safe_float(
        trade_plan.get("edge_risk_long" if action == "LONG" else "edge_risk_short", float("nan"))
    )
    if not np.isfinite(edge_risk):
        width = _safe_float(q90 - q10)
        edge_risk = _safe_float(edge / width) if np.isfinite(width) and width > 1e-12 else float("nan")

    exec_assumptions = _resolve_execution_assumptions(cfg, market=market, timeframe=timeframe)
    fee_bps = _safe_float(exec_assumptions.get("fee_bps"))
    slippage_bps = _safe_float(exec_assumptions.get("slippage_bps"))
    impact_bps = _safe_float(exec_assumptions.get("impact_bps"))
    delay_bars = _safe_int(exec_assumptions.get("delay_bars"), default=1)
    fill_price_mode = _safe_text(exec_assumptions.get("fill_price_mode", "next_open")) or "next_open"
    cost_profile = _safe_text(exec_assumptions.get("cost_profile", "dynamic_v1")) or "dynamic_v1"
    cost_bps = fee_bps + slippage_bps + max(impact_bps, 0.0)
    ks_cfg = (cfg.get("kill_switch", {}) if isinstance(cfg, dict) else {}) or {}

    exp_market = _safe_text(row.get("expected_date_market"))
    exp_utc = _safe_text(row.get("expected_date_utc"))
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    generated_market = _safe_text(row.get("generated_at_market"))
    if not generated_market:
        generated_market = _safe_text(row.get("generated_at_utc")) or generated_utc
    valid_until = exp_market if exp_market else exp_utc

    decision_id = str(uuid.uuid4())
    gate_fields = _derive_gate_fields(action=action, reasons=reasons, trade_plan=trade_plan, row=row)
    packet = {
        "decision_id": decision_id,
        "market": market,
        "symbol": symbol,
        "instrument_id": instrument_id,
        "provider": _safe_text(row.get("provider")),
        "timeframe": timeframe,
        "horizon": horizon,
        "action": action,
        "signal_strength": _safe_text(trade_plan.get("signal_strength", "")),
        "confidence_score": conf,
        "risk_level": _safe_text(row.get("risk_level", trade_plan.get("risk_level", "medium"))),
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "p_up": p_up,
        "p_down": p_down,
        "q10_change_pct": q10,
        "q50_change_pct": q50,
        "q90_change_pct": q90,
        "expected_edge_pct": edge,
        "edge_risk": edge_risk,
        "start_window_top1": _safe_text(row.get("start_window_top1", row.get("start_window_name"))),
        "cost_bps": cost_bps,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "impact_bps": impact_bps,
        "delay_bars": delay_bars,
        "fill_price_mode": fill_price_mode,
        "fill_price_source": "latest_price_proxy",
        "cost_profile": cost_profile,
        "cost_profile_key": _safe_text(exec_assumptions.get("profile_key", market)),
        "cost_assumption": (
            f"double_side fee={fee_bps:.2f}bps + slippage={slippage_bps:.2f}bps "
            f"+ impact={impact_bps:.2f}bps; delay_bars={delay_bars}; fill={fill_price_mode}; profile={cost_profile}"
        ),
        "valid_until": valid_until,
        "generated_at_market_tz": generated_market,
        "generated_at_utc": generated_utc,
        "model_version": _safe_text(row.get("prediction_method", "baseline_momentum_quantile")),
        "data_version": _safe_text(row.get("price_source", "snapshot_v1")),
        "config_hash": _config_hash(cfg),
        "git_commit": _safe_text(git_commit) or "-",
        "reasons": reasons,
        "gate_status": gate_fields["gate_status"],
        "blocked_reason": gate_fields["blocked_reason"],
        "health_grade": gate_fields["health_grade"],
        "risk_budget_left": gate_fields["risk_budget_left"],
        "notes": _safe_text(trade_plan.get("action_reason", "")),
        "policy_position_size": _safe_float(row.get("policy_position_size", trade_plan.get("policy_position_size"))),
        "kill_switch_enabled": bool(ks_cfg.get("enabled", True)),
        "kill_switch_env_var": _safe_text(ks_cfg.get("env_var", "DISABLE_TRADING")) or "DISABLE_TRADING",
        "kill_switch_recovery_health_checks_required": int(_safe_float(ks_cfg.get("recovery_health_checks_required", 3))),
        "kill_switch_trial_position_scale": _safe_float(ks_cfg.get("trial_position_scale", 0.25)),
        "kill_switch_trial_windows": int(_safe_float(ks_cfg.get("trial_windows", 1))),
        "kill_switch_admin_role": _safe_text(ks_cfg.get("admin_role", "ops_admin")) or "ops_admin",
        "kill_switch_operator_role_env": _safe_text(ks_cfg.get("operator_role_env", "TRADING_OPERATOR_ROLE")) or "TRADING_OPERATOR_ROLE",
    }
    return packet


def persist_decision_packet(packet: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_id = _safe_text(packet.get("decision_id")) or str(uuid.uuid4())
    packet = dict(packet)
    packet["decision_id"] = packet_id
    packet["reasons"] = packet.get("reasons", [])

    json_path = output_dir / f"decision_packet_{packet_id}.json"
    json_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")

    log_path = output_dir / "decision_packets_log.csv"
    row = dict(packet)
    row["reasons"] = ";".join(row.get("reasons", [])) if isinstance(row.get("reasons"), list) else str(row.get("reasons", ""))
    row["blocked_reason"] = (
        ";".join(row.get("blocked_reason", []))
        if isinstance(row.get("blocked_reason"), list)
        else str(row.get("blocked_reason", ""))
    )
    row_df = pd.DataFrame([row])
    if log_path.exists():
        old = pd.read_csv(log_path)
        merged = pd.concat([old, row_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["decision_id"], keep="last")
    else:
        merged = row_df
    merged.to_csv(log_path, index=False, encoding="utf-8-sig")

    gates_log_path = output_dir / "gates_audit_log.jsonl"
    gate_payload = {
        "timestamp_utc": packet.get("generated_at_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
        "decision_id": packet.get("decision_id"),
        "market": packet.get("market"),
        "symbol": packet.get("symbol"),
        "action": packet.get("action"),
        "gate_status": packet.get("gate_status", "PASS"),
        "blocked_reason": packet.get("blocked_reason", []),
        "health_grade": packet.get("health_grade", "medium"),
        "risk_budget_left": packet.get("risk_budget_left"),
        "config_hash": packet.get("config_hash"),
        "git_commit": packet.get("git_commit"),
    }
    with gates_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(gate_payload, ensure_ascii=False) + "\n")

    latest_path = output_dir / "decision_packet_latest.json"
    latest_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"json": json_path, "log": log_path, "latest": latest_path}
