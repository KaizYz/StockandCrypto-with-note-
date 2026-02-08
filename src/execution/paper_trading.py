from __future__ import annotations

import json
import os
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
    return str(value or "").strip()


def _safe_int(value: Any, default: int = 0) -> int:
    out = _safe_float(value)
    if np.isfinite(out):
        return int(round(out))
    return int(default)


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _load_table(path: Path, columns: List[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[columns].copy()


def _save_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_row(df: pd.DataFrame, row: Dict[str, Any]) -> pd.DataFrame:
    row_df = pd.DataFrame([row], columns=df.columns)
    if df.empty:
        return row_df
    return pd.concat([df, row_df], ignore_index=True)


def _position_side_to_sign(side: str) -> float:
    return 1.0 if str(side).lower() == "long" else -1.0


def load_execution_artifacts(output_dir: Path) -> Dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    orders_cols = [
        "order_id",
        "decision_id",
        "market",
        "symbol",
        "side",
        "qty",
        "order_type",
        "status",
        "requested_price",
        "filled_price",
        "delay_bars",
        "fill_model",
        "created_at_utc",
        "filled_at_utc",
        "reason_code",
    ]
    fills_cols = [
        "fill_id",
        "order_id",
        "decision_id",
        "market",
        "symbol",
        "side",
        "qty",
        "price",
        "fee_bps",
        "slippage_bps",
        "impact_bps",
        "delay_bars",
        "fill_model",
        "cost_pct",
        "fill_time_utc",
    ]
    positions_cols = [
        "position_id",
        "market",
        "symbol",
        "side",
        "qty",
        "entry_price",
        "entry_time_utc",
        "stop_loss",
        "take_profit",
        "decision_id_open",
        "status",
        "exit_price",
        "exit_time_utc",
        "exit_reason",
        "gross_pnl_pct",
        "net_pnl_pct",
        "fees_pct",
    ]
    daily_pnl_cols = [
        "date_utc",
        "trades_closed",
        "realized_net_pnl_pct",
        "realized_gross_pnl_pct",
        "fees_pct",
        "cumulative_realized_net_pnl_pct",
        "generated_at_utc",
    ]
    return {
        "orders": _load_table(output_dir / "paper_orders.csv", orders_cols),
        "fills": _load_table(output_dir / "paper_fills.csv", fills_cols),
        "positions": _load_table(output_dir / "paper_positions.csv", positions_cols),
        "daily_pnl": _load_table(output_dir / "paper_daily_pnl.csv", daily_pnl_cols),
    }


def _calc_execution_cost(packet: Dict[str, Any], *, qty: float) -> Dict[str, float]:
    # Prefer explicit packet fields; fallback to legacy aggregate cost_bps split.
    fee_bps = _safe_float(packet.get("fee_bps"))
    slippage_bps = _safe_float(packet.get("slippage_bps"))
    impact_bps = _safe_float(packet.get("impact_bps"))
    if not np.isfinite(fee_bps) or not np.isfinite(slippage_bps):
        total_bps = max(_safe_float(packet.get("cost_bps")), 0.0)
        fee_bps = total_bps * 0.5
        slippage_bps = total_bps * 0.5
    if not np.isfinite(impact_bps):
        impact_bps = 0.0
    fee_bps = max(fee_bps, 0.0)
    slippage_bps = max(slippage_bps, 0.0)
    impact_bps = max(impact_bps, 0.0)
    # Dynamic adjustment by uncertainty/volatility proxy and order size.
    q10 = _safe_float(packet.get("q10_change_pct"))
    q90 = _safe_float(packet.get("q90_change_pct"))
    width_pct = abs(q90 - q10) if np.isfinite(q10) and np.isfinite(q90) else float("nan")
    vol_proxy = _safe_float(packet.get("volatility_proxy"))
    if not np.isfinite(vol_proxy):
        if np.isfinite(width_pct):
            # q change pct is percentage points in current project convention.
            vol_proxy = max(width_pct / 100.0, 0.0)
        else:
            vol_proxy = 0.0
    vol_proxy = float(np.clip(vol_proxy, 0.0, 0.5))
    qty_safe = max(_safe_float(qty), 0.0)
    qty_scale = np.sqrt(max(qty_safe, 1.0))
    slip_scale = 1.0 + min(vol_proxy * 10.0, 3.0)
    impact_scale = 1.0 + min(vol_proxy * 8.0, 3.0)
    slippage_bps_dyn = slippage_bps * slip_scale
    impact_bps_dyn = impact_bps * impact_scale * qty_scale
    total_bps = fee_bps + slippage_bps_dyn + impact_bps_dyn
    cost_pct = total_bps / 10000.0
    return {
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps_dyn,
        "impact_bps": impact_bps_dyn,
        "slippage_scale": slip_scale,
        "impact_scale": impact_scale,
        "volatility_proxy": vol_proxy,
        "qty_scale": qty_scale,
        "total_bps": total_bps,
        "cost_pct": cost_pct,
    }


def _resolve_fill_price(*, latest_price: float, requested_price: float, fill_mode: str) -> float:
    mode = str(fill_mode or "next_open").strip().lower()
    if mode == "mid":
        if np.isfinite(latest_price) and np.isfinite(requested_price):
            return float((latest_price + requested_price) / 2.0)
        return float(latest_price if np.isfinite(latest_price) else requested_price)
    if mode == "vwap":
        # Paper MVP: no orderbook/tick vwap stream, fallback to latest proxy.
        return float(latest_price if np.isfinite(latest_price) else requested_price)
    # next_open / unknown -> latest proxy in paper runtime.
    return float(latest_price if np.isfinite(latest_price) else requested_price)


def _close_position_row(pos: pd.Series, exit_price: float, reason: str, cost_pct: float) -> Dict[str, Any]:
    sign = _position_side_to_sign(str(pos.get("side", "long")))
    entry = _safe_float(pos.get("entry_price"))
    gross = _safe_float(((exit_price - entry) / entry) * sign) if np.isfinite(entry) and entry > 0 else float("nan")
    net = _safe_float(gross - cost_pct) if np.isfinite(gross) else float("nan")
    fees_pct = cost_pct
    updated = dict(pos)
    updated["status"] = "closed"
    updated["exit_price"] = exit_price
    updated["exit_time_utc"] = _utc_now_text()
    updated["exit_reason"] = reason
    updated["gross_pnl_pct"] = gross
    updated["net_pnl_pct"] = net
    updated["fees_pct"] = fees_pct
    return updated


def _maybe_close_open_positions(
    positions: pd.DataFrame,
    *,
    market: str,
    symbol: str,
    latest_price: float,
    fallback_cost_pct: float,
) -> pd.DataFrame:
    if positions.empty or not np.isfinite(latest_price):
        return positions
    out_rows = []
    for _, pos in positions.iterrows():
        status = str(pos.get("status", "closed")).lower()
        if status != "open":
            out_rows.append(dict(pos))
            continue
        if str(pos.get("market", "")) != market or str(pos.get("symbol", "")) != symbol:
            out_rows.append(dict(pos))
            continue
        side = str(pos.get("side", "long")).lower()
        sl = _safe_float(pos.get("stop_loss"))
        tp = _safe_float(pos.get("take_profit"))
        should_close = False
        close_reason = ""
        if side == "long":
            if np.isfinite(sl) and latest_price <= sl:
                should_close = True
                close_reason = "stop_loss"
            elif np.isfinite(tp) and latest_price >= tp:
                should_close = True
                close_reason = "take_profit"
        else:
            if np.isfinite(sl) and latest_price >= sl:
                should_close = True
                close_reason = "stop_loss"
            elif np.isfinite(tp) and latest_price <= tp:
                should_close = True
                close_reason = "take_profit"

        if should_close:
            out_rows.append(
                _close_position_row(
                    pos,
                    exit_price=latest_price,
                    reason=close_reason,
                    cost_pct=fallback_cost_pct,
                )
            )
        else:
            out_rows.append(dict(pos))
    return pd.DataFrame(out_rows, columns=positions.columns)


def _save_daily_pnl(positions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    cols = [
        "date_utc",
        "trades_closed",
        "realized_net_pnl_pct",
        "realized_gross_pnl_pct",
        "fees_pct",
        "cumulative_realized_net_pnl_pct",
        "generated_at_utc",
    ]
    if positions.empty:
        out = pd.DataFrame(columns=cols)
        _save_table(output_dir / "paper_daily_pnl.csv", out)
        return out

    closed = positions[positions["status"].astype(str) == "closed"].copy()
    if closed.empty:
        out = pd.DataFrame(columns=cols)
        _save_table(output_dir / "paper_daily_pnl.csv", out)
        return out

    closed["exit_time_utc_dt"] = pd.to_datetime(closed.get("exit_time_utc"), utc=True, errors="coerce")
    closed = closed.dropna(subset=["exit_time_utc_dt"])
    if closed.empty:
        out = pd.DataFrame(columns=cols)
        _save_table(output_dir / "paper_daily_pnl.csv", out)
        return out

    closed["date_utc"] = closed["exit_time_utc_dt"].dt.strftime("%Y-%m-%d")
    closed["net_pnl_pct"] = pd.to_numeric(closed.get("net_pnl_pct"), errors="coerce").fillna(0.0)
    closed["gross_pnl_pct"] = pd.to_numeric(closed.get("gross_pnl_pct"), errors="coerce").fillna(0.0)
    closed["fees_pct"] = pd.to_numeric(closed.get("fees_pct"), errors="coerce").fillna(0.0)
    agg = (
        closed.groupby("date_utc", as_index=False)
        .agg(
            trades_closed=("position_id", "count"),
            realized_net_pnl_pct=("net_pnl_pct", "sum"),
            realized_gross_pnl_pct=("gross_pnl_pct", "sum"),
            fees_pct=("fees_pct", "sum"),
        )
        .sort_values("date_utc")
    )
    agg["cumulative_realized_net_pnl_pct"] = agg["realized_net_pnl_pct"].cumsum()
    agg["generated_at_utc"] = _utc_now_text()
    out = agg[cols].copy()
    _save_table(output_dir / "paper_daily_pnl.csv", out)
    return out


def _load_kill_switch_state(output_dir: Path) -> Dict[str, Any]:
    state_path = output_dir / "kill_switch.state.json"
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {
        "active": False,
        "reason": "default",
        "release_request": False,
        "requested_by": "",
        "requester_role": "",
        "trial_windows_remaining": 0,
        "trial_position_scale": 1.0,
    }


def _save_kill_switch_state(output_dir: Path, state: Dict[str, Any]) -> None:
    path = output_dir / "kill_switch.state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_operator_role(packet: Dict[str, Any]) -> str:
    env_key = _safe_text(packet.get("kill_switch_operator_role_env", "TRADING_OPERATOR_ROLE")) or "TRADING_OPERATOR_ROLE"
    role = _safe_text(os.getenv(env_key, ""))
    return role.lower()


def _update_health_streak(output_dir: Path, packet: Dict[str, Any]) -> Dict[str, Any]:
    path = output_dir / "health_checks_streak.json"
    prev = {"consecutive_pass": 0, "last_pass": False}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                prev.update(payload)
        except Exception:
            pass

    blocked = packet.get("blocked_reason", [])
    if not isinstance(blocked, list):
        blocked = [_safe_text(blocked)] if _safe_text(blocked) else []
    prefixes = {p.split(":", 1)[0] for p in blocked if ":" in p}
    data_ok = "data" not in prefixes
    risk_ok = "risk" not in prefixes
    model_ok = (_safe_text(packet.get("health_grade", "medium")).lower() != "poor") and ("model" not in prefixes)
    is_pass = bool(data_ok and risk_ok and model_ok)
    streak = int(prev.get("consecutive_pass", 0)) + 1 if is_pass else 0
    out = {
        "consecutive_pass": streak,
        "last_pass": is_pass,
        "last_check_utc": _utc_now_text(),
        "data_ok": data_ok,
        "model_ok": model_ok,
        "risk_ok": risk_ok,
    }
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _resolve_kill_switch_active(output_dir: Path, *, env_var: str = "DISABLE_TRADING", enabled: bool = True) -> Dict[str, Any]:
    if not enabled:
        return {"active": False, "reason": "disabled", "env_active": False, "file_active": False, "state": {}}
    state = _load_kill_switch_state(output_dir)
    file_active = bool(state.get("active", False))
    env_key = str(env_var or "DISABLE_TRADING").strip() or "DISABLE_TRADING"
    env_raw = str(os.getenv(env_key, "")).strip().lower()
    env_active = env_raw in {"1", "true", "yes", "on"}
    active = bool(file_active or env_active)
    reason = str(state.get("reason", "")).strip()
    if env_active:
        reason = f"env:{env_key}"
    if not reason:
        reason = "state_file" if file_active else "inactive"
    return {"active": active, "reason": reason, "env_active": env_active, "file_active": file_active, "state": state}


def _attempt_kill_switch_recovery(
    output_dir: Path,
    *,
    kill_switch: Dict[str, Any],
    packet: Dict[str, Any],
    health_streak: Dict[str, Any],
) -> Dict[str, Any]:
    state = dict(kill_switch.get("state", {}) or {})
    if not bool(kill_switch.get("active", False)):
        return kill_switch
    if bool(kill_switch.get("env_active", False)):
        return kill_switch
    if not bool(state.get("release_request", False)):
        return kill_switch

    admin_role = _safe_text(packet.get("kill_switch_admin_role", "ops_admin")).lower() or "ops_admin"
    requester_role = _safe_text(state.get("requester_role")).lower() or _read_operator_role(packet)
    required_checks = max(_safe_int(packet.get("kill_switch_recovery_health_checks_required", 3), default=3), 1)
    current_streak = _safe_int(health_streak.get("consecutive_pass", 0), default=0)
    now = _utc_now_text()

    if requester_role != admin_role:
        _append_jsonl(
            output_dir / "kill_switch_recovery_log.jsonl",
            {
                "timestamp_utc": now,
                "event": "recovery_denied",
                "reason": "role_not_allowed",
                "required_role": admin_role,
                "requester_role": requester_role,
                "required_consecutive_pass": required_checks,
                "current_consecutive_pass": current_streak,
            },
        )
        return kill_switch
    if current_streak < required_checks:
        _append_jsonl(
            output_dir / "kill_switch_recovery_log.jsonl",
            {
                "timestamp_utc": now,
                "event": "recovery_denied",
                "reason": "health_checks_not_enough",
                "required_consecutive_pass": required_checks,
                "current_consecutive_pass": current_streak,
            },
        )
        return kill_switch

    trial_scale = _safe_float(packet.get("kill_switch_trial_position_scale", 0.25))
    if not np.isfinite(trial_scale) or trial_scale <= 0:
        trial_scale = 0.25
    trial_windows = max(_safe_int(packet.get("kill_switch_trial_windows", 1), default=1), 1)
    state.update(
        {
            "active": False,
            "reason": "recovered_by_admin",
            "release_request": False,
            "released_at_utc": now,
            "released_by": _safe_text(state.get("requested_by", "")),
            "requester_role": requester_role,
            "trial_windows_remaining": trial_windows,
            "trial_position_scale": trial_scale,
        }
    )
    _save_kill_switch_state(output_dir, state)
    _append_jsonl(
        output_dir / "kill_switch_recovery_log.jsonl",
        {
            "timestamp_utc": now,
            "event": "kill_switch_off",
            "reason": "recovered_by_admin",
            "required_consecutive_pass": required_checks,
            "current_consecutive_pass": current_streak,
            "trial_windows_remaining": trial_windows,
            "trial_position_scale": trial_scale,
            "released_by": _safe_text(state.get("released_by", "")),
            "requester_role": requester_role,
        },
    )
    kill_switch = dict(kill_switch)
    kill_switch.update({"active": False, "reason": "recovered_by_admin", "state": state, "file_active": False})
    return kill_switch


def _apply_trial_position_scale(
    output_dir: Path,
    *,
    kill_switch: Dict[str, Any],
    qty: float,
    action: str,
) -> tuple[float, bool, float]:
    state = dict(kill_switch.get("state", {}) or {})
    if action not in {"LONG", "SHORT"}:
        return qty, False, 1.0
    if bool(kill_switch.get("active", False)):
        return qty, False, 1.0
    remaining = _safe_int(state.get("trial_windows_remaining", 0), default=0)
    if remaining <= 0:
        return qty, False, 1.0
    scale = _safe_float(state.get("trial_position_scale", 1.0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return max(qty * scale, 0.0), True, float(scale)


def _consume_trial_window_if_needed(output_dir: Path, *, kill_switch: Dict[str, Any], used: bool) -> None:
    if not used:
        return
    state = dict(kill_switch.get("state", {}) or {})
    remaining = _safe_int(state.get("trial_windows_remaining", 0), default=0)
    if remaining <= 0:
        return
    state["trial_windows_remaining"] = max(remaining - 1, 0)
    _save_kill_switch_state(output_dir, state)


def _record_kill_switch_transition(output_dir: Path, active_info: Dict[str, Any]) -> None:
    marker_path = output_dir / ".kill_switch_last_seen.json"
    prev_active = None
    if marker_path.exists():
        try:
            prev_active = bool(json.loads(marker_path.read_text(encoding="utf-8")).get("active"))
        except Exception:
            prev_active = None
    now_active = bool(active_info.get("active", False))
    if prev_active is None or prev_active == now_active:
        marker_path.write_text(json.dumps({"active": now_active}, ensure_ascii=False), encoding="utf-8")
        return
    event_path = output_dir / ("kill_switch_events.jsonl" if now_active else "kill_switch_recovery_log.jsonl")
    _append_jsonl(
        event_path,
        {
            "timestamp_utc": _utc_now_text(),
            "event": "kill_switch_on" if now_active else "kill_switch_off",
            "reason": str(active_info.get("reason", "")),
            "env_active": bool(active_info.get("env_active", False)),
            "file_active": bool(active_info.get("file_active", False)),
        },
    )
    marker_path.write_text(json.dumps({"active": now_active}, ensure_ascii=False), encoding="utf-8")


def _append_run_log(output_dir: Path, payload: Dict[str, Any]) -> None:
    _append_jsonl(output_dir / "paper_run_log.jsonl", payload)


def apply_decision_to_paper_book(
    packet: Dict[str, Any],
    *,
    latest_price: float | None,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_execution_artifacts(output_dir)
    orders = artifacts["orders"]
    fills = artifacts["fills"]
    positions = artifacts["positions"]
    trace_id = str(uuid.uuid4())

    decision_id = str(packet.get("decision_id", "")).strip()
    if not decision_id:
        _append_run_log(
            output_dir,
            {
                "timestamp_utc": _utc_now_text(),
                "trace_id": trace_id,
                "decision_id": decision_id,
                "status": "error",
                "message": "decision_id is required",
            },
        )
        return {"status": "error", "message": "decision_id is required", "artifacts": artifacts}
    if not orders.empty and decision_id in orders["decision_id"].astype(str).tolist():
        _append_run_log(
            output_dir,
            {
                "timestamp_utc": _utc_now_text(),
                "trace_id": trace_id,
                "decision_id": decision_id,
                "status": "skipped",
                "message": "duplicate decision_id (idempotent skip)",
            },
        )
        return {"status": "skipped", "message": "duplicate decision_id (idempotent skip)", "artifacts": artifacts}

    market = str(packet.get("market", "unknown"))
    symbol = str(packet.get("symbol", ""))
    action = str(packet.get("action", "WAIT")).upper()
    qty = _safe_float(packet.get("policy_position_size"))
    if not np.isfinite(qty) or qty <= 0:
        qty = 1.0
    req_price = _safe_float(packet.get("entry"))
    latest_px = _safe_float(latest_price)
    if not np.isfinite(latest_px):
        latest_px = req_price
    if not np.isfinite(latest_px):
        _append_run_log(
            output_dir,
            {
                "timestamp_utc": _utc_now_text(),
                "trace_id": trace_id,
                "decision_id": decision_id,
                "market": market,
                "symbol": symbol,
                "action": action,
                "status": "error",
                "message": "no valid market price / entry price",
            },
        )
        return {"status": "error", "message": "no valid market price / entry price", "artifacts": artifacts}

    fill_mode = str(packet.get("fill_price_mode", "next_open")).strip().lower() or "next_open"
    delay_bars = int(max(_safe_float(packet.get("delay_bars")), 0.0)) if np.isfinite(_safe_float(packet.get("delay_bars"))) else 0
    px = _resolve_fill_price(latest_price=latest_px, requested_price=req_price, fill_mode=fill_mode)
    positions = _maybe_close_open_positions(
        positions,
        market=market,
        symbol=symbol,
        latest_price=px,
        fallback_cost_pct=max(_safe_float(packet.get("cost_bps")), 0.0) / 10000.0,
    )
    kill_switch = _resolve_kill_switch_active(
        output_dir,
        env_var=str(packet.get("kill_switch_env_var", "DISABLE_TRADING")),
        enabled=bool(packet.get("kill_switch_enabled", True)),
    )
    health_streak = _update_health_streak(output_dir, packet)
    kill_switch = _attempt_kill_switch_recovery(
        output_dir,
        kill_switch=kill_switch,
        packet=packet,
        health_streak=health_streak,
    )
    _record_kill_switch_transition(output_dir, kill_switch)
    qty_scaled, in_trial_window, trial_scale = _apply_trial_position_scale(
        output_dir,
        kill_switch=kill_switch,
        qty=qty,
        action=action,
    )
    if in_trial_window:
        qty = qty_scaled
    cost = _calc_execution_cost(packet, qty=qty)

    order_id = str(uuid.uuid4())
    now = _utc_now_text()
    gate_status = str(packet.get("gate_status", "")).strip().upper()

    if action not in {"LONG", "SHORT"}:
        order_row = {
            "order_id": order_id,
            "decision_id": decision_id,
            "market": market,
            "symbol": symbol,
            "side": "wait",
            "qty": 0.0,
            "order_type": "none",
            "status": "skipped",
            "requested_price": req_price,
            "filled_price": np.nan,
            "delay_bars": delay_bars,
            "fill_model": fill_mode,
            "created_at_utc": now,
            "filled_at_utc": "",
            "reason_code": "wait_signal",
        }
        orders = _append_row(orders, order_row)
        _save_table(output_dir / "paper_orders.csv", orders)
        _save_table(output_dir / "paper_fills.csv", fills)
        _save_table(output_dir / "paper_positions.csv", positions)
        daily = _save_daily_pnl(positions, output_dir)
        _append_run_log(
            output_dir,
            {
                "timestamp_utc": now,
                "trace_id": trace_id,
                "decision_id": decision_id,
                "order_id": order_id,
                "market": market,
                "symbol": symbol,
                "action": action,
                "gate_status": gate_status or "PASS",
                "kill_switch_active": bool(kill_switch.get("active", False)),
                "health_consecutive_pass": _safe_int(health_streak.get("consecutive_pass", 0), default=0),
                "delay_bars": delay_bars,
                "fill_model": fill_mode,
                "status": "ok",
                "message": "WAIT decision recorded (no order placed)",
            },
        )
        return {
            "status": "ok",
            "message": "WAIT decision recorded (no order placed)",
            "artifacts": {"orders": orders, "fills": fills, "positions": positions, "daily_pnl": daily},
        }

    if bool(kill_switch.get("active", False)) or gate_status == "BLOCKED":
        reason_code = "kill_switch_active" if bool(kill_switch.get("active", False)) else "gate_blocked"
        order_row = {
            "order_id": order_id,
            "decision_id": decision_id,
            "market": market,
            "symbol": symbol,
            "side": action.lower(),
            "qty": qty,
            "order_type": "market",
            "status": "blocked",
            "requested_price": req_price,
            "filled_price": np.nan,
            "delay_bars": delay_bars,
            "fill_model": fill_mode,
            "created_at_utc": now,
            "filled_at_utc": "",
            "reason_code": reason_code,
        }
        orders = _append_row(orders, order_row)
        _save_table(output_dir / "paper_orders.csv", orders)
        _save_table(output_dir / "paper_fills.csv", fills)
        _save_table(output_dir / "paper_positions.csv", positions)
        if bool(kill_switch.get("active", False)):
            _append_jsonl(
                output_dir / "kill_switch_events.jsonl",
                {
                    "timestamp_utc": now,
                    "event": "blocked_order",
                    "decision_id": decision_id,
                    "market": market,
                    "symbol": symbol,
                    "action": action,
                    "reason": str(kill_switch.get("reason", "kill_switch_active")),
                },
            )
        daily = _save_daily_pnl(positions, output_dir)
        _append_run_log(
            output_dir,
            {
                "timestamp_utc": now,
                "trace_id": trace_id,
                "decision_id": decision_id,
                "order_id": order_id,
                "market": market,
                "symbol": symbol,
                "action": action,
                "gate_status": gate_status or ("BLOCKED" if reason_code == "gate_blocked" else "PASS"),
                "kill_switch_active": bool(kill_switch.get("active", False)),
                "health_consecutive_pass": _safe_int(health_streak.get("consecutive_pass", 0), default=0),
                "delay_bars": delay_bars,
                "fill_model": fill_mode,
                "status": "blocked",
                "message": reason_code,
            },
        )
        return {
            "status": "skipped",
            "message": f"{reason_code}: no new order placed",
            "artifacts": {"orders": orders, "fills": fills, "positions": positions, "daily_pnl": daily},
        }

    side = "long" if action == "LONG" else "short"
    order_row = {
        "order_id": order_id,
        "decision_id": decision_id,
        "market": market,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "order_type": "market",
        "status": "filled",
        "requested_price": req_price,
        "filled_price": px,
        "delay_bars": delay_bars,
        "fill_model": fill_mode,
        "created_at_utc": now,
        "filled_at_utc": now,
        "reason_code": "paper_fill",
    }
    fill_id = str(uuid.uuid4())
    fill_row = {
        "fill_id": fill_id,
        "order_id": order_id,
        "decision_id": decision_id,
        "market": market,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": px,
        "fee_bps": cost["fee_bps"],
        "slippage_bps": cost["slippage_bps"],
        "impact_bps": cost["impact_bps"],
        "delay_bars": delay_bars,
        "fill_model": fill_mode,
        "cost_pct": cost["cost_pct"],
        "fill_time_utc": now,
    }

    position_row = {
        "position_id": str(uuid.uuid4()),
        "market": market,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry_price": px,
        "entry_time_utc": now,
        "stop_loss": _safe_float(packet.get("sl")),
        "take_profit": _safe_float(packet.get("tp1")),
        "decision_id_open": decision_id,
        "status": "open",
        "exit_price": np.nan,
        "exit_time_utc": "",
        "exit_reason": "",
        "gross_pnl_pct": np.nan,
        "net_pnl_pct": np.nan,
        "fees_pct": np.nan,
    }

    orders = _append_row(orders, order_row)
    fills = _append_row(fills, fill_row)
    positions = _append_row(positions, position_row)

    _save_table(output_dir / "paper_orders.csv", orders)
    _save_table(output_dir / "paper_fills.csv", fills)
    _save_table(output_dir / "paper_positions.csv", positions)
    _consume_trial_window_if_needed(output_dir, kill_switch=kill_switch, used=in_trial_window)
    daily = _save_daily_pnl(positions, output_dir)
    _append_run_log(
        output_dir,
        {
            "timestamp_utc": now,
            "trace_id": trace_id,
            "decision_id": decision_id,
            "order_id": order_id,
            "market": market,
            "symbol": symbol,
            "action": action,
            "gate_status": gate_status or "PASS",
            "kill_switch_active": bool(kill_switch.get("active", False)),
            "health_consecutive_pass": _safe_int(health_streak.get("consecutive_pass", 0), default=0),
            "delay_bars": delay_bars,
            "fill_model": fill_mode,
            "filled_price": px,
            "cost_bps": cost["total_bps"],
            "trial_position_applied": bool(in_trial_window),
            "trial_scale": trial_scale,
            "status": "ok",
            "message": f"{action} order filled in paper book",
        },
    )
    return {
        "status": "ok",
        "message": f"{action} order filled in paper book",
        "artifacts": {"orders": orders, "fills": fills, "positions": positions, "daily_pnl": daily},
    }


def summarize_execution(artifacts: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    positions = artifacts.get("positions", pd.DataFrame())
    if positions.empty:
        return {
            "open_positions": 0.0,
            "closed_positions": 0.0,
            "win_rate": float("nan"),
            "avg_net_pnl_pct": float("nan"),
            "total_net_pnl_pct": float("nan"),
        }
    open_n = float((positions["status"].astype(str) == "open").sum())
    closed = positions[positions["status"].astype(str) == "closed"].copy()
    if closed.empty:
        return {
            "open_positions": open_n,
            "closed_positions": 0.0,
            "win_rate": float("nan"),
            "avg_net_pnl_pct": float("nan"),
            "total_net_pnl_pct": float("nan"),
        }
    net = pd.to_numeric(closed["net_pnl_pct"], errors="coerce")
    win_rate = float((net > 0).mean()) if net.notna().any() else float("nan")
    return {
        "open_positions": open_n,
        "closed_positions": float(len(closed)),
        "win_rate": win_rate,
        "avg_net_pnl_pct": float(net.mean()) if net.notna().any() else float("nan"),
        "total_net_pnl_pct": float(net.sum()) if net.notna().any() else float("nan"),
    }
