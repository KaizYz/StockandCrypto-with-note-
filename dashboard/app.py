from __future__ import annotations

import json
import hashlib
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.markets.session_forecast import build_session_forecast_bundle
from src.markets.snapshot import build_market_snapshot_from_instruments
from src.markets.universe import get_universe_catalog, load_universe
from src.models.policy import apply_policy_frame
from src.execution import (
    apply_decision_to_paper_book,
    build_decision_packet,
    load_execution_artifacts,
    persist_decision_packet,
    summarize_execution,
)
from src.utils.config import load_config


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=600, show_spinner=False)
def _load_main_config_cached(config_path: str = "configs/config.yaml") -> Dict[str, object]:
    return load_config(config_path)


def _ui_lang() -> str:
    lang = str(st.session_state.get("ui_lang", "zh")).strip().lower()
    return "en" if lang.startswith("en") else "zh"


def _t(zh_text: str, en_text: str) -> str:
    return zh_text if _ui_lang() == "zh" else en_text


@st.cache_data(ttl=180, show_spinner=False)
def _load_backtest_artifacts(processed_dir_str: str) -> Dict[str, pd.DataFrame]:
    root = Path(processed_dir_str) / "backtest"
    return {
        "metrics_summary": _load_csv(root / "metrics_summary.csv"),
        "metrics_by_fold": _load_csv(root / "metrics_by_fold.csv"),
        "compare": _load_csv(root / "compare_baselines.csv"),
        "equity": _load_csv(root / "equity.csv"),
        "trades": _load_csv(root / "trades.csv"),
        "latest_signals": _load_csv(root / "latest_signals.csv"),
    }


def _normalize_symbol_token(text: object) -> str:
    token = str(text or "").strip().lower()
    return token.replace("/", "").replace("-", "").replace("_", "")


@st.cache_data(ttl=300, show_spinner=False)
def _run_single_symbol_backtest_cached(
    market: str,
    symbol: str,
    provider: str,
    fallback_symbol: str = "",
    config_path: str = "configs/config.yaml",
) -> Dict[str, pd.DataFrame]:
    from src.evaluation.backtest_multi_market import run_single_symbol_backtest

    result = run_single_symbol_backtest(
        config_path=config_path,
        market=market,
        symbol=symbol,
        provider=provider,
        fallback_symbol=fallback_symbol,
    )
    return result


@st.cache_data(ttl=180, show_spinner=False)
def _load_symbol_signal_context_cached(
    market: str,
    symbol: str,
    provider: str,
    fallback_symbol: str = "",
    config_path: str = "configs/config.yaml",
) -> pd.DataFrame:
    from src.evaluation.backtest_multi_market import _build_model_like_signals, _fetch_daily_bars

    cfg = load_config(config_path)
    bt_cfg = cfg.get("backtest_multi_market", {})
    lookback_days_cfg = bt_cfg.get(
        "lookback_days",
        {"crypto": 540, "cn_equity": 1200, "us_equity": 1200},
    )
    mk = str(market)
    sym = str(symbol)
    prov = str(provider or ("binance" if mk == "crypto" else "yahoo"))
    lb_days = int(lookback_days_cfg.get(mk, 720))
    try:
        bars, _ = _fetch_daily_bars(
            market=mk,
            symbol=sym,
            provider=prov,
            lookback_days=lb_days,
            cfg=cfg,
        )
    except Exception:
        if mk == "crypto":
            fs = str(fallback_symbol or "").strip().upper()
            if not fs and prov == "coingecko":
                guess = str(sym).strip().upper()
                if guess.isalpha() and len(guess) <= 8:
                    fs = f"{guess}USDT"
            if fs.endswith("USDT"):
                bars, _ = _fetch_daily_bars(
                    market=mk,
                    symbol=fs,
                    provider="binance",
                    lookback_days=lb_days,
                    cfg=cfg,
                )
                sym = fs
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    bars = bars.dropna(subset=["timestamp_utc", "open", "close"]).sort_values("timestamp_utc").reset_index(drop=True)
    if bars.empty:
        return pd.DataFrame()
    modeled = _build_model_like_signals(bars, cfg=cfg, market=mk)
    if modeled.empty:
        return pd.DataFrame()
    latest = modeled.tail(1).copy()
    latest["symbol"] = sym
    latest["market"] = mk
    latest["provider"] = prov
    return latest


def _reason_token_cn(token: object) -> str:
    key = str(token or "").strip().lower()
    mapping = {
        "ema_bull_cross": "EMA金叉",
        "ema_bear_cross": "EMA死叉",
        "macd_golden_cross": "MACD金叉",
        "macd_dead_cross": "MACD死叉",
        "supertrend_bullish": "SuperTrend看涨",
        "supertrend_bearish": "SuperTrend看跌",
        "lux_ms_bullish": "ML Adaptive SuperTrend看涨",
        "lux_ms_bearish": "ML Adaptive SuperTrend看跌",
        "volume_surge": "放量",
        "rsi_oversold": "RSI超卖",
        "rsi_overbought": "RSI超买",
        "kdj_golden_cross": "KDJ金叉",
        "kdj_dead_cross": "KDJ死叉",
        "news_positive": "新闻偏利好",
        "news_negative": "新闻偏利空",
        "bos_up": "BOS向上突破",
        "bos_down": "BOS向下突破",
        "choch_bull": "CHOCH转多",
        "choch_bear": "CHOCH转空",
        "ema_trend_up": "EMA趋势上行",
        "ema_trend_down": "EMA趋势下行",
        "long_signal": "方向与幅度满足做多条件",
        "short_signal": "方向与幅度满足做空条件",
        "short_disallowed": "当前市场禁做空",
        "short_edge_below_threshold": "做空优势不足",
        "threshold_not_met": "阈值未触发",
        "probability_neutral": "方向概率中性",
        "position_below_minimum": "仓位低于最小阈值",
        "signal_neutral": "信号中性",
        "flat": "观望",
    }
    return mapping.get(key, str(token or "-"))


def _reason_token_en(token: object) -> str:
    key = str(token or "").strip().lower()
    mapping = {
        "ema_bull_cross": "EMA golden cross",
        "ema_bear_cross": "EMA death cross",
        "macd_golden_cross": "MACD golden cross",
        "macd_dead_cross": "MACD death cross",
        "supertrend_bullish": "SuperTrend bullish",
        "supertrend_bearish": "SuperTrend bearish",
        "lux_ms_bullish": "ML Adaptive SuperTrend bullish",
        "lux_ms_bearish": "ML Adaptive SuperTrend bearish",
        "volume_surge": "Volume surge",
        "rsi_oversold": "RSI oversold",
        "rsi_overbought": "RSI overbought",
        "kdj_golden_cross": "KDJ golden cross",
        "kdj_dead_cross": "KDJ death cross",
        "news_positive": "News sentiment positive",
        "news_negative": "News sentiment negative",
        "bos_up": "BOS breakout up",
        "bos_down": "BOS breakout down",
        "choch_bull": "CHOCH bullish shift",
        "choch_bear": "CHOCH bearish shift",
        "ema_trend_up": "EMA trend up",
        "ema_trend_down": "EMA trend down",
        "long_signal": "Direction + magnitude satisfy long conditions",
        "short_signal": "Direction + magnitude satisfy short conditions",
        "short_disallowed": "Shorting disallowed in current market",
        "short_edge_below_threshold": "Short edge below threshold",
        "threshold_not_met": "Threshold not met",
        "probability_neutral": "Direction probability is neutral",
        "position_below_minimum": "Position size below minimum threshold",
        "signal_neutral": "Signal neutral",
        "flat": "Wait",
    }
    return mapping.get(key, str(token or "-"))


def _reason_token_text(token: object) -> str:
    return _reason_token_cn(token) if _ui_lang() == "zh" else _reason_token_en(token)


def _format_reason_tokens_cn(reason_text: object) -> str:
    raw = str(reason_text or "").strip()
    if not raw:
        return "-"
    joiner = "；" if _ui_lang() == "zh" else "; "
    return joiner.join(_reason_token_text(t) for t in raw.split(";") if str(t).strip())


@st.cache_data(ttl=15, show_spinner=False)
def _fetch_live_btc_price() -> float | None:
    try:
        r = requests.get(
            "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT",
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        return float(data["price"])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _load_universe_cached(market: str, pool_key: str) -> pd.DataFrame:
    return load_universe(market, pool_key)


@st.cache_data(ttl=30, show_spinner=False)
def _build_selected_snapshot_cached(
    instrument_id: str,
    name: str,
    market: str,
    symbol: str,
    provider: str,
    timezone: str,
    horizon_unit: str,
    horizon_steps: int,
    history_lookback_days: int,
    config_path: str = "configs/config.yaml",
    schema_version: str = "dashboard_pages_v1",
) -> pd.DataFrame:
    inst = [
        {
            "id": instrument_id,
            "name": name,
            "market": market,
            "symbol": symbol,
            "provider": provider,
            "timezone": timezone,
            "horizon_unit": horizon_unit,
            "horizon_steps": horizon_steps,
            "history_lookback_days": history_lookback_days,
        }
    ]
    return build_market_snapshot_from_instruments(inst, config_path=config_path)


@st.cache_data(ttl=300, show_spinner=False)
def _build_session_bundle_cached(
    symbol: str,
    exchange: str,
    market_type: str,
    mode: str,
    horizon_hours: int,
    lookforward_days: int,
    refresh_token: int = 0,
    config_path: str = "configs/config.yaml",
    schema_version: str = "session_page_v1",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    _ = refresh_token
    bundle = build_session_forecast_bundle(
        symbol=symbol,
        exchange=exchange,
        market_type=market_type,
        mode=mode,
        horizon_hours=horizon_hours,
        lookforward_days=lookforward_days,
        config_path=config_path,
    )
    return bundle.hourly, bundle.blocks, bundle.daily, bundle.metadata


def _is_finite_number(x: object) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _safe_float(x: object) -> float:
    try:
        out = float(x)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


@st.cache_data(ttl=3600, show_spinner=False)
def _get_git_hash_short_cached() -> str:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        )
        return out.strip() or "-"
    except Exception:
        return "-"


def _signal_strength_label(edge_pp: float, weak_pp: float, strong_pp: float) -> str:
    if not np.isfinite(edge_pp):
        return "-"
    if edge_pp < weak_pp:
        return "弱"
    if edge_pp < strong_pp:
        return "中"
    return "强"


def _append_signal_strength_columns(df: pd.DataFrame, weak_pp: float, strong_pp: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["p_up"] = pd.to_numeric(out.get("p_up"), errors="coerce")
    out["signal_strength"] = (out["p_up"] - 0.5).abs()
    out["signal_strength_pp"] = out["signal_strength"] * 100.0
    out["signal_strength_score"] = (out["signal_strength"] * 200.0).clip(0.0, 100.0)
    out["signal_strength_label"] = out["signal_strength_pp"].map(
        lambda x: _signal_strength_label(float(x), weak_pp=weak_pp, strong_pp=strong_pp)
        if _is_finite_number(x)
        else "-"
    )
    return out


def _append_edge_columns(df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    cost_pct = float(cost_bps) / 10000.0
    out["q10_change_pct"] = pd.to_numeric(out.get("q10_change_pct"), errors="coerce")
    out["q50_change_pct"] = pd.to_numeric(out.get("q50_change_pct"), errors="coerce")
    out["q90_change_pct"] = pd.to_numeric(out.get("q90_change_pct"), errors="coerce")
    width = (out["q90_change_pct"] - out["q10_change_pct"]).abs()
    out["edge_score"] = out["q50_change_pct"] - cost_pct
    out["edge_score_short"] = (-out["q50_change_pct"]) - cost_pct
    out["edge_risk"] = out["edge_score"] / width.where(width > 1e-12, np.nan)
    out["edge_risk_short"] = out["edge_score_short"] / width.where(width > 1e-12, np.nan)
    return out


def _format_signal_strength(label: object, edge_pp: object, score: object) -> str:
    if not (_is_finite_number(edge_pp) and _is_finite_number(score)):
        return "-"
    return f"{_signal_strength_text(label)} ({float(edge_pp):.2f}pp / {float(score):.0f})"


def _rank_metric_options() -> Dict[str, str]:
    return {
        "p_up": _t("按 P(up)", "By P(up)"),
        "q50_change_pct": _t("按 q50_change_pct", "By q50_change_pct"),
        "volatility_score": _t("按 volatility_score", "By volatility_score"),
        "edge_score": _t("按 edge_score（推荐）", "By edge_score (recommended)"),
        "edge_risk": _t("按 edge_risk（高级）", "By edge_risk (advanced)"),
    }


def _sort_by_rank(df: pd.DataFrame, rank_key: str, side: str) -> pd.DataFrame:
    if df.empty:
        return df

    if side == "up":
        if rank_key == "p_up":
            return df.sort_values("p_up", ascending=False)
        if rank_key == "q50_change_pct":
            return df.sort_values("q50_change_pct", ascending=False)
        if rank_key == "volatility_score":
            return df.sort_values("volatility_score", ascending=False)
        if rank_key == "edge_risk":
            return df.sort_values("edge_risk", ascending=False)
        return df.sort_values("edge_score", ascending=False)

    if side == "down":
        if rank_key == "p_up":
            return df.sort_values("p_up", ascending=True)
        if rank_key == "q50_change_pct":
            return df.sort_values("q50_change_pct", ascending=True)
        if rank_key == "volatility_score":
            return df.sort_values("volatility_score", ascending=False)
        if rank_key == "edge_risk":
            return df.sort_values("edge_risk_short", ascending=False)
        return df.sort_values("edge_score_short", ascending=False)

    if rank_key == "p_up":
        return df.sort_values("signal_strength", ascending=False)
    if rank_key == "q50_change_pct":
        return df.sort_values("q50_change_pct", ascending=False, key=lambda s: s.abs())
    if rank_key == "edge_risk":
        return df.sort_values("edge_risk", ascending=False, key=lambda s: s.abs())
    if rank_key == "edge_score":
        return df.sort_values("edge_score", ascending=False, key=lambda s: s.abs())
    return df.sort_values("volatility_score", ascending=False)


def _style_signed_value(v: object) -> str:
    if not _is_finite_number(v):
        return ""
    value = float(v)
    if value > 0:
        return "color: #22c55e; font-weight: 600;"
    if value < 0:
        return "color: #ef4444; font-weight: 600;"
    return "color: #94a3b8;"


def _style_strength_label(v: object) -> str:
    text = str(v or "").lower()
    if ("强" in text) or ("strong" in text):
        return "color: #22c55e; font-weight: 600;"
    if ("中" in text) or ("medium" in text):
        return "color: #f59e0b; font-weight: 600;"
    if ("弱" in text) or ("weak" in text):
        return "color: #94a3b8; font-weight: 600;"
    return ""


def _signal_strength_text(label: object) -> str:
    lv = str(label or "")
    if lv in {"强", "strong"}:
        return _t("强", "Strong")
    if lv in {"中", "medium"}:
        return _t("中", "Medium")
    if lv in {"弱", "weak"}:
        return _t("弱", "Weak")
    return str(label or "-")


def _signal_strength_human_text(label: object) -> str:
    lv = str(label or "")
    if lv in {"强", "strong"}:
        return _t("强信号（可执行）", "Strong Signal (Executable)")
    if lv in {"中", "medium"}:
        return _t("中等信号（需风控）", "Medium Signal (Risk-control needed)")
    if lv in {"弱", "weak"}:
        return _t("弱信号（≈随机）", "Weak Signal (~coin flip)")
    return "-"


def _render_signal_badge(label: object) -> None:
    lv = str(label or "").lower()
    if lv in {"强", "strong"}:
        bg, fg = "#163a2f", "#86efac"
    elif lv in {"中", "medium"}:
        bg, fg = "#3f3113", "#fcd34d"
    else:
        bg, fg = "#1f2937", "#cbd5e1"
    text = _signal_strength_human_text(lv)
    st.markdown(
        (
            "<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
            f"background:{bg};color:{fg};font-size:12px;font-weight:700'>{text}</span>"
        ),
        unsafe_allow_html=True,
    )


def _auc_binary(y_true: pd.Series, y_score: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(y_score, errors="coerce")
    mask = y.notna() & s.notna()
    if mask.sum() < 3:
        return float("nan")
    y = y[mask].astype(int)
    s = s[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s.rank(method="average")
    sum_ranks_pos = float(ranks[y == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _compute_recent_hourly_reliability(
    *,
    symbol: str,
    horizon_hours: int,
    window_days: int = 30,
    cfg: Dict[str, object] | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame, str]:
    cfg_local = cfg or _load_main_config_cached("configs/config.yaml")
    model_symbol = str(((cfg_local.get("data", {}) if isinstance(cfg_local, dict) else {}).get("symbol", "BTCUSDT")))
    if str(symbol).upper() != model_symbol.upper():
        return {}, pd.DataFrame(), _t(
            f"近期模型可信度面板当前仅支持主模型币种：{model_symbol}。",
            f"Recent reliability panel currently supports only the primary model symbol: {model_symbol}.",
        )

    path = Path("data/processed/predictions_hourly.csv")
    if not path.exists():
        return {}, pd.DataFrame(), _t(
            "缺少 predictions_hourly.csv，无法计算近期可信度。",
            "Missing predictions_hourly.csv; cannot compute recent reliability.",
        )

    df = pd.read_csv(path)
    p_col = f"dir_h{int(horizon_hours)}_p_up"
    q10_col = f"ret_h{int(horizon_hours)}_q0.1"
    q90_col = f"ret_h{int(horizon_hours)}_q0.9"
    required = {"close", p_col, q10_col, q90_col}
    if not required.issubset(df.columns):
        return {}, pd.DataFrame(), _t(
            f"预测文件缺少列：{', '.join(sorted(required - set(df.columns)))}",
            f"Prediction file missing columns: {', '.join(sorted(required - set(df.columns)))}",
        )

    ts_col = "timestamp_market" if "timestamp_market" in df.columns else "timestamp_utc"
    if ts_col not in df.columns:
        return {}, pd.DataFrame(), _t("预测文件缺少时间列。", "Prediction file missing time column.")

    work = df[[ts_col, "close", p_col, q10_col, q90_col]].copy()
    work[ts_col] = pd.to_datetime(work[ts_col], errors="coerce")
    work = work.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work["realized_ret"] = work["close"].shift(-int(horizon_hours)) / work["close"] - 1.0
    work["y"] = (work["realized_ret"] > 0).astype(float)
    work["p"] = pd.to_numeric(work[p_col], errors="coerce")
    work["q10"] = pd.to_numeric(work[q10_col], errors="coerce")
    work["q90"] = pd.to_numeric(work[q90_col], errors="coerce")
    work = work.dropna(subset=["realized_ret", "p", "q10", "q90"])
    if work.empty:
        return {}, pd.DataFrame(), _t("可用于评估的样本不足。", "Insufficient samples for evaluation.")

    window_rows = max(24, int(window_days) * 24)
    w = work.tail(window_rows).copy()
    if w.empty:
        return {}, pd.DataFrame(), _t("滚动窗口样本为空。", "Rolling-window sample is empty.")

    pred_up = (w["p"] >= 0.5).astype(float)
    acc = float((pred_up == w["y"]).mean())
    auc = float(_auc_binary(w["y"], w["p"]))
    brier = float(((w["p"] - w["y"]) ** 2).mean())
    cov = float(((w["realized_ret"] >= w["q10"]) & (w["realized_ret"] <= w["q90"])).mean())
    width = float((w["q90"] - w["q10"]).mean())

    # Baselines on the same rolling window.
    naive_p = pd.Series(0.5, index=w.index, dtype=float)
    naive_pred = (naive_p >= 0.5).astype(float)
    prev_ret = w["close"] / w["close"].shift(1) - 1.0
    prev_p = (prev_ret > 0).astype(float).fillna(0.5)
    prev_pred = (prev_p >= 0.5).astype(float)

    rows = [
        {
            "模型": _t(f"当前模型(h={int(horizon_hours)}h)", f"Current Model (h={int(horizon_hours)}h)"),
            "样本范围": _t(f"近{int(window_days)}天滚动", f"Rolling {int(window_days)} days"),
            "Accuracy": acc,
            "AUC": auc,
            "Brier": brier,
        },
        {
            "模型": "Naive(0.5)",
            "样本范围": _t(f"近{int(window_days)}天滚动", f"Rolling {int(window_days)} days"),
            "Accuracy": float((naive_pred == w["y"]).mean()),
            "AUC": float("nan"),
            "Brier": float(((naive_p - w["y"]) ** 2).mean()),
        },
        {
            "模型": "Prev-bar",
            "样本范围": _t(f"近{int(window_days)}天滚动", f"Rolling {int(window_days)} days"),
            "Accuracy": float((prev_pred == w["y"]).mean()),
            "AUC": float(_auc_binary(w["y"], prev_p)),
            "Brier": float(((prev_p - w["y"]) ** 2).mean()),
        },
    ]

    wf_path = Path("data/processed/metrics_walk_forward_summary.csv")
    if wf_path.exists():
        wf = pd.read_csv(wf_path)
        req_cols = {"branch", "horizon", "task", "metric", "model", "mean"}
        if req_cols.issubset(wf.columns):
            wf = wf[
                (wf["branch"] == "hourly")
                & (pd.to_numeric(wf["horizon"], errors="coerce") == int(horizon_hours))
                & (wf["task"] == "direction")
                & (wf["metric"].isin(["accuracy", "roc_auc"]))
            ].copy()
        else:
            wf = pd.DataFrame()
        if not wf.empty:
            for model_name in ["baseline_logistic", "mvp_lightgbm", "baseline_prev_bar"]:
                sub = wf[wf["model"] == model_name]
                if sub.empty:
                    continue
                acc_m = pd.to_numeric(sub.loc[sub["metric"] == "accuracy", "mean"], errors="coerce")
                auc_m = pd.to_numeric(sub.loc[sub["metric"] == "roc_auc", "mean"], errors="coerce")
                rows.append(
                    {
                        "模型": model_name,
                        "样本范围": "walk_forward_summary(全样本)",
                        "Accuracy": float(acc_m.iloc[0]) if not acc_m.empty else float("nan"),
                        "AUC": float(auc_m.iloc[0]) if not auc_m.empty else float("nan"),
                        "Brier": float("nan"),
                    }
                )

    summary = {
        "accuracy": acc,
        "auc": auc,
        "brier": brier,
        "coverage": cov,
        "width": width,
        "samples": float(len(w)),
    }
    compare_df = pd.DataFrame(rows)
    return summary, compare_df, ""


@st.cache_data(ttl=300, show_spinner=False)
def _compute_recent_symbol_reliability_cached(
    *,
    market: str,
    symbol: str,
    provider: str,
    fallback_symbol: str = "",
    horizon_steps: int = 1,
    window_days: int = 30,
    config_path: str = "configs/config.yaml",
) -> Tuple[Dict[str, float], pd.DataFrame, str]:
    from src.evaluation.backtest_multi_market import _build_model_like_signals, _fetch_daily_bars

    cfg = load_config(config_path)
    bt_cfg = cfg.get("backtest_multi_market", {})
    lookback_days_cfg = bt_cfg.get(
        "lookback_days",
        {"crypto": 540, "cn_equity": 1200, "us_equity": 1200},
    )
    mk = str(market)
    sym = str(symbol)
    prov = str(provider or ("binance" if mk == "crypto" else "yahoo"))
    lb_days = int(lookback_days_cfg.get(mk, 720))

    try:
        bars, _ = _fetch_daily_bars(
            market=mk,
            symbol=sym,
            provider=prov,
            lookback_days=lb_days,
            cfg=cfg,
        )
    except Exception:
        if mk == "crypto":
            fs = str(fallback_symbol or "").strip().upper()
            if not fs and prov == "coingecko":
                guess = str(sym).strip().upper()
                if guess.isalpha() and len(guess) <= 8:
                    fs = f"{guess}USDT"
            if fs.endswith("USDT"):
                bars, _ = _fetch_daily_bars(
                    market=mk,
                    symbol=fs,
                    provider="binance",
                    lookback_days=lb_days,
                    cfg=cfg,
                )
                sym = fs
            else:
                return {}, pd.DataFrame(), _t(
                    "无法获取该标的历史数据，无法计算近期可信度。",
                    "Failed to fetch historical data for this symbol; cannot compute recent reliability.",
                )
        else:
            return {}, pd.DataFrame(), _t(
                "无法获取该标的历史数据，无法计算近期可信度。",
                "Failed to fetch historical data for this symbol; cannot compute recent reliability.",
            )

    bars = bars.dropna(subset=["timestamp_utc", "open", "close"]).sort_values("timestamp_utc").reset_index(drop=True)
    if bars.empty:
        return {}, pd.DataFrame(), _t("历史数据为空。", "Historical data is empty.")

    modeled = _build_model_like_signals(bars, cfg=cfg, market=mk)
    if modeled.empty:
        return {}, pd.DataFrame(), _t("信号建模结果为空。", "Signal modeling output is empty.")

    h = max(1, int(horizon_steps))
    work = modeled.copy()
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work["realized_ret_h"] = work["close"].shift(-h) / work["close"] - 1.0
    p_raw = pd.to_numeric(work.get("p_up"), errors="coerce")
    decay = float(np.exp(-0.12 * max(0, h - 1)))
    work["p_up_h"] = 0.5 + (p_raw - 0.5) * decay
    q10 = pd.to_numeric(work.get("q10_change_pct"), errors="coerce")
    q50 = pd.to_numeric(work.get("q50_change_pct"), errors="coerce")
    q90 = pd.to_numeric(work.get("q90_change_pct"), errors="coerce")
    work["q10_h"] = np.power(1.0 + q10, h) - 1.0
    work["q50_h"] = np.power(1.0 + q50, h) - 1.0
    work["q90_h"] = np.power(1.0 + q90, h) - 1.0
    work["y"] = (work["realized_ret_h"] > 0).astype(float)
    work = work.dropna(subset=["realized_ret_h", "p_up_h", "q10_h", "q90_h", "y"])
    if work.empty:
        return {}, pd.DataFrame(), _t("有效评估样本不足。", "Insufficient valid samples for evaluation.")

    window_rows = max(20, int(window_days))
    w = work.tail(window_rows).copy()
    if w.empty:
        return {}, pd.DataFrame(), _t("滚动窗口为空。", "Rolling window is empty.")

    pred_up = (w["p_up_h"] >= 0.5).astype(float)
    acc = float((pred_up == w["y"]).mean())
    auc = float(_auc_binary(w["y"], w["p_up_h"]))
    brier = float(((w["p_up_h"] - w["y"]) ** 2).mean())
    coverage = float(((w["realized_ret_h"] >= w["q10_h"]) & (w["realized_ret_h"] <= w["q90_h"])).mean())
    width = float((w["q90_h"] - w["q10_h"]).mean())

    naive_p = pd.Series(0.5, index=w.index, dtype=float)
    naive_pred = (naive_p >= 0.5).astype(float)
    prev_ret_h = w["close"] / w["close"].shift(h) - 1.0
    prev_p = (prev_ret_h > 0).astype(float).fillna(0.5)
    prev_pred = (prev_p >= 0.5).astype(float)
    rows = [
        {
            "模型": _t(f"当前模型(h={h}d)", f"Current Model (h={h}d)"),
            "样本范围": _t(f"近{int(window_days)}天滚动", f"Rolling {int(window_days)} days"),
            "Accuracy": acc,
            "AUC": auc,
            "Brier": brier,
            "Coverage": coverage,
        },
        {
            "模型": "Naive(0.5)",
            "样本范围": _t(f"近{int(window_days)}天滚动", f"Rolling {int(window_days)} days"),
            "Accuracy": float((naive_pred == w["y"]).mean()),
            "AUC": float("nan"),
            "Brier": float(((naive_p - w["y"]) ** 2).mean()),
            "Coverage": float("nan"),
        },
        {
            "模型": "Prev-h bar",
            "样本范围": _t(f"近{int(window_days)}天滚动", f"Rolling {int(window_days)} days"),
            "Accuracy": float((prev_pred == w["y"]).mean()),
            "AUC": float(_auc_binary(w["y"], prev_p)),
            "Brier": float(((prev_p - w["y"]) ** 2).mean()),
            "Coverage": float("nan"),
        },
    ]

    summary = {
        "accuracy": acc,
        "auc": auc,
        "brier": brier,
        "coverage": coverage,
        "width": width,
        "samples": float(len(w)),
    }
    return summary, pd.DataFrame(rows), ""


def _model_health_grade(summary: Dict[str, float] | None) -> str:
    if not summary:
        return "中"
    brier = _safe_float(summary.get("brier"))
    coverage = _safe_float(summary.get("coverage"))
    if not (np.isfinite(brier) and np.isfinite(coverage)):
        return "中"
    if brier <= 0.20 and 0.72 <= coverage <= 0.88:
        return "良"
    if brier <= 0.28 and 0.60 <= coverage <= 0.95:
        return "中"
    return "差"


def _model_health_text(level: str) -> str:
    lv = str(level or "").strip()
    if _ui_lang() == "zh":
        return lv or "中"
    mapping = {"良": "Good", "中": "Medium", "差": "Poor"}
    return mapping.get(lv, lv or "Medium")


def _parse_horizon_label(label: str) -> Tuple[str, int]:
    text = str(label or "").strip().lower()
    if text.endswith("h"):
        try:
            return "hour", max(1, int(text[:-1]))
        except Exception:
            return "hour", 1
    if text.endswith("d"):
        try:
            return "day", max(1, int(text[:-1]))
        except Exception:
            return "day", 1
    return "day", 1


def _default_lookback_days(market: str) -> int:
    if market == "crypto":
        return 365
    return 730


def _build_snapshot_fresh(inst: list[dict[str, object]]) -> pd.DataFrame:
    snapshot_mod = importlib.import_module("src.markets.snapshot")
    snapshot_mod = importlib.reload(snapshot_mod)
    return snapshot_mod.build_market_snapshot_from_instruments(inst, config_path="configs/config.yaml")


def _ensure_snapshot_factors(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    row = df.iloc[0]
    factor_cols = [
        "size_factor",
        "value_factor",
        "growth_factor",
        "momentum_factor",
        "reversal_factor",
        "low_vol_factor",
    ]
    has_any_factor = any(_is_finite_number(row.get(c)) for c in factor_cols)
    if has_any_factor:
        return df

    horizon_unit, horizon_steps = _parse_horizon_label(str(row.get("horizon_label", "1d")))
    fallback = [
        {
            "id": str(row.get("instrument_id", "unknown")),
            "name": str(row.get("name", "unknown")),
            "market": str(row.get("market", "unknown")),
            "symbol": str(row.get("symbol", "")),
            "provider": str(row.get("provider", "")),
            "timezone": str(row.get("timezone", "UTC")),
            "horizon_unit": horizon_unit,
            "horizon_steps": horizon_steps,
            "history_lookback_days": _default_lookback_days(str(row.get("market", ""))),
        }
    ]
    try:
        refreshed = _build_snapshot_fresh(fallback)
        if not refreshed.empty:
            return refreshed
    except Exception:
        pass
    return df


def _infer_horizons(columns: List[str]) -> List[int]:
    horizons = set()
    for c in columns:
        if c.startswith("dir_h") and c.endswith("_p_up"):
            mid = c.replace("dir_h", "").replace("_p_up", "")
            try:
                horizons.add(int(mid))
            except Exception:
                pass
    return sorted(horizons)


def _format_price(x: float | int | None) -> str:
    try:
        value = float(x)
        if not np.isfinite(value):
            return "-"
        return f"${value:,.2f}"
    except Exception:
        return "-"


def _format_change_pct(x: float | int | None) -> str:
    try:
        value = float(x)
        if not np.isfinite(value):
            return "-"
        return f"{value:+.2%}"
    except Exception:
        return "-"


def _format_float(x: float | int | None, digits: int = 4) -> str:
    try:
        value = float(x)
        if not np.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"
    except Exception:
        return "-"


def _render_big_value(label: str, value: str, *, caption: str = "") -> None:
    st.markdown(f"**{label}**")
    st.markdown(
        (
            "<div style='font-size:3rem;font-weight:700;line-height:1.15;"
            "word-break:break-word;overflow-wrap:anywhere;'>"
            f"{value}</div>"
        ),
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def _trend_cn(label: str) -> str:
    mapping = {
        "bullish": "偏多",
        "bearish": "偏空",
        "sideways": "震荡",
    }
    return mapping.get(str(label), str(label))


def _trend_text(label: str) -> str:
    cn = _trend_cn(label)
    if _ui_lang() == "zh":
        return cn
    mapping = {
        "偏多": "Bullish",
        "偏空": "Bearish",
        "震荡": "Sideways",
        "趋势偏多": "Bullish Trend",
        "趋势偏空": "Bearish Trend",
        "趋势混合": "Mixed Trend",
    }
    return mapping.get(cn, str(label))


def _risk_cn(label: str) -> str:
    mapping = {
        "low": "低",
        "medium": "中",
        "high": "高",
        "extreme": "极高",
    }
    return mapping.get(str(label), str(label))


def _risk_text(label: str) -> str:
    cn = _risk_cn(label)
    if _ui_lang() == "zh":
        return cn
    mapping = {"低": "Low", "中": "Medium", "高": "High", "极高": "Extreme"}
    return mapping.get(cn, str(label))


def _policy_action_cn(label: str) -> str:
    mapping = {
        "Long": "做多",
        "Short": "做空",
        "Flat": "观望",
    }
    return mapping.get(str(label), str(label))


def _policy_action_text(label: str) -> str:
    cn = _policy_action_cn(label)
    if _ui_lang() == "zh":
        return cn
    mapping = {"做多": "Long", "做空": "Short", "观望": "Wait"}
    return mapping.get(cn, str(label))


def _session_display_name(session_name: str) -> str:
    key = str(session_name)
    if _ui_lang() == "zh":
        mapping = {"asia": "亚盘", "europe": "欧盘", "us": "美盘"}
    else:
        mapping = {"asia": "Asia", "europe": "Europe", "us": "US"}
    return mapping.get(key, key)


def _render_hourly_heatmap(
    hourly_df: pd.DataFrame, value_col: str, title: str, *, horizon_hours: int
) -> None:
    if hourly_df.empty or value_col not in hourly_df.columns:
        st.info(_t("暂无热力图数据。", "No heatmap data available."))
        return
    work = hourly_df.copy()
    work = work.sort_values("hour_bj")
    x = [f"{int(h):02d}:00" for h in work["hour_bj"].tolist()]
    z_values = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0).tolist()
    fig = go.Figure(
        data=go.Heatmap(
            z=[z_values],
            x=x,
            y=[title],
            colorscale="Blues",
            colorbar=dict(title=title),
        )
    )
    fig.update_layout(
        title=_t(
            f"24小时热力图：从该小时开始的未来{int(horizon_hours)}h {title}（北京时间）",
            f"24h Heatmap: Future {int(horizon_hours)}h from each hour start ({title}, Asia/Shanghai)",
        ),
        xaxis_title=_t("小时", "Hour"),
        yaxis_title=_t("指标", "Metric"),
        template="plotly_white",
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_top_tables(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    top_n: int,
    *,
    rank_key: str,
    cost_bps: float,
    weak_pp: float,
    strong_pp: float,
    horizon_hours: int,
) -> None:
    rank_name = _rank_metric_options().get(rank_key, rank_key)
    st.caption(
        _t(
            f"榜单统一排序：{rank_name} | 成本估计：{float(cost_bps):.1f} bps | "
            f"小时级语义：从该小时起算未来 {int(horizon_hours)}h。",
            f"Ranking: {rank_name} | Cost estimate: {float(cost_bps):.1f} bps | "
            f"Hourly semantics: future {int(horizon_hours)}h from each hour start.",
        )
    )

    col_trend = _t("趋势", "Trend")
    col_risk = _t("风险", "Risk")
    col_pup = _t("上涨概率", "P(up)")
    col_pdown = _t("下跌概率", "P(down)")
    col_q50 = _t("预期涨跌幅", "Expected Change")
    col_target = _t("目标价格(q50)", "Target Price (q50)")
    col_strength = _t("信号强弱", "Signal Strength")
    col_edge = _t("机会值(edge)", "Edge")
    col_edge_risk = _t("风险调整机会", "Risk-adjusted Edge")
    col_edge_short = _t("空头机会(edge)", "Short Edge")
    col_edge_risk_short = _t("空头风险调整机会", "Short Risk-adjusted Edge")
    col_action = _t("策略动作", "Policy Action")
    col_pos = _t("建议仓位", "Position Size")
    col_net_edge = _t("预期净优势", "Expected Net Edge")
    col_conf = _t("置信度", "Confidence")
    hour_section_up = _t("小时级：最可能上涨 Top N", "Hourly: Top N Upside")
    hour_section_down = _t("小时级：最可能下跌 Top N", "Hourly: Top N Downside")
    hour_section_vol = _t("小时级：最可能大波动 Top N", "Hourly: Top N Volatility")
    day_section_up = _t("日线级：最可能上涨 Top N", "Daily: Top N Upside")
    day_section_down = _t("日线级：最可能下跌 Top N", "Daily: Top N Downside")
    day_section_vol = _t("日线级：最可能大波动 Top N", "Daily: Top N Volatility")

    if hourly_df.empty:
        st.info(_t("暂无小时级榜单数据。", "No hourly ranking data available."))
    else:
        h = hourly_df.copy()
        h = _append_signal_strength_columns(h, weak_pp=weak_pp, strong_pp=strong_pp)
        h = _append_edge_columns(h, cost_bps=cost_bps)
        h[col_trend] = h["trend_label"].map(_trend_text)
        h[col_risk] = h["risk_level"].map(_risk_text)
        h[col_pup] = h["p_up"].map(lambda x: _format_change_pct(x).replace("+", ""))
        h[col_pdown] = h["p_down"].map(lambda x: _format_change_pct(x).replace("+", ""))
        h[col_q50] = h["q50_change_pct"].map(_format_change_pct)
        h[col_target] = h["target_price_q50"].map(_format_price)
        h[col_conf] = h["confidence_score"].map(lambda x: _format_float(x, 1))
        h[col_strength] = h.apply(
            lambda r: _format_signal_strength(
                r.get("signal_strength_label", "-"),
                r.get("signal_strength_pp"),
                r.get("signal_strength_score"),
            ),
            axis=1,
        )
        h[col_edge] = h["edge_score"].map(_format_change_pct)
        h[col_edge_risk] = h["edge_risk"].map(lambda x: _format_float(x, 3))
        h[col_edge_short] = h["edge_score_short"].map(_format_change_pct)
        h[col_edge_risk_short] = h["edge_risk_short"].map(lambda x: _format_float(x, 3))
        if "policy_action" in h.columns:
            h[col_action] = h["policy_action"].map(_policy_action_text)
            h[col_pos] = h["policy_position_size"].map(
                lambda x: f"{float(x):.1%}" if _is_finite_number(x) else "-"
            )
            h[col_net_edge] = h["policy_expected_edge_pct"].map(_format_change_pct)

        up_rank = _sort_by_rank(h, rank_key=rank_key, side="up")
        down_rank = _sort_by_rank(h, rank_key=rank_key, side="down")
        vol_rank = _sort_by_rank(h, rank_key=rank_key, side="vol")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**{hour_section_up}（{rank_name}）**")
            cols = [
                "hour_label",
                col_pup,
                col_q50,
                col_target,
                col_strength,
                col_edge,
                col_edge_risk,
                col_action,
                col_pos,
                col_net_edge,
                col_trend,
                col_risk,
                col_conf,
            ]
            cols = [c for c in cols if c in h.columns]
            st.dataframe(up_rank[cols].head(top_n), use_container_width=True, hide_index=True)
        with c2:
            st.markdown(f"**{hour_section_down}（{rank_name}）**")
            cols = [
                "hour_label",
                col_pdown,
                col_q50,
                col_target,
                col_strength,
                col_edge_short,
                col_edge_risk_short,
                col_action,
                col_pos,
                col_net_edge,
                col_trend,
                col_risk,
                col_conf,
            ]
            cols = [c for c in cols if c in h.columns]
            st.dataframe(down_rank[cols].head(top_n), use_container_width=True, hide_index=True)
        with c3:
            st.markdown(f"**{hour_section_vol}（{rank_name}）**")
            cols = [
                "hour_label",
                col_q50,
                col_target,
                col_strength,
                col_edge,
                col_edge_risk,
                col_action,
                col_pos,
                col_net_edge,
                col_trend,
                col_risk,
                col_conf,
            ]
            cols = [c for c in cols if c in h.columns]
            st.dataframe(vol_rank[cols].head(top_n), use_container_width=True, hide_index=True)

    st.markdown("---")
    if daily_df.empty:
        st.info(_t("暂无日线级榜单数据。", "No daily ranking data available."))
    else:
        d = daily_df.copy()
        d = _append_signal_strength_columns(d, weak_pp=weak_pp, strong_pp=strong_pp)
        d = _append_edge_columns(d, cost_bps=cost_bps)
        d[col_trend] = d["trend_label"].map(_trend_text)
        d[col_risk] = d["risk_level"].map(_risk_text)
        d[col_pup] = d["p_up"].map(lambda x: _format_change_pct(x).replace("+", ""))
        d[col_pdown] = d["p_down"].map(lambda x: _format_change_pct(x).replace("+", ""))
        d[col_q50] = d["q50_change_pct"].map(_format_change_pct)
        d[col_target] = d["target_price_q50"].map(_format_price)
        d[col_conf] = d["confidence_score"].map(lambda x: _format_float(x, 1))
        d[col_strength] = d.apply(
            lambda r: _format_signal_strength(
                r.get("signal_strength_label", "-"),
                r.get("signal_strength_pp"),
                r.get("signal_strength_score"),
            ),
            axis=1,
        )
        d[col_edge] = d["edge_score"].map(_format_change_pct)
        d[col_edge_risk] = d["edge_risk"].map(lambda x: _format_float(x, 3))
        d[col_edge_short] = d["edge_score_short"].map(_format_change_pct)
        d[col_edge_risk_short] = d["edge_risk_short"].map(lambda x: _format_float(x, 3))
        if "policy_action" in d.columns:
            d[col_action] = d["policy_action"].map(_policy_action_text)
            d[col_pos] = d["policy_position_size"].map(
                lambda x: f"{float(x):.1%}" if _is_finite_number(x) else "-"
            )
            d[col_net_edge] = d["policy_expected_edge_pct"].map(_format_change_pct)

        up_rank = _sort_by_rank(d, rank_key=rank_key, side="up")
        down_rank = _sort_by_rank(d, rank_key=rank_key, side="down")
        vol_rank = _sort_by_rank(d, rank_key=rank_key, side="vol")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**{day_section_up}（{rank_name}）**")
            cols = [
                "date_bj",
                col_pup,
                col_q50,
                col_target,
                col_strength,
                col_edge,
                col_edge_risk,
                col_action,
                col_pos,
                col_net_edge,
                col_trend,
                col_risk,
                col_conf,
            ]
            cols = [c for c in cols if c in d.columns]
            st.dataframe(up_rank[cols].head(top_n), use_container_width=True, hide_index=True)
        with c2:
            st.markdown(f"**{day_section_down}（{rank_name}）**")
            cols = [
                "date_bj",
                col_pdown,
                col_q50,
                col_target,
                col_strength,
                col_edge_short,
                col_edge_risk_short,
                col_action,
                col_pos,
                col_net_edge,
                col_trend,
                col_risk,
                col_conf,
            ]
            cols = [c for c in cols if c in d.columns]
            st.dataframe(down_rank[cols].head(top_n), use_container_width=True, hide_index=True)
        with c3:
            st.markdown(f"**{day_section_vol}（{rank_name}）**")
            cols = [
                "date_bj",
                col_q50,
                col_target,
                col_strength,
                col_edge,
                col_edge_risk,
                col_action,
                col_pos,
                col_net_edge,
                col_trend,
                col_risk,
                col_conf,
            ]
            cols = [c for c in cols if c in d.columns]
            st.dataframe(vol_rank[cols].head(top_n), use_container_width=True, hide_index=True)


def _render_crypto_session_page() -> None:
    st.header(_t("交易时间段预测（Crypto）", "Session Forecast (Crypto)"))
    st.caption(
        _t(
            "北京时间24小时制；支持亚盘/欧盘/美盘、关键小时概率与未来N天日线预测。",
            "Beijing-time 24h view with Asia/Europe/US sessions, key-hour probabilities, and next-N-day daily forecast.",
        )
    )

    if "session_refresh_token" not in st.session_state:
        st.session_state["session_refresh_token"] = 0

    cfg = {}
    try:
        import yaml

        cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}
    fc = cfg.get("forecast_config", {})
    source_cfg = fc.get("data_source", {})
    strength_cfg = fc.get("signal_strength", {})
    rank_cfg = fc.get("ranking", {})

    symbols = fc.get("symbols", {}).get("default", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    exchanges = source_cfg.get("exchanges", ["binance", "bybit"])
    market_types = source_cfg.get("market_types", ["perp", "spot"])
    default_exchange = source_cfg.get("default_exchange", "binance")
    default_market_type = source_cfg.get("default_market_type", "perp")
    default_horizon = int(fc.get("hourly", {}).get("horizon_hours", 4))
    default_days = int(fc.get("daily", {}).get("lookforward_days", 14))
    weak_pp = float(strength_cfg.get("weak_threshold_pp", 2.0))
    strong_pp = float(strength_cfg.get("strong_threshold_pp", 5.0))
    default_cost_bps = float(rank_cfg.get("cost_bps", 8.0))

    f1, f2, f3, f4, f5 = st.columns([2, 1, 1, 1, 1])
    symbol = f1.selectbox(_t("币种", "Symbol"), options=symbols, index=0, key="session_symbol")
    exchange = f2.selectbox(
        _t("数据源", "Exchange"),
        options=exchanges,
        index=exchanges.index(default_exchange) if default_exchange in exchanges else 0,
        key="session_exchange",
    )
    market_type = f3.selectbox(
        _t("市场类型", "Market Type"),
        options=market_types,
        index=market_types.index(default_market_type) if default_market_type in market_types else 0,
        key="session_market_type",
    )
    mode = f4.selectbox(_t("模式", "Mode"), options=["forecast", "seasonality"], index=0, key="session_mode")
    horizon_hours = int(
        f5.selectbox(_t("小时周期", "Horizon (hour)"), options=[4], index=0 if default_horizon == 4 else 0, key="session_horizon")
    )

    c1, c2 = st.columns([1, 3])
    lookforward_days = int(c1.slider(_t("未来N天（日线）", "Next N Days (daily)"), 7, 30, default_days, 1, key="session_daily_n"))
    compare_view = bool(c2.checkbox(_t("对照视图：Forecast vs Seasonality", "Compare View: Forecast vs Seasonality"), value=False, key="session_compare_view"))
    if c2.button(_t("刷新并重算", "Refresh and Recompute"), key="session_refresh_btn"):
        st.session_state["session_refresh_token"] += 1

    try:
        hourly_df, blocks_df, daily_df, meta = _build_session_bundle_cached(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            mode=mode,
            horizon_hours=horizon_hours,
            lookforward_days=lookforward_days,
            refresh_token=int(st.session_state["session_refresh_token"]),
        )
    except Exception as exc:
        st.error(_t(f"时段预测计算失败：{exc}", f"Session forecast failed: {exc}"))
        return

    mode_actual = str(meta.get("mode_actual", mode))
    if mode_actual != mode:
        st.warning(_t(f"请求模式是 `{mode}`，当前自动降级为 `{mode_actual}`。", f"Requested mode `{mode}` downgraded to `{mode_actual}`."))
    st.caption(
        _t(
            f"最新价格：{_format_price(meta.get('current_price'))} | 更新时间（北京时间）：{meta.get('data_updated_at_bj', '-')} | 模式/周期：{mode_actual} / {int(horizon_hours)}h",
            f"Latest Price: {_format_price(meta.get('current_price'))} | Updated (Asia/Shanghai): {meta.get('data_updated_at_bj', '-')} | Mode/Horizon: {mode_actual} / {int(horizon_hours)}h",
        )
    )

    data_version_seed = (
        f"{meta.get('symbol', '-')}"
        f"|{meta.get('exchange_actual', '-')}"
        f"|{meta.get('market_type', '-')}"
        f"|{meta.get('data_updated_at_bj', '-')}"
        f"|{meta.get('data_source_actual', '-')}"
    )
    data_version = hashlib.sha1(data_version_seed.encode("utf-8")).hexdigest()[:12]
    with st.expander(_t("数据 & 模型信息", "Data & Model Info"), expanded=False):
        st.markdown(
            _t(
                f"- 数据源：{meta.get('exchange_actual', '-')} / {meta.get('market_type', '-')} / {meta.get('symbol', '-')}\n"
                f"- 请求数据源：{meta.get('exchange', '-')}\n"
                f"- 数据更新时间（北京时间）：{meta.get('data_updated_at_bj', '-')}\n"
                f"- 预测生成时间（北京时间）：{meta.get('forecast_generated_at_bj', '-')}\n"
                f"- horizon={int(horizon_hours)}h / mode={mode_actual}\n"
                f"- model_version：{meta.get('model_version', '-')}\n"
                f"- data_version：{data_version}\n"
                f"- git_hash：{_get_git_hash_short_cached()}",
                f"- Source: {meta.get('exchange_actual', '-')} / {meta.get('market_type', '-')} / {meta.get('symbol', '-')}\n"
                f"- Requested source: {meta.get('exchange', '-')}\n"
                f"- Data updated (Asia/Shanghai): {meta.get('data_updated_at_bj', '-')}\n"
                f"- Forecast generated (Asia/Shanghai): {meta.get('forecast_generated_at_bj', '-')}\n"
                f"- horizon={int(horizon_hours)}h / mode={mode_actual}\n"
                f"- model_version: {meta.get('model_version', '-')}\n"
                f"- data_version: {data_version}\n"
                f"- git_hash: {_get_git_hash_short_cached()}",
            )
        )
        st.caption(f"Data Source Detail: {meta.get('data_source_actual', '-')}")

    # Session cards
    if blocks_df.empty:
        st.info(_t("暂无时段汇总数据。", "No session summary data available."))
    else:
        cards = _append_signal_strength_columns(blocks_df.copy(), weak_pp=weak_pp, strong_pp=strong_pp)
        cards["session_name_cn"] = cards.get("session_name_cn", cards["session_name"].map(_session_display_name))
        cards = cards.sort_values(
            "session_name", key=lambda s: s.map({"asia": 0, "europe": 1, "us": 2}).fillna(9)
        )
        cols = st.columns(3)
        for idx, (_, row) in enumerate(cards.iterrows()):
            col = cols[idx % 3]
            with col:
                st.markdown(f"**{row.get('session_name_cn', '-') }（{row.get('session_hours', '-') }）**")
                st.metric(_t("上涨概率", "P(up)"), _format_change_pct(row.get("p_up")).replace("+", ""))
                st.metric(_t("下跌概率", "P(down)"), _format_change_pct(row.get("p_down")).replace("+", ""))
                st.metric(_t("预期涨跌幅(q50)", "Expected Change (q50)"), _format_change_pct(row.get("q50_change_pct")))
                st.metric(_t("目标价格(q50)", "Target Price (q50)"), _format_price(row.get("target_price_q50")))
                _render_signal_badge(row.get("signal_strength_label", "-"))
                st.caption(
                    _t("信号强度：", "Signal strength: ")
                    + _format_signal_strength(
                        row.get("signal_strength_label", "-"),
                        row.get("signal_strength_pp"),
                        row.get("signal_strength_score"),
                    )
                )
                st.caption(
                    _t(
                        f"趋势：{_trend_text(str(row.get('trend_label', '-')))} | "
                        f"风险：{_risk_text(str(row.get('risk_level', '-')))} | "
                        f"置信度：{_format_float(row.get('confidence_score'), 1)}",
                        f"Trend: {_trend_text(str(row.get('trend_label', '-')))} | "
                        f"Risk: {_risk_text(str(row.get('risk_level', '-')))} | "
                        f"Confidence: {_format_float(row.get('confidence_score'), 1)}",
                    )
                )
                if "policy_action" in row.index:
                    st.caption(
                        _t(
                            f"策略：{_policy_action_text(str(row.get('policy_action', 'Flat')))} | "
                            f"仓位：{(float(row.get('policy_position_size')) if _is_finite_number(row.get('policy_position_size')) else 0.0):.1%} | "
                            f"净优势：{_format_change_pct(row.get('policy_expected_edge_pct'))}",
                            f"Action: {_policy_action_text(str(row.get('policy_action', 'Flat')))} | "
                            f"Size: {(float(row.get('policy_position_size')) if _is_finite_number(row.get('policy_position_size')) else 0.0):.1%} | "
                            f"Net edge: {_format_change_pct(row.get('policy_expected_edge_pct'))}",
                        )
                    )

    st.caption(
        _t(
            f"信号解释：`弱信号` (<{weak_pp:.1f}pp) 接近抛硬币，谨慎解读；"
            f"`中信号` ({weak_pp:.1f}-{strong_pp:.1f}pp)；`强信号` (>{strong_pp:.1f}pp)。",
            f"Signal guide: `Weak` (<{weak_pp:.1f}pp) is close to coin flip; "
            f"`Medium` ({weak_pp:.1f}-{strong_pp:.1f}pp); `Strong` (>{strong_pp:.1f}pp).",
        )
    )

    if compare_view:
        compare_mode = "seasonality" if mode == "forecast" else "forecast"
        try:
            cmp_hourly, cmp_blocks, _, cmp_meta = _build_session_bundle_cached(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                mode=compare_mode,
                horizon_hours=horizon_hours,
                lookforward_days=lookforward_days,
                refresh_token=int(st.session_state["session_refresh_token"]),
            )
            st.markdown(f"### {_t('Forecast vs Seasonality 对照（同参数）', 'Forecast vs Seasonality (same params)')}")
            st.caption(
                _t(
                    f"当前模式：{mode_actual} | 对照模式：{cmp_meta.get('mode_actual', compare_mode)} | "
                    "重点看 Δp_up（Forecast - Seasonality）。",
                    f"Current mode: {mode_actual} | Compare mode: {cmp_meta.get('mode_actual', compare_mode)} | "
                    "Focus on Δp_up (Forecast - Seasonality).",
                )
            )
            if not blocks_df.empty and not cmp_blocks.empty:
                left = blocks_df[["session_name", "session_name_cn", "p_up", "q50_change_pct"]].copy()
                right = cmp_blocks[["session_name", "p_up", "q50_change_pct"]].copy()
                merged = left.merge(right, on="session_name", how="inner", suffixes=("_main", "_cmp"))
                merged["Δp_up"] = merged["p_up_main"] - merged["p_up_cmp"]
                merged["Δq50"] = merged["q50_change_pct_main"] - merged["q50_change_pct_cmp"]
                show = merged.rename(
                    columns={
                        "session_name_cn": "时段",
                        "p_up_main": f"{mode_actual} p_up",
                        "p_up_cmp": f"{cmp_meta.get('mode_actual', compare_mode)} p_up",
                        "q50_change_pct_main": f"{mode_actual} q50",
                        "q50_change_pct_cmp": f"{cmp_meta.get('mode_actual', compare_mode)} q50",
                    }
                )
                fmt = {
                    c: "{:.2%}"
                    for c in [
                        f"{mode_actual} p_up",
                        f"{cmp_meta.get('mode_actual', compare_mode)} p_up",
                        f"{mode_actual} q50",
                        f"{cmp_meta.get('mode_actual', compare_mode)} q50",
                        "Δp_up",
                        "Δq50",
                    ]
                    if c in show.columns
                }
                styled_cmp = show.style.format(fmt, na_rep="-")
                delta_cols = [c for c in ["Δp_up", "Δq50"] if c in show.columns]
                if delta_cols:
                    styled_cmp = styled_cmp.applymap(_style_signed_value, subset=delta_cols)
                st.dataframe(styled_cmp, use_container_width=True, hide_index=True)
                st.caption(_t("若 Δp_up 绝对值较大，说明模型观点与历史季节性节奏有明显偏离。", "A large absolute Δp_up means model view significantly deviates from historical seasonality."))
            else:
                st.info(_t("对照视图数据不足。", "Insufficient data for compare view."))
        except Exception as exc:
            st.warning(_t(f"对照视图构建失败：{exc}", f"Failed to build compare view: {exc}"))

    st.markdown("---")
    tab_up = _t("上涨概率", "P(up)")
    tab_down = _t("下跌概率", "P(down)")
    tab_vol = _t("波动强度", "Volatility")
    tab_conf = _t("置信度", "Confidence")
    tab1, tab2, tab3, tab4 = st.tabs([tab_up, tab_down, tab_vol, tab_conf])
    with tab1:
        _render_hourly_heatmap(hourly_df, "p_up", tab_up, horizon_hours=horizon_hours)
    with tab2:
        _render_hourly_heatmap(hourly_df, "p_down", tab_down, horizon_hours=horizon_hours)
    with tab3:
        _render_hourly_heatmap(hourly_df, "volatility_score", tab_vol, horizon_hours=horizon_hours)
    with tab4:
        _render_hourly_heatmap(hourly_df, "confidence_score", tab_conf, horizon_hours=horizon_hours)

    st.markdown("---")
    st.subheader(_t("未来N天日线预测", "Next N-day Daily Forecast"))
    oneway_state = float(st.session_state.get("session_cost_bps_oneway", max(0.0, default_cost_bps / 2.0)))
    default_side = _t("双边(开+平)", "Round-trip (open+close)")
    cost_side_state = str(st.session_state.get("session_cost_side", default_side))
    is_round_trip = cost_side_state.startswith("双边") or cost_side_state.startswith("Round-trip")
    cost_bps_state = oneway_state * (2.0 if is_round_trip else 1.0)
    if daily_df.empty:
        st.info(_t("暂无日线预测数据。", "No daily forecast data."))
    else:
        d = _append_signal_strength_columns(daily_df.copy(), weak_pp=weak_pp, strong_pp=strong_pp)
        d = _append_edge_columns(d, cost_bps=cost_bps_state)
        col_trend = _t("趋势", "Trend")
        col_risk = _t("风险", "Risk")
        col_strength = _t("信号强弱", "Signal Strength")
        col_strength_score = _t("强度分(0-100)", "Strength(0-100)")
        col_edge = _t("机会值(edge)", "Edge")
        col_edge_risk = _t("风险调整机会", "Edge/Risk")
        col_vol = _t("波动强度", "Volatility")
        col_action = _t("策略动作", "Policy Action")
        col_pos = _t("建议仓位", "Position Size")
        col_net_edge = _t("预期净优势", "Expected Net Edge")
        d[col_trend] = d["trend_label"].map(_trend_text)
        d[col_risk] = d["risk_level"].map(_risk_text)
        d[col_strength] = d["signal_strength_label"].map(_signal_strength_text)
        d[col_strength_score] = d["signal_strength_score"]
        d[col_edge] = d["edge_score"]
        d[col_edge_risk] = d["edge_risk"]
        d[col_vol] = pd.to_numeric(d.get("volatility_score"), errors="coerce")
        if "policy_action" in d.columns:
            d[col_action] = d["policy_action"].map(_policy_action_text)
            d[col_pos] = d["policy_position_size"]
            d[col_net_edge] = pd.to_numeric(d["policy_expected_edge_pct"], errors="coerce")

        show_cols = [
            "date_bj",
            "day_of_week",
            "p_up",
            "p_down",
            "q50_change_pct",
            "target_price_q10",
            "target_price_q50",
            "target_price_q90",
            "start_window_top1",
            col_vol,
            col_strength,
            col_strength_score,
            col_edge,
            col_edge_risk,
            col_action,
            col_pos,
            col_net_edge,
            col_trend,
            col_risk,
            "confidence_score",
        ]
        show_cols = [c for c in show_cols if c in d.columns]
        d_view = d[show_cols].rename(
            columns={
                "p_up": _t("上涨概率", "P(up)"),
                "p_down": _t("下跌概率", "P(down)"),
                "q50_change_pct": _t("预期涨跌幅(q50)", "Expected Change (q50)"),
                "target_price_q10": _t("目标价格(q10)", "Target Price (q10)"),
                "target_price_q50": _t("目标价格(q50)", "Target Price (q50)"),
                "target_price_q90": _t("目标价格(q90)", "Target Price (q90)"),
                "start_window_top1": "start_window_top1",
                "confidence_score": _t("置信度", "Confidence"),
                col_pos: col_pos,
                col_net_edge: col_net_edge,
            }
        )
        st.caption(_t("关键列已置前（date_bj/day_of_week）并将趋势/风险/置信度放在表尾，便于快速扫读。", "Key columns moved forward (date_bj/day_of_week); trend/risk/confidence placed at end for quick scanning."))
        format_map_all = {
            _t("上涨概率", "P(up)"): "{:.2%}",
            _t("下跌概率", "P(down)"): "{:.2%}",
            _t("预期涨跌幅(q50)", "Expected Change (q50)"): "{:+.2%}",
            _t("目标价格(q10)", "Target Price (q10)"): "${:,.2f}",
            _t("目标价格(q50)", "Target Price (q50)"): "${:,.2f}",
            _t("目标价格(q90)", "Target Price (q90)"): "${:,.2f}",
            col_vol: "{:.2%}",
            col_strength_score: "{:.0f}",
            col_edge: "{:+.2%}",
            col_edge_risk: "{:+.3f}",
            col_pos: "{:.1%}",
            col_net_edge: "{:+.2%}",
            _t("置信度", "Confidence"): "{:.1f}",
        }
        format_map = {k: v for k, v in format_map_all.items() if k in d_view.columns}
        styled = d_view.style.format(format_map, na_rep="-")
        signed_cols = [c for c in [_t("预期涨跌幅(q50)", "Expected Change (q50)"), col_edge, col_edge_risk] if c in d_view.columns]
        if signed_cols:
            styled = styled.applymap(_style_signed_value, subset=signed_cols)
        if col_strength in d_view.columns:
            styled = styled.applymap(_style_strength_label, subset=[col_strength])
        if col_vol in d_view.columns:
            # pandas Styler.background_gradient depends on matplotlib.
            # Degrade gracefully when matplotlib is not installed.
            try:
                import matplotlib  # noqa: F401

                styled = styled.background_gradient(cmap="YlOrRd", subset=[col_vol])
            except Exception:
                pass
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")
    t1, t2, t3 = st.columns([1, 1, 1.2])
    top_n = int(t1.slider(_t("榜单显示 Top N", "Top N rows"), 3, 12, 5, 1, key="session_topn"))
    rank_key = t2.selectbox(
        _t("榜单排序标准", "Ranking Metric"),
        options=list(_rank_metric_options().keys()),
        index=list(_rank_metric_options().keys()).index("edge_score"),
        format_func=lambda k: _rank_metric_options().get(k, k),
        key="session_rank_key",
    )
    cost_mode_labels = {
        "双边(开+平)": _t("双边(开+平)", "Round-trip (open+close)"),
        "单边": _t("单边", "One-way"),
    }
    cost_mode = t3.selectbox(
        _t("成本口径", "Cost Type"),
        options=["双边(开+平)", "单边"],
        index=0,
        key="session_cost_side",
        format_func=lambda x: cost_mode_labels.get(str(x), str(x)),
        help=_t("默认使用双边成本，更保守。", "Round-trip cost is default (more conservative)."),
    )
    c31, c32 = st.columns([1, 1])
    one_way_cost_bps = float(
        c31.number_input(
            _t("单边成本（bps）", "One-way Cost (bps)"),
            min_value=0.0,
            max_value=100.0,
            value=max(0.0, cost_bps_state / 2.0),
            step=1.0,
            key="session_cost_bps_oneway",
        )
    )
    is_round_trip_mode = str(cost_mode).startswith("双边") or str(cost_mode).startswith("Round-trip")
    cost_bps = one_way_cost_bps * (2.0 if is_round_trip_mode else 1.0)
    c32.metric(_t("当前用于计算的成本", "Cost used in calculation"), f"{cost_bps:.1f} bps")
    with st.expander(_t("ⓘ edge 公式说明", "ⓘ Edge Formula"), expanded=False):
        st.markdown(
            _t(
                "- `edge_score = q50_change_pct - cost_bps/10000`\n"
                "- `edge_risk = edge_score / (q90_change_pct - q10_change_pct)`\n"
                "- 当前 `cost_bps` 口径："
                + ("双边（开仓+平仓）" if is_round_trip_mode else "单边")
                + "。",
                "- `edge_score = q50_change_pct - cost_bps/10000`\n"
                "- `edge_risk = edge_score / (q90_change_pct - q10_change_pct)`\n"
                "- Current `cost_bps` mode: "
                + ("Round-trip (open+close)" if is_round_trip_mode else "One-way")
                + ".",
            )
        )

    _render_top_tables(
        hourly_df=hourly_df,
        daily_df=daily_df,
        top_n=top_n,
        rank_key=rank_key,
        cost_bps=cost_bps,
        weak_pp=weak_pp,
        strong_pp=strong_pp,
        horizon_hours=horizon_hours,
    )

    with st.expander(_t("近期模型可信度（滚动30天）", "Recent Model Reliability (rolling 30d)"), expanded=False):
        summary_30d, baseline_df, err_msg = _compute_recent_hourly_reliability(
            symbol=str(symbol),
            horizon_hours=int(horizon_hours),
            window_days=30,
            cfg=cfg,
        )
        if err_msg:
            st.info(err_msg)
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Direction Accuracy", f"{summary_30d.get('accuracy', float('nan')):.2%}")
            m2.metric("ROC-AUC", f"{summary_30d.get('auc', float('nan')):.3f}")
            m3.metric("Brier", f"{summary_30d.get('brier', float('nan')):.4f}")
            m4, m5, m6 = st.columns(3)
            m4.metric(_t("Coverage(80%目标)", "Coverage (target 80%)"), f"{summary_30d.get('coverage', float('nan')):.2%}")
            m5.metric(_t("平均区间宽度(q90-q10)", "Average interval width (q90-q10)"), f"{summary_30d.get('width', float('nan')):.2%}")
            m6.metric(_t("样本数", "Samples"), f"{int(summary_30d.get('samples', 0))}")
            st.caption(_t("说明：Coverage 目标通常接近 80%；Brier 越低越好。", "Coverage is usually expected near 80%; lower Brier is better."))
            if not baseline_df.empty:
                st.markdown(f"**{_t('Baseline 对比（含 Naive / Prev-bar / 已有模型汇总）', 'Baseline Comparison (Naive / Prev-bar / model summary)')}**")
                st.dataframe(
                    baseline_df.style.format(
                        {
                            "Accuracy": "{:.2%}",
                            "AUC": "{:.3f}",
                            "Brier": "{:.4f}",
                        },
                        na_rep="-",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    with st.expander(_t("如何解读这个页面？", "How to read this page?"), expanded=False):
        st.markdown(
            _t(
                f"- 小时级语义：`从该小时开始的未来{int(horizon_hours)}h`，不是“该小时内必涨/必跌”。\n"
                "- 看 `P(up)/P(down)` 判断方向概率。\n"
                "- `信号强弱 = |P(up)-0.5|`：弱信号接近 coin flip，不要过度解读。\n"
                "- 看 `预期涨跌幅(q50)` 判断幅度。\n"
                "- 看 `波动强度` 和 `风险等级` 判断风险。\n"
                "- 看 `edge_score / edge_risk` 判断机会值与风险调整后性价比。\n"
                "- 看 `策略动作/建议仓位/预期净优势` 判断是否值得参与。\n"
                "- `Forecast` 与 `Seasonality` 是两种口径，分歧本身也是信息。",
                f"- Hourly semantics: future `{int(horizon_hours)}h` return from each hour start, not guaranteed up/down within that hour.\n"
                "- Use `P(up)/P(down)` for direction probability.\n"
                "- `Signal strength = |P(up)-0.5|`: weak signals are close to coin flip.\n"
                "- Use `q50 change` for expected magnitude.\n"
                "- Use `volatility` and `risk level` for risk.\n"
                "- Use `edge_score / edge_risk` for risk-adjusted opportunity.\n"
                "- Use `policy action / position size / expected net edge` for participation decision.\n"
                "- `Forecast` and `Seasonality` are two views; divergence itself is informative.",
            )
        )


def _render_projection_chart(
    *,
    current_price: float,
    q10_change_pct: float,
    q50_change_pct: float,
    q90_change_pct: float,
    expected_date_label: str,
    title: str,
    entry_price: float | None = None,
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
) -> None:
    if not pd.notna(current_price):
        return
    if not (pd.notna(q10_change_pct) and pd.notna(q50_change_pct) and pd.notna(q90_change_pct)):
        return

    q10_price = float(current_price * (1.0 + q10_change_pct))
    q50_price = float(current_price * (1.0 + q50_change_pct))
    q90_price = float(current_price * (1.0 + q90_change_pct))
    x = ["现在", expected_date_label if expected_date_label else "预期日期"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[current_price, q90_price],
            mode="lines+markers",
            name="q90",
            line=dict(width=1),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[current_price, q10_price],
            mode="lines+markers",
            name="q10",
            line=dict(width=1),
            marker=dict(size=6),
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.20)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[current_price, q50_price],
            mode="lines+markers",
            name="q50",
            line=dict(width=3),
            marker=dict(size=7),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="价格",
        template="plotly_white",
        height=320,
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    entry_v = _safe_float(entry_price if entry_price is not None else current_price)
    sl_v = _safe_float(stop_loss_price)
    tp_v = _safe_float(take_profit_price)
    if np.isfinite(entry_v):
        fig.add_hline(
            y=entry_v,
            line_dash="dot",
            line_color="#94a3b8",
            annotation_text="P0/Entry",
            annotation_position="top left",
        )
    if np.isfinite(sl_v):
        fig.add_hline(
            y=sl_v,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="SL",
            annotation_position="top left",
        )
    if np.isfinite(tp_v):
        fig.add_hline(
            y=tp_v,
            line_dash="dash",
            line_color="#22c55e",
            annotation_text="TP",
            annotation_position="top left",
        )
    st.plotly_chart(fig, use_container_width=True)


def _expected_date(latest_market: str, latest_utc: str, branch_name: str, horizon: int) -> str:
    ts_market = pd.to_datetime(latest_market, errors="coerce")
    if pd.notna(ts_market):
        if branch_name == "hourly":
            exp_ts = ts_market + pd.Timedelta(hours=int(horizon))
        else:
            exp_ts = ts_market + pd.Timedelta(days=int(horizon))
        return exp_ts.strftime("%Y-%m-%d %H:%M:%S %z")

    ts = pd.to_datetime(latest_utc, utc=True, errors="coerce")
    if pd.isna(ts):
        return "-"
    if branch_name == "hourly":
        exp_ts = ts + pd.Timedelta(hours=int(horizon))
    else:
        exp_ts = ts + pd.Timedelta(days=int(horizon))
    return exp_ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def _render_factor_explain() -> None:
    with st.expander(_t("指标解释（给非量化用户）", "Metric Explanation (for non-quants)"), expanded=False):
        st.markdown(
            _t(
                "- `市值因子`：规模相关，通常越大越稳。\n"
                "- `价值因子`：估值便宜程度（越高通常越便宜）。\n"
                "- `成长因子`：增长能力（盈利/营收或价格增长代理）。\n"
                "- `动能因子`：近期趋势强弱。\n"
                "- `反转因子`：短期是否有回撤后反弹特征。\n"
                "- `低波动因子`：波动越低，数值通常越好。",
                "- `Size factor`: scale related, usually larger is more stable.\n"
                "- `Value factor`: valuation cheapness (higher is usually cheaper).\n"
                "- `Growth factor`: growth ability (earnings/revenue or price-growth proxy).\n"
                "- `Momentum factor`: recent trend strength.\n"
                "- `Reversal factor`: short-term rebound-after-pullback behavior.\n"
                "- `Low-vol factor`: lower volatility usually scores better.",
            )
        )


def _factor_label_text(label_cn: str) -> str:
    if _ui_lang() == "zh":
        return label_cn
    mapping = {
        "市值因子": "Size",
        "价值因子": "Value",
        "成长因子": "Growth",
        "动能因子": "Momentum",
        "反转因子": "Reversal",
        "低波动因子": "Low Volatility",
    }
    return mapping.get(label_cn, label_cn)


def _render_factor_top_contributions(row: pd.Series) -> None:
    factor_map = {
        "市值因子": _safe_float(row.get("size_factor")),
        "价值因子": _safe_float(row.get("value_factor")),
        "成长因子": _safe_float(row.get("growth_factor")),
        "动能因子": _safe_float(row.get("momentum_factor")),
        "反转因子": _safe_float(row.get("reversal_factor")),
        "低波动因子": _safe_float(row.get("low_vol_factor")),
    }
    factor_col = _t("因子", "Factor")
    contrib_col = _t("贡献", "Contribution")
    rows = [{factor_col: _factor_label_text(k), contrib_col: v} for k, v in factor_map.items() if np.isfinite(v)]
    if not rows:
        return
    df = pd.DataFrame(rows)
    pos = df[df[contrib_col] > 0].sort_values(contrib_col, ascending=False).head(3).copy()
    neg = df[df[contrib_col] < 0].sort_values(contrib_col, ascending=True).head(3).copy()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{_t('Top 正贡献因子', 'Top Positive Contributors')}**")
        if pos.empty:
            st.caption(_t("暂无显著正贡献因子。", "No significant positive contributors."))
        else:
            pos[contrib_col] = pos[contrib_col].map(_format_change_pct)
            st.dataframe(pos, use_container_width=True, hide_index=True)
    with c2:
        st.markdown(f"**{_t('Top 负贡献因子', 'Top Negative Contributors')}**")
        if neg.empty:
            st.caption(_t("暂无显著负贡献因子。", "No significant negative contributors."))
        else:
            neg[contrib_col] = neg[contrib_col].map(_format_change_pct)
            st.dataframe(neg, use_container_width=True, hide_index=True)
    st.caption(
        _t(
            "解释示例：成长/动能为正通常支持趋势延续；反转为负表示短线反弹支持偏弱。",
            "Example: positive growth/momentum tends to support trend continuation; negative reversal implies weak short-term rebound support.",
        )
    )


def _render_core_field_explain() -> None:
    with st.expander(_t("这4个核心字段是什么意思？", "What do these 4 core fields mean?"), expanded=False):
        st.markdown(
            _t(
                "- `当前价格`：当前可交易市场的最新成交价。\n"
                "- `预测价格`：模型给出的目标价格（默认看中位数 q50）。\n"
                "- `预计涨跌幅`：从当前价格到预测价格的变化比例。\n"
                "- `预期日期`：这次预测对应的目标时间点（完整时间，含时区）。",
                "- `Current Price`: latest tradable market price.\n"
                "- `Predicted Price`: model target price (q50 by default).\n"
                "- `Expected Change`: percentage change from current to predicted price.\n"
                "- `Expected Date`: target timestamp for this prediction (full datetime with timezone).",
            )
        )


def _task_label(task: str) -> str:
    mapping = {
        "direction": "方向预测（涨/跌）",
        "magnitude": "幅度预测（涨跌幅）",
        "magnitude_quantile": "区间预测（q10/q50/q90）",
    }
    return mapping.get(str(task), str(task))


def _branch_label(branch: str) -> str:
    mapping = {"hourly": "小时", "daily": "日线"}
    return mapping.get(str(branch), str(branch))


def _model_label(model: str) -> str:
    text = str(model or "")
    mapping = {
        "baseline_prev_bar": "基线模型（延续上一根K线方向）",
        "baseline_logistic": "基线逻辑回归",
        "mvp_lightgbm": "LightGBM（增强版）",
        "mvp_lightgbm_quantile_q50": "LightGBM 中位数预测（q50）",
        "mvp_lightgbm_quantile": "LightGBM 分位数预测（q10/q50/q90）",
    }
    if text in mapping:
        return mapping[text]
    if text.startswith("baseline_"):
        return f"基线模型（{text}）"
    return text


def _metric_meta(metric: str) -> tuple[str, str]:
    key = str(metric or "")
    mapping = {
        "accuracy": ("准确率", "越高越好（>50% 说明优于随机猜测）"),
        "f1": ("F1综合分", "越高越好（综合看精确率和召回率）"),
        "precision": ("精确率", "越高越好（模型说会涨时，真正上涨的比例）"),
        "recall": ("召回率", "越高越好（实际上涨里被识别出来的比例）"),
        "roc_auc": ("AUC", "越高越好（0.5 约等于随机）"),
        "mae": ("平均绝对误差", "越低越好（越接近真实值）"),
        "rmse": ("均方根误差", "越低越好（对大误差更敏感）"),
        "sign_accuracy": ("方向正确率", "越高越好（只看方向是否预测对）"),
    }
    if key in mapping:
        return mapping[key]
    if key.startswith("pinball_"):
        return ("分位数误差", "越低越好（区间预测误差）")
    return (key, "结合策略目标解读")


def _format_metric_value(metric: str, value: object) -> str:
    if not _is_finite_number(value):
        return "-"
    v = float(value)
    ratio_metrics = {"accuracy", "f1", "precision", "recall", "roc_auc", "sign_accuracy"}
    if str(metric) in ratio_metrics:
        return f"{v:.2%}"
    return f"{v:.4f}"


def _render_model_metrics_readable(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        st.info(_t("暂无可展示的模型评估结果。", "No model evaluation results to display."))
        return

    required_cols = {"branch", "task", "model", "horizon", "metric", "mean", "std"}
    missing = required_cols - set(metrics.columns)
    if missing:
        st.dataframe(metrics, use_container_width=True)
        return

    st.markdown(f"**{_t('Model Metrics（通俗版）', 'Model Metrics (Plain-language)')}**")
    with st.expander(_t("怎么看这些指标？", "How to read these metrics?"), expanded=False):
        st.markdown(
            _t(
                "- `准确率/F1/AUC`：越高越好。\n"
                "- `MAE/RMSE/分位数误差`：越低越好。\n"
                "- `std(波动)`：越低代表越稳定。\n"
                "- 如果方向指标一般但幅度误差小，适合做区间预期，不适合单独做买卖信号。",
                "- `Accuracy/F1/AUC`: higher is better.\n"
                "- `MAE/RMSE/Pinball`: lower is better.\n"
                "- `std` (stability): lower is more stable.\n"
                "- If direction metrics are average but magnitude error is low, use it for range expectation rather than standalone trade signal.",
            )
        )

    work = metrics.copy()
    work["branch"] = work["branch"].astype(str)
    work["task"] = work["task"].astype(str)
    work["_horizon_norm"] = (
        pd.to_numeric(work["horizon"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    f1, f2, f3 = st.columns(3)
    all_opt = _t("全部", "All")
    branch_options = [all_opt] + sorted(work["branch"].dropna().unique().tolist())
    task_options = [all_opt] + sorted(work["task"].dropna().unique().tolist())
    horizon_values = sorted(work["_horizon_norm"].dropna().unique().tolist())
    horizon_options = [all_opt] + [str(h) for h in horizon_values]

    selected_branch = f1.selectbox(_t("分支", "Branch"), branch_options, index=0, key="metrics_branch_filter")
    selected_task = f2.selectbox(_t("任务", "Task"), task_options, index=0, key="metrics_task_filter")
    selected_horizon = f3.selectbox(_t("周期", "Horizon"), horizon_options, index=0, key="metrics_horizon_filter")

    view = work.copy()
    if selected_branch != all_opt:
        view = view[view["branch"] == selected_branch]
    if selected_task != all_opt:
        view = view[view["task"] == selected_task]
    if selected_horizon != all_opt:
        view = view[view["_horizon_norm"] == selected_horizon]

    if view.empty:
        st.info(_t("当前筛选条件下没有指标记录。", "No metric rows under current filters."))
    else:
        out = pd.DataFrame(
            {
                "分支": view["branch"].map(_branch_label),
                "任务": view["task"].map(_task_label),
                "模型": view["model"].map(_model_label),
                "预测周期": view["_horizon_norm"],
                "指标": view["metric"].map(lambda x: _metric_meta(str(x))[0]),
                "平均表现": [
                    _format_metric_value(str(m), v) for m, v in zip(view["metric"], view["mean"])
                ],
                "稳定性(std)": view["std"].map(lambda x: _format_metric_value("std", x)),
                "解读": view["metric"].map(lambda x: _metric_meta(str(x))[1]),
            }
        )
        st.dataframe(out, use_container_width=True, hide_index=True)

    with st.expander(_t("查看原始 Model Metrics 表", "View raw Model Metrics table"), expanded=False):
        st.dataframe(metrics, use_container_width=True)


def _render_trade_signal_block(signal_row: pd.Series, *, header: str = "开单信号与理由") -> None:
    if signal_row is None or len(signal_row) == 0:
        return
    st.markdown(f"**{header}**")
    action_raw = str(signal_row.get("trade_signal", signal_row.get("policy_action", "Flat")))
    action_cn = _policy_action_text(action_raw)
    trend = str(signal_row.get("trade_trend_context", "mixed"))
    trend_cn = _trend_text({"bullish": "趋势偏多", "bearish": "趋势偏空", "mixed": "趋势混合"}.get(trend, trend))
    stop_price = _safe_float(signal_row.get("trade_stop_loss_price"))
    take_price = _safe_float(signal_row.get("trade_take_profit_price"))
    rr_ratio = _safe_float(signal_row.get("trade_rr_ratio"))
    support_score = _safe_float(signal_row.get("trade_support_score"))
    stop_text = _format_price(stop_price)
    take_text = _format_price(take_price)
    if action_raw == "Flat":
        stop_text = _t("不适用（观望）", "N/A (Wait)")
        take_text = _t("不适用（观望）", "N/A (Wait)")

    c1, c2, c3, c4, c5 = st.columns([1.6, 1.3, 1.2, 1.2, 1.0])
    with c1:
        _render_big_value(_t("当前信号", "Current Signal"), action_cn, caption=_t("观望 = 暂不下单", "Wait = no trade now"))
    with c2:
        _render_big_value(_t("趋势判断", "Trend"), trend_cn)
    with c3:
        _render_big_value(_t("止损价", "Stop Loss"), stop_text)
    with c4:
        _render_big_value(_t("止盈价", "Take Profit"), take_text)
    with c5:
        _render_big_value(_t("盈亏比(RR)", "Risk/Reward (RR)"), _format_float(rr_ratio, 2))
    if _is_finite_number(support_score):
        st.caption(
            _t(
                f"技术共振分数: {_format_float(support_score, 0)}（>0 偏多，<0 偏空）",
                f"Technical confluence score: {_format_float(support_score, 0)} (>0 bullish, <0 bearish)",
            )
        )
    reason_text = _format_reason_tokens_cn(signal_row.get("trade_reason_tokens", signal_row.get("policy_reason", "-")))
    st.write(_t(f"开单理由: {reason_text}", f"Reason: {reason_text}"))
    st.caption(
        _t(
            "说明: 该理由由 EMA/MACD/SuperTrend/成交量/BOS/CHOCH 自动生成，用于解释信号，不构成投资建议。",
            "Note: reason is auto-generated from EMA/MACD/SuperTrend/Volume/BOS/CHOCH for interpretability only, not investment advice.",
        )
    )


def _ensure_policy_for_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_snapshot_factors(df)
    if df.empty:
        return df
    work = df.copy()
    try:
        cfg = _load_main_config_cached()
        if "market_type" not in work.columns:
            work["market_type"] = np.where(work["market"].astype(str).eq("crypto"), "spot", "cash")
        if "p_up" not in work.columns:
            work["p_up"] = pd.to_numeric(work.get("policy_p_up_used"), errors="coerce")
        else:
            work["p_up"] = pd.to_numeric(work["p_up"], errors="coerce")
            fallback_p = pd.to_numeric(work.get("policy_p_up_used"), errors="coerce")
            work["p_up"] = work["p_up"].where(work["p_up"].notna(), fallback_p)
        if "volatility_score" not in work.columns and {"q90_change_pct", "q10_change_pct"}.issubset(work.columns):
            work["volatility_score"] = (
                pd.to_numeric(work["q90_change_pct"], errors="coerce")
                - pd.to_numeric(work["q10_change_pct"], errors="coerce")
            )
        if "confidence_score" not in work.columns:
            conf = (2.0 * (pd.to_numeric(work["p_up"], errors="coerce") - 0.5).abs()).clip(0.0, 1.0) * 100.0
            work["confidence_score"] = conf.fillna(50.0)
        if "risk_level" not in work.columns:
            width = pd.to_numeric(work.get("volatility_score"), errors="coerce").abs()
            work["risk_level"] = np.where(
                width < 0.02,
                "low",
                np.where(width < 0.05, "medium", np.where(width < 0.10, "high", "extreme")),
            )
        work = apply_policy_frame(
            work,
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
    except Exception:
        pass
    return work


def _find_policy_backtest_row(
    *,
    processed_dir: Path,
    market: str,
    symbol: str,
    aliases: List[str] | None = None,
) -> pd.Series | None:
    artifacts = _load_backtest_artifacts(str(processed_dir))
    summary = artifacts.get("metrics_summary", pd.DataFrame())
    if summary.empty:
        return None
    alias_tokens = {_normalize_symbol_token(symbol)}
    for a in aliases or []:
        alias_tokens.add(_normalize_symbol_token(a))
    sub = summary[(summary["market"].astype(str) == str(market))].copy()
    if sub.empty or "symbol" not in sub.columns:
        return None
    sub = sub[sub["symbol"].map(_normalize_symbol_token).isin(alias_tokens)]
    if sub.empty:
        return None
    policy = sub[sub["strategy"].astype(str) == "policy"].head(1)
    if policy.empty:
        return None
    return policy.iloc[0]


def _reliability_level_text(summary: Dict[str, float] | None) -> str:
    if not summary:
        return _t("暂无", "N/A")
    brier = float(summary.get("brier", float("nan")))
    coverage = float(summary.get("coverage", float("nan")))
    if not np.isfinite(brier) or not np.isfinite(coverage):
        return _t("暂无", "N/A")
    if brier <= 0.18 and 0.72 <= coverage <= 0.88:
        return _t(
            f"高（Brier {brier:.3f}，Coverage {coverage:.1%}）",
            f"High (Brier {brier:.3f}, Coverage {coverage:.1%})",
        )
    if brier <= 0.24 and 0.65 <= coverage <= 0.92:
        return _t(
            f"中（Brier {brier:.3f}，Coverage {coverage:.1%}）",
            f"Medium (Brier {brier:.3f}, Coverage {coverage:.1%})",
        )
    return _t(
        f"低（Brier {brier:.3f}，Coverage {coverage:.1%}）",
        f"Low (Brier {brier:.3f}, Coverage {coverage:.1%})",
    )


def _entry_touch_state_path(processed_dir: Path | None = None) -> Path:
    root = processed_dir or Path("data/processed")
    out = root / "execution"
    out.mkdir(parents=True, exist_ok=True)
    return out / "entry_touch_state.json"


def _load_entry_touch_state(processed_dir: Path | None = None) -> Dict[str, Dict[str, object]]:
    path = _entry_touch_state_path(processed_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def _save_entry_touch_state(state: Dict[str, Dict[str, object]], processed_dir: Path | None = None) -> None:
    path = _entry_touch_state_path(processed_dir)
    if len(state) > 2000:
        items = sorted(
            state.items(),
            key=lambda kv: str(kv[1].get("updated_at_utc", "")),
            reverse=True,
        )[:2000]
        state = dict(items)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_entry_band_pct(market: str) -> float:
    mk = str(market or "").strip().lower()
    if mk == "crypto":
        return 0.0015
    if mk == "cn_equity":
        return 0.0010
    if mk == "us_equity":
        return 0.0008
    return 0.0010


def _plan_level_side(
    *,
    action: str,
    q50: float,
    p_up: float,
    edge_risk_long: float,
    edge_risk_short: float,
) -> str:
    if action in {"LONG", "SHORT"}:
        return action
    if np.isfinite(edge_risk_long) and np.isfinite(edge_risk_short):
        return "LONG" if edge_risk_long >= edge_risk_short else "SHORT"
    if np.isfinite(q50):
        return "LONG" if q50 >= 0 else "SHORT"
    if np.isfinite(p_up):
        return "LONG" if p_up >= 0.5 else "SHORT"
    return "LONG"


def _compute_plan_levels(
    *,
    side: str,
    entry: float,
    q10: float,
    q50: float,
    q90: float,
    atr_proxy_pct: float,
    atr_mult: float,
    tp_mode: str,
    sl_mode: str,
) -> Tuple[float, float, float, float]:
    if not np.isfinite(entry):
        return float("nan"), float("nan"), float("nan"), float("nan")
    sl = float("nan")
    tp = float("nan")
    tp2 = float("nan")
    rr = float("nan")
    if side == "LONG":
        atr_stop_ret = -atr_mult * atr_proxy_pct
        q10_ret = q10 if np.isfinite(q10) else float("nan")
        if sl_mode == "max_q10_atr":
            sl_ret = max(q10_ret if np.isfinite(q10_ret) else -999.0, atr_stop_ret)
        elif sl_mode == "q10_pref":
            sl_ret = q10_ret if np.isfinite(q10_ret) else atr_stop_ret
        else:
            sl_ret = atr_stop_ret if np.isfinite(atr_stop_ret) else q10_ret
        sl = entry * (1.0 + sl_ret)
        if tp_mode == "q90":
            tp_ret = q90 if np.isfinite(q90) else q50
        elif tp_mode == "mid":
            if np.isfinite(q50) and np.isfinite(q90):
                tp_ret = 0.5 * (q50 + q90)
            else:
                tp_ret = q50 if np.isfinite(q50) else q90
        else:
            tp_ret = q50 if np.isfinite(q50) else q90
        tp = entry * (1.0 + tp_ret) if np.isfinite(tp_ret) else float("nan")
        tp2 = entry * (1.0 + q90) if np.isfinite(q90) else float("nan")
        risk = entry - sl
        reward = tp - entry
        rr = _safe_float(reward / risk) if np.isfinite(risk) and risk > 1e-12 else float("nan")
    else:
        atr_stop_ret = atr_mult * atr_proxy_pct
        q90_ret = q90 if np.isfinite(q90) else float("nan")
        if sl_mode == "max_q10_atr":
            sl_ret = min(q90_ret if np.isfinite(q90_ret) else 999.0, atr_stop_ret)
        elif sl_mode == "q10_pref":
            sl_ret = q90_ret if np.isfinite(q90_ret) else atr_stop_ret
        else:
            sl_ret = atr_stop_ret if np.isfinite(atr_stop_ret) else q90_ret
        sl = entry * (1.0 + sl_ret)
        if tp_mode == "q90":
            tp_ret = q10 if np.isfinite(q10) else q50
        elif tp_mode == "mid":
            if np.isfinite(q10) and np.isfinite(q50):
                tp_ret = 0.5 * (q10 + q50)
            else:
                tp_ret = q50 if np.isfinite(q50) else q10
        else:
            tp_ret = q50 if np.isfinite(q50) else q10
        tp = entry * (1.0 + tp_ret) if np.isfinite(tp_ret) else float("nan")
        tp2 = entry * (1.0 + q10) if np.isfinite(q10) else float("nan")
        risk = sl - entry
        reward = entry - tp
        rr = _safe_float(reward / risk) if np.isfinite(risk) and risk > 1e-12 else float("nan")
    return sl, tp, tp2, rr


def _build_trade_decision_plan(
    row: pd.Series,
    *,
    cfg: Dict[str, object] | None = None,
    risk_profile: str = "标准",
    model_health: str = "中",
    event_risk: bool = False,
    persist_touch: bool = True,
    processed_dir: Path | None = None,
) -> Dict[str, object]:
    cfg_local = cfg or _load_main_config_cached("configs/config.yaml")
    policy_cfg = (cfg_local.get("policy", {}) if isinstance(cfg_local, dict) else {})
    th_cfg = policy_cfg.get("thresholds", {})
    ex_cfg = policy_cfg.get("execution", {})
    p_bull = float(th_cfg.get("p_bull", 0.55))
    p_bear = float(th_cfg.get("p_bear", 0.45))
    conf_min = float((cfg_local.get("decision", {}) if isinstance(cfg_local, dict) else {}).get("confidence_min", 60.0))
    fee_bps = float(ex_cfg.get("fee_bps", 10.0))
    slippage_bps = float(ex_cfg.get("slippage_bps", 10.0))
    cost_bps = fee_bps + slippage_bps  # 双边成本：开+平
    cost_pct = cost_bps / 10000.0

    current_price = _safe_float(row.get("current_price"))
    p_up = _safe_float(row.get("p_up", row.get("policy_p_up_used")))
    q10 = _safe_float(row.get("q10_change_pct"))
    q50 = _safe_float(row.get("q50_change_pct", row.get("predicted_change_pct")))
    q90 = _safe_float(row.get("q90_change_pct"))
    conf = _safe_float(row.get("confidence_score"))
    risk_level = str(row.get("risk_level", "medium"))
    allow_short = bool(row.get("policy_allow_short", True))
    horizon_label = str(row.get("horizon_label", "4h"))
    market = str(row.get("market", "")).strip().lower()
    symbol = str(row.get("symbol", row.get("snapshot_symbol", ""))).strip()
    price_source = str(row.get("price_source", "-"))
    price_timestamp_market = str(
        row.get(
            "price_timestamp_market",
            row.get("timestamp_market", row.get("latest_market", row.get("generated_at_market", "-"))),
        )
    )
    price_timestamp_utc = str(
        row.get(
            "price_timestamp_utc",
            row.get("timestamp_utc", row.get("latest_utc", row.get("generated_at_utc", "-"))),
        )
    )
    signal_time_utc = str(row.get("generated_at_utc", row.get("timestamp_utc", price_timestamp_utc)))
    valid_until = str(row.get("expected_date_market", row.get("expected_date_utc", "-")))
    entry_candidates = [
        _safe_float(row.get("signal_entry_price")),
        _safe_float(row.get("entry_price_snapshot")),
        _safe_float(row.get("plan_entry")),
        _safe_float(row.get("entry")),
        current_price,
    ]
    entry_seed = next((x for x in entry_candidates if np.isfinite(x)), float("nan"))
    entry_band_cfg = ((cfg_local.get("decision", {}) if isinstance(cfg_local, dict) else {}).get("entry_band_pct", {}))
    entry_band_pct = float("nan")
    if isinstance(entry_band_cfg, dict):
        entry_band_pct = _safe_float(entry_band_cfg.get(market, entry_band_cfg.get("default")))
    elif _is_finite_number(entry_band_cfg):
        entry_band_pct = float(entry_band_cfg)
    if not np.isfinite(entry_band_pct) or entry_band_pct <= 0:
        entry_band_pct = _default_entry_band_pct(market)
    entry_band_pct = float(np.clip(entry_band_pct, 0.0002, 0.02))
    key_raw = "|".join(
        [
            market or "-",
            symbol or "-",
            horizon_label or "-",
            valid_until or "-",
            f"{_safe_float(q50):.6f}",
            f"{_safe_float(p_up):.6f}",
            str(risk_profile),
        ]
    )
    signal_key = hashlib.sha1(key_raw.encode("utf-8")).hexdigest()[:20]
    entry = entry_seed
    entry_touched_at = ""
    if persist_touch:
        state = _load_entry_touch_state(processed_dir)
        rec = state.get(signal_key, {})
        rec_entry = _safe_float(rec.get("entry_price"))
        if np.isfinite(rec_entry):
            entry = rec_entry
        entry_touched_at = str(rec.get("entry_touched_at", "")).strip()
    else:
        state = {}
    entry_gap_pct = (
        _safe_float((current_price - entry) / entry)
        if np.isfinite(current_price) and np.isfinite(entry) and abs(entry) > 1e-12
        else float("nan")
    )
    entry_touched = bool(np.isfinite(entry_gap_pct) and abs(entry_gap_pct) <= entry_band_pct)
    if entry_touched and not entry_touched_at:
        tz_name = str(row.get("timezone", "Asia/Shanghai")) or "Asia/Shanghai"
        try:
            entry_touched_at = pd.Timestamp.now(tz=tz_name).strftime("%Y-%m-%d %H:%M:%S %z")
        except Exception:
            entry_touched_at = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    if persist_touch:
        state[signal_key] = {
            "market": market,
            "symbol": symbol,
            "horizon_label": horizon_label,
            "entry_price": entry,
            "entry_band_pct": entry_band_pct,
            "entry_touched_at": entry_touched_at,
            "signal_time_utc": signal_time_utc,
            "valid_until": valid_until,
            "updated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        _save_entry_touch_state(state, processed_dir)
    width = _safe_float(q90 - q10)
    if not np.isfinite(width):
        width = _safe_float(row.get("volatility_score"))
    atr_proxy_pct = float(np.clip(max(0.003, (abs(width) / 2.0) if np.isfinite(width) else 0.003), 0.003, 0.03))

    edge_long = _safe_float(q50 - cost_pct)
    edge_short = _safe_float((-q50) - cost_pct)
    edge_risk_long = _safe_float(edge_long / width) if np.isfinite(width) and width > 1e-12 else float("nan")
    edge_risk_short = _safe_float(edge_short / width) if np.isfinite(width) and width > 1e-12 else float("nan")

    model_health_norm = str(model_health).strip().lower()
    model_health_ok = model_health_norm in {"良", "中", "good", "medium"}
    long_checks = [
        (_t("p_up >= 阈值", "p_up >= threshold"), np.isfinite(p_up) and p_up >= p_bull),
        (_t("edge_score > 0（覆盖成本）", "edge_score > 0 (covers cost)"), np.isfinite(edge_long) and edge_long > 0),
        (_t("confidence >= 最低阈值", "confidence >= minimum"), np.isfinite(conf) and conf >= conf_min),
        (_t("风险非极高", "risk is not extreme"), str(risk_level) != "extreme"),
        (_t("模型健康非差", "model health is not bad"), model_health_ok),
        (_t("无重大事件风险", "no major event risk"), not bool(event_risk)),
    ]
    short_checks = [
        (_t("p_up <= 阈值", "p_up <= threshold"), np.isfinite(p_up) and p_up <= p_bear),
        (_t("edge_score > 0（覆盖成本）", "edge_score > 0 (covers cost)"), np.isfinite(edge_short) and edge_short > 0),
        (_t("confidence >= 最低阈值", "confidence >= minimum"), np.isfinite(conf) and conf >= conf_min),
        (_t("允许做空", "shorting allowed"), allow_short),
        (_t("风险非极高", "risk is not extreme"), str(risk_level) != "extreme"),
        (_t("模型健康非差", "model health is not bad"), model_health_ok),
        (_t("无重大事件风险", "no major event risk"), not bool(event_risk)),
    ]
    long_ok = all(ok for _, ok in long_checks)
    short_ok = all(ok for _, ok in short_checks)

    action = "WAIT"
    if long_ok and not short_ok:
        action = "LONG"
    elif short_ok and not long_ok:
        action = "SHORT"
    elif long_ok and short_ok:
        action = "LONG" if edge_risk_long >= edge_risk_short else "SHORT"

    # entry is fixed at signal snapshot (or persisted state), not refreshed with current price.
    profile_key = {
        "conservative": "conservative",
        "standard": "standard",
        "aggressive": "aggressive",
        "保守": "conservative",
        "标准": "standard",
        "激进": "aggressive",
    }.get(str(risk_profile or "standard").strip().lower(), "standard")
    profile_display = {
        "conservative": _t("保守", "Conservative"),
        "standard": _t("标准", "Standard"),
        "aggressive": _t("激进", "Aggressive"),
    }.get(profile_key, _t("标准", "Standard"))

    action_reason = _t("规则未全部满足，建议观望。", "Entry rules are not fully met. Suggested action: Wait.")
    if profile_key == "conservative":
        atr_mult = 1.0
        tp_mode = "q50"
        sl_mode = "max_q10_atr"
    elif profile_key == "aggressive":
        atr_mult = 2.0
        tp_mode = "q90"
        sl_mode = "q10_pref"
    else:
        atr_mult = 1.5
        tp_mode = "mid"
        sl_mode = "atr_pref"
    plan_side = _plan_level_side(
        action=action,
        q50=q50,
        p_up=p_up,
        edge_risk_long=edge_risk_long,
        edge_risk_short=edge_risk_short,
    )
    sl, tp, tp2, rr = _compute_plan_levels(
        side=plan_side,
        entry=entry,
        q10=q10,
        q50=q50,
        q90=q90,
        atr_proxy_pct=atr_proxy_pct,
        atr_mult=atr_mult,
        tp_mode=tp_mode,
        sl_mode=sl_mode,
    )
    if action == "LONG" and np.isfinite(entry):
        atr_stop_ret = -atr_mult * atr_proxy_pct
        q10_ret = q10 if np.isfinite(q10) else float("nan")
        if sl_mode == "max_q10_atr":
            sl_ret = max(q10_ret if np.isfinite(q10_ret) else -999.0, atr_stop_ret)
        elif sl_mode == "q10_pref":
            sl_ret = q10_ret if np.isfinite(q10_ret) else atr_stop_ret
        else:
            sl_ret = atr_stop_ret if np.isfinite(atr_stop_ret) else q10_ret
        sl = entry * (1.0 + sl_ret)
        if tp_mode == "q90":
            tp_ret = q90 if np.isfinite(q90) else q50
        elif tp_mode == "mid":
            if np.isfinite(q50) and np.isfinite(q90):
                tp_ret = 0.5 * (q50 + q90)
            else:
                tp_ret = q50 if np.isfinite(q50) else q90
        else:
            tp_ret = q50 if np.isfinite(q50) else q90
        tp = entry * (1.0 + tp_ret) if np.isfinite(tp_ret) else float("nan")
        tp2 = entry * (1.0 + q90) if np.isfinite(q90) else float("nan")
        risk = entry - sl
        reward = tp - entry
        rr = _safe_float(reward / risk) if np.isfinite(risk) and risk > 1e-12 else float("nan")
        action_reason = _t(
            "做多条件满足：方向概率、edge、置信度与风险过滤均通过。",
            "Long conditions satisfied: direction probability, edge, confidence, and risk filters all pass.",
        )
    elif action == "SHORT" and np.isfinite(entry):
        atr_stop_ret = atr_mult * atr_proxy_pct
        q90_ret = q90 if np.isfinite(q90) else float("nan")
        if sl_mode == "max_q10_atr":
            sl_ret = min(q90_ret if np.isfinite(q90_ret) else 999.0, atr_stop_ret)
        elif sl_mode == "q10_pref":
            sl_ret = q90_ret if np.isfinite(q90_ret) else atr_stop_ret
        else:
            sl_ret = atr_stop_ret if np.isfinite(atr_stop_ret) else q90_ret
        sl = entry * (1.0 + sl_ret)
        if tp_mode == "q90":
            tp_ret = q10 if np.isfinite(q10) else q50
        elif tp_mode == "mid":
            if np.isfinite(q10) and np.isfinite(q50):
                tp_ret = 0.5 * (q10 + q50)
            else:
                tp_ret = q50 if np.isfinite(q50) else q10
        else:
            tp_ret = q50 if np.isfinite(q50) else q10
        tp = entry * (1.0 + tp_ret) if np.isfinite(tp_ret) else float("nan")
        tp2 = entry * (1.0 + q10) if np.isfinite(q10) else float("nan")
        risk = sl - entry
        reward = entry - tp
        rr = _safe_float(reward / risk) if np.isfinite(risk) and risk > 1e-12 else float("nan")
        action_reason = _t(
            "做空条件满足：方向概率、edge、置信度与风险过滤均通过。",
            "Short conditions satisfied: direction probability, edge, confidence, and risk filters all pass.",
        )

    if action == "WAIT" and entry_touched:
        action_reason = _t(
            "已到价，但规则未完全通过，建议继续观察。",
            "Entry touched, but rule checks are not fully passed. Keep waiting.",
        )

    if action == "LONG":
        selected_checks = long_checks
    elif action == "SHORT":
        selected_checks = short_checks
    else:
        selected_checks = long_checks if sum(1 for _, ok in long_checks if ok) >= sum(1 for _, ok in short_checks if ok) else short_checks
    checks_passed = int(sum(1 for _, ok in selected_checks if ok))
    checks_total = int(len(selected_checks))
    failed_checks = [label for label, ok in selected_checks if not ok]
    gate_blocked = not (np.isfinite(current_price) and np.isfinite(entry))
    valid_until_ts = pd.to_datetime(valid_until, utc=True, errors="coerce")
    expired = bool(pd.notna(valid_until_ts) and pd.Timestamp.now(tz="UTC") > valid_until_ts)
    if gate_blocked:
        trade_status = "BLOCKED"
    elif expired:
        trade_status = "EXPIRED"
    elif action in {"LONG", "SHORT"}:
        trade_status = "READY" if entry_touched else "WAIT_ENTRY"
    else:
        trade_status = "WAIT_RULES"
    status_text_map = {
        "READY": _t("可执行", "Ready"),
        "WAIT_ENTRY": _t("等待到价", "Waiting Entry"),
        "WAIT_RULES": _t("规则未通过", "Rules Not Passed"),
        "BLOCKED": _t("阻断", "Blocked"),
        "EXPIRED": _t("已过期", "Expired"),
    }
    if trade_status == "READY":
        trade_status_note = _t("已到价 + 规则通过，可执行。", "Entry touched + rules passed, executable.")
    elif trade_status == "WAIT_ENTRY":
        trade_status_note = _t("规则已通过，等待到价。", "Rules passed, waiting for entry touch.")
    elif trade_status == "WAIT_RULES" and entry_touched:
        fail_text = " / ".join(failed_checks[:3]) if failed_checks else _t("未知", "unknown")
        trade_status_note = _t(f"已到价，但规则未通过：{fail_text}", f"Entry touched but rules failed: {fail_text}")
    elif trade_status == "WAIT_RULES":
        fail_text = " / ".join(failed_checks[:3]) if failed_checks else _t("未知", "unknown")
        trade_status_note = _t(f"规则未通过：{fail_text}", f"Rules failed: {fail_text}")
    elif trade_status == "EXPIRED":
        trade_status_note = _t("信号已过期，请刷新重算。", "Signal expired. Refresh and recompute.")
    else:
        trade_status_note = _t("价格/数据缺失，暂不可执行。", "Price/data missing. Execution blocked.")

    strength = _signal_strength_label(abs(_safe_float(p_up - 0.5)) * 100.0 if np.isfinite(p_up) else float("nan"), 2.0, 5.0)
    return {
        "action": action,
        "action_cn": {"LONG": _t("做多", "Long"), "SHORT": _t("做空", "Short"), "WAIT": _t("观望", "Wait")}.get(action, _t("观望", "Wait")),
        "action_reason": action_reason,
        "entry": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "take_profit_2": tp2,
        "rr": rr,
        "plan_side": plan_side,
        "plan_side_text": _t("做多预案", "Long plan") if plan_side == "LONG" else _t("做空预案", "Short plan"),
        "entry_band_pct": entry_band_pct,
        "entry_gap_pct": entry_gap_pct,
        "entry_touched": entry_touched,
        "entry_touched_at": entry_touched_at,
        "trade_status": trade_status,
        "trade_status_text": status_text_map.get(trade_status, trade_status),
        "trade_status_note": trade_status_note,
        "signal_key": signal_key,
        "signal_time_utc": signal_time_utc,
        "valid_until": valid_until,
        "price_source": price_source,
        "price_timestamp_market": price_timestamp_market,
        "price_timestamp_utc": price_timestamp_utc,
        "failed_checks": failed_checks,
        "horizon_label": horizon_label,
        "risk_level": risk_level,
        "confidence_score": conf,
        "p_up": p_up,
        "p_down": 1.0 - p_up if np.isfinite(p_up) else float("nan"),
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "edge_long": edge_long,
        "edge_short": edge_short,
        "edge_risk_long": edge_risk_long,
        "edge_risk_short": edge_risk_short,
        "cost_bps": cost_bps,
        "risk_profile": profile_display,
        "model_health": model_health,
        "event_risk": bool(event_risk),
        "long_checks": long_checks,
        "short_checks": short_checks,
        "selected_checks": selected_checks,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "signal_strength": strength,
        "signal_strength_text": _signal_strength_human_text(strength),
        "policy_reason": str(row.get("policy_reason", "-")),
    }


def _render_trade_decision_summary(
    *,
    plan: Dict[str, object],
    reliability_summary: Dict[str, float] | None,
    backtest_policy_row: pd.Series | None = None,
) -> None:
    st.markdown(f"## {_t('交易决策卡（3秒结论）', 'Decision Card (3-second view)')}")
    action = str(plan.get("action", "WAIT"))
    action_cn = str(plan.get("action_cn", _t("观望", "Wait")))
    action_display = f"{action} / {action_cn}" if _ui_lang() == "zh" else action
    color = {"LONG": "#22c55e", "SHORT": "#ef4444", "WAIT": "#f59e0b"}.get(action, "#94a3b8")
    st.markdown(
        (
            "<div style='padding:12px 14px;border:1px solid rgba(148,163,184,.25);border-radius:12px;'>"
            f"<div style='font-size:13px;color:#94a3b8'>{_t('最终建议', 'Final Suggestion')}</div>"
            f"<div style='font-size:42px;font-weight:800;color:{color};line-height:1.1'>{action_display}</div>"
            f"<div style='margin-top:6px;color:#cbd5e1'>{plan.get('action_reason','-')}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        _t(
            f"当前满足 {int(plan.get('checks_passed', 0))}/{int(plan.get('checks_total', 0))} 条开仓条件 -> 建议 {str(plan.get('action_cn', _t('观望', 'Wait')))}（{str(plan.get('risk_profile', _t('标准', 'Standard')))}）",
            f"Now passing {int(plan.get('checks_passed', 0))}/{int(plan.get('checks_total', 0))} entry checks -> Suggested action: {str(plan.get('action', 'WAIT'))}",
        )
    )

    status = str(plan.get("trade_status", "WAIT_RULES"))
    status_text = str(plan.get("trade_status_text", status))
    status_note = str(plan.get("trade_status_note", "-"))
    st.info(f"{_t('交易状态', 'Trade Status')}: {status_text} | {status_note}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(_t("推荐入场", "Suggested Entry"), _format_price(plan.get("entry")))
    c2.metric(_t("止损", "Stop Loss"), _format_price(plan.get("stop_loss")))
    c3.metric(_t("止盈(TP1)", "Take Profit (TP1)"), _format_price(plan.get("take_profit")))
    c4.metric(_t("盈亏比 R:R", "Risk/Reward R:R"), _format_float(plan.get("rr"), 2))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric(_t("持仓周期", "Holding Horizon"), str(plan.get("horizon_label", "4h")))
    c6.metric(_t("风险等级", "Risk Level"), _risk_text(str(plan.get("risk_level", "-"))))
    c7.metric(_t("信号强度", "Signal Strength"), str(plan.get("signal_strength_text", "-")))
    c8.metric(_t("模型可信度", "Model Reliability"), _reliability_level_text(reliability_summary))
    c9, c10, c11, c12 = st.columns(4)
    c9.metric(_t("止盈(TP2)", "Take Profit (TP2)"), _format_price(plan.get("take_profit_2")))
    c10.metric("Long Edge", _format_change_pct(plan.get("edge_long")))
    c11.metric("Short Edge", _format_change_pct(plan.get("edge_short")))
    active_edge_risk = (
        plan.get("edge_risk_long")
        if str(plan.get("action", "WAIT")) == "LONG"
        else (plan.get("edge_risk_short") if str(plan.get("action", "WAIT")) == "SHORT" else float("nan"))
    )
    c12.metric("Edge/Risk", _format_float(active_edge_risk, 3))
    c13, c14, c15, c16 = st.columns(4)
    c13.metric(_t("距入场(%)", "Entry Gap (%)"), _format_change_pct(plan.get("entry_gap_pct")))
    c14.metric(
        _t("到价触发", "Entry Touched"),
        _t("已触发", "Touched") if bool(plan.get("entry_touched", False)) else _t("未触发", "Not yet"),
    )
    c15.metric(
        _t("首次触发时间", "First Touch Time"),
        str(plan.get("entry_touched_at", "-")) if str(plan.get("entry_touched_at", "")).strip() else "-",
    )
    c16.metric(_t("预案方向", "Plan Side"), str(plan.get("plan_side_text", "-")))
    st.caption(
        _t(
            f"P(up): {_format_change_pct(plan.get('p_up')).replace('+','')} | "
            f"P(down): {_format_change_pct(plan.get('p_down')).replace('+','')} | "
            f"成本口径: 双边 {float(plan.get('cost_bps', 0.0)):.1f} bps（开+平）",
            f"P(up): {_format_change_pct(plan.get('p_up')).replace('+','')} | "
            f"P(down): {_format_change_pct(plan.get('p_down')).replace('+','')} | "
            f"Cost basis: round-trip {float(plan.get('cost_bps', 0.0)):.1f} bps (open+close)",
        )
    )
    st.caption(
        _t(
            f"价格源: {plan.get('price_source', '-')} | 市场时间: {plan.get('price_timestamp_market', '-')} | UTC: {plan.get('price_timestamp_utc', '-')} | "
            f"信号时间(UTC): {plan.get('signal_time_utc', '-')} | 有效期: {plan.get('valid_until', '-')}",
            f"Price source: {plan.get('price_source', '-')} | Market ts: {plan.get('price_timestamp_market', '-')} | UTC: {plan.get('price_timestamp_utc', '-')} | "
            f"Signal time (UTC): {plan.get('signal_time_utc', '-')} | Valid until: {plan.get('valid_until', '-')}",
        )
    )
    if backtest_policy_row is not None:
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric(_t("近回测胜率", "Backtest Win Rate"), _format_change_pct(backtest_policy_row.get("win_rate")).replace("+", ""))
        b2.metric("Profit Factor", _format_float(backtest_policy_row.get("profit_factor"), 2))
        b3.metric("Avg Win/Loss", _format_float(backtest_policy_row.get("avg_win_loss_ratio"), 2))
        b4.metric(_t("最大回撤", "Max Drawdown"), _format_change_pct(backtest_policy_row.get("max_drawdown")))
        b5.metric(_t("夏普", "Sharpe"), _format_float(backtest_policy_row.get("sharpe"), 2))
        b6.metric(_t("交易次数", "Trades"), f"{int(_safe_float(backtest_policy_row.get('trades_count')))}")
        st.caption(
            _t(
                f"Expectancy: {_format_float(backtest_policy_row.get('expectancy'), 4)} | "
                f"总收益: {_format_change_pct(backtest_policy_row.get('total_return'))} | "
                f"波动率: {_format_float(backtest_policy_row.get('volatility'), 4)}",
                f"Expectancy: {_format_float(backtest_policy_row.get('expectancy'), 4)} | "
                f"Total Return: {_format_change_pct(backtest_policy_row.get('total_return'))} | "
                f"Volatility: {_format_float(backtest_policy_row.get('volatility'), 4)}",
            )
        )


def _render_rule_checklist(plan: Dict[str, object]) -> None:
    st.markdown(f"**{_t('信号触发规则（当前判定）', 'Signal Trigger Rules (Current)')}**")
    lcol, scol = st.columns(2)
    with lcol:
        st.markdown(_t("`Long` 触发条件", "`Long` Conditions"))
        for label, ok in plan.get("long_checks", []):
            st.markdown(f"- {'✅' if ok else '❌'} {label}")
    with scol:
        st.markdown(_t("`Short` 触发条件", "`Short` Conditions"))
        for label, ok in plan.get("short_checks", []):
            st.markdown(f"- {'✅' if ok else '❌'} {label}")
    st.markdown(_t("`当前建议` 对应条件", "Checks for current suggestion"))
    for label, ok in plan.get("selected_checks", []):
        st.markdown(f"- {'✅' if ok else '❌'} {label}")
    st.caption(
        _t(
            "规则：Long 需满足 p_up、edge、置信度、风险过滤；Short 同理并要求允许做空。未满足则观望。",
            "Rule: Long requires p_up, edge, confidence, and risk filters; Short is analogous and also requires shorting to be allowed. Otherwise Wait.",
        )
    )


def _execution_output_dir(processed_dir: Path | None = None) -> Path:
    root = processed_dir or Path("data/processed")
    out = root / "execution"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_kill_switch_state_dashboard(out_dir: Path) -> Dict[str, object]:
    path = out_dir / "kill_switch.state.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
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
        "trial_position_scale": 0.25,
    }


def _save_kill_switch_state_dashboard(out_dir: Path, state: Dict[str, object]) -> None:
    path = out_dir / "kill_switch.state.json"
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl_dashboard(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _utc_now_text_dashboard() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")


def _load_health_streak_dashboard(out_dir: Path) -> Dict[str, object]:
    path = out_dir / "health_checks_streak.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {"consecutive_pass": 0, "last_pass": False, "last_check_utc": "-"}


def _render_kill_switch_control_panel(out_dir: Path) -> None:
    cfg = _load_main_config_cached("configs/config.yaml")
    ks_cfg = (cfg.get("kill_switch", {}) if isinstance(cfg, dict) else {}) or {}
    required_checks = int(_safe_float(ks_cfg.get("recovery_health_checks_required", 3)))
    default_trial_scale = float(_safe_float(ks_cfg.get("trial_position_scale", 0.25)))
    default_trial_windows = int(_safe_float(ks_cfg.get("trial_windows", 1)))
    admin_role = str(ks_cfg.get("admin_role", "ops_admin"))

    state = _load_kill_switch_state_dashboard(out_dir)
    health = _load_health_streak_dashboard(out_dir)

    st.subheader(_t("Kill Switch 控制台", "Kill Switch Console"))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(_t("当前状态", "Current Status"), "ACTIVE" if bool(state.get("active", False)) else "INACTIVE")
    k2.metric(_t("恢复申请", "Recovery Request"), _t("已提交", "Submitted") if bool(state.get("release_request", False)) else _t("未提交", "Not Submitted"))
    k3.metric(_t("连续健康通过", "Consecutive Health Pass"), f"{int(_safe_float(health.get('consecutive_pass', 0)))} / {required_checks}")
    k4.metric(_t("试运行窗口剩余", "Trial Windows Left"), f"{int(_safe_float(state.get('trial_windows_remaining', 0)))}")
    st.caption(
        f"reason={state.get('reason', '-')}"
        f" | requested_by={state.get('requested_by', '-')}"
        f" | requester_role={state.get('requester_role', '-')}"
        f" | trial_scale={_format_float(state.get('trial_position_scale', default_trial_scale), 2)}"
        f" | admin_role={admin_role}"
    )

    c1, c2 = st.columns(2)
    operator_name = c1.text_input(_t("操作人", "Operator"), value=str(state.get("requested_by", "") or "youka"), key="ks_operator_name")
    role_options = ["ops_admin", "trader", "viewer"]
    current_role = str(state.get("requester_role", "")).lower()
    role_index = role_options.index(current_role) if current_role in role_options else 0
    operator_role = c2.selectbox(
        _t("操作角色", "Operator Role"),
        options=role_options,
        index=role_index,
        key="ks_operator_role",
    )
    reason = st.text_input(_t("操作原因", "Reason"), value="manual_ops", key="ks_reason")

    b1, b2, b3 = st.columns(3)
    if b1.button(_t("触发 Kill Switch（停止开新仓）", "Activate Kill Switch (block new entries)"), key="ks_activate_btn", use_container_width=True):
        state.update(
            {
                "active": True,
                "reason": reason or "manual_switch_on",
                "release_request": False,
                "requested_by": operator_name,
                "requester_role": operator_role,
                "trial_windows_remaining": 0,
                "trial_position_scale": float(default_trial_scale),
            }
        )
        _save_kill_switch_state_dashboard(out_dir, state)
        _append_jsonl_dashboard(
            out_dir / "kill_switch_events.jsonl",
            {
                "timestamp_utc": _utc_now_text_dashboard(),
                "event": "manual_switch_on",
                "reason": reason,
                "operator": operator_name,
                "operator_role": operator_role,
            },
        )
        st.success(_t("已触发 Kill Switch。", "Kill Switch activated."))
        st.rerun()

    if b2.button(_t("提交恢复申请（不立即解除）", "Submit recovery request (no immediate release)"), key="ks_request_recovery_btn", use_container_width=True):
        state.update(
            {
                "release_request": True,
                "requested_by": operator_name,
                "requester_role": operator_role,
                "reason": reason or state.get("reason", "recovery_requested"),
            }
        )
        _save_kill_switch_state_dashboard(out_dir, state)
        _append_jsonl_dashboard(
            out_dir / "kill_switch_recovery_log.jsonl",
            {
                "timestamp_utc": _utc_now_text_dashboard(),
                "event": "recovery_requested",
                "reason": reason,
                "operator": operator_name,
                "operator_role": operator_role,
                "required_consecutive_pass": required_checks,
                "current_consecutive_pass": int(_safe_float(health.get("consecutive_pass", 0))),
            },
        )
        st.info(_t("恢复申请已提交。后续执行步骤会按健康检查自动判断是否解除。", "Recovery request submitted. Future execution steps will auto-check health and decide release."))
        st.rerun()

    if b3.button(_t("管理员立即解除（应急）", "Admin force release (emergency)"), key="ks_force_off_btn", use_container_width=True):
        if operator_role != admin_role:
            st.error(_t(f"仅 {admin_role} 可执行强制解除。", f"Only {admin_role} can force release."))
        else:
            state.update(
                {
                    "active": False,
                    "release_request": False,
                    "reason": "manual_force_off",
                    "requested_by": operator_name,
                    "requester_role": operator_role,
                    "trial_windows_remaining": max(default_trial_windows, 1),
                    "trial_position_scale": float(default_trial_scale),
                }
            )
            _save_kill_switch_state_dashboard(out_dir, state)
            _append_jsonl_dashboard(
                out_dir / "kill_switch_recovery_log.jsonl",
                {
                    "timestamp_utc": _utc_now_text_dashboard(),
                    "event": "manual_force_off",
                    "reason": reason,
                    "operator": operator_name,
                    "operator_role": operator_role,
                    "trial_windows_remaining": max(default_trial_windows, 1),
                    "trial_position_scale": float(default_trial_scale),
                },
            )
            st.success(_t("已强制解除 Kill Switch，并设置试运行窗口。", "Kill Switch force released and trial window configured."))
            st.rerun()


def _packet_preview_row(packet: Dict[str, object]) -> Dict[str, object]:
    return {
        "decision_id": packet.get("decision_id"),
        "market": packet.get("market"),
        "symbol": packet.get("symbol"),
        "action": packet.get("action"),
        "entry": _format_price(packet.get("entry")),
        "sl": _format_price(packet.get("sl")),
        "tp1": _format_price(packet.get("tp1")),
        "rr": _format_float(packet.get("rr"), 2),
        "trade_status": packet.get("trade_status"),
        "edge": _format_change_pct(packet.get("expected_edge_pct")),
        "edge_risk": _format_float(packet.get("edge_risk"), 3),
        "confidence": _format_float(packet.get("confidence_score"), 1),
        "risk_level": _risk_text(str(packet.get("risk_level", "-"))),
        "valid_until": packet.get("valid_until"),
        "model_version": packet.get("model_version"),
        "data_version": packet.get("data_version"),
        "config_hash": packet.get("config_hash"),
        "git_commit": packet.get("git_commit"),
        "reasons": " | ".join(packet.get("reasons", [])) if isinstance(packet.get("reasons"), list) else str(packet.get("reasons", "")),
    }


def _render_decision_packet_and_execution(
    *,
    row: pd.Series,
    trade_plan: Dict[str, object] | None,
    section_prefix: str,
    processed_dir: Path | None = None,
) -> None:
    cfg = _load_main_config_cached("configs/config.yaml")
    packet = build_decision_packet(
        row=row,
        trade_plan=trade_plan or {},
        cfg=cfg,
        git_commit=_get_git_hash_short_cached(),
    )
    out_dir = _execution_output_dir(processed_dir)
    key_suffix = f"{str(packet.get('market','m'))}_{str(packet.get('symbol','s'))}_{str(packet.get('decision_id','d'))[:8]}"

    st.markdown(f"### {_t('DecisionPacket（统一决策输出）', 'DecisionPacket (Unified Decision Output)')}")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric(_t("最终动作", "Final Action"), str(packet.get("action", "WAIT")))
    p2.metric("Entry", _format_price(packet.get("entry")))
    p3.metric("TP1 / SL", f"{_format_price(packet.get('tp1'))} / {_format_price(packet.get('sl'))}")
    p4.metric("RR", _format_float(packet.get("rr"), 2))
    p5, p6, p7 = st.columns(3)
    p5.metric(_t("交易状态", "Trade Status"), str(packet.get("trade_status", "-")))
    p6.metric(_t("距入场(%)", "Entry Gap (%)"), _format_change_pct(packet.get("entry_gap_pct")))
    p7.metric(
        _t("到价触发", "Entry Touched"),
        _t("已触发", "Touched") if bool(packet.get("entry_touched", False)) else _t("未触发", "Not yet"),
    )
    st.caption(
        f"config_hash={packet.get('config_hash','-')} | git={packet.get('git_commit','-')} | "
        f"cost={float(_safe_float(packet.get('cost_bps'))):.1f}bps | valid_until={packet.get('valid_until','-')} | "
        f"gate={packet.get('gate_status','PASS')}"
    )
    blocked = packet.get("blocked_reason", [])
    if isinstance(blocked, list) and blocked:
        st.warning(_t("Gate 阻断原因: ", "Gate blocked reasons: ") + " | ".join(str(x) for x in blocked))

    btn1, btn2 = st.columns(2)
    if btn1.button(_t("保存 DecisionPacket", "Save DecisionPacket"), key=f"save_dp_{key_suffix}", use_container_width=True):
        paths = persist_decision_packet(packet, out_dir)
        st.success(_t(f"已保存 DecisionPacket: `{paths['json']}`", f"DecisionPacket saved: `{paths['json']}`"))

    latest_price = _safe_float(row.get("current_price"))
    if btn2.button(_t("Paper Trading 执行一步", "Run one Paper Trading step"), key=f"paper_exec_{key_suffix}", use_container_width=True):
        persist_decision_packet(packet, out_dir)
        res = apply_decision_to_paper_book(packet, latest_price=latest_price, output_dir=out_dir)
        if str(res.get("status")) in {"ok", "skipped"}:
            st.success(str(res.get("message", "paper trading step completed")))
        else:
            st.error(str(res.get("message", "paper trading step failed")))

    with st.expander(_t("查看 DecisionPacket JSON", "View DecisionPacket JSON"), expanded=False):
        st.json(packet)

    artifacts = load_execution_artifacts(out_dir)
    stats = summarize_execution(artifacts)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open Positions", f"{int(_safe_float(stats.get('open_positions')))}")
    c2.metric("Closed Positions", f"{int(_safe_float(stats.get('closed_positions')))}")
    c3.metric("Paper Win Rate", _format_change_pct(stats.get("win_rate")).replace("+", ""))
    c4.metric("Avg Net PnL", _format_change_pct(stats.get("avg_net_pnl_pct")))
    c5.metric("Total Net PnL", _format_change_pct(stats.get("total_net_pnl_pct")))

    orders = artifacts.get("orders", pd.DataFrame())
    if not orders.empty:
        view = orders.sort_values("created_at_utc", ascending=False).head(15).copy()
        st.markdown(f"**{_t('最近 Paper Orders', 'Recent Paper Orders')}**")
        st.dataframe(view, use_container_width=True, hide_index=True)

    positions = artifacts.get("positions", pd.DataFrame())
    if not positions.empty:
        open_pos = positions[positions["status"].astype(str) == "open"].copy()
        closed_pos = positions[positions["status"].astype(str) == "closed"].copy()
        st.markdown(f"**{_t('当前持仓（Open）', 'Current Positions (Open)')}**")
        if open_pos.empty:
            st.info(_t("当前无持仓。", "No open positions."))
        else:
            st.dataframe(open_pos.sort_values("entry_time_utc", ascending=False), use_container_width=True, hide_index=True)
        st.markdown(f"**{_t('最近平仓（Closed）', 'Recently Closed Positions')}**")
        if closed_pos.empty:
            st.info(_t("暂无平仓记录。", "No closed position records yet."))
        else:
            st.dataframe(
                closed_pos.sort_values("exit_time_utc", ascending=False).head(15),
                use_container_width=True,
                hide_index=True,
            )


def _render_symbol_backtest_section(
    *,
    processed_dir: Path,
    market: str,
    symbol: str,
    symbol_aliases: List[str] | None = None,
    provider: str | None = None,
    fallback_symbol: str | None = None,
    title: str = "回测结果（开单效果）",
) -> None:
    artifacts = _load_backtest_artifacts(str(processed_dir))
    summary = artifacts.get("metrics_summary", pd.DataFrame())
    compare = artifacts.get("compare", pd.DataFrame())
    equity = artifacts.get("equity", pd.DataFrame())
    by_fold = artifacts.get("metrics_by_fold", pd.DataFrame())
    trades = artifacts.get("trades", pd.DataFrame())
    latest_signals = artifacts.get("latest_signals", pd.DataFrame())
    if summary.empty:
        summary = pd.DataFrame(columns=["market", "symbol", "strategy"])
    alias_tokens = {_normalize_symbol_token(symbol)}
    for a in symbol_aliases or []:
        alias_tokens.add(_normalize_symbol_token(a))

    def _match_symbol(df: pd.DataFrame, col: str = "symbol") -> pd.Series:
        return df[col].map(_normalize_symbol_token).isin(alias_tokens)

    sub = summary[
        (summary["market"].astype(str) == str(market))
        & _match_symbol(summary, col="symbol")
    ].copy()

    # On-demand single-symbol backtest fallback if precomputed table has no matching row.
    fallback_note = ""
    if sub.empty:
        try:
            prov = str(provider or ("binance" if str(market) == "crypto" and str(symbol).upper().endswith("USDT") else "yahoo"))
            with st.spinner(_t("该标的不在预计算回测样本中，正在即时回测...", "Symbol not in precomputed sample, running on-demand backtest...")):
                realtime = _run_single_symbol_backtest_cached(
                    market=str(market),
                    symbol=str(symbol),
                    provider=prov,
                    fallback_symbol=str(fallback_symbol or ""),
                )
            summary_rt = realtime.get("metrics_summary", pd.DataFrame())
            compare_rt = realtime.get("compare_baselines", pd.DataFrame())
            equity_rt = realtime.get("equity", pd.DataFrame())
            by_fold_rt = realtime.get("metrics_by_fold", pd.DataFrame())
            trades_rt = realtime.get("trades", pd.DataFrame())
            latest_signal_rt = realtime.get("latest_signal", pd.DataFrame())
            if not summary_rt.empty:
                sub = summary_rt[
                    (summary_rt["market"].astype(str) == str(market))
                    & _match_symbol(summary_rt, col="symbol")
                ].copy()
                compare = compare_rt
                equity = equity_rt
                by_fold = by_fold_rt
                trades = trades_rt
                latest_signals = latest_signal_rt
                fallback_note = _t(
                    "已为当前标的即时补跑回测（未写入全量回测文件，仅用于本页展示）。",
                    "On-demand backtest generated for current symbol (not persisted to global backtest artifacts).",
                )
        except Exception as exc:
            fallback_note = _t(f"即时回测失败：{exc}", f"On-demand backtest failed: {exc}")

    if sub.empty:
        st.info(_t("该标的暂无回测记录。可能是历史数据不足或数据源不可用。", "No backtest record for this symbol. History may be insufficient or data source unavailable."))
        if fallback_note:
            st.caption(fallback_note)
        return

    st.markdown("---")
    st.subheader(title)
    if fallback_note:
        st.caption(fallback_note)

    sig_view = latest_signals[
        (latest_signals["market"].astype(str) == str(market))
        & _match_symbol(latest_signals, col="symbol")
    ].copy()
    if not sig_view.empty:
        _render_trade_signal_block(sig_view.iloc[-1], header=_t("当前可执行信号（回测口径）", "Current Executable Signal (Backtest view)"))

    policy_row = sub[sub["strategy"] == "policy"].head(1)
    if not policy_row.empty:
        row = policy_row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(_t("策略总收益", "Strategy Return"), _format_change_pct(row.get("total_return")))
        c2.metric(_t("策略夏普", "Strategy Sharpe"), _format_float(row.get("sharpe"), 2))
        c3.metric(_t("最大回撤", "Max Drawdown"), _format_change_pct(row.get("max_drawdown")))
        c4.metric(_t("胜率", "Win Rate"), _format_change_pct(row.get("win_rate")).replace("+", ""))
        st.caption(
            f"{_t('盈亏比', 'Avg Win/Loss')}: {_format_float(row.get('avg_win_loss_ratio'), 2)} | "
            f"Profit Factor: {_format_float(row.get('profit_factor'), 2)} | "
            f"{_t('交易次数', 'Trades')}: {int(_safe_float(row.get('trades_count')))}"
        )

    show = sub.copy()
    show["策略"] = show["strategy"].map(
        {
            "policy": "策略信号",
            "buy_hold": "买入并持有",
            "ma_crossover": "均线交叉",
            "naive_prev_bar": "前一日方向",
            "lux_machine_supertrend": "机器学习自适应SuperTrend",
        }
    ).fillna(show["strategy"])
    show["总收益"] = show["total_return"].map(_format_change_pct)
    show["夏普"] = show["sharpe"].map(lambda x: _format_float(x, 2))
    show["最大回撤"] = show["max_drawdown"].map(_format_change_pct)
    show["胜率"] = show["win_rate"].map(lambda x: _format_change_pct(x).replace("+", ""))
    show["盈亏比"] = show["avg_win_loss_ratio"].map(lambda x: _format_float(x, 2))
    show["PF"] = show["profit_factor"].map(lambda x: _format_float(x, 2))
    show_cols = ["策略", "总收益", "夏普", "最大回撤", "胜率", "盈亏比", "PF"]
    st.dataframe(show[show_cols], use_container_width=True, hide_index=True)

    cmp = compare[
        (compare["market"].astype(str) == str(market))
        & _match_symbol(compare, col="symbol")
    ].copy()
    if not cmp.empty:
        cmp["相对总收益提升"] = cmp["delta_total_return"].map(_format_change_pct)
        cmp["相对夏普提升"] = cmp["delta_sharpe"].map(lambda x: _format_float(x, 2))
        cmp["相对回撤变化"] = cmp["delta_max_drawdown"].map(_format_change_pct)
        cmp["基准策略"] = cmp["baseline"].map(
            {
                "buy_hold": _t("买入并持有", "Buy & Hold"),
                "ma_crossover": _t("均线交叉", "MA Crossover"),
                "naive_prev_bar": _t("前一日方向", "Previous-bar Direction"),
                "lux_machine_supertrend": _t("机器学习自适应SuperTrend", "ML Adaptive SuperTrend"),
            }
        ).fillna(cmp["baseline"])
        st.markdown(f"**{_t('与基准对比（策略信号 - 基准）', 'vs Baselines (Policy - Baseline)')}**")
        col_base = _t("基准策略", "Baseline")
        col_delta_ret = _t("相对总收益提升", "Delta Return")
        col_delta_sharpe = _t("相对夏普提升", "Delta Sharpe")
        col_delta_dd = _t("相对回撤变化", "Delta Drawdown")
        cmp_view = cmp.rename(
            columns={
                "基准策略": col_base,
                "相对总收益提升": col_delta_ret,
                "相对夏普提升": col_delta_sharpe,
                "相对回撤变化": col_delta_dd,
            }
        )
        st.dataframe(
            cmp_view[[col_base, col_delta_ret, col_delta_sharpe, col_delta_dd]],
            use_container_width=True,
            hide_index=True,
        )

    sub_eq = equity[
        (equity["market"].astype(str) == str(market))
        & _match_symbol(equity, col="symbol")
    ].copy()
    if not sub_eq.empty:
        sub_eq["timestamp_utc"] = pd.to_datetime(sub_eq["timestamp_utc"], utc=True, errors="coerce")
        sub_eq = sub_eq.dropna(subset=["timestamp_utc"]).sort_values(["strategy", "fold", "timestamp_utc"])
        rows: List[pd.DataFrame] = []
        for strategy, sdf in sub_eq.groupby("strategy", dropna=False):
            running = 1.0
            frames: List[pd.DataFrame] = []
            for _, fold_df in sdf.groupby("fold", dropna=False):
                part = fold_df.sort_values("timestamp_utc").copy()
                part["eq_chain"] = running * (1.0 + pd.to_numeric(part["strategy_ret"], errors="coerce").fillna(0.0)).cumprod()
                running = float(part["eq_chain"].iloc[-1]) if not part.empty else running
                frames.append(part)
            if frames:
                merged = pd.concat(frames, ignore_index=True)
                merged["strategy"] = strategy
                rows.append(merged[["timestamp_utc", "strategy", "eq_chain"]])
        if rows:
            eq_plot = pd.concat(rows, ignore_index=True)
            fig = go.Figure()
            for strategy, sdf in eq_plot.groupby("strategy", dropna=False):
                label = {
                    "policy": _t("策略信号", "Policy Signal"),
                    "buy_hold": _t("买入并持有", "Buy & Hold"),
                    "ma_crossover": _t("均线交叉", "MA Crossover"),
                    "naive_prev_bar": _t("前一日方向", "Previous-bar Direction"),
                    "lux_machine_supertrend": _t("机器学习自适应SuperTrend", "ML Adaptive SuperTrend"),
                }.get(str(strategy), str(strategy))
                fig.add_trace(
                    go.Scatter(
                        x=sdf["timestamp_utc"],
                        y=sdf["eq_chain"],
                        mode="lines",
                        name=label,
                    )
                )
            fig.update_layout(
                title=_t(f"{symbol} 回测资金曲线（Walk-forward 串联）", f"{symbol} Backtest Equity Curve (Walk-forward chain)"),
                xaxis_title=_t("时间", "Time"),
                yaxis_title=_t("净值", "Equity"),
                template="plotly_white",
                height=340,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    fold_view = by_fold[
        (by_fold["market"].astype(str) == str(market))
        & _match_symbol(by_fold, col="symbol")
    ].copy()
    if not fold_view.empty:
        col_total = _t("总收益", "Total Return")
        col_sharpe = _t("夏普", "Sharpe")
        col_mdd = _t("最大回撤", "Max Drawdown")
        col_strategy = _t("策略", "Strategy")
        fold_view[col_total] = fold_view["total_return"].map(_format_change_pct)
        fold_view[col_sharpe] = fold_view["sharpe"].map(lambda x: _format_float(x, 2))
        fold_view[col_mdd] = fold_view["max_drawdown"].map(_format_change_pct)
        fold_view[col_strategy] = fold_view["strategy"].astype(str)
        st.markdown(f"**{_t('分折（fold）表现', 'Fold-level Performance')}**")
        st.dataframe(
            fold_view[["fold", col_strategy, col_total, col_sharpe, col_mdd]].sort_values([col_strategy, "fold"]),
            use_container_width=True,
            hide_index=True,
        )

    trade_view = trades[
        (trades["market"].astype(str) == str(market))
        & _match_symbol(trades, col="symbol")
    ].copy()
    if not trade_view.empty:
        trade_view = trade_view.sort_values("entry_time", ascending=False).head(30).copy()
        col_reason = _t("开单理由", "Entry Reason")
        col_side = _t("方向", "Side")
        col_signal = _t("信号", "Signal")
        col_entry = _t("进场价", "Entry Price")
        col_sl = _t("止损价", "Stop Loss")
        col_tp = _t("止盈价", "Take Profit")
        col_pnl = _t("净收益", "Net PnL")
        col_exit = _t("退出原因", "Exit Reason")
        if "entry_signal_reason" in trade_view.columns:
            trade_view[col_reason] = trade_view["entry_signal_reason"].map(_format_reason_tokens_cn)
        elif "reason" in trade_view.columns:
            trade_view[col_reason] = trade_view["reason"].map(_format_reason_tokens_cn)
        trade_view[col_side] = trade_view.get("side", pd.Series(["-"] * len(trade_view))).map(
            {"long": _t("做多", "Long"), "short": _t("做空", "Short")}
        ).fillna("-")
        trade_view[col_signal] = trade_view.get("entry_signal", pd.Series(["-"] * len(trade_view))).map(
            _policy_action_text
        )
        trade_view[col_entry] = trade_view["entry_price"].map(_format_price)
        trade_view[col_sl] = trade_view.get("stop_loss_price", pd.Series([np.nan] * len(trade_view))).map(
            _format_price
        )
        trade_view[col_tp] = trade_view.get("take_profit_price", pd.Series([np.nan] * len(trade_view))).map(
            _format_price
        )
        trade_view[col_pnl] = trade_view.get("net_pnl_pct", pd.Series([np.nan] * len(trade_view))).map(
            _format_change_pct
        )
        trade_view[col_exit] = trade_view.get("exit_reason", pd.Series(["-"] * len(trade_view))).map(
            {
                "stop_loss": _t("止损", "Stop Loss"),
                "take_profit": _t("止盈", "Take Profit"),
                "time_exit": _t("时间平仓", "Time Exit"),
                "flat": _t("无持仓（未开单）", "No position (no trade)"),
            }
        ).fillna("-")
        st.markdown(f"**{_t('最近开单记录（含止盈止损）', 'Recent Trades (with TP/SL)')}**")
        show_cols = [
            "entry_time",
            col_side,
            col_signal,
            col_entry,
            col_sl,
            col_tp,
            col_pnl,
            col_exit,
            col_reason,
        ]
        show_cols = [c for c in show_cols if c in trade_view.columns]
        st.dataframe(trade_view[show_cols], use_container_width=True, hide_index=True)


def _render_snapshot_result(
    df: pd.DataFrame,
    title_prefix: str,
    trade_plan: Dict[str, object] | None = None,
) -> None:
    work = _ensure_policy_for_snapshot(df)
    if work.empty:
        st.warning(_t("当前选择未能生成预测快照。", "No snapshot generated for current selection."))
        return
    row = work.iloc[0]
    if pd.isna(row.get("current_price")):
        st.error(_t(f"价格获取失败: {row.get('price_source', '-')}", f"Failed to fetch price: {row.get('price_source', '-')}"))
        if "error_message" in row and pd.notna(row.get("error_message")):
            st.caption(str(row.get("error_message")))
        return

    delta_abs = row.get("predicted_change_abs")
    delta_text = f"{float(delta_abs):+,.2f}" if _is_finite_number(delta_abs) else "-"
    action_text = _policy_action_cn(str(row.get("policy_action", "Flat")))
    s1, s2 = st.columns(2)
    with s1:
        _render_big_value(_t("当前价格", "Current Price"), _format_price(row.get("current_price")))
    with s2:
        _render_big_value(_t("预测价格", "Predicted Price"), _format_price(row.get("predicted_price")))
    s3, s4 = st.columns(2)
    with s3:
        _render_big_value(_t("预计涨跌幅", "Expected Change"), _format_change_pct(row.get("predicted_change_pct")), caption=_t(f"价差: {delta_text}", f"Delta: {delta_text}"))
    with s4:
        _render_big_value(_t("策略动作", "Policy Action"), _policy_action_text(action_text), caption=_t("观望 = 暂不下单", "Wait = no trade now"))
    expected_date_full = str(row.get("expected_date_market", "-"))
    st.markdown(f"**{_t('预期日期（完整）', 'Expected Date (full)')}**")
    st.code(expected_date_full)
    st.caption(
        _t(
            f"价格源: {row.get('price_source', '-')} | 预测方法: {row.get('prediction_method', '-')}",
            f"Price source: {row.get('price_source', '-')} | Prediction method: {row.get('prediction_method', '-')}",
        )
    )
    if "policy_position_size" in row.index:
        policy_reason_text = _format_reason_tokens_cn(row.get("policy_reason", "-"))
        st.caption(
            _t(
                f"建议仓位: {(float(row.get('policy_position_size')) if _is_finite_number(row.get('policy_position_size')) else 0.0):.1%} | "
                f"预期净优势: {_format_change_pct(row.get('policy_expected_edge_pct'))} | "
                f"策略理由: {policy_reason_text}",
                f"Suggested size: {(float(row.get('policy_position_size')) if _is_finite_number(row.get('policy_position_size')) else 0.0):.1%} | "
                f"Expected net edge: {_format_change_pct(row.get('policy_expected_edge_pct'))} | "
                f"Policy reason: {policy_reason_text}",
            )
        )
    _render_decision_packet_and_execution(
        row=row,
        trade_plan=trade_plan or {},
        section_prefix=title_prefix,
        processed_dir=Path("data/processed"),
    )
    signal_ctx = pd.DataFrame()
    try:
        market = str(row.get("market", ""))
        symbol = str(row.get("symbol", ""))
        provider = str(row.get("provider", "yahoo"))
        fallback_symbol = str(row.get("fallback_symbol", ""))
        if market == "crypto" and not fallback_symbol and provider == "coingecko":
            sym_guess = symbol.upper().strip()
            if sym_guess.isalpha() and len(sym_guess) <= 8:
                fallback_symbol = f"{sym_guess}USDT"
        signal_ctx = _load_symbol_signal_context_cached(
            market=market,
            symbol=symbol,
            provider=provider,
            fallback_symbol=fallback_symbol,
        )
    except Exception:
        signal_ctx = pd.DataFrame()
    if not signal_ctx.empty:
        _render_trade_signal_block(signal_ctx.iloc[-1], header=_t("开单信号与风控计划", "Trade Signal & Risk Plan"))
    _render_core_field_explain()

    st.markdown(f"**{_t('量化因子（风险 + 市场行为）', 'Quant Factors (Risk + Market Behavior)')}**")
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    f1.metric(_t("市值因子", "Size"), _format_float(row.get("size_factor"), digits=3))
    f2.metric(_t("价值因子", "Value"), _format_float(row.get("value_factor"), digits=4))
    f3.metric(_t("成长因子", "Growth"), _format_change_pct(row.get("growth_factor")))
    f4.metric(_t("动能因子", "Momentum"), _format_change_pct(row.get("momentum_factor")))
    f5.metric(_t("反转因子", "Reversal"), _format_change_pct(row.get("reversal_factor")))
    f6.metric(_t("低波动因子", "Low Volatility"), _format_change_pct(row.get("low_vol_factor")))

    market_cap = row.get("market_cap_usd")
    market_cap_text = _format_price(market_cap) if _is_finite_number(market_cap) else "-"
    st.caption(
        _t(
            "风险因子来源: "
            f"size={row.get('size_factor_source', '-')}, "
            f"value={row.get('value_factor_source', '-')}, "
            f"growth={row.get('growth_factor_source', '-')} | "
            f"Market Cap(USD): {market_cap_text}",
            "Risk factor source: "
            f"size={row.get('size_factor_source', '-')}, "
            f"value={row.get('value_factor_source', '-')}, "
            f"growth={row.get('growth_factor_source', '-')} | "
            f"Market Cap(USD): {market_cap_text}",
        )
    )
    _render_factor_explain()

    _render_projection_chart(
        current_price=float(row.get("current_price")),
        q10_change_pct=float(row.get("q10_change_pct")),
        q50_change_pct=float(row.get("q50_change_pct")),
        q90_change_pct=float(row.get("q90_change_pct")),
        expected_date_label=expected_date_full,
        title=_t(f"{title_prefix} 预测可视化图（q10 / q50 / q90）", f"{title_prefix} Projection Chart (q10 / q50 / q90)"),
        entry_price=(
            float(trade_plan.get("entry"))
            if isinstance(trade_plan, dict) and _is_finite_number(trade_plan.get("entry"))
            else None
        ),
        stop_loss_price=(
            float(trade_plan.get("stop_loss"))
            if isinstance(trade_plan, dict) and _is_finite_number(trade_plan.get("stop_loss"))
            else None
        ),
        take_profit_price=(
            float(trade_plan.get("take_profit"))
            if isinstance(trade_plan, dict) and _is_finite_number(trade_plan.get("take_profit"))
            else None
        ),
    )


def _render_branch(branch_name: str, df: pd.DataFrame, live_price: float | None = None) -> None:
    st.subheader(f"{branch_name.capitalize()} Branch")
    if df.empty:
        st.warning(f"No predictions found for {branch_name}.")
        return

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    latest = df.iloc[-1]
    st.caption(f"Latest UTC: {latest.get('timestamp_utc', '-')}")

    horizons = _infer_horizons(df.columns.tolist())
    if not horizons:
        st.warning("No horizon prediction columns found.")
        return

    selected_h = st.selectbox(
        f"{branch_name} horizon",
        options=horizons,
        index=0,
        key=f"{branch_name}_horizon",
    )

    c1, c2, c3 = st.columns(3)
    p_up = float(latest.get(f"dir_h{selected_h}_p_up", 0.0))
    p_down = float(latest.get(f"dir_h{selected_h}_p_down", 1.0 - p_up))
    c1.metric("P(up)", f"{p_up:.2%}")
    c2.metric("P(down)", f"{p_down:.2%}")
    c3.metric("Start Window", str(latest.get("start_window_name", "W?")))
    st.caption(
        _t(
            "说明：P(up)/P(down)是方向概率；预测价格是幅度模型结果，两者短期可能不完全一致。",
            "Note: P(up)/P(down) is directional probability; predicted price comes from magnitude model. They may diverge in short term.",
        )
    )

    q10_col = f"ret_h{selected_h}_q0.1"
    q50_col = f"ret_h{selected_h}_q0.5"
    q90_col = f"ret_h{selected_h}_q0.9"
    model_base_price = float(latest.get("close", float("nan")))
    current_price = float(live_price) if live_price is not None else model_base_price
    pred_ret_q50 = float(latest.get(q50_col, float("nan")))
    pred_price = (
        current_price * (1.0 + pred_ret_q50)
        if pd.notna(current_price) and pd.notna(pred_ret_q50)
        else float("nan")
    )
    delta_abs = (
        pred_price - current_price
        if pd.notna(pred_price) and pd.notna(current_price)
        else float("nan")
    )
    pred_price_base = (
        model_base_price * (1.0 + pred_ret_q50)
        if pd.notna(model_base_price) and pd.notna(pred_ret_q50)
        else float("nan")
    )
    expected_date = _expected_date(
        str(latest.get("timestamp_market", "")),
        str(latest.get("timestamp_utc", "")),
        branch_name,
        selected_h,
    )

    policy_row = {}
    try:
        cfg = _load_main_config_cached()
        branch_market_type = "perp" if branch_name == "hourly" else "spot"
        policy_input = pd.DataFrame(
            [
                {
                    "market": "crypto",
                    "market_type": branch_market_type,
                    "p_up": p_up,
                    "q10_change_pct": float(latest.get(q10_col, float("nan"))),
                    "q50_change_pct": pred_ret_q50,
                    "q90_change_pct": float(latest.get(q90_col, float("nan"))),
                    "volatility_score": float(latest.get(q90_col, float("nan")))
                    - float(latest.get(q10_col, float("nan"))),
                    "confidence_score": 50.0,
                    "current_price": current_price,
                    "risk_level": "medium",
                }
            ]
        )
        policy_eval = apply_policy_frame(policy_input, cfg)
        if not policy_eval.empty:
            policy_row = policy_eval.iloc[0].to_dict()
    except Exception:
        policy_row = {}

    delta_text = f"{delta_abs:+,.2f}" if _is_finite_number(delta_abs) else "-"
    action_text = _policy_action_cn(str(policy_row.get("policy_action", "Flat")))
    p1, p2 = st.columns(2)
    with p1:
        _render_big_value(_t("当前价格", "Current Price"), _format_price(current_price))
    with p2:
        _render_big_value(_t("预测价格 (q50)", "Predicted Price (q50)"), _format_price(pred_price))
    p3, p4 = st.columns(2)
    with p3:
        _render_big_value(_t("预计涨跌幅", "Expected Change"), _format_change_pct(pred_ret_q50), caption=_t(f"价差: {delta_text}", f"Delta: {delta_text}"))
    with p4:
        _render_big_value(_t("策略动作", "Policy Action"), _policy_action_text(action_text), caption=_t("观望 = 暂不下单", "Wait = no trade now"))
    st.markdown(f"**{_t('预期日期（完整）', 'Expected Date (full)')}**")
    st.code(expected_date)
    st.caption(
        _t(
            f"模型基准价（最后收盘）: {_format_price(model_base_price)} | "
            f"按基准价口径预测价: {_format_price(pred_price_base)}",
            f"Model base price (last close): {_format_price(model_base_price)} | "
            f"Prediction on base-price convention: {_format_price(pred_price_base)}",
        )
    )
    if policy_row:
        policy_reason_text = _format_reason_tokens_cn(policy_row.get("policy_reason", "-"))
        st.caption(
            _t(
                f"建议仓位: {(float(policy_row.get('policy_position_size')) if _is_finite_number(policy_row.get('policy_position_size')) else 0.0):.1%} | "
                f"预期净优势: {_format_change_pct(policy_row.get('policy_expected_edge_pct'))} | "
                f"策略理由: {policy_reason_text}",
                f"Suggested size: {(float(policy_row.get('policy_position_size')) if _is_finite_number(policy_row.get('policy_position_size')) else 0.0):.1%} | "
                f"Expected net edge: {_format_change_pct(policy_row.get('policy_expected_edge_pct'))} | "
                f"Policy reason: {policy_reason_text}",
            )
        )

    _render_projection_chart(
        current_price=float(current_price),
        q10_change_pct=float(latest.get(q10_col, float("nan"))),
        q50_change_pct=float(latest.get(q50_col, float("nan"))),
        q90_change_pct=float(latest.get(q90_col, float("nan"))),
        expected_date_label=expected_date,
        title=_t(f"{branch_name.capitalize()} 预测可视化（q10 / q50 / q90）", f"{branch_name.capitalize()} Projection (q10 / q50 / q90)"),
    )


def _render_btc_model_detail_section(
    btc_live: float | None,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> None:
    st.markdown("---")
    st.subheader(_t("BTC 模型详情（Hourly / Daily）", "BTC Model Detail (Hourly / Daily)"))
    if btc_live is not None:
        st.info(_t(f"BTC 实时价格 (Binance.US): **${btc_live:,.2f}**", f"BTC Live Price (Binance.US): **${btc_live:,.2f}**"))
    else:
        st.warning(_t("BTC 实时价格获取失败（不影响模型详情展示）。", "Failed to fetch BTC live price (model detail still available)."))

    btc_signal_ctx = _load_symbol_signal_context_cached(
        market="crypto",
        symbol="BTCUSDT",
        provider="binance",
        fallback_symbol="BTCUSDT",
    )
    if not btc_signal_ctx.empty:
        _render_trade_signal_block(btc_signal_ctx.iloc[-1], header=_t("BTC 当前开单信号（技术触发 + 风控）", "BTC Current Signal (technical + risk)"))

    left, right = st.columns(2)
    with left:
        _render_branch("hourly", hourly_df, live_price=btc_live)
    with right:
        _render_branch("daily", daily_df, live_price=btc_live)


def _build_btc_model_snapshot_from_hourly(
    *,
    hourly_df: pd.DataFrame,
    btc_live: float | None,
    fallback_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    if hourly_df.empty:
        return pd.DataFrame()

    required_cols = {
        "timestamp_utc",
        "timestamp_market",
        "close",
        "ret_h4_q0.1",
        "ret_h4_q0.5",
        "ret_h4_q0.9",
    }
    if not required_cols.issubset(set(hourly_df.columns)):
        return pd.DataFrame()

    df = hourly_df.sort_values("timestamp_utc").reset_index(drop=True)
    latest = df.iloc[-1]
    model_base_price = _safe_float(latest.get("close"))
    live_price = _safe_float(btc_live)
    current_price = live_price if np.isfinite(live_price) else model_base_price

    q10 = _safe_float(latest.get("ret_h4_q0.1"))
    q50 = _safe_float(latest.get("ret_h4_q0.5"))
    q90 = _safe_float(latest.get("ret_h4_q0.9"))
    p_up = _safe_float(latest.get("dir_h4_p_up"))
    p_down = _safe_float(latest.get("dir_h4_p_down"))
    if not (np.isfinite(current_price) and np.isfinite(q10) and np.isfinite(q50) and np.isfinite(q90)):
        return pd.DataFrame()

    predicted_price = current_price * (1.0 + q50)
    width = q90 - q10
    confidence_score = float(np.clip((2.0 * abs(p_up - 0.5)) * 100.0, 0.0, 100.0)) if np.isfinite(p_up) else 50.0
    risk_level = "medium"
    if np.isfinite(width):
        if width < 0.02:
            risk_level = "low"
        elif width < 0.05:
            risk_level = "medium"
        elif width < 0.10:
            risk_level = "high"
        else:
            risk_level = "extreme"
    expected_date_market = _expected_date(
        str(latest.get("timestamp_market", "")),
        str(latest.get("timestamp_utc", "")),
        "hourly",
        4,
    )

    ts_utc = pd.to_datetime(str(latest.get("timestamp_utc", "")), utc=True, errors="coerce")
    if pd.notna(ts_utc):
        expected_date_utc = (ts_utc + pd.Timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        expected_date_utc = "-"

    base = {
        "instrument_id": "btc",
        "name": "Bitcoin",
        "market": "crypto",
        "symbol": "BTCUSDT",
        "provider": "binance",
        "timezone": "Asia/Shanghai",
        "horizon_label": "4h",
        "current_price": current_price,
        "predicted_price": predicted_price,
        "predicted_change_pct": q50,
        "predicted_change_abs": predicted_price - current_price,
        "expected_date_market": expected_date_market,
        "expected_date_utc": expected_date_utc,
        "p_up": p_up,
        "p_down": p_down if np.isfinite(p_down) else (1.0 - p_up if np.isfinite(p_up) else float("nan")),
        "volatility_score": width,
        "confidence_score": confidence_score,
        "risk_level": risk_level,
        "start_window_top1": str(latest.get("start_window_name", "W?")),
        "model_base_price": model_base_price,
        "q10_change_pct": q10,
        "q50_change_pct": q50,
        "q90_change_pct": q90,
        "price_source": "binance_ticker_live" if np.isfinite(live_price) else "model_last_close",
        "prediction_method": "mvp_lightgbm_quantile_q50 (hourly h=4)",
        "size_factor": float("nan"),
        "value_factor": float("nan"),
        "growth_factor": float("nan"),
        "momentum_factor": float("nan"),
        "reversal_factor": float("nan"),
        "low_vol_factor": float("nan"),
        "size_factor_source": "na",
        "value_factor_source": "na",
        "growth_factor_source": "na",
        "market_cap_usd": float("nan"),
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
    }

    # Reuse factor fields from fallback snapshot if available.
    if not fallback_snapshot.empty:
        ref = fallback_snapshot.iloc[0]
        factor_cols = [
            "size_factor",
            "value_factor",
            "growth_factor",
            "momentum_factor",
            "reversal_factor",
            "low_vol_factor",
            "size_factor_source",
            "value_factor_source",
            "growth_factor_source",
            "market_cap_usd",
        ]
        for col in factor_cols:
            if col in ref:
                base[col] = ref.get(col)

    return pd.DataFrame([base])


def _build_snapshot_for_selected(
    *,
    market: str,
    row: pd.Series,
) -> pd.DataFrame:
    provider = str(row.get("provider", "yahoo"))
    snapshot_symbol = str(row.get("snapshot_symbol", row.get("symbol", "")))
    if market == "crypto":
        return _build_selected_snapshot_cached(
            instrument_id=str(snapshot_symbol).lower(),
            name=str(row.get("name", snapshot_symbol)),
            market="crypto",
            symbol=snapshot_symbol,
            provider=provider,
            timezone="Asia/Shanghai",
            horizon_unit="hour",
            horizon_steps=4,
            history_lookback_days=365,
        )
    if market == "cn_equity":
        code = str(row.get("code", row.get("symbol", ""))).lower()
        return _build_selected_snapshot_cached(
            instrument_id=code,
            name=str(row.get("name", row.get("symbol", ""))),
            market="cn_equity",
            symbol=str(row.get("symbol", "")),
            provider="yahoo",
            timezone="Asia/Shanghai",
            horizon_unit="day",
            horizon_steps=3,
            history_lookback_days=730,
        )
    return _build_selected_snapshot_cached(
        instrument_id=str(row.get("symbol", "")).lower(),
        name=str(row.get("name", row.get("symbol", ""))),
        market="us_equity",
        symbol=str(row.get("symbol", "")),
        provider="yahoo",
        timezone="America/New_York",
        horizon_unit="day",
        horizon_steps=3,
        history_lookback_days=730,
    )


def _render_crypto_page(
    *,
    processed_dir: Path,
    btc_live: float | None,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> None:
    st.header(_t("Crypto 页面", "Crypto Page"))
    cfg = _load_main_config_cached("configs/config.yaml")
    catalog = get_universe_catalog()["crypto"]
    pool_key = st.selectbox(
        _t("选择加密池", "Select Crypto Universe"),
        options=list(catalog.keys()),
        format_func=lambda k: catalog[k],
        key="crypto_pool_page",
    )
    uni = _load_universe_cached("crypto", pool_key)
    choice = st.selectbox(_t("选择币种", "Select Symbol"), uni["display"].tolist(), key="crypto_symbol_page")
    row = uni[uni["display"] == choice].iloc[0]
    snap = _build_snapshot_for_selected(market="crypto", row=row)

    snapshot_symbol = str(row.get("snapshot_symbol", "")).upper()
    symbol = str(row.get("symbol", "")).upper()
    is_btc = snapshot_symbol == "BTCUSDT" or symbol == "BTC"
    selected_snap = snap
    btc_main_snap_ready = False
    if is_btc:
        model_snap = _build_btc_model_snapshot_from_hourly(
            hourly_df=hourly_df,
            btc_live=btc_live,
            fallback_snapshot=snap,
        )
        if not model_snap.empty:
            selected_snap = model_snap
            btc_main_snap_ready = True

    # P0: 顶部决策卡（结论 -> 规则 -> 风险）
    selected_snap_for_plan = _ensure_policy_for_snapshot(selected_snap)
    bt_symbol = str(row.get("snapshot_symbol", row.get("symbol", "")))
    bt_policy_row = _find_policy_backtest_row(
        processed_dir=processed_dir,
        market="crypto",
        symbol=bt_symbol,
        aliases=[
            str(row.get("symbol", "")),
            str(row.get("snapshot_symbol", "")),
            str(row.get("name", "")),
        ],
    )
    if not selected_snap_for_plan.empty:
        plan_row = selected_snap_for_plan.iloc[0]
        plan = _build_trade_decision_plan(plan_row, cfg=cfg)
        horizon_unit, horizon_steps = _parse_horizon_label(str(plan_row.get("horizon_label", "4h")))
        horizon_hours = int(horizon_steps) if horizon_unit == "hour" else max(1, int(horizon_steps) * 24)
        rel_summary, _, rel_msg = _compute_recent_hourly_reliability(
            symbol=str(snapshot_symbol or symbol),
            horizon_hours=horizon_hours,
            window_days=30,
            cfg=cfg,
        )
        rel_7d, _, rel_7d_msg = _compute_recent_hourly_reliability(
            symbol=str(snapshot_symbol or symbol),
            horizon_hours=horizon_hours,
            window_days=7,
            cfg=cfg,
        )
        st.markdown("---")
        _render_trade_decision_summary(
            plan=plan,
            reliability_summary=rel_summary if not rel_msg else {},
            backtest_policy_row=bt_policy_row,
        )
        _render_rule_checklist(plan)
        if rel_msg:
            st.caption(_t(f"模型可信度补充：{rel_msg}", f"Reliability note: {rel_msg}"))
        elif not rel_7d_msg and rel_7d:
            acc_30 = _safe_float(rel_summary.get("accuracy"))
            acc_7 = _safe_float(rel_7d.get("accuracy"))
            brier_30 = _safe_float(rel_summary.get("brier"))
            brier_7 = _safe_float(rel_7d.get("brier"))
            if np.isfinite(acc_7) and np.isfinite(acc_30) and np.isfinite(brier_7) and np.isfinite(brier_30):
                if acc_7 + 0.03 < acc_30 or brier_7 > brier_30 + 0.03:
                    st.warning(
                        _t(
                            f"近期退化提示：7天表现弱于30天（Acc {acc_7:.1%} vs {acc_30:.1%}，"
                            f"Brier {brier_7:.3f} vs {brier_30:.3f}）。建议降低仓位。",
                            f"Recent degradation: 7d performance is weaker than 30d (Acc {acc_7:.1%} vs {acc_30:.1%}, "
                            f"Brier {brier_7:.3f} vs {brier_30:.3f}). Consider smaller position size.",
                        )
                    )
                else:
                    st.caption(
                        _t(
                            f"近期稳定：7天 vs 30天（Acc {acc_7:.1%}/{acc_30:.1%}，"
                            f"Brier {brier_7:.3f}/{brier_30:.3f}）。",
                            f"Recent stability: 7d vs 30d (Acc {acc_7:.1%}/{acc_30:.1%}, "
                            f"Brier {brier_7:.3f}/{brier_30:.3f}).",
                        )
                    )
        edge_abs = _safe_float(plan.get("edge_long"))
        q50 = _safe_float(plan.get("q50"))
        if np.isfinite(q50) and np.isfinite(edge_abs) and abs(q50) < abs(edge_abs):
            st.warning(_t("冲突提示：方向概率可能存在，但预期幅度不足以覆盖成本，建议观望或轻仓。", "Conflict: directional probability exists, but expected magnitude does not cover cost. Prefer wait or small size."))
        with st.expander(_t("决策公式说明（可审计）", "Decision Formula (auditable)"), expanded=False):
            st.markdown(
                _t(
                    "- Long 触发：`p_up>=阈值` 且 `edge_score>0` 且 `confidence>=阈值` 且 `风险非极高`\n"
                    "- Short 触发：`p_up<=阈值` 且 `edge_score>0` 且 `confidence>=阈值` 且 `允许做空`\n"
                    "- `edge_score = q50 - cost_bps/10000`（双边成本）\n"
                    "- Long 止损默认优先用 q10，下沿为正时使用 ATR 代理；止盈默认 q50，RR = (TP-entry)/(entry-SL)\n"
                    "- Short 止损默认优先用 q90，上沿为负时使用 ATR 代理；止盈默认 q50，RR = (entry-TP)/(SL-entry)",
                    "- Long trigger: `p_up>=threshold` and `edge_score>0` and `confidence>=threshold` and `risk != extreme`\n"
                    "- Short trigger: `p_up<=threshold` and `edge_score>0` and `confidence>=threshold` and `short_allowed`\n"
                    "- `edge_score = q50 - cost_bps/10000` (round-trip cost)\n"
                    "- Long SL uses q10 first; when lower band > 0, ATR proxy is used. TP uses q50 by default.\n"
                    "- Short SL uses q90 first; when upper band < 0, ATR proxy is used. TP uses q50 by default.",
                )
            )

    st.markdown("---")
    if btc_main_snap_ready:
        _render_snapshot_result(selected_snap, title_prefix="Crypto（BTC主模型）")
        st.caption(_t("当前 BTC 顶部卡片与下方 BTC 模型详情已统一口径：Hourly h=4。", "BTC top card and BTC model detail now use the same convention: Hourly h=4."))
    else:
        _render_snapshot_result(snap, title_prefix="Crypto")
        if not is_btc:
            st.caption(_t("当前币种卡片使用快照基线口径（非 BTC 主模型）。", "Current symbol card uses snapshot baseline convention (not the BTC primary model)."))

    _render_btc_model_detail_section(btc_live=btc_live, hourly_df=hourly_df, daily_df=daily_df)

    st.markdown("---")
    st.subheader(_t("模型效果解读", "Model Metrics Interpretation"))
    metrics_path = processed_dir / "metrics_walk_forward_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        _render_model_metrics_readable(metrics)
    else:
        st.info(_t("未找到模型评估结果（metrics_walk_forward_summary.csv）。", "Model metrics file not found (metrics_walk_forward_summary.csv)."))

    _render_symbol_backtest_section(
        processed_dir=processed_dir,
        market="crypto",
        symbol=bt_symbol,
        symbol_aliases=[
            str(row.get("symbol", "")),
            str(row.get("snapshot_symbol", "")),
            str(row.get("name", "")),
        ],
        provider=str(row.get("provider", "binance")),
        fallback_symbol=(
            f"{str(row.get('symbol', '')).upper()}USDT"
            if str(row.get("provider", "binance")) == "coingecko"
            else str(row.get("snapshot_symbol", ""))
        ),
        title=_t("Crypto 回测结果（该币种）", "Crypto Backtest (selected symbol)"),
    )


def _render_cn_page() -> None:
    st.header(_t("A股 页面", "CN A-share Page"))
    cfg = _load_main_config_cached("configs/config.yaml")
    catalog = get_universe_catalog()["cn_equity"]
    pool_key = st.selectbox(
        _t("选择A股股票池", "Select A-share Universe"),
        options=list(catalog.keys()),
        format_func=lambda k: catalog[k],
        key="cn_pool_page",
    )
    uni = _load_universe_cached("cn_equity", pool_key)
    choice = st.selectbox(_t("选择A股标的", "Select A-share Symbol"), uni["display"].tolist(), key="cn_symbol_page")
    row = uni[uni["display"] == choice].iloc[0]
    snap = _build_snapshot_for_selected(market="cn_equity", row=row)

    snap_for_plan = _ensure_policy_for_snapshot(snap)
    risk_col1, risk_col2 = st.columns(2)
    risk_profile_map = {"conservative": "保守", "standard": "标准", "aggressive": "激进"}
    risk_profile_key = risk_col1.selectbox(
        _t("风险偏好", "Risk Profile"),
        ["conservative", "standard", "aggressive"],
        index=1,
        format_func=lambda k: _t(risk_profile_map.get(k, "标准"), {"conservative": "Conservative", "standard": "Standard", "aggressive": "Aggressive"}.get(k, "Standard")),
        key="cn_risk_profile",
    )
    risk_profile = risk_profile_map.get(risk_profile_key, "标准")
    event_risk = bool(
        risk_col2.checkbox(_t("未来3天有重大事件风险（政策/财报/宏观）", "Major event risk in next 3 days (policy/earnings/macro)"), value=False, key="cn_event_risk")
    )

    bt_symbol = str(row.get("symbol", ""))
    bt_policy_row = _find_policy_backtest_row(
        processed_dir=Path("data/processed"),
        market="cn_equity",
        symbol=bt_symbol,
        aliases=[str(row.get("code", "")), str(row.get("name", ""))],
    )

    trade_plan: Dict[str, object] = {}
    rel_summary: Dict[str, float] = {}
    rel_compare = pd.DataFrame()
    rel_msg = ""
    if not snap_for_plan.empty:
        snap_row = snap_for_plan.iloc[0]
        h_unit, h_steps = _parse_horizon_label(str(snap_row.get("horizon_label", "3d")))
        h_days = max(1, int(h_steps) if h_unit == "day" else int(np.ceil(int(h_steps) / 24)))
        rel_summary, rel_compare, rel_msg = _compute_recent_symbol_reliability_cached(
            market="cn_equity",
            symbol=str(row.get("symbol", "")),
            provider="yahoo",
            fallback_symbol=str(row.get("symbol", "")),
            horizon_steps=h_days,
            window_days=30,
        )
        health_grade = _model_health_grade(rel_summary if not rel_msg else {})
        trade_plan = _build_trade_decision_plan(
            snap_row,
            cfg=cfg,
            risk_profile=risk_profile,
            model_health=health_grade,
            event_risk=event_risk,
        )
        st.markdown("---")
        _render_trade_decision_summary(
            plan=trade_plan,
            reliability_summary=rel_summary if not rel_msg else {},
            backtest_policy_row=bt_policy_row,
        )
        _render_rule_checklist(trade_plan)
        if rel_msg:
            st.caption(_t(f"模型健康补充：{rel_msg}", f"Reliability note: {rel_msg}"))
        with st.expander(_t("模型健康与校准（近30天）", "Model Health & Calibration (30d)"), expanded=False):
            if rel_msg:
                st.info(rel_msg)
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Brier", _format_float(rel_summary.get("brier"), 4))
                c2.metric(_t("Coverage(80%目标)", "Coverage (target 80%)"), _format_change_pct(rel_summary.get("coverage")).replace("+", ""))
                c3.metric("AUC", _format_float(rel_summary.get("auc"), 3))
                c4, c5, c6 = st.columns(3)
                c4.metric("Accuracy", _format_change_pct(rel_summary.get("accuracy")).replace("+", ""))
                c5.metric(_t("区间宽度", "Interval Width"), _format_change_pct(rel_summary.get("width")))
                c6.metric(_t("样本数", "Samples"), f"{int(rel_summary.get('samples', 0))}")
                st.caption(_t(f"模型健康等级：{_model_health_grade(rel_summary)}", f"Model health grade: {_model_health_text(_model_health_grade(rel_summary))}"))
                if not rel_compare.empty:
                    st.dataframe(
                        rel_compare.style.format(
                            {"Accuracy": "{:.2%}", "AUC": "{:.3f}", "Brier": "{:.4f}", "Coverage": "{:.2%}"},
                            na_rep="-",
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    st.markdown("---")
    _render_snapshot_result(snap, title_prefix=_t("A股", "CN A-share"), trade_plan=trade_plan if trade_plan else None)
    if not snap_for_plan.empty:
        with st.expander(_t("因子贡献摘要（Top 3 正/负）", "Factor Contribution Summary (Top 3 +/-)"), expanded=False):
            _render_factor_top_contributions(snap_for_plan.iloc[0])

    _render_symbol_backtest_section(
        processed_dir=Path("data/processed"),
        market="cn_equity",
        symbol=bt_symbol,
        symbol_aliases=[str(row.get("code", "")), str(row.get("name", ""))],
        provider="yahoo",
        fallback_symbol=str(row.get("symbol", "")),
        title=_t("A股 回测结果（该标的）", "CN A-share Backtest (selected symbol)"),
    )


def _render_us_page() -> None:
    st.header(_t("美股 页面", "US Equity Page"))
    cfg = _load_main_config_cached("configs/config.yaml")
    catalog = get_universe_catalog()["us_equity"]
    pool_key = st.selectbox(
        _t("选择美股股票池", "Select US Universe"),
        options=list(catalog.keys()),
        format_func=lambda k: catalog[k],
        key="us_pool_page",
    )
    uni = _load_universe_cached("us_equity", pool_key)
    choice = st.selectbox(_t("选择美股标的", "Select US Symbol"), uni["display"].tolist(), key="us_symbol_page")
    row = uni[uni["display"] == choice].iloc[0]
    snap = _build_snapshot_for_selected(market="us_equity", row=row)
    snap_for_plan = _ensure_policy_for_snapshot(snap)

    risk_col1, risk_col2 = st.columns(2)
    risk_profile_map = {"conservative": "保守", "standard": "标准", "aggressive": "激进"}
    risk_profile_key = risk_col1.selectbox(
        _t("风险偏好", "Risk Profile"),
        ["conservative", "standard", "aggressive"],
        index=1,
        format_func=lambda k: _t(risk_profile_map.get(k, "标准"), {"conservative": "Conservative", "standard": "Standard", "aggressive": "Aggressive"}.get(k, "Standard")),
        key="us_risk_profile",
    )
    risk_profile = risk_profile_map.get(risk_profile_key, "标准")
    event_risk = bool(
        risk_col2.checkbox(_t("未来3天有重大事件风险（财报/宏观）", "Major event risk in next 3 days (earnings/macro)"), value=False, key="us_event_risk")
    )

    bt_symbol = str(row.get("symbol", ""))
    bt_policy_row = _find_policy_backtest_row(
        processed_dir=Path("data/processed"),
        market="us_equity",
        symbol=bt_symbol,
        aliases=[str(row.get("name", ""))],
    )

    trade_plan: Dict[str, object] = {}
    rel_summary: Dict[str, float] = {}
    rel_compare = pd.DataFrame()
    rel_msg = ""
    if not snap_for_plan.empty:
        snap_row = snap_for_plan.iloc[0]
        h_unit, h_steps = _parse_horizon_label(str(snap_row.get("horizon_label", "3d")))
        h_days = max(1, int(h_steps) if h_unit == "day" else int(np.ceil(int(h_steps) / 24)))
        rel_summary, rel_compare, rel_msg = _compute_recent_symbol_reliability_cached(
            market="us_equity",
            symbol=str(row.get("symbol", "")),
            provider="yahoo",
            fallback_symbol=str(row.get("symbol", "")),
            horizon_steps=h_days,
            window_days=30,
        )
        health_grade = _model_health_grade(rel_summary if not rel_msg else {})
        trade_plan = _build_trade_decision_plan(
            snap_row,
            cfg=cfg,
            risk_profile=risk_profile,
            model_health=health_grade,
            event_risk=event_risk,
        )
        st.markdown("---")
        _render_trade_decision_summary(
            plan=trade_plan,
            reliability_summary=rel_summary if not rel_msg else {},
            backtest_policy_row=bt_policy_row,
        )
        _render_rule_checklist(trade_plan)
        if rel_msg:
            st.caption(_t(f"模型健康补充：{rel_msg}", f"Reliability note: {rel_msg}"))
        with st.expander(_t("模型健康与校准（近30天）", "Model Health & Calibration (30d)"), expanded=False):
            if rel_msg:
                st.info(rel_msg)
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Brier", _format_float(rel_summary.get("brier"), 4))
                c2.metric(_t("Coverage(80%目标)", "Coverage (target 80%)"), _format_change_pct(rel_summary.get("coverage")).replace("+", ""))
                c3.metric("AUC", _format_float(rel_summary.get("auc"), 3))
                c4, c5, c6 = st.columns(3)
                c4.metric("Accuracy", _format_change_pct(rel_summary.get("accuracy")).replace("+", ""))
                c5.metric(_t("区间宽度", "Interval Width"), _format_change_pct(rel_summary.get("width")))
                c6.metric(_t("样本数", "Samples"), f"{int(rel_summary.get('samples', 0))}")
                st.caption(_t(f"模型健康等级：{_model_health_grade(rel_summary)}", f"Model health grade: {_model_health_text(_model_health_grade(rel_summary))}"))
                if not rel_compare.empty:
                    st.dataframe(
                        rel_compare.style.format(
                            {"Accuracy": "{:.2%}", "AUC": "{:.3f}", "Brier": "{:.4f}", "Coverage": "{:.2%}"},
                            na_rep="-",
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    st.markdown("---")
    _render_snapshot_result(snap, title_prefix=_t("美股", "US Equity"), trade_plan=trade_plan if trade_plan else None)
    if not snap_for_plan.empty:
        with st.expander(_t("因子贡献摘要（Top 3 正/负）", "Factor Contribution Summary (Top 3 +/-)"), expanded=False):
            _render_factor_top_contributions(snap_for_plan.iloc[0])
    _render_symbol_backtest_section(
        processed_dir=Path("data/processed"),
        market="us_equity",
        symbol=bt_symbol,
        symbol_aliases=[str(row.get("name", ""))],
        provider="yahoo",
        fallback_symbol=str(row.get("symbol", "")),
        title=_t("美股 回测结果（该标的）", "US Equity Backtest (selected symbol)"),
    )


def _status_cn(status: str) -> str:
    mapping = {"Active": "可执行", "Watch": "观察", "Retired": "暂停"}
    return mapping.get(status, status)


def _status_text(status: str) -> str:
    cn = _status_cn(status)
    if _ui_lang() == "zh":
        return cn
    mapping = {"可执行": "Executable", "观察": "Watch", "暂停": "Paused"}
    return mapping.get(cn, str(status))


def _action_cn(action: str) -> str:
    mapping = {"Keep/Open": "持有或新开", "Monitor/Reduce": "观察或减仓", "Remove": "移除"}
    return mapping.get(action, action)


def _action_text(action: str) -> str:
    cn = _action_cn(action)
    if _ui_lang() == "zh":
        return cn
    mapping = {"持有或新开": "Keep/Open", "观察或减仓": "Monitor/Reduce", "移除": "Remove"}
    return mapping.get(cn, str(action))


def _alert_cn(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "无"
    mapping = {
        "prediction_unavailable": "预测不可用",
        "predicted_change_non_positive": "预测涨跌幅<=0",
        "history_too_short": "历史数据不足",
        "missing_rate_too_high": "缺失率过高",
        "liquidity_insufficient": "流动性不足",
        "risk_factor_missing": "风险因子缺失",
        "behavior_factor_missing": "行为因子缺失",
    }
    return "；".join(mapping.get(x, x) for x in text.split(";") if x.strip())


def _market_cn(market: str) -> str:
    mapping = {"crypto": "加密", "cn_equity": "A股", "us_equity": "美股"}
    return mapping.get(str(market), str(market))


def _market_text(market: str) -> str:
    cn = _market_cn(market)
    if _ui_lang() == "zh":
        return cn
    mapping = {"加密": "Crypto", "A股": "CN A-share", "美股": "US Equity"}
    return mapping.get(cn, str(market))


def _split_alert_codes(text: object) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(";") if x.strip()]


def _alert_tag_cn(code: str) -> str:
    mapping = {
        "prediction_unavailable": "无预测",
        "predicted_change_non_positive": "净优势不足",
        "history_too_short": "历史不足",
        "missing_rate_too_high": "缺失偏高",
        "liquidity_insufficient": "流动性不足",
        "risk_factor_missing": "风险因子缺失",
        "behavior_factor_missing": "行为因子缺失",
    }
    return mapping.get(str(code), str(code))


def _alert_tag_text(code: str) -> str:
    cn = _alert_tag_cn(code)
    if _ui_lang() == "zh":
        return cn
    mapping = {
        "无预测": "No Prediction",
        "净优势不足": "Insufficient Edge",
        "历史不足": "Insufficient History",
        "缺失偏高": "High Missing Rate",
        "流动性不足": "Low Liquidity",
        "风险因子缺失": "Missing Risk Factors",
        "行为因子缺失": "Missing Behavior Factors",
    }
    return mapping.get(cn, str(code))


def _alert_fix_action_cn(code: str) -> str:
    mapping = {
        "prediction_unavailable": "补齐该标的预测流水后重跑 tracking。",
        "predicted_change_non_positive": "等待净优势转正或切换到反向/观望策略。",
        "history_too_short": "补足历史K线（建议>=365日线，或>=60小时级样本）。",
        "missing_rate_too_high": "修复数据缺失（更换源/补拉缺口）后再评估。",
        "liquidity_insufficient": "提高流动性后再纳入，或切换更高流动性标的。",
        "risk_factor_missing": "补齐市值/估值/成长因子快照。",
        "behavior_factor_missing": "补齐OHLCV后重算动量/反转/低波动因子。",
    }
    return mapping.get(str(code), "检查数据源与特征计算流程。")


def _alert_fix_action_text(code: str) -> str:
    cn = _alert_fix_action_cn(code)
    if _ui_lang() == "zh":
        return cn
    mapping = {
        "补齐该标的预测流水后重跑 tracking。": "Backfill prediction pipeline for this symbol and rerun tracking.",
        "等待净优势转正或切换到反向/观望策略。": "Wait for positive net edge, or switch to reverse/wait policy.",
        "补足历史K线（建议>=365日线，或>=60小时级样本）。": "Backfill historical bars (>=365 daily or >=60 hourly samples).",
        "修复数据缺失（更换源/补拉缺口）后再评估。": "Fix data gaps (source switch/backfill) before reevaluation.",
        "提高流动性后再纳入，或切换更高流动性标的。": "Include after liquidity improves, or switch to more liquid symbols.",
        "补齐市值/估值/成长因子快照。": "Backfill market-cap/value/growth factor snapshots.",
        "补齐OHLCV后重算动量/反转/低波动因子。": "Backfill OHLCV and recompute momentum/reversal/low-vol factors.",
        "检查数据源与特征计算流程。": "Check data source and feature computation pipeline.",
    }
    return mapping.get(cn, cn)


def _confidence_bucket_cn(score: float) -> str:
    if not np.isfinite(score):
        return "未知"
    if score >= 80:
        return "高"
    if score >= 60:
        return "中"
    return "低"


def _confidence_bucket_text(score: float) -> str:
    cn = _confidence_bucket_cn(score)
    if _ui_lang() == "zh":
        return cn
    mapping = {"未知": "Unknown", "高": "High", "中": "Medium", "低": "Low"}
    return mapping.get(cn, cn)


def _risk_level_from_score(score: float) -> str:
    if not np.isfinite(score):
        return "high"
    if score <= 0.35:
        return "low"
    if score <= 0.55:
        return "medium"
    if score <= 0.75:
        return "high"
    return "extreme"


def _prepare_tracking_table(
    ranked: pd.DataFrame,
    coverage: pd.DataFrame,
    cost_bps: float,
    snapshot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if ranked.empty:
        return ranked.copy()

    work = ranked.copy()
    for c in ["market", "instrument_id", "name", "display", "symbol", "alerts", "policy_reason"]:
        if c not in work.columns:
            work[c] = ""
    work["market"] = work["market"].astype(str)
    work["instrument_id"] = work["instrument_id"].astype(str)

    if not coverage.empty:
        cov = coverage.copy()
        for c in ["market", "instrument_id"]:
            if c in cov.columns:
                cov[c] = cov[c].astype(str)
        cov_cols = [
            "market",
            "instrument_id",
            "prediction_available",
            "history_missing_rate",
            "hard_filter_pass",
            "alerts",
        ]
        cov_cols = [c for c in cov_cols if c in cov.columns]
        key_cols = [c for c in ["market", "instrument_id"] if c in cov_cols]
        cov = cov[cov_cols]
        if key_cols:
            cov = cov.drop_duplicates(subset=key_cols, keep="last")
            work = work.merge(cov, on=key_cols, how="left", suffixes=("", "_cov"))
        else:
            work["prediction_available"] = np.nan
            work["history_missing_rate"] = np.nan
            work["hard_filter_pass"] = np.nan
    else:
        work["prediction_available"] = np.nan
        work["history_missing_rate"] = np.nan
        work["hard_filter_pass"] = np.nan

    snap_df = snapshot.copy() if isinstance(snapshot, pd.DataFrame) else pd.DataFrame()
    if not snap_df.empty:
        for c in ["market", "instrument_id"]:
            if c in snap_df.columns:
                snap_df[c] = snap_df[c].astype(str)
        snap_keep = [
            "market",
            "instrument_id",
            "provider",
            "symbol",
            "snapshot_symbol",
            "horizon_label",
            "current_price",
            "predicted_price",
            "predicted_change_pct",
            "q10_change_pct",
            "q50_change_pct",
            "q90_change_pct",
            "p_up",
            "confidence_score",
            "volatility_score",
            "policy_allow_short",
            "policy_reason",
            "news_score_30m",
            "news_score_120m",
            "news_score_1440m",
            "news_count_30m",
            "news_burst_zscore",
            "news_pos_neg_ratio",
            "news_conflict_score",
            "news_event_risk",
            "news_gate_pass",
            "news_risk_level",
            "news_reason_codes",
            "news_latest_headlines",
            "news_latest_providers",
        ]
        snap_keep = [c for c in snap_keep if c in snap_df.columns]
        if {"market", "instrument_id"}.issubset(set(snap_keep)):
            snap_df = snap_df[snap_keep].drop_duplicates(subset=["market", "instrument_id"], keep="last")
            work = work.merge(snap_df, on=["market", "instrument_id"], how="left", suffixes=("", "_snap"))
            coalesce_cols = [
                "provider",
                "symbol",
                "snapshot_symbol",
                "horizon_label",
                "current_price",
                "predicted_price",
                "predicted_change_pct",
                "q10_change_pct",
                "q50_change_pct",
                "q90_change_pct",
                "p_up",
                "confidence_score",
                "volatility_score",
                "policy_allow_short",
                "policy_reason",
                "news_score_30m",
                "news_score_120m",
                "news_score_1440m",
                "news_count_30m",
                "news_burst_zscore",
                "news_pos_neg_ratio",
                "news_conflict_score",
                "news_event_risk",
                "news_gate_pass",
                "news_risk_level",
                "news_reason_codes",
                "news_latest_headlines",
                "news_latest_providers",
            ]
            for c in coalesce_cols:
                sc = f"{c}_snap"
                if sc not in work.columns:
                    continue
                if c not in work.columns:
                    work[c] = work[sc]
                else:
                    work[c] = work[c].where(work[c].notna(), work[sc])

    work["predicted_change_pct"] = pd.to_numeric(work.get("predicted_change_pct"), errors="coerce")
    work["q10_change_pct"] = pd.to_numeric(work.get("q10_change_pct"), errors="coerce")
    work["q50_change_pct"] = pd.to_numeric(work.get("q50_change_pct"), errors="coerce")
    work["q90_change_pct"] = pd.to_numeric(work.get("q90_change_pct"), errors="coerce")
    work["p_up"] = pd.to_numeric(work.get("p_up"), errors="coerce")
    work["confidence_score"] = pd.to_numeric(work.get("confidence_score"), errors="coerce")
    work["volatility_score"] = pd.to_numeric(work.get("volatility_score"), errors="coerce")
    work["current_price"] = pd.to_numeric(work.get("current_price"), errors="coerce")
    work["predicted_price"] = pd.to_numeric(work.get("predicted_price"), errors="coerce")
    work["policy_expected_edge_pct"] = pd.to_numeric(work.get("policy_expected_edge_pct"), errors="coerce")
    work["total_score"] = pd.to_numeric(work.get("total_score"), errors="coerce")
    work["liquidity_score"] = pd.to_numeric(work.get("liquidity_score"), errors="coerce")
    work["data_quality_score"] = pd.to_numeric(work.get("data_quality_score"), errors="coerce")
    work["history_score"] = pd.to_numeric(work.get("history_score"), errors="coerce")
    work["coverage_score"] = pd.to_numeric(work.get("coverage_score"), errors="coerce")
    work["factor_support_count"] = pd.to_numeric(work.get("factor_support_count"), errors="coerce").fillna(0.0)
    work["history_missing_rate"] = pd.to_numeric(work.get("history_missing_rate"), errors="coerce").fillna(0.0).clip(lower=0.0)
    work["news_score_30m"] = pd.to_numeric(work.get("news_score_30m"), errors="coerce").fillna(0.0)
    work["news_score_120m"] = pd.to_numeric(work.get("news_score_120m"), errors="coerce").fillna(0.0)
    work["news_score_1440m"] = pd.to_numeric(work.get("news_score_1440m"), errors="coerce").fillna(0.0)
    work["news_count_30m"] = pd.to_numeric(work.get("news_count_30m"), errors="coerce").fillna(0.0)
    work["news_burst_zscore"] = pd.to_numeric(work.get("news_burst_zscore"), errors="coerce").fillna(0.0)
    work["news_pos_neg_ratio"] = pd.to_numeric(work.get("news_pos_neg_ratio"), errors="coerce").fillna(1.0)
    work["news_conflict_score"] = pd.to_numeric(work.get("news_conflict_score"), errors="coerce").fillna(0.0)
    news_risk_series = work.get("news_risk_level", pd.Series(["low"] * len(work), index=work.index))
    work["news_risk_level"] = news_risk_series.fillna("low").astype(str).str.lower()
    news_reason_series = work.get("news_reason_codes", pd.Series([""] * len(work), index=work.index))
    work["news_reason_codes"] = news_reason_series.fillna("").astype(str)
    news_head_series = work.get("news_latest_headlines", pd.Series([""] * len(work), index=work.index))
    work["news_latest_headlines"] = news_head_series.fillna("").astype(str)
    news_provider_series = work.get("news_latest_providers", pd.Series([""] * len(work), index=work.index))
    work["news_latest_providers"] = news_provider_series.fillna("").astype(str)

    pred_flag_raw = work.get("prediction_available", pd.Series([np.nan] * len(work), index=work.index))
    pred_flag = (
        pred_flag_raw.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    )
    pred_flag = pred_flag.where(pred_flag.notna(), work["predicted_change_pct"].notna())
    hard_flag_raw = work.get("hard_filter_pass", pd.Series([np.nan] * len(work), index=work.index))
    hard_flag = hard_flag_raw.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    hard_flag = hard_flag.fillna(True)
    news_gate_raw = work.get("news_gate_pass", pd.Series([np.nan] * len(work), index=work.index))
    news_gate_flag = news_gate_raw.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    news_gate_flag = news_gate_flag.fillna(True)
    news_event_raw = work.get("news_event_risk", pd.Series([np.nan] * len(work), index=work.index))
    news_event_flag = news_event_raw.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    news_event_flag = news_event_flag.fillna(False)
    work["prediction_available_flag"] = pred_flag.astype(bool)
    work["hard_filter_pass_flag"] = hard_flag.astype(bool)
    work["news_gate_pass_flag"] = news_gate_flag.astype(bool)
    work["news_event_risk_flag"] = news_event_flag.astype(bool)

    cost_pct = float(cost_bps) / 10000.0
    fallback_edge = work["predicted_change_pct"] - cost_pct
    work["edge_score"] = work["policy_expected_edge_pct"].where(work["policy_expected_edge_pct"].notna(), fallback_edge)
    work["edge_score_short"] = (-work["predicted_change_pct"]) - cost_pct

    total_norm = (work["total_score"] / 100.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    quality_norm = (work["data_quality_score"] / 20.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    factor_norm = (work["factor_support_count"] / 3.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    conviction = (work["predicted_change_pct"].abs() * 20.0).clip(lower=0.0, upper=1.0).fillna(0.0)

    confidence = (
        100.0
        * (
            0.48 * total_norm
            + 0.20 * quality_norm
            + 0.15 * factor_norm
            + 0.17 * conviction
        )
        - (work["history_missing_rate"] * 30.0)
        - ((~work["hard_filter_pass_flag"]).astype(float) * 10.0)
    )
    work["confidence_score_est"] = confidence.clip(lower=0.0, upper=100.0)
    work["confidence_score"] = work["confidence_score"].where(work["confidence_score"].notna(), work["confidence_score_est"])

    risk_score = (
        (1.0 - total_norm) * 0.45
        + (1.0 - quality_norm) * 0.20
        + (1.0 - factor_norm) * 0.15
        + (1.0 - conviction) * 0.08
        + work["history_missing_rate"].clip(upper=1.0) * 0.12
        + ((~work["hard_filter_pass_flag"]).astype(float) * 0.22)
    ).clip(lower=0.0, upper=1.0)
    work["risk_score"] = risk_score
    work["risk_level"] = work["risk_score"].map(_risk_level_from_score)
    work["risk_level_cn"] = work["risk_level"].map(_risk_cn)
    work["risk_rank"] = work["risk_level"].map({"low": 0, "medium": 1, "high": 2, "extreme": 3}).fillna(9)

    denom = work["risk_score"].clip(lower=0.08)
    work["edge_risk"] = work["edge_score"] / denom
    work["edge_risk_short"] = work["edge_score_short"] / denom

    work["市场"] = work["market"].map(_market_text)
    work["原始状态"] = work["status"].map(_status_text).fillna("-")
    work["建议动作"] = work["recommended_action"].map(_action_text).fillna("-")
    work["策略动作"] = work["policy_action"].map(_policy_action_cn).fillna("观望")
    work["display_name"] = work["display"].fillna("").astype(str)
    work["display_name"] = work["display_name"].where(work["display_name"].str.strip() != "", work["name"].astype(str))
    work["display_name"] = work["display_name"].where(work["display_name"].str.strip() != "", work["symbol"].astype(str))
    work["track_key"] = work["market"].astype(str) + "|" + work["instrument_id"].astype(str)

    work["active_edge_score"] = np.where(work["策略动作"] == "做空", work["edge_score_short"], work["edge_score"])
    work["active_edge_risk"] = np.where(work["策略动作"] == "做空", work["edge_risk_short"], work["edge_risk"])
    work["inputs_ready"] = (
        work["prediction_available_flag"].astype(bool)
        & work["hard_filter_pass_flag"].astype(bool)
        & work["predicted_change_pct"].notna()
    )

    def _status_rule_row(r: pd.Series) -> Tuple[str, str]:
        conf_v = _safe_float(r.get("confidence_score_est"))
        edge_v = _safe_float(r.get("active_edge_score"))
        risk_v = str(r.get("risk_level", "high"))
        inputs_ok = bool(r.get("inputs_ready", False))
        hard_ok = bool(r.get("hard_filter_pass_flag", False))
        news_ok = bool(r.get("news_gate_pass_flag", True))
        news_event = bool(r.get("news_event_risk_flag", False))
        news_risk = str(r.get("news_risk_level", "low")).lower()
        tags: List[str] = []
        tags.append("inputs_ready" if inputs_ok else "insufficient_inputs")
        tags.append("edge_ok" if np.isfinite(edge_v) and edge_v > 0 else _t("edge不足", "edge_low"))
        tags.append("confidence_ok" if np.isfinite(conf_v) and conf_v >= 70 else _t("confidence偏低", "confidence_low"))
        tags.append("risk_ok" if risk_v in {"low", "medium"} else _t("risk偏高", "risk_high"))
        tags.append("news_ok" if news_ok else "news_block")
        if news_event:
            tags.append("news_event_risk")
        if news_risk not in {"low", "medium"}:
            tags.append("news_risk_high")
        if not hard_ok:
            tags.append("hard_filter_fail")

        if (
            inputs_ok
            and hard_ok
            and news_ok
            and np.isfinite(conf_v)
            and conf_v >= 70
            and np.isfinite(edge_v)
            and edge_v > 0
            and risk_v in {"low", "medium"}
            and news_risk in {"low", "medium"}
        ):
            return "可执行", " | ".join(tags)
        if (not inputs_ok) or (not hard_ok) or (not news_ok) or news_event or (np.isfinite(conf_v) and conf_v < 40) or risk_v == "extreme":
            return "暂停", " | ".join(tags)
        return "观察", " | ".join(tags)

    rule_tuple = work.apply(_status_rule_row, axis=1)
    work["状态(规则)"] = rule_tuple.map(lambda x: x[0] if isinstance(x, tuple) else "观察")
    work["状态触发标签"] = rule_tuple.map(lambda x: x[1] if isinstance(x, tuple) else "-")

    def _row_fix_actions(r: pd.Series) -> str:
        codes = _split_alert_codes(r.get("alerts"))
        if not codes and "alerts_cov" in r.index:
            codes = _split_alert_codes(r.get("alerts_cov"))
        if not codes:
            status_rule = str(r.get("状态(规则)", "观察"))
            if status_rule == "可执行":
                return _t("无需修复（当前可执行）。", "No fix needed (currently executable).")
            if status_rule == "观察":
                return _t("继续跟踪，等待净优势/置信度进一步改善。", "Keep tracking; wait for better net edge/confidence.")
            return _t("检查该标的数据源与预测流水是否齐全。", "Check whether data source and prediction pipeline are complete.")
        actions = []
        for c in codes:
            act = _alert_fix_action_text(c)
            if act not in actions:
                actions.append(act)
        return "；".join(actions) if _ui_lang() == "zh" else "; ".join(actions)

    def _row_alert_tags(r: pd.Series) -> str:
        codes = _split_alert_codes(r.get("alerts"))
        if not codes and "alerts_cov" in r.index:
            codes = _split_alert_codes(r.get("alerts_cov"))
        if not codes:
            return _t("无", "None")
        tags = []
        for c in codes:
            t = _alert_tag_text(c)
            if t not in tags:
                tags.append(t)
        return "、".join(tags) if _ui_lang() == "zh" else " / ".join(tags)

    def _short_reason_line(r: pd.Series) -> str:
        action_tag = {"做多": "long_signal", "做空": "short_signal", "观望": "wait_signal"}.get(
            str(r.get("策略动作", "观望")),
            "wait_signal",
        )
        conf_txt = _confidence_bucket_text(_safe_float(r.get("confidence_score_est")))
        risk_txt = _risk_text(str(r.get("risk_level", "high")))
        reason_raw = _format_reason_tokens_cn(r.get("policy_reason", "-"))
        reason_core = "-"
        if isinstance(reason_raw, str) and reason_raw.strip() and reason_raw != "-":
            reason_core = reason_raw.split("；")[0].split(";")[0].strip()
        if reason_core == "-":
            return (
                f"{action_tag} + {conf_txt}置信度 + {risk_txt}风险"
                if _ui_lang() == "zh"
                else f"{action_tag} + {conf_txt} confidence + {risk_txt} risk"
            )
        return (
            f"{action_tag} + {reason_core} + {conf_txt}置信度"
            if _ui_lang() == "zh"
            else f"{action_tag} + {reason_core} + {conf_txt} confidence"
        )

    work["告警标签"] = work.apply(_row_alert_tags, axis=1)
    work["修复动作"] = work.apply(_row_fix_actions, axis=1)
    work["短原因"] = work.apply(_short_reason_line, axis=1)
    return work


def _render_tracking_page(processed_dir: Path) -> None:
    st.header(_t("Selection / Research / Tracking 页面", "Selection / Research / Tracking"))
    tracking_dir = processed_dir / "tracking"
    ranked = _load_csv(tracking_dir / "ranked_universe.csv")
    actions = _load_csv(tracking_dir / "tracking_actions.csv")
    coverage = _load_csv(tracking_dir / "coverage_matrix.csv")
    snapshot = _load_csv(tracking_dir / "tracking_snapshot.csv")
    report_path = tracking_dir / "data_quality_report.md"

    if ranked.empty:
        st.info(
            _t(
                "还没有 tracking 结果。请先运行 `python -m src.markets.tracking --config configs/config.yaml`。",
                "No tracking result yet. Run `python -m src.markets.tracking --config configs/config.yaml` first.",
            )
        )
        return

    ctrl1, ctrl2 = st.columns([1, 1])
    cost_bps = float(
        ctrl1.number_input(
            _t("成本估计（bps，双边：开+平）", "Cost estimate (bps, round-trip: open+close)"),
            min_value=0.0,
            max_value=200.0,
            value=8.0,
            step=1.0,
            key="track_cost_bps",
        )
    )
    top_k = int(ctrl2.slider(_t("Top Opportunities 每组条数", "Top Opportunities rows/group"), 5, 10, 5, 1, key="track_top_opps_k"))

    prepared = _prepare_tracking_table(ranked=ranked, coverage=coverage, cost_bps=cost_bps, snapshot=snapshot)
    if prepared.empty:
        st.info(_t("tracking 数据为空。", "Tracking data is empty."))
        return

    cfg_main = _load_main_config_cached("configs/config.yaml")

    def _model_health_from_conf(conf_value: object) -> str:
        c = _safe_float(conf_value)
        if np.isfinite(c) and c >= 80:
            return "good"
        if np.isfinite(c) and c >= 60:
            return "medium"
        return "poor"

    def _enrich_plan_row(r: pd.Series) -> pd.Series:
        x = r.copy()
        conf_s = _safe_float(x.get("confidence_score"))
        conf_e = _safe_float(x.get("confidence_score_est"))
        if not np.isfinite(conf_s) and np.isfinite(conf_e):
            x["confidence_score"] = conf_e
        p_up = _safe_float(x.get("p_up"))
        if not np.isfinite(p_up):
            pa = str(x.get("policy_action", "Flat"))
            if pa == "Long":
                x["p_up"] = 0.55
            elif pa == "Short":
                x["p_up"] = 0.45
            else:
                x["p_up"] = 0.50
        q50 = _safe_float(x.get("q50_change_pct"))
        if not np.isfinite(q50):
            x["q50_change_pct"] = _safe_float(x.get("predicted_change_pct"))
            q50 = _safe_float(x.get("q50_change_pct"))
        q10 = _safe_float(x.get("q10_change_pct"))
        q90 = _safe_float(x.get("q90_change_pct"))
        if not (np.isfinite(q10) and np.isfinite(q90)):
            vol = _safe_float(x.get("volatility_score"))
            spread = max(0.01, abs(vol) if np.isfinite(vol) else 0.0, abs(q50) * 1.5 if np.isfinite(q50) else 0.01)
            x["q10_change_pct"] = (q50 - spread) if np.isfinite(q50) else -spread
            x["q90_change_pct"] = (q50 + spread) if np.isfinite(q50) else spread

        plan = _build_trade_decision_plan(
            x,
            cfg=cfg_main,
            risk_profile="standard",
            model_health=_model_health_from_conf(x.get("confidence_score")),
            event_risk=False,
            persist_touch=True,
            processed_dir=processed_dir,
        )
        return pd.Series(
            {
                "plan_entry": plan.get("entry"),
                "plan_stop_loss": plan.get("stop_loss"),
                "plan_take_profit": plan.get("take_profit"),
                "plan_rr": plan.get("rr"),
                "plan_action_text": plan.get("action_cn"),
                "plan_action_reason": plan.get("action_reason"),
                "plan_trade_status": plan.get("trade_status_text"),
                "plan_trade_status_note": plan.get("trade_status_note"),
                "plan_entry_gap_pct": plan.get("entry_gap_pct"),
                "plan_entry_touched": plan.get("entry_touched"),
                "plan_entry_touched_at": plan.get("entry_touched_at"),
                "plan_price_source": plan.get("price_source"),
                "plan_price_ts_market": plan.get("price_timestamp_market"),
                "plan_price_ts_utc": plan.get("price_timestamp_utc"),
            }
        )

    plan_enriched = prepared.apply(_enrich_plan_row, axis=1)
    prepared = pd.concat([prepared, plan_enriched], axis=1)

    total = len(prepared)
    executable_n = int((prepared["状态(规则)"] == "可执行").sum())
    watch_n = int((prepared["状态(规则)"] == "观察").sum())
    paused_n = int((prepared["状态(规则)"] == "暂停").sum())
    pred_cov = float(prepared["prediction_available_flag"].astype(bool).mean())
    hard_pass = float(prepared["hard_filter_pass_flag"].astype(bool).mean())
    avg_missing = float(pd.to_numeric(prepared["history_missing_rate"], errors="coerce").fillna(0.0).mean())

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(_t("候选总数", "Total Candidates"), f"{total}")
    k2.metric(_t("可执行", "Executable"), f"{executable_n}")
    k3.metric(_t("观察", "Watch"), f"{watch_n}")
    k4.metric(_t("暂停", "Paused"), f"{paused_n}")
    k5.metric(_t("预测可用率", "Prediction Coverage"), f"{pred_cov:.1%}")
    k6.metric(_t("硬门槛通过率", "Hard-filter Pass Rate"), f"{hard_pass:.1%}")
    st.caption(
        _t(
            f"平均缺失率：{avg_missing:.1%} | edge_score口径：优先 `policy_expected_edge_pct`，缺失时退化为 `predicted_change_pct - cost_bps/10000`。",
            f"Average missing rate: {avg_missing:.1%} | edge_score uses `policy_expected_edge_pct` first, fallback to `predicted_change_pct - cost_bps/10000`.",
        )
    )

    st.subheader(_t("Top Opportunities（先看这里）", "Top Opportunities (start here)"))
    top_long = prepared[(prepared["状态(规则)"] == "可执行") & (prepared["策略动作"] == "做多")].sort_values(
        "active_edge_risk", ascending=False
    )
    top_short = prepared[(prepared["状态(规则)"] == "可执行") & (prepared["策略动作"] == "做空")].sort_values(
        "active_edge_risk", ascending=False
    )
    top_watch = prepared[prepared["状态(规则)"] == "观察"].sort_values("active_edge_risk", ascending=False)

    def _show_opps(df: pd.DataFrame) -> pd.DataFrame:
        show = df.head(top_k).copy()
        if show.empty:
            return show
        out = pd.DataFrame(
            {
                "market": show["市场"],
                "symbol": show["display_name"],
                "final_action": show["策略动作"].map(_policy_action_text),
                _t("净优势(edge)", "edge_score"): show["active_edge_score"].map(_format_change_pct),
                _t("盈亏比(RR)", "RR"): show["plan_rr"].map(lambda x: _format_float(x, 2)),
                _t("风险", "risk"): show["risk_level"].map(_risk_text),
                _t("置信度", "confidence"): show["confidence_score_est"].map(lambda x: _format_float(x, 1)),
                "reason": show["短原因"],
            }
        )
        return out

    c_long, c_short, c_watch = st.columns(3)
    with c_long:
        st.markdown(f"**{_t('Top 做多机会（可执行）', 'Top Long Opportunities (Executable)')}**")
        show = _show_opps(top_long)
        if show.empty:
            st.info(_t("暂无满足条件的做多机会。", "No executable long opportunities."))
        else:
            st.dataframe(show, use_container_width=True, hide_index=True)
    with c_short:
        st.markdown(f"**{_t('Top 做空机会（可执行）', 'Top Short Opportunities (Executable)')}**")
        show = _show_opps(top_short)
        if show.empty:
            st.info(_t("暂无满足条件的做空机会。", "No executable short opportunities."))
        else:
            st.dataframe(show, use_container_width=True, hide_index=True)
    with c_watch:
        st.markdown(f"**{_t('Top 观察名单（潜力但未满足）', 'Top Watchlist (potential, not yet executable)')}**")
        show = _show_opps(top_watch)
        if show.empty:
            st.info(_t("暂无观察名单。", "Watchlist is empty."))
        else:
            st.dataframe(show, use_container_width=True, hide_index=True)

    with st.expander(_t("状态判定规则（可执行/观察/暂停）", "Status Rule (Executable/Watch/Paused)"), expanded=False):
        st.markdown(
            _t(
                "- `可执行`：`confidence>=70` 且 `active_edge_score>0` 且 `risk_level<=中` 且 `inputs_ready=true`。\n"
                "- `观察`：方向存在但 edge/置信度/风险尚未满足执行门槛。\n"
                "- `暂停`：insufficient_inputs、hard_filter_fail、风险极高或置信度过低。\n"
                "- 触发标签示例：`inputs_ready | edge_ok | confidence_ok | risk_ok`。",
                "- `Executable`: `confidence>=70` and `active_edge_score>0` and `risk_level<=medium` and `inputs_ready=true`.\n"
                "- `Watch`: direction exists but edge/confidence/risk do not pass execution threshold.\n"
                "- `Paused`: insufficient_inputs, hard_filter_fail, extreme risk, or very low confidence.\n"
                "- Example tags: `inputs_ready | edge_ok | confidence_ok | risk_ok`.",
            )
        )

    st.subheader(_t("Screener（筛选 + 排序）", "Screener (Filter + Sort)"))
    flt1, flt2, flt3, flt4 = st.columns([1, 1, 1, 1])
    all_opt = _t("全部", "All")
    market_options = [all_opt] + sorted(prepared["市场"].dropna().unique().tolist())
    status_labels = {
        "all": all_opt,
        "exec": _t("可执行", "Executable"),
        "watch": _t("观察", "Watch"),
        "paused": _t("暂停", "Paused"),
    }
    action_labels = {
        "all": all_opt,
        "long": _t("做多", "Long"),
        "short": _t("做空", "Short"),
        "wait": _t("观望", "Wait"),
    }
    preset_labels = {
        "all": all_opt,
        "conservative": _t("保守策略", "Conservative"),
        "aggressive": _t("激进策略", "Aggressive"),
        "low_risk": _t("低风险", "Low Risk"),
        "high_conf": _t("高置信度", "High Confidence"),
    }
    market_sel = flt1.selectbox(_t("市场", "Market"), market_options, 0, key="track_market_v2")
    status_sel = flt2.selectbox(
        _t("规则状态", "Rule Status"),
        options=list(status_labels.keys()),
        index=0,
        format_func=lambda k: status_labels.get(k, str(k)),
        key="track_status_v2",
    )
    action_sel = flt3.selectbox(
        _t("策略动作", "Policy Action"),
        options=list(action_labels.keys()),
        index=0,
        format_func=lambda k: action_labels.get(k, str(k)),
        key="track_action_v2",
    )
    preset_sel = flt4.selectbox(
        _t("快速预设", "Quick Preset"),
        options=list(preset_labels.keys()),
        index=0,
        format_func=lambda k: preset_labels.get(k, str(k)),
        key="track_preset_v2",
    )

    chip1, chip2, chip3, chip4, chip5 = st.columns(5)
    only_exec = chip1.toggle(_t("只看可执行", "Executable only"), value=False, key="track_chip_exec")
    only_long = chip2.toggle(_t("只看做多", "Long only"), value=False, key="track_chip_long")
    only_short = chip3.toggle(_t("只看做空", "Short only"), value=False, key="track_chip_short")
    exclude_paused = chip4.toggle(_t("排除暂停", "Exclude paused"), value=False, key="track_chip_no_pause")
    high_liq_only = chip5.toggle(_t("高流动性", "High liquidity"), value=False, key="track_chip_liq")

    sort_col1, sort_col2, sort_col3 = st.columns([2, 1, 1])
    sort_map = {
        _t("按 edge_risk（默认）", "By edge_risk (default)"): "active_edge_risk",
        _t("按 edge_score", "By edge_score"): "active_edge_score",
        _t("按 confidence", "By confidence"): "confidence_score_est",
        _t("按 risk_level", "By risk_level"): "risk_rank",
        _t("按 liquidity_score", "By liquidity_score"): "liquidity_score",
        _t("按 data_quality_score", "By data_quality_score"): "data_quality_score",
    }
    sort_label = sort_col1.selectbox(_t("排序方式", "Sort by"), list(sort_map.keys()), 0, key="track_sort_key")
    sort_key = sort_map[sort_label]
    sort_desc = bool(sort_col2.toggle(_t("降序", "Descending"), value=True, key="track_sort_desc"))
    top_n = int(sort_col3.slider(_t("主表展示前N", "Top N in main table"), 20, 500, 120, 10, key="track_topn_v2"))

    view = prepared.copy()
    if market_sel != all_opt:
        view = view[view["市场"] == market_sel]
    if status_sel == "exec":
        view = view[view["状态(规则)"] == "可执行"]
    elif status_sel == "watch":
        view = view[view["状态(规则)"] == "观察"]
    elif status_sel == "paused":
        view = view[view["状态(规则)"] == "暂停"]

    if action_sel == "long":
        view = view[view["策略动作"] == "做多"]
    elif action_sel == "short":
        view = view[view["策略动作"] == "做空"]
    elif action_sel == "wait":
        view = view[view["策略动作"] == "观望"]

    if preset_sel == "conservative":
        view = view[
            (view["状态(规则)"] == "可执行")
            & (view["risk_level"].isin(["low", "medium"]))
            & (view["confidence_score_est"] >= 75)
            & (view["active_edge_score"] > 0)
        ]
    elif preset_sel == "aggressive":
        view = view[
            (view["状态(规则)"] != "暂停")
            & (view["active_edge_score"].abs() > 0)
            & (view["confidence_score_est"] >= 55)
        ]
    elif preset_sel == "low_risk":
        view = view[view["risk_level"].isin(["low", "medium"])]
    elif preset_sel == "high_conf":
        view = view[view["confidence_score_est"] >= 80]

    if only_exec:
        view = view[view["状态(规则)"] == "可执行"]
    if only_long and not only_short:
        view = view[view["策略动作"] == "做多"]
    if only_short and not only_long:
        view = view[view["策略动作"] == "做空"]
    if exclude_paused:
        view = view[view["状态(规则)"] != "暂停"]
    if high_liq_only and "liquidity_score" in view.columns:
        liq_threshold = float(pd.to_numeric(prepared["liquidity_score"], errors="coerce").quantile(0.7))
        view = view[pd.to_numeric(view["liquidity_score"], errors="coerce") >= liq_threshold]

    if sort_key in view.columns:
        if sort_key == "risk_rank":
            view = view.sort_values([sort_key, "active_edge_risk"], ascending=[True, False])
        else:
            view = view.sort_values(sort_key, ascending=not sort_desc)

    show = view.head(top_n).copy()
    show["置信度"] = show["confidence_score_est"].map(lambda x: _format_float(x, 1))
    show["风险等级"] = show["risk_level"].map(_risk_text)
    show["机会值(edge)"] = show["active_edge_score"].map(_format_change_pct)
    show["风险调整(edge_risk)"] = show["active_edge_risk"].map(lambda x: _format_float(x, 3))
    show["预计涨跌幅"] = show["predicted_change_pct"].map(_format_change_pct)
    show["总分(0-100)"] = show["total_score"].map(lambda x: _format_float(x, 1))
    show["数据质量"] = show["data_quality_score"].map(lambda x: _format_float(x, 1))
    show["流动性"] = show["liquidity_score"].map(lambda x: _format_float(x, 1))
    show["因子支持数"] = show["factor_support_count"].map(lambda x: _format_float(x, 0))
    show["状态(规则)"] = show["状态(规则)"].map(_status_text)
    show["策略动作"] = show["策略动作"].map(_policy_action_text)
    show = show.rename(columns={"display_name": "标的", "symbol": "代码"})

    main_cols = [
        "市场",
        "标的",
        "代码",
        "状态(规则)",
        "策略动作",
        "原始状态",
        "置信度",
        "风险等级",
        "机会值(edge)",
        "风险调整(edge_risk)",
        "预计涨跌幅",
        "总分(0-100)",
        "数据质量",
        "流动性",
        "因子支持数",
        "告警标签",
        "状态触发标签",
        "修复动作",
        "短原因",
    ]
    main_cols = [c for c in main_cols if c in show.columns]
    display_show = show[main_cols].copy()
    if _ui_lang() != "zh":
        rename_map = {
            "市场": "market",
            "标的": "symbol",
            "代码": "ticker",
            "状态(规则)": "rule_status",
            "策略动作": "policy_action",
            "原始状态": "raw_status",
            "置信度": "confidence",
            "风险等级": "risk_level",
            "机会值(edge)": "edge_score",
            "风险调整(edge_risk)": "edge_risk",
            "预计涨跌幅": "pred_change_pct",
            "总分(0-100)": "total_score_0_100",
            "数据质量": "data_quality",
            "流动性": "liquidity",
            "因子支持数": "factor_support_count",
            "告警标签": "alerts",
            "状态触发标签": "rule_tags",
            "修复动作": "fix_action",
            "短原因": "short_reason",
        }
        display_show = display_show.rename(columns={k: v for k, v in rename_map.items() if k in display_show.columns})
    st.dataframe(display_show, use_container_width=True, hide_index=True)
    export_cols = [c for c in show.columns if c not in {"track_key"}]
    st.download_button(
        _t("Download CSV（当前筛选）", "Download CSV (current filter)"),
        data=show[export_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="tracking_screener_filtered.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.subheader(_t("单标的展开详情（Drill-down）", "Single-symbol Drill-down"))
    detail_source = view if not view.empty else prepared
    detail_options = detail_source["track_key"].dropna().astype(str).tolist()
    detail_name_col = "标的" if "标的" in detail_source.columns else "display_name"
    detail_map = {
        k: f"{detail_source.loc[detail_source['track_key'] == k, '市场'].iloc[0]} | {detail_source.loc[detail_source['track_key'] == k, detail_name_col].iloc[0]}"
        for k in detail_options
    }
    if detail_options:
        selected_key = st.selectbox(
            _t("选择标的", "Select Symbol"),
            detail_options,
            index=0,
            format_func=lambda x: detail_map.get(x, x),
            key="track_drill_symbol",
        )
        row = detail_source.loc[detail_source["track_key"] == selected_key].iloc[0]
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric(_t("规则状态", "Rule Status"), _status_text(str(row.get("状态(规则)", "-"))))
        d2.metric(_t("策略动作", "Policy Action"), _policy_action_text(str(row.get("策略动作", "-"))))
        d3.metric(_t("机会值(edge)", "Edge Score"), _format_change_pct(row.get("active_edge_score")))
        d4.metric(_t("风险调整(edge_risk)", "Risk-adjusted Edge"), _format_float(row.get("active_edge_risk"), 3))
        d5.metric(_t("置信度", "Confidence"), _format_float(row.get("confidence_score_est"), 1))
        d6.metric(_t("风险", "Risk"), _risk_text(str(row.get("risk_level", "-"))))
        e1, e2, e3, e4 = st.columns(4)
        e1.metric(_t("当前价格", "Current Price"), _format_price(row.get("current_price")))
        e2.metric(_t("预测价格", "Predicted Price"), _format_price(row.get("predicted_price")))
        e3.metric(_t("预测涨跌幅", "Predicted Change"), _format_change_pct(row.get("predicted_change_pct")))
        e4.metric(_t("因子支持数", "Factor Support Count"), _format_float(row.get("factor_support_count"), 0))

        st.markdown(f"**{_t('开单计划（Entry / SL / TP / RR）', 'Execution Plan (Entry / SL / TP / RR)')}**")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric(_t("推荐入场", "Suggested Entry"), _format_price(row.get("plan_entry")))
        p2.metric(_t("止损价", "Stop Loss"), _format_price(row.get("plan_stop_loss")))
        p3.metric(_t("止盈价", "Take Profit"), _format_price(row.get("plan_take_profit")))
        p4.metric(_t("盈亏比(RR)", "Risk/Reward (RR)"), _format_float(row.get("plan_rr"), 2))
        p5, p6, p7, p8 = st.columns(4)
        p5.metric(_t("交易状态", "Trade Status"), str(row.get("plan_trade_status", "-")))
        p6.metric(_t("距入场(%)", "Entry Gap (%)"), _format_change_pct(row.get("plan_entry_gap_pct")))
        p7.metric(
            _t("到价触发", "Entry Touched"),
            _t("已触发", "Touched") if bool(row.get("plan_entry_touched", False)) else _t("未触发", "Not yet"),
        )
        p8.metric(
            _t("首次触发时间", "First Touch Time"),
            str(row.get("plan_entry_touched_at", "-")) if str(row.get("plan_entry_touched_at", "")).strip() else "-",
        )
        if str(row.get("plan_trade_status_note", "")).strip():
            st.caption(str(row.get("plan_trade_status_note", "-")))
        st.caption(
            _t(
                f"价格源: {row.get('plan_price_source', '-')} | 市场时间: {row.get('plan_price_ts_market', '-')} | UTC: {row.get('plan_price_ts_utc', '-')}",
                f"Price source: {row.get('plan_price_source', '-')} | Market ts: {row.get('plan_price_ts_market', '-')} | UTC: {row.get('plan_price_ts_utc', '-')}",
            )
        )
        st.caption(
            _t(
                "说明：止损/止盈/RR 基于 q10/q50/q90 区间与成本口径自动计算，用于执行参考。",
                "Note: SL/TP/RR are auto-derived from q10/q50/q90 and cost assumptions, for execution reference.",
            )
        )

        tech_reason_text = _format_reason_tokens_cn(row.get("policy_reason", "-"))
        signal_ctx = pd.DataFrame()
        try:
            market_key = str(row.get("market", ""))
            symbol_key = str(row.get("symbol", ""))
            provider_key = str(row.get("provider", "yahoo"))
            fallback_symbol = str(row.get("snapshot_symbol", symbol_key))
            if market_key and symbol_key:
                signal_ctx = _load_symbol_signal_context_cached(
                    market=market_key,
                    symbol=symbol_key,
                    provider=provider_key,
                    fallback_symbol=fallback_symbol,
                )
        except Exception:
            signal_ctx = pd.DataFrame()
        if not signal_ctx.empty:
            srow = signal_ctx.iloc[-1]
            tech_reason_text = _format_reason_tokens_cn(
                srow.get("trade_reason_tokens", srow.get("policy_reason", row.get("policy_reason", "-")))
            )
            t1, t2, t3 = st.columns(3)
            t1.metric(_t("技术止损", "Technical Stop"), _format_price(srow.get("trade_stop_loss_price")))
            t2.metric(_t("技术止盈", "Technical Take Profit"), _format_price(srow.get("trade_take_profit_price")))
            t3.metric(_t("技术RR", "Technical RR"), _format_float(srow.get("trade_rr_ratio"), 2))

            sig_flags: List[str] = []
            if bool(srow.get("ema_cross_up", False)):
                sig_flags.append(_t("EMA金叉", "EMA golden cross"))
            if bool(srow.get("ema_cross_down", False)):
                sig_flags.append(_t("EMA死叉", "EMA death cross"))
            if bool(srow.get("macd_cross_up", False)):
                sig_flags.append(_t("MACD金叉", "MACD golden cross"))
            if bool(srow.get("macd_cross_down", False)):
                sig_flags.append(_t("MACD死叉", "MACD death cross"))
            if "rsi14" in srow.index and _is_finite_number(srow.get("rsi14")):
                rsi = float(srow.get("rsi14"))
                if rsi <= 30:
                    sig_flags.append(_t(f"RSI超卖({rsi:.1f})", f"RSI oversold ({rsi:.1f})"))
                elif rsi >= 70:
                    sig_flags.append(_t(f"RSI超买({rsi:.1f})", f"RSI overbought ({rsi:.1f})"))
                else:
                    sig_flags.append(_t(f"RSI中性({rsi:.1f})", f"RSI neutral ({rsi:.1f})"))
            if {"kdj_k", "kdj_d", "kdj_j"}.issubset(set(srow.index)):
                kdj_k = _safe_float(srow.get("kdj_k"))
                kdj_d = _safe_float(srow.get("kdj_d"))
                if np.isfinite(kdj_k) and np.isfinite(kdj_d):
                    if kdj_k > kdj_d:
                        sig_flags.append(_t("KDJ金叉结构", "KDJ bullish structure"))
                    elif kdj_k < kdj_d:
                        sig_flags.append(_t("KDJ死叉结构", "KDJ bearish structure"))
            if sig_flags:
                st.caption(
                    _t("技术状态：", "Technical state: ")
                    + " | ".join(sig_flags)
                )

        st.markdown(f"**{_t('做单理由（技术触发）', 'Trade Rationale (technical triggers)')}**")
        st.write(tech_reason_text if str(tech_reason_text).strip() else "-")
        st.caption(
            _t(
                "已接入：EMA / MACD / SuperTrend / BOS / CHOCH / 放量（按数据可用性触发）。",
                "Integrated: EMA / MACD / SuperTrend / BOS / CHOCH / Volume-surge (when data is available).",
            )
        )
        news_gate_raw = str(row.get("news_gate_pass", row.get("news_gate_pass_flag", True))).strip().lower()
        news_gate = news_gate_raw not in {"0", "false", "no", "n", "off"}
        news_event_raw = str(row.get("news_event_risk", row.get("news_event_risk_flag", False))).strip().lower()
        news_event = news_event_raw in {"1", "true", "yes", "y", "on"}
        news_risk_level = _risk_text(str(row.get("news_risk_level", "low")))
        news_score_30m = _safe_float(row.get("news_score_30m"))
        news_score_2h = _safe_float(row.get("news_score_2h", row.get("news_score_120m")))
        news_burst = _safe_float(row.get("news_burst_zscore"))
        news_count_30m = _safe_float(row.get("news_count_30m"))
        news_reason_codes = str(row.get("news_reason_codes", "")).strip()
        if any(k in row.index for k in ["news_score_30m", "news_score_120m", "news_score_2h", "news_gate_pass"]):
            st.caption(
                _t(
                    f"新闻门控：{'通过' if news_gate else '阻断'} | 新闻风险：{news_risk_level} | "
                    f"2h情绪分：{_format_float(news_score_2h, 3)} | 30m情绪分：{_format_float(news_score_30m, 3)} | "
                    f"30m条数：{_format_float(news_count_30m, 0)} | burst_z：{_format_float(news_burst, 2)} | "
                    f"事件风险：{'是' if news_event else '否'}",
                    f"News gate: {'PASS' if news_gate else 'BLOCK'} | News risk: {news_risk_level} | "
                    f"2h score: {_format_float(news_score_2h, 3)} | 30m score: {_format_float(news_score_30m, 3)} | "
                    f"30m count: {_format_float(news_count_30m, 0)} | burst_z: {_format_float(news_burst, 2)} | "
                    f"event risk: {'yes' if news_event else 'no'}",
                )
            )
            if news_reason_codes:
                st.caption(_t(f"新闻原因码：{news_reason_codes}", f"News reason codes: {news_reason_codes}"))
            headlines = str(row.get("news_latest_headlines", "")).strip()
            if headlines:
                st.caption(_t(f"最近新闻：{headlines}", f"Latest headlines: {headlines}"))
        else:
            st.caption(_t("新闻因子：当前无有效新闻信号。", "News factors: no valid news signal at this moment."))

        st.caption(
            _t(
                f"短原因：{row.get('短原因', '-')} | 触发标签：{row.get('状态触发标签', '-')} | "
                f"告警：{row.get('告警标签', '无')}",
                f"Short reason: {row.get('短原因', '-')} | Rule tags: {row.get('状态触发标签', '-')} | "
                f"Alerts: {row.get('告警标签', 'none')}",
            )
        )
        st.caption(_t(f"修复建议：{row.get('修复动作', '-')}", f"Fix action: {row.get('修复动作', '-')}"))
        if _is_finite_number(row.get("policy_position_size")):
            st.caption(_t(f"建议仓位：{float(row.get('policy_position_size')):.1%}", f"Suggested size: {float(row.get('policy_position_size')):.1%}"))
    else:
        st.info(_t("当前筛选条件下没有标的可展开。", "No symbol available under current filters."))

    st.subheader(_t("动作建议明细（含暂停原因 / 修复建议）", "Action Details (incl. pause reasons / fixes)"))
    action_view = prepared.copy()
    action_view["建议仓位"] = action_view["policy_position_size"].map(
        lambda x: f"{float(x):.1%}" if _is_finite_number(x) else "-"
    )
    action_view["预期净优势"] = action_view["active_edge_score"].map(_format_change_pct)
    action_view["预计涨跌幅"] = action_view["predicted_change_pct"].map(_format_change_pct)
    action_view["推荐入场"] = action_view["plan_entry"].map(_format_price)
    action_view["止损价"] = action_view["plan_stop_loss"].map(_format_price)
    action_view["止盈价"] = action_view["plan_take_profit"].map(_format_price)
    action_view["盈亏比(RR)"] = action_view["plan_rr"].map(lambda x: _format_float(x, 2))
    action_view["交易状态"] = action_view.get("plan_trade_status", pd.Series(["-"] * len(action_view), index=action_view.index)).astype(str)
    action_view["距入场(%)"] = action_view.get("plan_entry_gap_pct", pd.Series([np.nan] * len(action_view), index=action_view.index)).map(_format_change_pct)
    action_view["到价触发"] = action_view.get("plan_entry_touched", pd.Series([False] * len(action_view), index=action_view.index)).map(
        lambda v: _t("已触发", "Touched") if bool(v) else _t("未触发", "Not yet")
    )
    action_view["首次触发时间"] = action_view.get("plan_entry_touched_at", pd.Series([""] * len(action_view), index=action_view.index)).map(
        lambda v: str(v) if str(v).strip() else "-"
    )
    news_gate_series = action_view.get("news_gate_pass_flag")
    if news_gate_series is None:
        news_gate_series = action_view.get("news_gate")
    if news_gate_series is None:
        news_gate_series = pd.Series([True] * len(action_view), index=action_view.index)
    action_view["新闻门控"] = news_gate_series.map(
        lambda v: _t("通过", "PASS")
        if str(v).strip().lower() in {"1", "true", "yes", "y", "on"}
        else _t("阻断", "BLOCK")
    )

    news_risk_series = action_view.get("news_risk_level")
    if news_risk_series is None:
        news_risk_series = pd.Series(["low"] * len(action_view), index=action_view.index)
    action_view["新闻风险"] = news_risk_series.map(_risk_text)

    news_score_2h_series = action_view.get("news_score_120m")
    if news_score_2h_series is None:
        news_score_2h_series = action_view.get("news_score_2h")
    if news_score_2h_series is None:
        news_score_2h_series = pd.Series([np.nan] * len(action_view), index=action_view.index)
    action_view["2h新闻情绪"] = news_score_2h_series.map(lambda x: _format_float(x, 3))

    news_reason_series = action_view.get("news_reason_codes")
    if news_reason_series is None:
        news_reason_series = pd.Series([""] * len(action_view), index=action_view.index)
    action_view["新闻原因码"] = news_reason_series.fillna("").astype(str)
    action_cols = [
        "市场",
        "display_name",
        "状态(规则)",
        "策略动作",
        "推荐入场",
        "止损价",
        "止盈价",
        "盈亏比(RR)",
        "交易状态",
        "距入场(%)",
        "到价触发",
        "首次触发时间",
        "建议仓位",
        "预期净优势",
        "预计涨跌幅",
        "新闻门控",
        "新闻风险",
        "2h新闻情绪",
        "新闻原因码",
        "短原因",
        "状态触发标签",
        "告警标签",
        "修复动作",
    ]
    action_cols = [c for c in action_cols if c in action_view.columns]
    action_view = action_view.rename(columns={"display_name": "标的"})
    action_cols = ["标的" if c == "display_name" else c for c in action_cols]
    action_display = action_view[action_cols].copy()
    if "状态(规则)" in action_display.columns:
        action_display["状态(规则)"] = action_display["状态(规则)"].map(_status_text)
    if "策略动作" in action_display.columns:
        action_display["策略动作"] = action_display["策略动作"].map(_policy_action_text)
    if _ui_lang() != "zh":
        action_display = action_display.rename(
            columns={
                "市场": "market",
                "标的": "symbol",
                "状态(规则)": "rule_status",
                "策略动作": "policy_action",
                "推荐入场": "entry",
                "止损价": "stop_loss",
                "止盈价": "take_profit",
                "盈亏比(RR)": "rr",
                "交易状态": "trade_status",
                "距入场(%)": "entry_gap_pct",
                "到价触发": "entry_touched",
                "首次触发时间": "first_touch_time",
                "建议仓位": "position_size",
                "预期净优势": "expected_net_edge",
                "预计涨跌幅": "pred_change_pct",
                "新闻门控": "news_gate",
                "新闻风险": "news_risk",
                "2h新闻情绪": "news_score_2h",
                "新闻原因码": "news_reason_codes",
                "短原因": "short_reason",
                "状态触发标签": "rule_tags",
                "告警标签": "alerts",
                "修复动作": "fix_action",
            }
        )
    st.dataframe(action_display.head(200), use_container_width=True, hide_index=True)

    st.subheader(_t("Tracking：Watchlist + 信号变化提醒", "Tracking: Watchlist + Signal Change Alerts"))
    watch_path = tracking_dir / "watchlist.csv"
    changes_path = tracking_dir / "signal_changes.csv"
    watch_cols = [
        "track_key",
        "market",
        "instrument_id",
        "name",
        "added_time_bj",
        "last_signal_change_bj",
        "last_status",
        "last_action",
        "last_review_note",
        "signal_change_7d",
    ]
    watch = _load_csv(watch_path)
    if watch.empty:
        watch = pd.DataFrame(columns=watch_cols)
    else:
        for c in watch_cols:
            if c not in watch.columns:
                watch[c] = ""
        watch = watch[watch_cols].copy()

    now_bj_text = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S %z")
    if not watch.empty:
        idx_map = prepared.set_index("track_key")
        change_records: List[Dict[str, object]] = []
        for i in watch.index:
            key = str(watch.at[i, "track_key"])
            if key not in idx_map.index:
                continue
            cur = idx_map.loc[key]
            new_status = str(cur.get("状态(规则)", "-"))
            new_action = str(cur.get("策略动作", "-"))
            prev_status = str(watch.at[i, "last_status"] or "").strip()
            prev_action = str(watch.at[i, "last_action"] or "").strip()
            watch.at[i, "market"] = str(cur.get("market", ""))
            watch.at[i, "instrument_id"] = str(cur.get("instrument_id", ""))
            watch.at[i, "name"] = str(cur.get("display_name", ""))
            watch.at[i, "last_status"] = new_status
            watch.at[i, "last_action"] = new_action
            if prev_status and ((prev_status != new_status) or (prev_action != new_action)):
                watch.at[i, "last_signal_change_bj"] = now_bj_text
                change_records.append(
                    {
                        "track_key": key,
                        "market": str(cur.get("market", "")),
                        "name": str(cur.get("display_name", "")),
                        "change_time_bj": now_bj_text,
                        "from_status": prev_status,
                        "to_status": new_status,
                        "from_action": prev_action,
                        "to_action": new_action,
                    }
                )
        if change_records:
            old_changes = _load_csv(changes_path)
            new_changes = pd.DataFrame(change_records)
            all_changes = new_changes if old_changes.empty else pd.concat([old_changes, new_changes], ignore_index=True)
            all_changes.to_csv(changes_path, index=False, encoding="utf-8-sig")
        watch.to_csv(watch_path, index=False, encoding="utf-8-sig")

    label_map = {
        str(r["track_key"]): f"{r['市场']} | {r['display_name']}"
        for _, r in prepared[["track_key", "市场", "display_name"]].drop_duplicates().iterrows()
    }
    default_keys = watch["track_key"].dropna().astype(str).tolist() if not watch.empty else []
    selected_keys = st.multiselect(
        _t("选择需要持续跟踪的标的", "Select symbols for continuous tracking"),
        options=list(label_map.keys()),
        default=[k for k in default_keys if k in label_map],
        format_func=lambda x: label_map.get(x, x),
        key="track_watch_keys_v2",
    )

    edit_base = pd.DataFrame({"track_key": selected_keys})
    if not edit_base.empty:
        edit_base = edit_base.merge(
            prepared[["track_key", "市场", "display_name", "状态(规则)", "策略动作"]].drop_duplicates("track_key"),
            on="track_key",
            how="left",
        )
        if not watch.empty:
            edit_base = edit_base.merge(watch[["track_key", "last_review_note"]], on="track_key", how="left")
        else:
            edit_base["last_review_note"] = ""
        edit_base = edit_base.rename(
            columns={
                "display_name": "标的",
                "状态(规则)": "当前状态",
                "策略动作": "当前动作",
                "last_review_note": "研究备注",
            }
        )
        edit_display = edit_base[["track_key", "市场", "标的", "当前状态", "当前动作", "研究备注"]].copy()
        edit_display["当前状态"] = edit_display["当前状态"].map(_status_text)
        edit_display["当前动作"] = edit_display["当前动作"].map(_policy_action_text)
        if _ui_lang() != "zh":
            edit_display = edit_display.rename(
                columns={
                    "市场": "market",
                    "标的": "symbol",
                    "当前状态": "current_status",
                    "当前动作": "current_action",
                    "研究备注": "research_note",
                }
            )
        edited = st.data_editor(
            edit_display,
            hide_index=True,
            use_container_width=True,
            key="track_watch_editor_v2",
        )
    else:
        edited = pd.DataFrame(columns=["track_key", "研究备注"])
        st.info(_t("还未选择跟踪标的。", "No symbols selected for tracking yet."))

    if st.button(_t("保存 Watchlist", "Save Watchlist"), key="track_watch_save_v2"):
        prev_watch = watch.set_index("track_key") if not watch.empty else pd.DataFrame().set_index(pd.Index([]))
        keep_rows: List[Dict[str, object]] = []
        change_records: List[Dict[str, object]] = []
        idx_map = prepared.set_index("track_key")
        now_text = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S %z")
        note_map = {}
        note_col = "研究备注" if "研究备注" in edited.columns else "research_note"
        if not edited.empty and "track_key" in edited.columns and note_col in edited.columns:
            note_map = dict(zip(edited["track_key"].astype(str), edited[note_col].fillna("").astype(str)))

        for key in selected_keys:
            if key not in idx_map.index:
                continue
            cur = idx_map.loc[key]
            prev = prev_watch.loc[key] if key in prev_watch.index else None
            prev_status = str(prev["last_status"]).strip() if prev is not None and "last_status" in prev else ""
            prev_action = str(prev["last_action"]).strip() if prev is not None and "last_action" in prev else ""
            new_status = str(cur.get("状态(规则)", "-"))
            new_action = str(cur.get("策略动作", "-"))
            added_time = str(prev["added_time_bj"]) if prev is not None and str(prev.get("added_time_bj", "")).strip() else now_text
            last_signal_change = (
                str(prev["last_signal_change_bj"]) if prev is not None and str(prev.get("last_signal_change_bj", "")).strip() else "-"
            )
            if prev_status and ((prev_status != new_status) or (prev_action != new_action)):
                last_signal_change = now_text
                change_records.append(
                    {
                        "track_key": key,
                        "market": str(cur.get("market", "")),
                        "name": str(cur.get("display_name", "")),
                        "change_time_bj": now_text,
                        "from_status": prev_status,
                        "to_status": new_status,
                        "from_action": prev_action,
                        "to_action": new_action,
                    }
                )
            keep_rows.append(
                {
                    "track_key": key,
                    "market": str(cur.get("market", "")),
                    "instrument_id": str(cur.get("instrument_id", "")),
                    "name": str(cur.get("display_name", "")),
                    "added_time_bj": added_time,
                    "last_signal_change_bj": last_signal_change,
                    "last_status": new_status,
                    "last_action": new_action,
                    "last_review_note": note_map.get(str(key), ""),
                    "signal_change_7d": 0,
                }
            )

        new_watch = pd.DataFrame(keep_rows, columns=watch_cols)
        if change_records:
            old_changes = _load_csv(changes_path)
            new_changes = pd.DataFrame(change_records)
            all_changes = new_changes if old_changes.empty else pd.concat([old_changes, new_changes], ignore_index=True)
            all_changes.to_csv(changes_path, index=False, encoding="utf-8-sig")

        changes_all = _load_csv(changes_path)
        if not changes_all.empty and "change_time_bj" in changes_all.columns:
            changes_all["change_time_bj"] = pd.to_datetime(changes_all["change_time_bj"], errors="coerce", utc=True).dt.tz_convert(
                "Asia/Shanghai"
            )
            window_start = pd.Timestamp.now(tz="Asia/Shanghai") - pd.Timedelta(days=7)
            recent7 = changes_all[changes_all["change_time_bj"] >= window_start]
            cnt7 = recent7.groupby("track_key").size().to_dict()
            new_watch["signal_change_7d"] = new_watch["track_key"].map(cnt7).fillna(0).astype(int)

        new_watch.to_csv(watch_path, index=False, encoding="utf-8-sig")
        st.success(_t(f"已保存 Watchlist：{len(new_watch)} 个标的。", f"Watchlist saved: {len(new_watch)} symbols."))
        st.rerun()

    watch_latest = _load_csv(watch_path)
    changes_latest = _load_csv(changes_path)
    if not watch_latest.empty:
        for c in watch_cols:
            if c not in watch_latest.columns:
                watch_latest[c] = ""
        if not changes_latest.empty and "change_time_bj" in changes_latest.columns:
            changes_latest["change_time_bj"] = pd.to_datetime(changes_latest["change_time_bj"], errors="coerce", utc=True).dt.tz_convert(
                "Asia/Shanghai"
            )
            recent7 = changes_latest[
                changes_latest["change_time_bj"] >= (pd.Timestamp.now(tz="Asia/Shanghai") - pd.Timedelta(days=7))
            ]
            cnt7 = recent7.groupby("track_key").size().to_dict()
            watch_latest["signal_change_7d"] = watch_latest["track_key"].astype(str).map(cnt7).fillna(0).astype(int)
        watch_show = watch_latest.copy()
        watch_show["市场"] = watch_show["market"].map(_market_text)
        watch_show = watch_show.rename(
            columns={
                "name": "标的",
                "added_time_bj": "加入时间",
                "last_signal_change_bj": "最近信号变化",
                "last_status": "当前状态",
                "last_action": "当前动作",
                "signal_change_7d": "近7天变化次数",
                "last_review_note": "研究备注",
            }
        )
        watch_show["当前状态"] = watch_show["当前状态"].map(_status_text)
        watch_show["当前动作"] = watch_show["当前动作"].map(_policy_action_text)
        watch_cols_show = ["市场", "标的", "加入时间", "最近信号变化", "当前状态", "当前动作", "近7天变化次数", "研究备注"]
        watch_cols_show = [c for c in watch_cols_show if c in watch_show.columns]
        watch_display = watch_show[watch_cols_show].copy()
        if _ui_lang() != "zh":
            watch_display = watch_display.rename(
                columns={
                    "市场": "market",
                    "标的": "symbol",
                    "加入时间": "added_time",
                    "最近信号变化": "last_signal_change",
                    "当前状态": "current_status",
                    "当前动作": "current_action",
                    "近7天变化次数": "signal_changes_7d",
                    "研究备注": "research_note",
                }
            )
        st.dataframe(watch_display, use_container_width=True, hide_index=True)

    st.markdown(f"**{_t('信号变化提醒（最近24小时）', 'Signal Change Alerts (last 24h)')}**")
    if not changes_latest.empty and "change_time_bj" in changes_latest.columns:
        changes_latest["change_time_bj"] = pd.to_datetime(changes_latest["change_time_bj"], errors="coerce", utc=True).dt.tz_convert(
            "Asia/Shanghai"
        )
        recent24 = changes_latest[
            changes_latest["change_time_bj"] >= (pd.Timestamp.now(tz="Asia/Shanghai") - pd.Timedelta(hours=24))
        ].copy()
        if recent24.empty:
            st.info(_t("最近24小时没有状态/动作变化。", "No status/action changes in the last 24 hours."))
        else:
            recent24["市场"] = recent24["market"].map(_market_text)
            recent24["变化时间"] = recent24["change_time_bj"].dt.strftime("%Y-%m-%d %H:%M:%S %z")
            recent24 = recent24.rename(
                columns={
                    "name": "标的",
                    "from_status": "原状态",
                    "to_status": "新状态",
                    "from_action": "原动作",
                    "to_action": "新动作",
                }
            )
            for c in ["原状态", "新状态"]:
                if c in recent24.columns:
                    recent24[c] = recent24[c].map(_status_text)
            for c in ["原动作", "新动作"]:
                if c in recent24.columns:
                    recent24[c] = recent24[c].map(_policy_action_text)
            recent24_show = recent24[["变化时间", "市场", "标的", "原状态", "新状态", "原动作", "新动作"]].copy()
            if _ui_lang() != "zh":
                recent24_show = recent24_show.rename(
                    columns={
                        "变化时间": "change_time",
                        "市场": "market",
                        "标的": "symbol",
                        "原状态": "from_status",
                        "新状态": "to_status",
                        "原动作": "from_action",
                        "新动作": "to_action",
                    }
                )
            st.dataframe(
                recent24_show,
                use_container_width=True,
                hide_index=True,
            )
            watch_label = _status_text("观察")
            exec_label = _status_text("可执行")
            paused_label = _status_text("暂停")
            promote_n = int(((recent24["原状态"] == watch_label) & (recent24["新状态"] == exec_label)).sum())
            degrade_n = int(((recent24["原状态"] == exec_label) & (recent24["新状态"] != exec_label)).sum())
            risk_up_n = int((recent24["新状态"] == paused_label).sum())
            a1, a2, a3 = st.columns(3)
            a1.metric(_t("观察 -> 可执行", "Watch -> Executable"), f"{promote_n}")
            a2.metric(_t("可执行 -> 降级", "Executable -> Downgrade"), f"{degrade_n}")
            a3.metric(_t("变为暂停", "Changed to Paused"), f"{risk_up_n}")
    else:
        st.info(_t("暂无变化日志。保存 watchlist 后将开始记录。", "No change logs yet. Logs start after watchlist is saved."))

    st.subheader(_t("数据覆盖率 / 缺口分析", "Data Coverage / Gap Analysis"))
    code_list: List[str] = []
    for raw in prepared.get("alerts", pd.Series([""] * len(prepared), index=prepared.index)).tolist():
        code_list.extend(_split_alert_codes(raw))
    if not code_list and "alerts_cov" in prepared.columns:
        for raw in prepared["alerts_cov"].tolist():
            code_list.extend(_split_alert_codes(raw))
    if code_list:
        gap_df = pd.Series(code_list).value_counts().rename_axis("code").reset_index(name="count")
        gap_df["问题"] = gap_df["code"].map(_alert_tag_text)
        gap_df["修复建议"] = gap_df["code"].map(_alert_fix_action_text)
        gap_df = gap_df[["问题", "count", "修复建议"]].rename(columns={"count": _t("数量", "count")})
        if _ui_lang() != "zh":
            gap_df = gap_df.rename(columns={"问题": "issue", "修复建议": "fix_action"})
        st.dataframe(gap_df.head(12), use_container_width=True, hide_index=True)
    else:
        st.success(_t("当前没有明显的数据覆盖告警。", "No obvious data coverage alerts currently."))

    if not actions.empty:
        with st.expander(_t("原始 tracking_actions 快照", "Raw tracking_actions snapshot"), expanded=False):
            st.dataframe(actions.head(200), use_container_width=True, hide_index=True)

    if report_path.exists():
        with st.expander(_t("数据质量报告", "Data Quality Report"), expanded=False):
            st.markdown(report_path.read_text(encoding="utf-8"))


def _render_execution_page(processed_dir: Path) -> None:
    st.header(_t("Paper Trading / Execution 页面", "Paper Trading / Execution"))
    out_dir = _execution_output_dir(processed_dir)
    artifacts = load_execution_artifacts(out_dir)
    stats = summarize_execution(artifacts)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open Positions", f"{int(_safe_float(stats.get('open_positions')))}")
    c2.metric("Closed Positions", f"{int(_safe_float(stats.get('closed_positions')))}")
    c3.metric("Win Rate", _format_change_pct(stats.get("win_rate")).replace("+", ""))
    c4.metric("Avg Net PnL", _format_change_pct(stats.get("avg_net_pnl_pct")))
    c5.metric("Total Net PnL", _format_change_pct(stats.get("total_net_pnl_pct")))
    _render_kill_switch_control_panel(out_dir)

    log_path = out_dir / "decision_packets_log.csv"
    if log_path.exists():
        st.subheader(_t("DecisionPacket 日志", "DecisionPacket Logs"))
        log_df = pd.read_csv(log_path)
        show_cols = [
            "decision_id",
            "market",
            "symbol",
            "action",
            "entry",
            "sl",
            "tp1",
            "rr",
            "expected_edge_pct",
            "edge_risk",
            "confidence_score",
            "risk_level",
            "valid_until",
            "model_version",
            "data_version",
            "config_hash",
            "git_commit",
            "reasons",
            "generated_at_utc",
        ]
        show_cols = [c for c in show_cols if c in log_df.columns]
        st.dataframe(log_df.sort_values("generated_at_utc", ascending=False)[show_cols].head(200), use_container_width=True, hide_index=True)
    else:
        st.info(_t("暂无 DecisionPacket 日志。", "No DecisionPacket logs yet."))

    orders = artifacts.get("orders", pd.DataFrame())
    fills = artifacts.get("fills", pd.DataFrame())
    positions = artifacts.get("positions", pd.DataFrame())
    daily_pnl = artifacts.get("daily_pnl", pd.DataFrame())

    st.subheader(_t("订单（Orders）", "Orders"))
    if orders.empty:
        st.info(_t("暂无订单。", "No orders yet."))
    else:
        st.dataframe(orders.sort_values("created_at_utc", ascending=False).head(300), use_container_width=True, hide_index=True)

    st.subheader(_t("成交（Fills）", "Fills"))
    if fills.empty:
        st.info(_t("暂无成交。", "No fills yet."))
    else:
        st.dataframe(fills.sort_values("fill_time_utc", ascending=False).head(300), use_container_width=True, hide_index=True)

    st.subheader(_t("持仓（Positions）", "Positions"))
    if positions.empty:
        st.info(_t("暂无持仓记录。", "No position records yet."))
    else:
        open_pos = positions[positions["status"].astype(str) == "open"]
        closed_pos = positions[positions["status"].astype(str) == "closed"]
        st.markdown(f"**{_t('当前持仓（Open）', 'Open Positions')}**")
        if open_pos.empty:
            st.info(_t("当前无 open 持仓。", "No open positions."))
        else:
            st.dataframe(open_pos.sort_values("entry_time_utc", ascending=False), use_container_width=True, hide_index=True)
        st.markdown(f"**{_t('已平仓（Closed）', 'Closed Positions')}**")
        if closed_pos.empty:
            st.info(_t("暂无 closed 记录。", "No closed records yet."))
        else:
            st.dataframe(closed_pos.sort_values("exit_time_utc", ascending=False).head(300), use_container_width=True, hide_index=True)

    st.subheader(_t("每日已实现盈亏（Daily Realized PnL）", "Daily Realized PnL"))
    if daily_pnl.empty:
        st.info(_t("暂无 `paper_daily_pnl.csv`。执行交易后会自动生成。", "`paper_daily_pnl.csv` not available yet. It will be generated after trades are executed."))
    else:
        st.dataframe(daily_pnl.sort_values("date_utc", ascending=False).head(180), use_container_width=True, hide_index=True)

    run_log_path = out_dir / "paper_run_log.jsonl"
    if run_log_path.exists():
        with st.expander(_t("Paper Run Log（最近 200 条）", "Paper Run Log (latest 200)"), expanded=False):
            rows = []
            for line in run_log_path.read_text(encoding="utf-8").splitlines()[-200:]:
                line = str(line).strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("timestamp_utc", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info(_t("run log empty", "run log empty"))

    gates_log_path = out_dir / "gates_audit_log.jsonl"
    if gates_log_path.exists():
        with st.expander(_t("Gate Audit Log（最近 200 条）", "Gate Audit Log (latest 200)"), expanded=False):
            rows = []
            for line in gates_log_path.read_text(encoding="utf-8").splitlines()[-200:]:
                line = str(line).strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("timestamp_utc", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info(_t("gate log empty", "gate log empty"))

    for title, path_name in [
        (_t("Kill Switch 事件（最近 200 条）", "Kill Switch Events (latest 200)"), "kill_switch_events.jsonl"),
        (_t("Kill Switch 恢复日志（最近 200 条）", "Kill Switch Recovery Log (latest 200)"), "kill_switch_recovery_log.jsonl"),
    ]:
        p = out_dir / path_name
        if p.exists():
            with st.expander(title, expanded=False):
                rows = []
                for line in p.read_text(encoding="utf-8").splitlines()[-200:]:
                    line = str(line).strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
                if rows:
                    show = pd.DataFrame(rows)
                    sort_col = "timestamp_utc" if "timestamp_utc" in show.columns else show.columns[0]
                    st.dataframe(show.sort_values(sort_col, ascending=False), use_container_width=True, hide_index=True)
                else:
                    st.info(_t("log empty", "log empty"))

    with st.expander(_t("维护操作", "Maintenance"), expanded=False):
        if st.button(_t("清空 Paper Trading 日志", "Clear Paper Trading logs"), key="clear_execution_logs", use_container_width=False):
            for fn in [
                "decision_packet_latest.json",
                "decision_packets_log.csv",
                "paper_orders.csv",
                "paper_fills.csv",
                "paper_positions.csv",
                "paper_daily_pnl.csv",
                "paper_run_log.jsonl",
                "gates_audit_log.jsonl",
                "kill_switch_events.jsonl",
                "kill_switch_recovery_log.jsonl",
                ".kill_switch_last_seen.json",
                "kill_switch.state.json",
                "health_checks_streak.json",
            ]:
                p = out_dir / fn
                if p.exists():
                    p.unlink()
            for p in out_dir.glob("decision_packet_*.json"):
                try:
                    p.unlink()
                except Exception:
                    pass
            st.success(_t("Execution 日志已清空。", "Execution logs cleared."))
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Multi-Market Forecast Dashboard", layout="wide")
    st.title("Multi-Market Forecast Dashboard")
    st.caption(
        _t(
            "页面导航：Crypto / A股 / 美股 / 交易时间段预测 / Selection-Research-Tracking / Paper Trading-Execution",
            "Navigation: Crypto / CN A-share / US Equity / Session Forecast / Selection-Research-Tracking / Paper Trading-Execution",
        )
    )
    st.markdown(
        """
<style>
div[data-testid="stMetricValue"] {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    lang_pick = st.sidebar.selectbox(
        "语言 / Language",
        options=["中文", "English"],
        index=0 if _ui_lang() == "zh" else 1,
    )
    st.session_state["ui_lang"] = "zh" if lang_pick == "中文" else "en"

    if st.button(_t("清理缓存并刷新", "Clear cache and reload"), use_container_width=False):
        st.cache_data.clear()
        st.rerun()

    processed_dir = Path("data/processed")
    hourly = _load_csv(processed_dir / "predictions_hourly.csv")
    daily = _load_csv(processed_dir / "predictions_daily.csv")
    btc_live = _fetch_live_btc_price()

    page_items = [
        ("crypto", _t("Crypto 页面", "Crypto Page")),
        ("cn", _t("A股 页面", "CN A-share Page")),
        ("us", _t("美股 页面", "US Equity Page")),
        ("session", _t("交易时间段预测（Crypto）", "Session Forecast (Crypto)")),
        ("tracking", _t("Selection / Research / Tracking 页面", "Selection / Research / Tracking")),
        ("execution", _t("Paper Trading / Execution 页面", "Paper Trading / Execution")),
    ]
    page_labels = [x[1] for x in page_items]
    page_choice = st.sidebar.radio(_t("页面", "Page"), options=page_labels, index=0)
    page_key = next((k for k, lbl in page_items if lbl == page_choice), "crypto")

    if page_key == "crypto":
        _render_crypto_page(
            processed_dir=processed_dir,
            btc_live=btc_live,
            hourly_df=hourly,
            daily_df=daily,
        )
    elif page_key == "cn":
        _render_cn_page()
    elif page_key == "us":
        _render_us_page()
    elif page_key == "session":
        _render_crypto_session_page()
    elif page_key == "execution":
        _render_execution_page(processed_dir)
    else:
        _render_tracking_page(processed_dir)


if __name__ == "__main__":
    main()
