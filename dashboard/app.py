from __future__ import annotations

import importlib
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
from src.utils.config import load_config


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=600, show_spinner=False)
def _load_main_config_cached(config_path: str = "configs/config.yaml") -> Dict[str, object]:
    return load_config(config_path)


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
        "volume_surge": "放量",
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


def _format_reason_tokens_cn(reason_text: object) -> str:
    raw = str(reason_text or "").strip()
    if not raw:
        return "-"
    return "；".join(_reason_token_cn(t) for t in raw.split(";") if str(t).strip())


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


def _risk_cn(label: str) -> str:
    mapping = {
        "low": "低",
        "medium": "中",
        "high": "高",
        "extreme": "极高",
    }
    return mapping.get(str(label), str(label))


def _policy_action_cn(label: str) -> str:
    mapping = {
        "Long": "做多",
        "Short": "做空",
        "Flat": "观望",
    }
    return mapping.get(str(label), str(label))


def _session_display_name(session_name: str) -> str:
    mapping = {"asia": "亚盘", "europe": "欧盘", "us": "美盘"}
    return mapping.get(str(session_name), str(session_name))


def _render_hourly_heatmap(hourly_df: pd.DataFrame, value_col: str, title: str) -> None:
    if hourly_df.empty or value_col not in hourly_df.columns:
        st.info("暂无热力图数据。")
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
        title=f"24小时热力图 - {title}（北京时间）",
        xaxis_title="小时",
        yaxis_title="指标",
        template="plotly_white",
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_top_tables(hourly_df: pd.DataFrame, daily_df: pd.DataFrame, top_n: int) -> None:
    if hourly_df.empty:
        st.info("暂无小时级榜单数据。")
    else:
        h = hourly_df.copy()
        h["趋势"] = h["trend_label"].map(_trend_cn)
        h["风险"] = h["risk_level"].map(_risk_cn)
        h["上涨概率"] = h["p_up"].map(lambda x: _format_change_pct(x).replace("+", ""))
        h["下跌概率"] = h["p_down"].map(lambda x: _format_change_pct(x).replace("+", ""))
        h["预期涨跌幅"] = h["q50_change_pct"].map(_format_change_pct)
        h["目标价格(q50)"] = h["target_price_q50"].map(_format_price)
        h["置信度"] = h["confidence_score"].map(lambda x: _format_float(x, 1))
        if "policy_action" in h.columns:
            h["策略动作"] = h["policy_action"].map(_policy_action_cn)
            h["建议仓位"] = h["policy_position_size"].map(
                lambda x: f"{float(x):.1%}" if _is_finite_number(x) else "-"
            )
            h["预期净优势"] = h["policy_expected_edge_pct"].map(_format_change_pct)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**小时级：最可能上涨 Top N**")
            cols = [
                "hour_label",
                "上涨概率",
                "预期涨跌幅",
                "目标价格(q50)",
                "策略动作",
                "建议仓位",
                "预期净优势",
                "趋势",
                "风险",
                "置信度",
            ]
            cols = [c for c in cols if c in h.columns]
            st.dataframe(h.sort_values("p_up", ascending=False)[cols].head(top_n), use_container_width=True)
        with c2:
            st.markdown("**小时级：最可能下跌 Top N**")
            cols = [
                "hour_label",
                "下跌概率",
                "预期涨跌幅",
                "目标价格(q50)",
                "策略动作",
                "建议仓位",
                "预期净优势",
                "趋势",
                "风险",
                "置信度",
            ]
            cols = [c for c in cols if c in h.columns]
            st.dataframe(h.sort_values("p_down", ascending=False)[cols].head(top_n), use_container_width=True)
        with c3:
            st.markdown("**小时级：最可能大波动 Top N**")
            cols = [
                "hour_label",
                "预期涨跌幅",
                "目标价格(q50)",
                "策略动作",
                "建议仓位",
                "预期净优势",
                "趋势",
                "风险",
                "置信度",
            ]
            cols = [c for c in cols if c in h.columns]
            st.dataframe(
                h.sort_values("volatility_score", ascending=False)[cols].head(top_n),
                use_container_width=True,
            )

    st.markdown("---")
    if daily_df.empty:
        st.info("暂无日线级榜单数据。")
    else:
        d = daily_df.copy()
        d["趋势"] = d["trend_label"].map(_trend_cn)
        d["风险"] = d["risk_level"].map(_risk_cn)
        d["上涨概率"] = d["p_up"].map(lambda x: _format_change_pct(x).replace("+", ""))
        d["下跌概率"] = d["p_down"].map(lambda x: _format_change_pct(x).replace("+", ""))
        d["预期涨跌幅"] = d["q50_change_pct"].map(_format_change_pct)
        d["目标价格(q50)"] = d["target_price_q50"].map(_format_price)
        d["置信度"] = d["confidence_score"].map(lambda x: _format_float(x, 1))
        if "policy_action" in d.columns:
            d["策略动作"] = d["policy_action"].map(_policy_action_cn)
            d["建议仓位"] = d["policy_position_size"].map(
                lambda x: f"{float(x):.1%}" if _is_finite_number(x) else "-"
            )
            d["预期净优势"] = d["policy_expected_edge_pct"].map(_format_change_pct)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**日线级：最可能上涨 Top N**")
            cols = [
                "date_bj",
                "上涨概率",
                "预期涨跌幅",
                "目标价格(q50)",
                "策略动作",
                "建议仓位",
                "预期净优势",
                "趋势",
                "风险",
                "置信度",
            ]
            cols = [c for c in cols if c in d.columns]
            st.dataframe(d.sort_values("p_up", ascending=False)[cols].head(top_n), use_container_width=True)
        with c2:
            st.markdown("**日线级：最可能下跌 Top N**")
            cols = [
                "date_bj",
                "下跌概率",
                "预期涨跌幅",
                "目标价格(q50)",
                "策略动作",
                "建议仓位",
                "预期净优势",
                "趋势",
                "风险",
                "置信度",
            ]
            cols = [c for c in cols if c in d.columns]
            st.dataframe(d.sort_values("p_down", ascending=False)[cols].head(top_n), use_container_width=True)
        with c3:
            st.markdown("**日线级：最可能大波动 Top N**")
            cols = [
                "date_bj",
                "预期涨跌幅",
                "目标价格(q50)",
                "策略动作",
                "建议仓位",
                "预期净优势",
                "趋势",
                "风险",
                "置信度",
            ]
            cols = [c for c in cols if c in d.columns]
            st.dataframe(
                d.sort_values("volatility_score", ascending=False)[cols].head(top_n),
                use_container_width=True,
            )


def _render_crypto_session_page() -> None:
    st.header("交易时间段预测（Crypto）")
    st.caption("北京时间24小时制；支持亚盘/欧盘/美盘、关键小时概率与未来N天日线预测。")

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

    symbols = fc.get("symbols", {}).get("default", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    exchanges = source_cfg.get("exchanges", ["binance", "bybit"])
    market_types = source_cfg.get("market_types", ["perp", "spot"])
    default_exchange = source_cfg.get("default_exchange", "binance")
    default_market_type = source_cfg.get("default_market_type", "perp")
    default_horizon = int(fc.get("hourly", {}).get("horizon_hours", 4))
    default_days = int(fc.get("daily", {}).get("lookforward_days", 14))

    f1, f2, f3, f4, f5 = st.columns([2, 1, 1, 1, 1])
    symbol = f1.selectbox("币种", options=symbols, index=0, key="session_symbol")
    exchange = f2.selectbox(
        "数据源",
        options=exchanges,
        index=exchanges.index(default_exchange) if default_exchange in exchanges else 0,
        key="session_exchange",
    )
    market_type = f3.selectbox(
        "市场类型",
        options=market_types,
        index=market_types.index(default_market_type) if default_market_type in market_types else 0,
        key="session_market_type",
    )
    mode = f4.selectbox("模式", options=["forecast", "seasonality"], index=0, key="session_mode")
    horizon_hours = int(
        f5.selectbox("小时周期", options=[4], index=0 if default_horizon == 4 else 0, key="session_horizon")
    )

    c1, c2 = st.columns([1, 3])
    lookforward_days = int(c1.slider("未来N天（日线）", 7, 30, default_days, 1, key="session_daily_n"))
    if c2.button("刷新并重算", key="session_refresh_btn"):
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
        st.error(f"时段预测计算失败：{exc}")
        return

    mode_actual = str(meta.get("mode_actual", mode))
    if mode_actual != mode:
        st.warning(f"请求模式是 `{mode}`，当前自动降级为 `{mode_actual}`。")
    st.caption(
        f"当前价格：{_format_price(meta.get('current_price'))} | "
        f"请求数据源：{meta.get('exchange', '-')} | 实际数据源：{meta.get('exchange_actual', '-') } | "
        f"预测生成：{meta.get('forecast_generated_at_bj', '-')} | "
        f"数据更新时间：{meta.get('data_updated_at_bj', '-')}"
    )
    st.caption(f"数据来源：{meta.get('data_source_actual', '-')}")

    # Session cards
    if blocks_df.empty:
        st.info("暂无时段汇总数据。")
    else:
        cards = blocks_df.copy()
        cards["session_name_cn"] = cards.get("session_name_cn", cards["session_name"].map(_session_display_name))
        cards = cards.sort_values(
            "session_name", key=lambda s: s.map({"asia": 0, "europe": 1, "us": 2}).fillna(9)
        )
        cols = st.columns(3)
        for idx, (_, row) in enumerate(cards.iterrows()):
            col = cols[idx % 3]
            with col:
                st.markdown(f"**{row.get('session_name_cn', '-') }（{row.get('session_hours', '-') }）**")
                st.metric("上涨概率", _format_change_pct(row.get("p_up")).replace("+", ""))
                st.metric("下跌概率", _format_change_pct(row.get("p_down")).replace("+", ""))
                st.metric("预期涨跌幅(q50)", _format_change_pct(row.get("q50_change_pct")))
                st.metric("目标价格(q50)", _format_price(row.get("target_price_q50")))
                st.caption(
                    f"趋势：{_trend_cn(str(row.get('trend_label', '-')))} | "
                    f"风险：{_risk_cn(str(row.get('risk_level', '-')))} | "
                    f"置信度：{_format_float(row.get('confidence_score'), 1)}"
                )
                if "policy_action" in row.index:
                    st.caption(
                        f"策略：{_policy_action_cn(str(row.get('policy_action', 'Flat')))} | "
                        f"仓位：{(float(row.get('policy_position_size')) if _is_finite_number(row.get('policy_position_size')) else 0.0):.1%} | "
                        f"净优势：{_format_change_pct(row.get('policy_expected_edge_pct'))}"
                    )

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["上涨概率", "下跌概率", "波动强度", "置信度"])
    with tab1:
        _render_hourly_heatmap(hourly_df, "p_up", "上涨概率")
    with tab2:
        _render_hourly_heatmap(hourly_df, "p_down", "下跌概率")
    with tab3:
        _render_hourly_heatmap(hourly_df, "volatility_score", "波动强度")
    with tab4:
        _render_hourly_heatmap(hourly_df, "confidence_score", "置信度")

    st.markdown("---")
    st.subheader("未来N天日线预测")
    if daily_df.empty:
        st.info("暂无日线预测数据。")
    else:
        d = daily_df.copy()
        d["趋势"] = d["trend_label"].map(_trend_cn)
        d["风险"] = d["risk_level"].map(_risk_cn)
        d["上涨概率"] = d["p_up"].map(lambda x: _format_change_pct(x).replace("+", ""))
        d["下跌概率"] = d["p_down"].map(lambda x: _format_change_pct(x).replace("+", ""))
        d["预期涨跌幅"] = d["q50_change_pct"].map(_format_change_pct)
        d["目标价格(q10)"] = d["target_price_q10"].map(_format_price)
        d["目标价格(q50)"] = d["target_price_q50"].map(_format_price)
        d["目标价格(q90)"] = d["target_price_q90"].map(_format_price)
        d["置信度"] = d["confidence_score"].map(lambda x: _format_float(x, 1))
        if "policy_action" in d.columns:
            d["策略动作"] = d["policy_action"].map(_policy_action_cn)
            d["建议仓位"] = d["policy_position_size"].map(
                lambda x: f"{float(x):.1%}" if _is_finite_number(x) else "-"
            )
            d["预期净优势"] = d["policy_expected_edge_pct"].map(_format_change_pct)
        show_cols = [
            "date_bj",
            "day_of_week",
            "上涨概率",
            "下跌概率",
            "预期涨跌幅",
            "目标价格(q10)",
            "目标价格(q50)",
            "目标价格(q90)",
            "策略动作",
            "建议仓位",
            "预期净优势",
            "趋势",
            "风险",
            "置信度",
            "start_window_top1",
        ]
        show_cols = [c for c in show_cols if c in d.columns]
        st.dataframe(d[show_cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    top_n = int(st.slider("榜单显示 Top N", 3, 12, 5, 1, key="session_topn"))
    _render_top_tables(hourly_df=hourly_df, daily_df=daily_df, top_n=top_n)

    with st.expander("如何解读这个页面？", expanded=False):
        st.markdown(
            "- 看 `P(up)/P(down)` 判断方向概率。\n"
            "- 看 `预期涨跌幅(q50)` 判断幅度。\n"
            "- 看 `波动强度` 和 `风险等级` 判断风险。\n"
            "- 看 `策略动作/建议仓位/预期净优势` 判断是否值得参与。\n"
            "- `Forecast` 与 `Seasonality` 是两种口径，分歧本身也是信息。"
        )


def _render_projection_chart(
    *,
    current_price: float,
    q10_change_pct: float,
    q50_change_pct: float,
    q90_change_pct: float,
    expected_date_label: str,
    title: str,
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
    with st.expander("指标解释（给非量化用户）", expanded=False):
        st.markdown(
            "- `市值因子`：规模相关，通常越大越稳。\n"
            "- `价值因子`：估值便宜程度（越高通常越便宜）。\n"
            "- `成长因子`：增长能力（盈利/营收或价格增长代理）。\n"
            "- `动能因子`：近期趋势强弱。\n"
            "- `反转因子`：短期是否有回撤后反弹特征。\n"
            "- `低波动因子`：波动越低，数值通常越好。"
        )


def _render_core_field_explain() -> None:
    with st.expander("这4个核心字段是什么意思？", expanded=False):
        st.markdown(
            "- `当前价格`：当前可交易市场的最新成交价。\n"
            "- `预测价格`：模型给出的目标价格（默认看中位数 q50）。\n"
            "- `预计涨跌幅`：从当前价格到预测价格的变化比例。\n"
            "- `预期日期`：这次预测对应的目标时间点（完整时间，含时区）。"
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
        st.info("暂无可展示的模型评估结果。")
        return

    required_cols = {"branch", "task", "model", "horizon", "metric", "mean", "std"}
    missing = required_cols - set(metrics.columns)
    if missing:
        st.dataframe(metrics, use_container_width=True)
        return

    st.markdown("**Model Metrics（通俗版）**")
    with st.expander("怎么看这些指标？", expanded=False):
        st.markdown(
            "- `准确率/F1/AUC`：越高越好。\n"
            "- `MAE/RMSE/分位数误差`：越低越好。\n"
            "- `std(波动)`：越低代表越稳定。\n"
            "- 如果方向指标一般但幅度误差小，适合做区间预期，不适合单独做买卖信号。"
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
    branch_options = ["全部"] + sorted(work["branch"].dropna().unique().tolist())
    task_options = ["全部"] + sorted(work["task"].dropna().unique().tolist())
    horizon_values = sorted(work["_horizon_norm"].dropna().unique().tolist())
    horizon_options = ["全部"] + [str(h) for h in horizon_values]

    selected_branch = f1.selectbox("分支", branch_options, index=0, key="metrics_branch_filter")
    selected_task = f2.selectbox("任务", task_options, index=0, key="metrics_task_filter")
    selected_horizon = f3.selectbox("周期", horizon_options, index=0, key="metrics_horizon_filter")

    view = work.copy()
    if selected_branch != "全部":
        view = view[view["branch"] == selected_branch]
    if selected_task != "全部":
        view = view[view["task"] == selected_task]
    if selected_horizon != "全部":
        view = view[view["_horizon_norm"] == selected_horizon]

    if view.empty:
        st.info("当前筛选条件下没有指标记录。")
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

    with st.expander("查看原始 Model Metrics 表", expanded=False):
        st.dataframe(metrics, use_container_width=True)


def _render_trade_signal_block(signal_row: pd.Series, *, header: str = "开单信号与理由") -> None:
    if signal_row is None or len(signal_row) == 0:
        return
    st.markdown(f"**{header}**")
    action_raw = str(signal_row.get("trade_signal", signal_row.get("policy_action", "Flat")))
    action_cn = _policy_action_cn(action_raw)
    trend = str(signal_row.get("trade_trend_context", "mixed"))
    trend_cn = {"bullish": "趋势偏多", "bearish": "趋势偏空", "mixed": "趋势混合"}.get(trend, trend)
    stop_price = _safe_float(signal_row.get("trade_stop_loss_price"))
    take_price = _safe_float(signal_row.get("trade_take_profit_price"))
    rr_ratio = _safe_float(signal_row.get("trade_rr_ratio"))
    support_score = _safe_float(signal_row.get("trade_support_score"))
    stop_text = _format_price(stop_price)
    take_text = _format_price(take_price)
    if action_raw == "Flat":
        stop_text = "不适用（观望）"
        take_text = "不适用（观望）"

    c1, c2, c3, c4, c5 = st.columns([1.6, 1.3, 1.2, 1.2, 1.0])
    with c1:
        _render_big_value("当前信号", action_cn, caption="观望 = 暂不下单")
    with c2:
        _render_big_value("趋势判断", trend_cn)
    with c3:
        _render_big_value("止损价", stop_text)
    with c4:
        _render_big_value("止盈价", take_text)
    with c5:
        _render_big_value("盈亏比(RR)", _format_float(rr_ratio, 2))
    if _is_finite_number(support_score):
        st.caption(f"技术共振分数: {_format_float(support_score, 0)}（>0 偏多，<0 偏空）")
    reason_text = _format_reason_tokens_cn(signal_row.get("trade_reason_tokens", signal_row.get("policy_reason", "-")))
    st.write(f"开单理由: {reason_text}")
    st.caption(
        "说明: 该理由由 EMA/MACD/SuperTrend/成交量/BOS/CHOCH 自动生成，"
        "用于解释信号，不构成投资建议。"
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
            with st.spinner("该标的不在预计算回测样本中，正在即时回测..."):
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
                fallback_note = "已为当前标的即时补跑回测（未写入全量回测文件，仅用于本页展示）。"
        except Exception as exc:
            fallback_note = f"即时回测失败：{exc}"

    if sub.empty:
        st.info("该标的暂无回测记录。可能是历史数据不足或数据源不可用。")
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
        _render_trade_signal_block(sig_view.iloc[-1], header="当前可执行信号（回测口径）")

    policy_row = sub[sub["strategy"] == "policy"].head(1)
    if not policy_row.empty:
        row = policy_row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("策略总收益", _format_change_pct(row.get("total_return")))
        c2.metric("策略夏普", _format_float(row.get("sharpe"), 2))
        c3.metric("最大回撤", _format_change_pct(row.get("max_drawdown")))
        c4.metric("胜率", _format_change_pct(row.get("win_rate")).replace("+", ""))
        st.caption(
            f"盈亏比: {_format_float(row.get('avg_win_loss_ratio'), 2)} | "
            f"Profit Factor: {_format_float(row.get('profit_factor'), 2)} | "
            f"交易次数: {int(_safe_float(row.get('trades_count')))}"
        )

    show = sub.copy()
    show["策略"] = show["strategy"].map(
        {
            "policy": "策略信号",
            "buy_hold": "买入并持有",
            "ma_crossover": "均线交叉",
            "naive_prev_bar": "前一日方向",
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
                "buy_hold": "买入并持有",
                "ma_crossover": "均线交叉",
                "naive_prev_bar": "前一日方向",
            }
        ).fillna(cmp["baseline"])
        st.markdown("**与基准对比（策略信号 - 基准）**")
        st.dataframe(
            cmp[["基准策略", "相对总收益提升", "相对夏普提升", "相对回撤变化"]],
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
                    "policy": "策略信号",
                    "buy_hold": "买入并持有",
                    "ma_crossover": "均线交叉",
                    "naive_prev_bar": "前一日方向",
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
                title=f"{symbol} 回测资金曲线（Walk-forward 串联）",
                xaxis_title="时间",
                yaxis_title="净值",
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
        fold_view["总收益"] = fold_view["total_return"].map(_format_change_pct)
        fold_view["夏普"] = fold_view["sharpe"].map(lambda x: _format_float(x, 2))
        fold_view["最大回撤"] = fold_view["max_drawdown"].map(_format_change_pct)
        fold_view["策略"] = fold_view["strategy"].astype(str)
        st.markdown("**分折（fold）表现**")
        st.dataframe(
            fold_view[["fold", "策略", "总收益", "夏普", "最大回撤"]].sort_values(["策略", "fold"]),
            use_container_width=True,
            hide_index=True,
        )

    trade_view = trades[
        (trades["market"].astype(str) == str(market))
        & _match_symbol(trades, col="symbol")
    ].copy()
    if not trade_view.empty:
        trade_view = trade_view.sort_values("entry_time", ascending=False).head(30).copy()
        if "entry_signal_reason" in trade_view.columns:
            trade_view["开单理由"] = trade_view["entry_signal_reason"].map(_format_reason_tokens_cn)
        elif "reason" in trade_view.columns:
            trade_view["开单理由"] = trade_view["reason"].map(_format_reason_tokens_cn)
        trade_view["方向"] = trade_view.get("side", pd.Series(["-"] * len(trade_view))).map(
            {"long": "做多", "short": "做空"}
        ).fillna("-")
        trade_view["信号"] = trade_view.get("entry_signal", pd.Series(["-"] * len(trade_view))).map(
            _policy_action_cn
        )
        trade_view["进场价"] = trade_view["entry_price"].map(_format_price)
        trade_view["止损价"] = trade_view.get("stop_loss_price", pd.Series([np.nan] * len(trade_view))).map(
            _format_price
        )
        trade_view["止盈价"] = trade_view.get("take_profit_price", pd.Series([np.nan] * len(trade_view))).map(
            _format_price
        )
        trade_view["净收益"] = trade_view.get("net_pnl_pct", pd.Series([np.nan] * len(trade_view))).map(
            _format_change_pct
        )
        trade_view["退出原因"] = trade_view.get("exit_reason", pd.Series(["-"] * len(trade_view))).map(
            {"stop_loss": "止损", "take_profit": "止盈", "time_exit": "时间平仓", "flat": "无持仓（未开单）"}
        ).fillna("-")
        st.markdown("**最近开单记录（含止盈止损）**")
        show_cols = [
            "entry_time",
            "方向",
            "信号",
            "进场价",
            "止损价",
            "止盈价",
            "净收益",
            "退出原因",
            "开单理由",
        ]
        show_cols = [c for c in show_cols if c in trade_view.columns]
        st.dataframe(trade_view[show_cols], use_container_width=True, hide_index=True)


def _render_snapshot_result(df: pd.DataFrame, title_prefix: str) -> None:
    df = _ensure_snapshot_factors(df)
    if df.empty:
        st.warning("当前选择未能生成预测快照。")
        return
    work = df.copy()
    try:
        cfg = _load_main_config_cached()
        if "market_type" not in work.columns:
            work["market_type"] = np.where(work["market"].astype(str).eq("crypto"), "spot", "cash")
        if "p_up" not in work.columns:
            work["p_up"] = float("nan")
        if "volatility_score" not in work.columns and {"q90_change_pct", "q10_change_pct"}.issubset(work.columns):
            work["volatility_score"] = (
                pd.to_numeric(work["q90_change_pct"], errors="coerce")
                - pd.to_numeric(work["q10_change_pct"], errors="coerce")
            )
        if "confidence_score" not in work.columns:
            work["confidence_score"] = 50.0
        if "risk_level" not in work.columns:
            work["risk_level"] = "medium"
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
    row = work.iloc[0]
    if pd.isna(row.get("current_price")):
        st.error(f"价格获取失败: {row.get('price_source', '-')}")
        if "error_message" in row and pd.notna(row.get("error_message")):
            st.caption(str(row.get("error_message")))
        return

    delta_abs = row.get("predicted_change_abs")
    delta_text = f"{float(delta_abs):+,.2f}" if _is_finite_number(delta_abs) else "-"
    action_text = _policy_action_cn(str(row.get("policy_action", "Flat")))
    s1, s2 = st.columns(2)
    with s1:
        _render_big_value("当前价格", _format_price(row.get("current_price")))
    with s2:
        _render_big_value("预测价格", _format_price(row.get("predicted_price")))
    s3, s4 = st.columns(2)
    with s3:
        _render_big_value("预计涨跌幅", _format_change_pct(row.get("predicted_change_pct")), caption=f"价差: {delta_text}")
    with s4:
        _render_big_value("策略动作", action_text, caption="观望 = 暂不下单")
    expected_date_full = str(row.get("expected_date_market", "-"))
    st.markdown("**预期日期（完整）**")
    st.code(expected_date_full)
    st.caption(
        f"价格源: {row.get('price_source', '-')} | 预测方法: {row.get('prediction_method', '-')}"
    )
    if "policy_position_size" in row.index:
        st.caption(
            f"建议仓位: {(float(row.get('policy_position_size')) if _is_finite_number(row.get('policy_position_size')) else 0.0):.1%} | "
            f"预期净优势: {_format_change_pct(row.get('policy_expected_edge_pct'))} | "
            f"策略理由: {row.get('policy_reason', '-')}"
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
        _render_trade_signal_block(signal_ctx.iloc[-1], header="开单信号与风控计划")
    _render_core_field_explain()

    st.markdown("**量化因子（风险 + 市场行为）**")
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    f1.metric("市值因子", _format_float(row.get("size_factor"), digits=3))
    f2.metric("价值因子", _format_float(row.get("value_factor"), digits=4))
    f3.metric("成长因子", _format_change_pct(row.get("growth_factor")))
    f4.metric("动能因子", _format_change_pct(row.get("momentum_factor")))
    f5.metric("反转因子", _format_change_pct(row.get("reversal_factor")))
    f6.metric("低波动因子", _format_change_pct(row.get("low_vol_factor")))

    market_cap = row.get("market_cap_usd")
    market_cap_text = _format_price(market_cap) if _is_finite_number(market_cap) else "-"
    st.caption(
        "风险因子来源: "
        f"size={row.get('size_factor_source', '-')}, "
        f"value={row.get('value_factor_source', '-')}, "
        f"growth={row.get('growth_factor_source', '-')} | "
        f"Market Cap(USD): {market_cap_text}"
    )
    _render_factor_explain()

    _render_projection_chart(
        current_price=float(row.get("current_price")),
        q10_change_pct=float(row.get("q10_change_pct")),
        q50_change_pct=float(row.get("q50_change_pct")),
        q90_change_pct=float(row.get("q90_change_pct")),
        expected_date_label=expected_date_full,
        title=f"{title_prefix} 预测可视化图（q10 / q50 / q90）",
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
    st.caption("说明：P(up)/P(down)是方向概率；预测价格是幅度模型结果，两者短期可能不完全一致。")

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
        _render_big_value("当前价格", _format_price(current_price))
    with p2:
        _render_big_value("预测价格 (q50)", _format_price(pred_price))
    p3, p4 = st.columns(2)
    with p3:
        _render_big_value("预计涨跌幅", _format_change_pct(pred_ret_q50), caption=f"价差: {delta_text}")
    with p4:
        _render_big_value("策略动作", action_text, caption="观望 = 暂不下单")
    st.markdown("**预期日期（完整）**")
    st.code(expected_date)
    st.caption(
        f"模型基准价（最后收盘）: {_format_price(model_base_price)} | "
        f"按基准价口径预测价: {_format_price(pred_price_base)}"
    )
    if policy_row:
        st.caption(
            f"建议仓位: {(float(policy_row.get('policy_position_size')) if _is_finite_number(policy_row.get('policy_position_size')) else 0.0):.1%} | "
            f"预期净优势: {_format_change_pct(policy_row.get('policy_expected_edge_pct'))} | "
            f"策略理由: {policy_row.get('policy_reason', '-')}"
        )

    _render_projection_chart(
        current_price=float(current_price),
        q10_change_pct=float(latest.get(q10_col, float("nan"))),
        q50_change_pct=float(latest.get(q50_col, float("nan"))),
        q90_change_pct=float(latest.get(q90_col, float("nan"))),
        expected_date_label=expected_date,
        title=f"{branch_name.capitalize()} 预测可视化（q10 / q50 / q90）",
    )


def _render_btc_model_detail_section(
    btc_live: float | None,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> None:
    st.markdown("---")
    st.subheader("BTC 模型详情（Hourly / Daily）")
    if btc_live is not None:
        st.info(f"BTC 实时价格 (Binance.US): **${btc_live:,.2f}**")
    else:
        st.warning("BTC 实时价格获取失败（不影响模型详情展示）。")

    btc_signal_ctx = _load_symbol_signal_context_cached(
        market="crypto",
        symbol="BTCUSDT",
        provider="binance",
        fallback_symbol="BTCUSDT",
    )
    if not btc_signal_ctx.empty:
        _render_trade_signal_block(btc_signal_ctx.iloc[-1], header="BTC 当前开单信号（技术触发 + 风控）")

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
    if not (np.isfinite(current_price) and np.isfinite(q10) and np.isfinite(q50) and np.isfinite(q90)):
        return pd.DataFrame()

    predicted_price = current_price * (1.0 + q50)
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
    st.header("Crypto 页面")
    catalog = get_universe_catalog()["crypto"]
    pool_key = st.selectbox(
        "选择加密池",
        options=list(catalog.keys()),
        format_func=lambda k: catalog[k],
        key="crypto_pool_page",
    )
    uni = _load_universe_cached("crypto", pool_key)
    choice = st.selectbox("选择币种", uni["display"].tolist(), key="crypto_symbol_page")
    row = uni[uni["display"] == choice].iloc[0]
    snap = _build_snapshot_for_selected(market="crypto", row=row)

    snapshot_symbol = str(row.get("snapshot_symbol", "")).upper()
    symbol = str(row.get("symbol", "")).upper()
    is_btc = snapshot_symbol == "BTCUSDT" or symbol == "BTC"
    if is_btc:
        model_snap = _build_btc_model_snapshot_from_hourly(
            hourly_df=hourly_df,
            btc_live=btc_live,
            fallback_snapshot=snap,
        )
        if not model_snap.empty:
            _render_snapshot_result(model_snap, title_prefix="Crypto（BTC主模型）")
            st.caption("当前 BTC 顶部卡片与下方 BTC 模型详情已统一口径：Hourly h=4。")
        else:
            _render_snapshot_result(snap, title_prefix="Crypto")
    else:
        _render_snapshot_result(snap, title_prefix="Crypto")
        st.caption("当前币种卡片使用快照基线口径（非 BTC 主模型）。")

    _render_btc_model_detail_section(btc_live=btc_live, hourly_df=hourly_df, daily_df=daily_df)

    st.markdown("---")
    st.subheader("模型效果解读")
    metrics_path = processed_dir / "metrics_walk_forward_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        _render_model_metrics_readable(metrics)
    else:
        st.info("未找到模型评估结果（metrics_walk_forward_summary.csv）。")

    bt_symbol = str(row.get("snapshot_symbol", row.get("symbol", "")))
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
        title="Crypto 回测结果（该币种）",
    )


def _render_cn_page() -> None:
    st.header("A股 页面")
    catalog = get_universe_catalog()["cn_equity"]
    pool_key = st.selectbox(
        "选择A股股票池",
        options=list(catalog.keys()),
        format_func=lambda k: catalog[k],
        key="cn_pool_page",
    )
    uni = _load_universe_cached("cn_equity", pool_key)
    choice = st.selectbox("选择A股标的", uni["display"].tolist(), key="cn_symbol_page")
    row = uni[uni["display"] == choice].iloc[0]
    snap = _build_snapshot_for_selected(market="cn_equity", row=row)
    _render_snapshot_result(snap, title_prefix="A股")
    _render_symbol_backtest_section(
        processed_dir=Path("data/processed"),
        market="cn_equity",
        symbol=str(row.get("symbol", "")),
        symbol_aliases=[str(row.get("code", "")), str(row.get("name", ""))],
        provider="yahoo",
        fallback_symbol=str(row.get("symbol", "")),
        title="A股 回测结果（该标的）",
    )


def _render_us_page() -> None:
    st.header("美股 页面")
    catalog = get_universe_catalog()["us_equity"]
    pool_key = st.selectbox(
        "选择美股股票池",
        options=list(catalog.keys()),
        format_func=lambda k: catalog[k],
        key="us_pool_page",
    )
    uni = _load_universe_cached("us_equity", pool_key)
    choice = st.selectbox("选择美股标的", uni["display"].tolist(), key="us_symbol_page")
    row = uni[uni["display"] == choice].iloc[0]
    snap = _build_snapshot_for_selected(market="us_equity", row=row)
    _render_snapshot_result(snap, title_prefix="美股")
    _render_symbol_backtest_section(
        processed_dir=Path("data/processed"),
        market="us_equity",
        symbol=str(row.get("symbol", "")),
        symbol_aliases=[str(row.get("name", ""))],
        provider="yahoo",
        fallback_symbol=str(row.get("symbol", "")),
        title="美股 回测结果（该标的）",
    )


def _status_cn(status: str) -> str:
    mapping = {"Active": "可执行", "Watch": "观察", "Retired": "暂停"}
    return mapping.get(status, status)


def _action_cn(action: str) -> str:
    mapping = {"Keep/Open": "持有或新开", "Monitor/Reduce": "观察或减仓", "Remove": "移除"}
    return mapping.get(action, action)


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


def _render_tracking_page(processed_dir: Path) -> None:
    st.header("Selection / Research / Tracking 页面")
    tracking_dir = processed_dir / "tracking"
    ranked = _load_csv(tracking_dir / "ranked_universe.csv")
    actions = _load_csv(tracking_dir / "tracking_actions.csv")
    coverage = _load_csv(tracking_dir / "coverage_matrix.csv")
    report_path = tracking_dir / "data_quality_report.md"

    if ranked.empty:
        st.info(
            "还没有 tracking 结果。请先运行 `python -m src.markets.tracking --config configs/config.yaml`。"
        )
        return

    ranked = ranked.copy()
    ranked["状态"] = ranked["status"].map(_status_cn)
    ranked["建议动作"] = ranked["recommended_action"].map(_action_cn)
    ranked["预计涨跌幅"] = ranked["predicted_change_pct"].map(_format_change_pct)
    ranked["总分(0-100)"] = ranked["total_score"].map(lambda x: _format_float(x, 1))
    ranked["告警说明"] = ranked["alerts"].map(_alert_cn)
    ranked["因子支持数(0-3)"] = ranked.get("factor_support_count", pd.Series([np.nan] * len(ranked))).fillna(0).astype(int)
    if "policy_action" in ranked.columns:
        ranked["策略动作"] = ranked["policy_action"].map(_policy_action_cn)
        ranked["建议仓位"] = ranked["policy_position_size"].map(
            lambda x: f"{float(x):.1%}" if _is_finite_number(x) else "-"
        )
        ranked["预期净优势"] = ranked["policy_expected_edge_pct"].map(_format_change_pct)
        ranked["策略理由"] = ranked["policy_reason"].fillna("-")

    total = len(ranked)
    active_n = int((ranked["状态"] == "可执行").sum())
    watch_n = int((ranked["状态"] == "观察").sum())
    retired_n = int((ranked["状态"] == "暂停").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("候选总数", f"{total}")
    c2.metric("可执行", f"{active_n}")
    c3.metric("观察", f"{watch_n}")
    c4.metric("暂停", f"{retired_n}")

    count_by_status = ranked["状态"].value_counts().reindex(["可执行", "观察", "暂停"]).fillna(0)
    fig_status = go.Figure(
        go.Bar(
            x=count_by_status.index.tolist(),
            y=count_by_status.values.tolist(),
            text=count_by_status.values.tolist(),
            textposition="outside",
        )
    )
    fig_status.update_layout(
        title="状态分布",
        xaxis_title="状态",
        yaxis_title="数量",
        template="plotly_white",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_status, use_container_width=True)
    if "策略动作" in ranked.columns:
        action_count = (
            ranked["策略动作"]
            .value_counts()
            .reindex(["做多", "做空", "观望"])
            .fillna(0)
        )
        fig_policy = go.Figure(
            go.Bar(
                x=action_count.index.tolist(),
                y=action_count.values.tolist(),
                text=action_count.values.tolist(),
                textposition="outside",
            )
        )
        fig_policy.update_layout(
            title="策略动作分布（Policy Layer）",
            xaxis_title="动作",
            yaxis_title="数量",
            template="plotly_white",
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_policy, use_container_width=True)

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
    market_options = ["全部"] + sorted(ranked["market"].dropna().unique().tolist())
    status_options = ["全部"] + ["可执行", "观察", "暂停"]
    mkt = filter_col1.selectbox("市场筛选", market_options, index=0, key="track_market_page")
    sts = filter_col2.selectbox("状态筛选", status_options, index=0, key="track_status_page")
    top_n = int(filter_col3.slider("展示前N条", 10, 300, 50, 10, key="track_topn_page"))

    view = ranked.copy()
    if mkt != "全部":
        view = view[view["market"] == mkt]
    if sts != "全部":
        view = view[view["状态"] == sts]
    view = view.sort_values("total_score", ascending=False)

    table_cols = [
        "market",
        "name",
        "状态",
        "建议动作",
        "策略动作",
        "建议仓位",
        "预期净优势",
        "总分(0-100)",
        "预计涨跌幅",
        "因子支持数(0-3)",
        "告警说明",
        "策略理由",
    ]
    table_cols = [c for c in table_cols if c in view.columns]
    st.dataframe(view[table_cols].head(top_n), use_container_width=True)

    with st.expander("术语说明（通俗版）", expanded=False):
        st.markdown(
            "- `预计涨跌幅`：模型预计未来窗口内，价格大概涨/跌多少。\n"
            "- `因子支持数(0-3)`：成长/动能/价值这3个方向里，有几个支持当前预测方向。\n"
            "- `策略动作`：Policy 层输出（做多/做空/观望）。`观望`= 当前不建议开单，不是做空。\n"
            "- `建议仓位`：0%-100%，反映信号强度和不确定性修正后的结果。\n"
            "- `预期净优势`：考虑手续费/滑点后的期望优势（越高越好）。\n"
            "- `总分(0-100)`：综合了流动性、数据质量、历史长度和覆盖能力。\n"
            "- `状态`：\n"
            "  - `可执行`：可作为优先候选\n"
            "  - `观察`：先观察，不建议直接执行\n"
            "  - `暂停`：当前不建议使用"
        )

    if not actions.empty:
        actions = actions.copy()
        actions["状态"] = actions["status"].map(_status_cn)
        actions["建议动作"] = actions["recommended_action"].map(_action_cn)
        actions["预计涨跌幅"] = actions["predicted_change_pct"].map(_format_change_pct)
        actions["告警说明"] = actions["alerts"].map(_alert_cn)
        if "policy_action" in actions.columns:
            actions["策略动作"] = actions["policy_action"].map(_policy_action_cn)
            actions["建议仓位"] = actions["policy_position_size"].map(
                lambda x: f"{float(x):.1%}" if _is_finite_number(x) else "-"
            )
            actions["预期净优势"] = actions["policy_expected_edge_pct"].map(_format_change_pct)
            actions["策略理由"] = actions["policy_reason"].fillna("-")
        st.subheader("动作建议明细")
        action_cols = [
            "market",
            "name",
            "状态",
            "建议动作",
            "策略动作",
            "建议仓位",
            "预期净优势",
            "预计涨跌幅",
            "策略理由",
            "告警说明",
        ]
        action_cols = [c for c in action_cols if c in actions.columns]
        st.dataframe(actions[action_cols].head(top_n), use_container_width=True)

    if not coverage.empty and "hard_filter_pass" in coverage.columns:
        pass_rate = float(coverage["hard_filter_pass"].astype(bool).mean())
        st.caption(f"硬门槛通过率: {pass_rate:.1%}")

    if report_path.exists():
        with st.expander("数据质量报告", expanded=False):
            st.markdown(report_path.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Multi-Market Forecast Dashboard", layout="wide")
    st.title("Multi-Market Forecast Dashboard")
    st.caption("页面导航：Crypto / A股 / 美股 / 交易时间段预测 / Selection-Research-Tracking")
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

    if st.button("Clear cache and reload", use_container_width=False):
        st.cache_data.clear()
        st.rerun()

    processed_dir = Path("data/processed")
    hourly = _load_csv(processed_dir / "predictions_hourly.csv")
    daily = _load_csv(processed_dir / "predictions_daily.csv")
    btc_live = _fetch_live_btc_price()

    page = st.sidebar.radio(
        "页面",
        options=[
            "Crypto 页面",
            "A股 页面",
            "美股 页面",
            "交易时间段预测（Crypto）",
            "Selection / Research / Tracking 页面",
        ],
        index=0,
    )

    if page == "Crypto 页面":
        _render_crypto_page(
            processed_dir=processed_dir,
            btc_live=btc_live,
            hourly_df=hourly,
            daily_df=daily,
        )
    elif page == "A股 页面":
        _render_cn_page()
    elif page == "美股 页面":
        _render_us_page()
    elif page == "交易时间段预测（Crypto）":
        _render_crypto_session_page()
    else:
        _render_tracking_page(processed_dir)


if __name__ == "__main__":
    main()
