from __future__ import annotations

import numpy as np
import pandas as pd

from src.markets.simulated_kline import (
    SimulationConfig,
    simulate_daily_future_ohlc,
    simulate_future_ohlc,
)


def _build_profile_full24() -> pd.DataFrame:
    hours = np.arange(24)
    p_up = 0.5 + 0.05 * np.sin(hours / 24.0 * np.pi * 2.0)
    q50 = 0.0006 * np.cos(hours / 24.0 * np.pi * 2.0)
    q10 = q50 - 0.003
    q90 = q50 + 0.003
    return pd.DataFrame(
        {
            "hour_bj": hours,
            "p_up": p_up,
            "q10_change_pct": q10,
            "q50_change_pct": q50,
            "q90_change_pct": q90,
            "is_trading_hour": 1,
        }
    )


def test_simulated_kline_reproducible_with_same_seed() -> None:
    profile = _build_profile_full24()
    cfg = SimulationConfig(n_steps=12, n_paths=200, seed=42)
    res1 = simulate_future_ohlc(
        profile,
        current_price=100.0,
        market_mode="crypto",
        reference_ts_bj="2026-02-08 12:00:00+0800",
        config=cfg,
    )
    res2 = simulate_future_ohlc(
        profile,
        current_price=100.0,
        market_mode="crypto",
        reference_ts_bj="2026-02-08 12:00:00+0800",
        config=cfg,
    )
    df1 = res1["sim_df"]
    df2 = res2["sim_df"]
    assert len(df1) == len(df2) == 12
    assert np.allclose(df1["close"].to_numpy(), df2["close"].to_numpy(), rtol=0, atol=1e-12)
    assert np.allclose(df1["open"].to_numpy(), df2["open"].to_numpy(), rtol=0, atol=1e-12)


def test_simulated_kline_ohlc_is_valid() -> None:
    profile = _build_profile_full24()
    res = simulate_future_ohlc(
        profile,
        current_price=100.0,
        market_mode="crypto",
        reference_ts_bj="2026-02-08 12:00:00+0800",
        config=SimulationConfig(n_steps=24, n_paths=300, seed=7),
    )
    df = res["sim_df"]
    assert not df.empty
    assert (df["high"] >= np.maximum(df["open"], df["close"])).all()
    assert (df["low"] <= np.minimum(df["open"], df["close"])).all()
    assert (df["low"] > 0).all()
    rep = res["sim_df_representative"]
    assert not rep.empty
    assert (rep["high"] >= np.maximum(rep["open"], rep["close"])).all()
    assert (rep["low"] <= np.minimum(rep["open"], rep["close"])).all()


def test_index_mode_filters_to_trading_hours_only() -> None:
    profile = _build_profile_full24()
    profile["is_trading_hour"] = profile["hour_bj"].between(8, 15).astype(int)
    res = simulate_future_ohlc(
        profile,
        current_price=4000.0,
        market_mode="index",
        active_session="asia",
        reference_ts_bj="2026-02-08 00:00:00+0800",
        config=SimulationConfig(n_steps=8, n_paths=120, seed=11),
    )
    df = res["sim_df"]
    assert not df.empty
    assert df["hour_bj"].between(8, 15).all()
    assert set(df["session_name"].unique().tolist()) == {"asia"}


def test_simulation_degrades_when_ops_limit_exceeded() -> None:
    profile = _build_profile_full24()
    cfg = SimulationConfig(n_steps=24, n_paths=1000, seed=101, max_ops=5000)
    res = simulate_future_ohlc(
        profile,
        current_price=100.0,
        market_mode="crypto",
        reference_ts_bj="2026-02-08 12:00:00+0800",
        config=cfg,
    )
    diag = res["diagnostics"]
    assert bool(diag.get("degrade_flag")) is True
    assert int(diag.get("n_paths_used", 0)) < int(diag.get("n_paths_requested", 0))
    assert int(diag.get("n_paths_used", 0)) * 24 <= cfg.max_ops


def test_daily_simulated_kline_outputs_daily_granularity() -> None:
    now = pd.Timestamp("2026-02-08 00:00:00", tz="Asia/Shanghai")
    daily = pd.DataFrame(
        {
            "date_bj": [(now + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(12)],
            "p_up": [0.55] * 12,
            "q10_change_pct": [-0.01] * 12,
            "q50_change_pct": [0.0025] * 12,
            "q90_change_pct": [0.015] * 12,
        }
    )
    res = simulate_daily_future_ohlc(
        daily,
        current_price=100.0,
        reference_ts_bj="2026-02-08 12:00:00+0800",
        config=SimulationConfig(n_steps=8, n_paths=200, seed=123),
    )
    df = res["sim_df"]
    rep = res["sim_df_representative"]
    assert len(df) == 8
    assert len(rep) == 8
    assert (df["session_name"] == "daily").all()
    assert res["diagnostics"].get("profile_granularity") == "daily"
