from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BJ_TZ = "Asia/Shanghai"

SESSION_HOURS = {
    "asia": set(range(8, 16)),
    "europe": set(range(16, 24)),
    "us": set(range(0, 8)),
}


@dataclass
class SimulationConfig:
    n_steps: int = 24
    n_paths: int = 300
    seed: int = 42
    agg: str = "mean"  # mean / median
    vol_scale: float = 1.0
    sigma_floor: float = 3e-4
    sigma_cap: float = 0.03
    wick_cap: float = 0.015
    wick_scale: float = 0.6
    max_ops: int = 50_000
    missing_ratio_hard_limit: float = 0.40
    fit_tol_p: float = 0.005
    fit_tol_q50: float = 0.001
    fit_tol_qband: float = 0.0015


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def _safe_ts_bj(x: Any) -> pd.Timestamp:
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp.now(tz=BJ_TZ)
    if ts.tzinfo is None:
        return ts.tz_localize(BJ_TZ)
    return ts.tz_convert(BJ_TZ)


def _session_from_hour_bj(hour_bj: int) -> str:
    h = int(hour_bj)
    if 8 <= h <= 15:
        return "asia"
    if 16 <= h <= 23:
        return "europe"
    return "us"


def _resolve_quantile_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    candidates = {
        "p_up": ["p_up"],
        "q10": ["q10_change_pct", "q10"],
        "q50": ["q50_change_pct", "q50"],
        "q90": ["q90_change_pct", "q90"],
    }
    for k, cols in candidates.items():
        for c in cols:
            if c in df.columns:
                mapping[k] = c
                break
    return mapping


def _normalize_hour_profile(
    step_profile: pd.DataFrame,
    *,
    market_mode: str,
    active_session: str | None,
    sigma_floor: float,
    missing_ratio_hard_limit: float,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if step_profile.empty:
        raise ValueError("step_profile is empty")

    df = step_profile.copy()
    if "hour_bj" not in df.columns:
        if "hour_label" in df.columns:
            h = pd.to_numeric(
                df["hour_label"].astype(str).str.extract(r"(\d{1,2})", expand=False),
                errors="coerce",
            )
            df["hour_bj"] = h
        else:
            raise ValueError("step_profile missing hour_bj/hour_label")

    df["hour_bj"] = pd.to_numeric(df["hour_bj"], errors="coerce")
    df = df.dropna(subset=["hour_bj"]).copy()
    if df.empty:
        raise ValueError("step_profile has no valid hour_bj")
    df["hour_bj"] = df["hour_bj"].astype(int) % 24

    if str(market_mode).strip().lower() == "index":
        if "is_trading_hour" in df.columns:
            mask = pd.to_numeric(df["is_trading_hour"], errors="coerce").fillna(0).astype(int) == 1
            df = df[mask].copy()
        if active_session:
            hs = SESSION_HOURS.get(str(active_session).strip().lower(), set())
            if hs:
                df = df[df["hour_bj"].isin(hs)].copy()
        if df.empty:
            raise ValueError("index profile has no tradable hours after filtering")

    qcols = _resolve_quantile_columns(df)
    required_map = {"p_up", "q10", "q50", "q90"}
    if not required_map.issubset(qcols.keys()):
        missing = sorted(required_map - set(qcols.keys()))
        raise ValueError(f"step_profile missing required columns: {missing}")

    out = (
        df.sort_values("hour_bj")
        .drop_duplicates(subset=["hour_bj"], keep="last")
        .reset_index(drop=True)
    )

    p_up_raw = pd.to_numeric(out[qcols["p_up"]], errors="coerce")
    q10_raw = pd.to_numeric(out[qcols["q10"]], errors="coerce")
    q50_raw = pd.to_numeric(out[qcols["q50"]], errors="coerce")
    q90_raw = pd.to_numeric(out[qcols["q90"]], errors="coerce")

    row_missing = p_up_raw.isna() | q10_raw.isna() | q50_raw.isna() | q90_raw.isna()
    missing_ratio = float(row_missing.mean()) if len(row_missing) > 0 else 1.0

    # fallback policy
    p_up = p_up_raw.fillna(0.5).clip(0.01, 0.99)
    q50 = q50_raw.interpolate(limit_direction="both").fillna(0.0)
    band = np.maximum(np.abs(q50) * 0.3, float(sigma_floor) * 1.2816)
    q10 = q10_raw.fillna(q50 - band)
    q90 = q90_raw.fillna(q50 + band)

    # quantile monotonic fix per row
    arr = np.column_stack(
        [
            pd.to_numeric(q10, errors="coerce").values,
            pd.to_numeric(q50, errors="coerce").values,
            pd.to_numeric(q90, errors="coerce").values,
        ]
    )
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr_sorted = np.sort(arr, axis=1)
    q10_fix = arr_sorted[:, 0]
    q50_fix = arr_sorted[:, 1]
    q90_fix = arr_sorted[:, 2]

    out["p_up"] = p_up.values
    out["q10"] = q10_fix
    out["q50"] = q50_fix
    out["q90"] = q90_fix

    if "session_name" not in out.columns:
        out["session_name"] = out["hour_bj"].map(_session_from_hour_bj)

    fallback_info = {
        "missing_ratio": missing_ratio,
        "fallback_p_up_count": int(p_up_raw.isna().sum()),
        "fallback_q50_count": int(q50_raw.isna().sum()),
        "fallback_q10_count": int(q10_raw.isna().sum()),
        "fallback_q90_count": int(q90_raw.isna().sum()),
        "quantile_reordered_count": int(
            np.sum((arr[:, 0] > arr[:, 1]) | (arr[:, 1] > arr[:, 2]))
        ),
        "insufficient_inputs": bool(missing_ratio > float(missing_ratio_hard_limit)),
    }
    return out[["hour_bj", "session_name", "p_up", "q10", "q50", "q90"]], fallback_info


def _build_future_step_table(
    profile_by_hour: pd.DataFrame,
    *,
    n_steps: int,
    reference_ts_bj: pd.Timestamp,
    market_mode: str,
) -> pd.DataFrame:
    df = profile_by_hour.copy()
    hour_map = {int(r["hour_bj"]): r for _, r in df.iterrows()}
    tradable_hours = sorted(hour_map.keys())
    if not tradable_hours:
        raise ValueError("No tradable hours in profile")

    cursor = reference_ts_bj.floor("h")
    rows: List[Dict[str, Any]] = []
    guard = 0
    max_guard = max(72, int(n_steps) * 24)
    while len(rows) < int(n_steps) and guard < max_guard:
        guard += 1
        cursor = cursor + pd.Timedelta(hours=1)
        h = int(cursor.hour)
        if str(market_mode).strip().lower() == "index" and h not in tradable_hours:
            continue
        if h not in hour_map:
            continue
        r = hour_map[h]
        rows.append(
            {
                "step_idx": len(rows) + 1,
                "ts_market": cursor,
                "ts_utc": cursor.tz_convert("UTC"),
                "hour_bj": h,
                "session_name": str(r["session_name"]),
                "p_up": _safe_float(r["p_up"]),
                "q10": _safe_float(r["q10"]),
                "q50": _safe_float(r["q50"]),
                "q90": _safe_float(r["q90"]),
            }
        )
    if len(rows) < int(n_steps):
        raise ValueError(f"Unable to build enough future steps: {len(rows)}/{int(n_steps)}")
    return pd.DataFrame(rows)


def build_simulation_summary(sim_df: pd.DataFrame, current_price: float) -> Dict[str, float]:
    if sim_df.empty or not np.isfinite(_safe_float(current_price)):
        return {}
    cp = float(current_price)
    end_price = _safe_float(sim_df["close"].iloc[-1])
    cum = pd.to_numeric(sim_df.get("cum_ret_from_start"), errors="coerce")
    up = float(cum.max()) if not cum.empty else float("nan")
    dd = float(cum.min()) if not cum.empty else float("nan")
    band_width_abs = pd.to_numeric(sim_df.get("close_q90"), errors="coerce") - pd.to_numeric(
        sim_df.get("close_q10"), errors="coerce"
    )
    band_width_abs_mean = float(band_width_abs.mean()) if len(band_width_abs) > 0 else float("nan")
    band_width_pct = band_width_abs / pd.to_numeric(sim_df.get("close"), errors="coerce")
    band_width_pct_mean = float(band_width_pct.mean()) if len(band_width_pct) > 0 else float("nan")
    return {
        "end_price": end_price,
        "end_ret_pct": (end_price / cp - 1.0) if np.isfinite(end_price) else float("nan"),
        "max_up_from_start": up,
        "max_dd_from_start": dd,
        "avg_band_width_abs": band_width_abs_mean,
        "avg_band_width_pct": band_width_pct_mean,
    }


def _build_sim_df_from_arrays(
    steps: pd.DataFrame,
    *,
    current_price: float,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    close_q10: np.ndarray,
    close_q90: np.ndarray,
    active_session_value: str,
) -> pd.DataFrame:
    out = steps.copy()
    out["open"] = np.asarray(open_arr, dtype=float)
    out["high"] = np.asarray(high_arr, dtype=float)
    out["low"] = np.asarray(low_arr, dtype=float)
    out["close"] = np.asarray(close_arr, dtype=float)
    out["close_q10"] = np.asarray(close_q10, dtype=float)
    out["close_q90"] = np.asarray(close_q90, dtype=float)
    out["mean_step_ret"] = out["close"] / out["open"] - 1.0
    out["cum_ret_from_start"] = out["close"] / float(current_price) - 1.0
    out["active_session"] = str(active_session_value or "")
    return out


def _pick_terminal_path_idx(terminal_prices: np.ndarray, q: float) -> int:
    arr = np.asarray(terminal_prices, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return 0
    target = float(np.nanquantile(arr, q))
    dist = np.abs(arr - target)
    return int(np.nanargmin(dist))


def _normalize_daily_profile(
    daily_profile: pd.DataFrame,
    *,
    sigma_floor: float,
    missing_ratio_hard_limit: float,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if daily_profile.empty:
        raise ValueError("daily_profile is empty")
    df = daily_profile.copy()
    qcols = _resolve_quantile_columns(df)
    required_map = {"q10", "q50", "q90"}
    if not required_map.issubset(qcols.keys()):
        missing = sorted(required_map - set(qcols.keys()))
        raise ValueError(f"daily_profile missing required columns: {missing}")

    if "date_bj" in df.columns:
        ts_market = pd.to_datetime(df["date_bj"], errors="coerce")
    elif "ts_market" in df.columns:
        ts_market = pd.to_datetime(df["ts_market"], errors="coerce")
    else:
        ts_market = pd.to_datetime(df.get("timestamp_market"), errors="coerce")
    if ts_market.isna().all():
        base = pd.Timestamp.now(tz=BJ_TZ).normalize()
        ts_market = pd.Series([base + pd.Timedelta(days=i + 1) for i in range(len(df))])

    ts_market = pd.to_datetime(ts_market, errors="coerce")
    if ts_market.dt.tz is None:
        ts_market = ts_market.dt.tz_localize(BJ_TZ)
    else:
        ts_market = ts_market.dt.tz_convert(BJ_TZ)
    df["ts_market"] = ts_market
    df = df.dropna(subset=["ts_market"]).copy()
    if df.empty:
        raise ValueError("daily_profile has no valid date")
    df = df.sort_values("ts_market").reset_index(drop=True)

    p_up_raw = pd.to_numeric(df[qcols.get("p_up", "")], errors="coerce") if "p_up" in qcols else pd.Series(np.nan, index=df.index)
    q10_raw = pd.to_numeric(df[qcols["q10"]], errors="coerce")
    q50_raw = pd.to_numeric(df[qcols["q50"]], errors="coerce")
    q90_raw = pd.to_numeric(df[qcols["q90"]], errors="coerce")

    row_missing = p_up_raw.isna() | q10_raw.isna() | q50_raw.isna() | q90_raw.isna()
    missing_ratio = float(row_missing.mean()) if len(row_missing) > 0 else 1.0

    p_up = p_up_raw.fillna(0.5).clip(0.01, 0.99)
    q50 = q50_raw.interpolate(limit_direction="both").fillna(0.0)
    band = np.maximum(np.abs(q50) * 0.3, float(sigma_floor) * 1.2816)
    q10 = q10_raw.fillna(q50 - band)
    q90 = q90_raw.fillna(q50 + band)

    arr = np.column_stack(
        [
            pd.to_numeric(q10, errors="coerce").values,
            pd.to_numeric(q50, errors="coerce").values,
            pd.to_numeric(q90, errors="coerce").values,
        ]
    )
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr_sorted = np.sort(arr, axis=1)

    out = pd.DataFrame(
        {
            "step_idx": np.arange(1, len(df) + 1, dtype=int),
            "ts_market": df["ts_market"].values,
            "ts_utc": pd.to_datetime(df["ts_market"], errors="coerce").dt.tz_convert("UTC").values,
            "hour_bj": pd.to_datetime(df["ts_market"], errors="coerce").dt.hour.astype(int).values,
            "session_name": "daily",
            "p_up": p_up.values,
            "q10": arr_sorted[:, 0],
            "q50": arr_sorted[:, 1],
            "q90": arr_sorted[:, 2],
            "date_bj": pd.to_datetime(df["ts_market"], errors="coerce").dt.strftime("%Y-%m-%d").values,
            "day_of_week": pd.to_datetime(df["ts_market"], errors="coerce").dt.day_name().values,
        }
    )

    fallback_info = {
        "missing_ratio": missing_ratio,
        "fallback_p_up_count": int(p_up_raw.isna().sum()),
        "fallback_q50_count": int(q50_raw.isna().sum()),
        "fallback_q10_count": int(q10_raw.isna().sum()),
        "fallback_q90_count": int(q90_raw.isna().sum()),
        "quantile_reordered_count": int(
            np.sum((arr[:, 0] > arr[:, 1]) | (arr[:, 1] > arr[:, 2]))
        ),
        "insufficient_inputs": bool(missing_ratio > float(missing_ratio_hard_limit)),
    }
    return out, fallback_info


def estimate_tp_sl_hit_prob(
    price_paths: np.ndarray,
    *,
    entry: float,
    tp: float,
    sl: float,
    side: str = "long",
) -> Dict[str, float]:
    if not isinstance(price_paths, np.ndarray) or price_paths.ndim != 2:
        return {
            "p_hit_tp_first": float("nan"),
            "p_hit_sl_first": float("nan"),
            "p_no_hit": float("nan"),
            "tp_hit_step_fastest": float("nan"),
            "tp_hit_step_median": float("nan"),
            "tp_hit_step_slowest": float("nan"),
            "sl_hit_step_fastest": float("nan"),
            "sl_hit_step_median": float("nan"),
            "sl_hit_step_slowest": float("nan"),
        }
    n = int(price_paths.shape[0])
    if n <= 0:
        return {
            "p_hit_tp_first": float("nan"),
            "p_hit_sl_first": float("nan"),
            "p_no_hit": float("nan"),
            "tp_hit_step_fastest": float("nan"),
            "tp_hit_step_median": float("nan"),
            "tp_hit_step_slowest": float("nan"),
            "sl_hit_step_fastest": float("nan"),
            "sl_hit_step_median": float("nan"),
            "sl_hit_step_slowest": float("nan"),
        }
    side_norm = str(side or "long").strip().lower()
    use_short = side_norm == "short"
    tp_first = 0
    sl_first = 0
    no_hit = 0
    tp_steps: List[int] = []
    sl_steps: List[int] = []
    for i in range(n):
        arr = price_paths[i]
        if use_short:
            tp_hits = np.where(arr <= float(tp))[0]
            sl_hits = np.where(arr >= float(sl))[0]
        else:
            tp_hits = np.where(arr >= float(tp))[0]
            sl_hits = np.where(arr <= float(sl))[0]
        first_tp = int(tp_hits[0]) if len(tp_hits) else 10**9
        first_sl = int(sl_hits[0]) if len(sl_hits) else 10**9
        if first_tp < first_sl:
            tp_first += 1
            tp_steps.append(first_tp + 1)
        elif first_sl < first_tp:
            sl_first += 1
            sl_steps.append(first_sl + 1)
        else:
            no_hit += 1
    def _step_stats(xs: List[int], stat: str) -> float:
        if not xs:
            return float("nan")
        arr = np.asarray(xs, dtype=float)
        if stat == "fast":
            return float(np.nanmin(arr))
        if stat == "slow":
            return float(np.nanmax(arr))
        return float(np.nanmedian(arr))
    return {
        "p_hit_tp_first": tp_first / n,
        "p_hit_sl_first": sl_first / n,
        "p_no_hit": no_hit / n,
        "tp_hit_step_fastest": _step_stats(tp_steps, "fast"),
        "tp_hit_step_median": _step_stats(tp_steps, "median"),
        "tp_hit_step_slowest": _step_stats(tp_steps, "slow"),
        "sl_hit_step_fastest": _step_stats(sl_steps, "fast"),
        "sl_hit_step_median": _step_stats(sl_steps, "median"),
        "sl_hit_step_slowest": _step_stats(sl_steps, "slow"),
    }


def _simulate_from_steps(
    steps: pd.DataFrame,
    *,
    current_price: float,
    config: SimulationConfig,
    fallback_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = config
    cp = _safe_float(current_price)
    if not np.isfinite(cp) or cp <= 0:
        raise ValueError("current_price is invalid")
    if steps.empty:
        raise ValueError("steps is empty")

    n_steps = int(len(steps))
    n_paths_req = int(max(10, cfg.n_paths))
    max_ops = int(max(1_000, cfg.max_ops))
    n_paths_use = n_paths_req
    degrade_flag = False
    degrade_reason = ""
    if n_paths_use * n_steps > max_ops:
        n_paths_use = max(50, max_ops // max(1, n_steps))
        n_paths_use = min(n_paths_use, n_paths_req)
        degrade_flag = True
        degrade_reason = "ops_limit"

    rng = np.random.default_rng(int(cfg.seed))

    p_up = pd.to_numeric(steps["p_up"], errors="coerce").fillna(0.5).clip(0.01, 0.99).to_numpy(dtype=float)
    q10 = pd.to_numeric(steps["q10"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    q50 = pd.to_numeric(steps["q50"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    q90 = pd.to_numeric(steps["q90"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    sigma_l = np.maximum((q50 - q10) / 1.2816, float(cfg.sigma_floor))
    sigma_r = np.maximum((q90 - q50) / 1.2816, float(cfg.sigma_floor))
    sigma_l = np.minimum(sigma_l * float(cfg.vol_scale), float(cfg.sigma_cap))
    sigma_r = np.minimum(sigma_r * float(cfg.vol_scale), float(cfg.sigma_cap))

    dir_up = rng.random((n_paths_use, n_steps)) < p_up.reshape(1, -1)
    z = np.abs(rng.standard_normal((n_paths_use, n_steps)))
    up_ret = q50.reshape(1, -1) + z * sigma_r.reshape(1, -1)
    dn_ret = q50.reshape(1, -1) - z * sigma_l.reshape(1, -1)
    ret = np.where(dir_up, np.maximum(up_ret, 0.0), np.minimum(dn_ret, 0.0))
    ret = np.clip(ret, -0.95, 0.95)

    lr = np.log1p(ret)
    cum_lr = np.cumsum(lr, axis=1)
    prices = np.concatenate(
        [np.full((n_paths_use, 1), cp, dtype=float), cp * np.exp(cum_lr)],
        axis=1,
    )
    open_paths = prices[:, :-1]
    close_paths = prices[:, 1:]

    sigma_step = ((sigma_l + sigma_r) / 2.0).reshape(1, -1)
    wick_noise = np.abs(rng.normal(0.0, float(cfg.wick_scale) * sigma_step, size=ret.shape))
    wick_noise = np.clip(wick_noise, 0.0, float(cfg.wick_cap))
    high_paths = np.maximum(open_paths, close_paths) * (1.0 + wick_noise)
    low_paths = np.minimum(open_paths, close_paths) * (1.0 - wick_noise)

    agg_mode = str(cfg.agg).strip().lower()
    if agg_mode == "median":
        close_agg = np.median(close_paths, axis=0)
    else:
        close_agg = np.mean(close_paths, axis=0)

    open_agg = np.empty(n_steps, dtype=float)
    open_agg[0] = cp
    if n_steps > 1:
        open_agg[1:] = close_agg[:-1]
    high_raw = np.quantile(high_paths, 0.70, axis=0)
    low_raw = np.quantile(low_paths, 0.30, axis=0)
    high_agg = np.maximum.reduce([high_raw, open_agg, close_agg])
    low_agg = np.minimum.reduce([low_raw, open_agg, close_agg])

    close_q10 = np.quantile(close_paths, 0.10, axis=0)
    close_q90 = np.quantile(close_paths, 0.90, axis=0)
    sim_df = _build_sim_df_from_arrays(
        steps,
        current_price=cp,
        open_arr=open_agg,
        high_arr=high_agg,
        low_arr=low_agg,
        close_arr=close_agg,
        close_q10=close_q10,
        close_q90=close_q90,
        active_session_value=str(steps.get("session_name", pd.Series([""])).iloc[0]) if "session_name" in steps.columns else "",
    )

    terminal_prices = close_paths[:, -1] if close_paths.size else np.array([], dtype=float)
    base_idx = _pick_terminal_path_idx(terminal_prices, 0.50)
    bull_idx = _pick_terminal_path_idx(terminal_prices, 0.90)
    bear_idx = _pick_terminal_path_idx(terminal_prices, 0.10)

    sim_df_representative = _build_sim_df_from_arrays(
        steps,
        current_price=cp,
        open_arr=open_paths[base_idx],
        high_arr=high_paths[base_idx],
        low_arr=low_paths[base_idx],
        close_arr=close_paths[base_idx],
        close_q10=close_q10,
        close_q90=close_q90,
        active_session_value=str(steps.get("session_name", pd.Series([""])).iloc[0]) if "session_name" in steps.columns else "",
    )
    sim_df_bull = _build_sim_df_from_arrays(
        steps,
        current_price=cp,
        open_arr=open_paths[bull_idx],
        high_arr=high_paths[bull_idx],
        low_arr=low_paths[bull_idx],
        close_arr=close_paths[bull_idx],
        close_q10=close_q10,
        close_q90=close_q90,
        active_session_value=str(steps.get("session_name", pd.Series([""])).iloc[0]) if "session_name" in steps.columns else "",
    )
    sim_df_bear = _build_sim_df_from_arrays(
        steps,
        current_price=cp,
        open_arr=open_paths[bear_idx],
        high_arr=high_paths[bear_idx],
        low_arr=low_paths[bear_idx],
        close_arr=close_paths[bear_idx],
        close_q10=close_q10,
        close_q90=close_q90,
        active_session_value=str(steps.get("session_name", pd.Series([""])).iloc[0]) if "session_name" in steps.columns else "",
    )

    sim_p_up = (ret > 0).mean(axis=0)
    sim_q10 = np.quantile(ret, 0.10, axis=0)
    sim_q50 = np.quantile(ret, 0.50, axis=0)
    sim_q90 = np.quantile(ret, 0.90, axis=0)

    err_p = np.abs(sim_p_up - p_up)
    err_q10 = np.abs(sim_q10 - q10)
    err_q50 = np.abs(sim_q50 - q50)
    err_q90 = np.abs(sim_q90 - q90)

    fit_warn = bool(
        (np.nanmax(err_p) > float(cfg.fit_tol_p))
        or (np.nanmax(err_q50) > float(cfg.fit_tol_q50))
        or (np.nanmax(err_q10) > float(cfg.fit_tol_qband))
        or (np.nanmax(err_q90) > float(cfg.fit_tol_qband))
    )

    summary = build_simulation_summary(sim_df, current_price=cp)
    diagnostics: Dict[str, Any] = {
        "n_steps": int(n_steps),
        "n_paths_requested": int(n_paths_req),
        "n_paths_used": int(n_paths_use),
        "degrade_flag": bool(degrade_flag),
        "degrade_reason": degrade_reason,
        "fit_error_p_up_max": float(np.nanmax(err_p)),
        "fit_error_q10_max": float(np.nanmax(err_q10)),
        "fit_error_q50_max": float(np.nanmax(err_q50)),
        "fit_error_q90_max": float(np.nanmax(err_q90)),
        "distribution_fit_warn": bool(fit_warn),
        "fit_tol_p": float(cfg.fit_tol_p),
        "fit_tol_q50": float(cfg.fit_tol_q50),
        "fit_tol_qband": float(cfg.fit_tol_qband),
        "representative_idx": int(base_idx),
        "bull_idx": int(bull_idx),
        "bear_idx": int(bear_idx),
    }
    if isinstance(fallback_info, dict):
        diagnostics.update(fallback_info)

    terminal_df = pd.DataFrame({"terminal_price": terminal_prices})
    return {
        "sim_df": sim_df,
        "sim_df_representative": sim_df_representative,
        "sim_df_bull": sim_df_bull,
        "sim_df_bear": sim_df_bear,
        "summary": summary,
        "diagnostics": diagnostics,
        "terminal_df": terminal_df,
        "paths_close": close_paths,
        "paths_open": open_paths,
        "paths_high": high_paths,
        "paths_low": low_paths,
    }


def simulate_future_ohlc(
    step_profile: pd.DataFrame,
    *,
    current_price: float,
    market_mode: str = "crypto",
    active_session: str | None = None,
    reference_ts_bj: Any = None,
    config: SimulationConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or SimulationConfig()
    cp = _safe_float(current_price)
    if not np.isfinite(cp) or cp <= 0:
        raise ValueError("current_price is invalid")

    normalized_profile, fallback_info = _normalize_hour_profile(
        step_profile,
        market_mode=str(market_mode),
        active_session=active_session,
        sigma_floor=float(cfg.sigma_floor),
        missing_ratio_hard_limit=float(cfg.missing_ratio_hard_limit),
    )
    if bool(fallback_info.get("insufficient_inputs")):
        raise ValueError(
            f"insufficient profile quality: missing_ratio={float(fallback_info.get('missing_ratio', 1.0)):.2%}"
        )

    ref_bj = _safe_ts_bj(reference_ts_bj)
    steps = _build_future_step_table(
        normalized_profile,
        n_steps=int(cfg.n_steps),
        reference_ts_bj=ref_bj,
        market_mode=str(market_mode),
    )
    return _simulate_from_steps(
        steps,
        current_price=cp,
        config=cfg,
        fallback_info=fallback_info,
    )

def simulate_daily_future_ohlc(
    daily_profile: pd.DataFrame,
    *,
    current_price: float,
    reference_ts_bj: Any = None,
    config: SimulationConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or SimulationConfig()
    cp = _safe_float(current_price)
    if not np.isfinite(cp) or cp <= 0:
        raise ValueError("current_price is invalid")

    normalized_daily, fallback_info = _normalize_daily_profile(
        daily_profile,
        sigma_floor=float(cfg.sigma_floor),
        missing_ratio_hard_limit=float(cfg.missing_ratio_hard_limit),
    )
    if bool(fallback_info.get("insufficient_inputs")):
        raise ValueError(
            f"insufficient daily profile quality: missing_ratio={float(fallback_info.get('missing_ratio', 1.0)):.2%}"
        )

    if normalized_daily.empty:
        raise ValueError("normalized daily profile is empty")

    n_steps_eff = int(min(int(cfg.n_steps), len(normalized_daily)))
    if n_steps_eff <= 0:
        raise ValueError("effective n_steps is invalid for daily simulation")
    steps = normalized_daily.head(n_steps_eff).copy().reset_index(drop=True)
    steps["step_idx"] = np.arange(1, len(steps) + 1, dtype=int)

    # If a reference timestamp is provided, keep date order but anchor tz parsing for consistency.
    _ = _safe_ts_bj(reference_ts_bj)
    res = _simulate_from_steps(
        steps,
        current_price=cp,
        config=cfg,
        fallback_info=fallback_info,
    )
    diag = dict(res.get("diagnostics", {}))
    diag["profile_granularity"] = "daily"
    diag["n_steps_effective"] = int(n_steps_eff)
    res["diagnostics"] = diag
    return res
