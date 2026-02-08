from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from src.ingestion.symbol_mapper import canonical_symbol, normalize_market
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json, write_csv


NS_PER_MIN = 60 * 1_000_000_000


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _safe_int(value: Any, default: int = 0) -> int:
    out = _safe_float(value)
    if np.isfinite(out):
        return int(round(out))
    return int(default)


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _to_dt_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _default_news_cfg() -> Dict[str, Any]:
    return {
        "output_dir": "data/processed/news",
        "raw_output_csv": "data/processed/news/news_raw.csv",
        "raw_output_parquet": "data/processed/news/news_raw.parquet",
        "features_output_csv": "data/processed/news/news_features_hourly.csv",
        "features_output_parquet": "data/processed/news/news_features_hourly.parquet",
        "latest_output_csv": "data/processed/news/news_features_latest.csv",
        "latest_output_json": "data/processed/news/news_features_latest.json",
        "feature_windows_minutes": [30, 120, 1440],
        "decay_tau_minutes": 360.0,
        "grid_lookback_days": 30,
        "dedup_symbol_latest_only": True,
    }


def _merge_news_cfg(cfg_news: Dict[str, Any]) -> Dict[str, Any]:
    return {**_default_news_cfg(), **(cfg_news or {})}


def _resolve_paths(cfg: Dict[str, Any], cfg_news: Dict[str, Any]) -> Dict[str, Path]:
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    processed_dir = Path(paths_cfg.get("processed_data_dir", "data/processed"))
    out_root = Path(cfg_news.get("output_dir", processed_dir / "news"))
    out_root = ensure_dir(out_root)
    return {
        "raw_csv": Path(cfg_news.get("raw_output_csv", out_root / "news_raw.csv")),
        "raw_parquet": Path(cfg_news.get("raw_output_parquet", out_root / "news_raw.parquet")),
        "feat_csv": Path(cfg_news.get("features_output_csv", out_root / "news_features_hourly.csv")),
        "feat_parquet": Path(cfg_news.get("features_output_parquet", out_root / "news_features_hourly.parquet")),
        "latest_csv": Path(cfg_news.get("latest_output_csv", out_root / "news_features_latest.csv")),
        "latest_json": Path(cfg_news.get("latest_output_json", out_root / "news_features_latest.json")),
    }


def _load_news_raw(paths: Dict[str, Path]) -> pd.DataFrame:
    if paths["raw_parquet"].exists():
        try:
            df = pd.read_parquet(paths["raw_parquet"])
            return df
        except Exception:
            pass
    if paths["raw_csv"].exists():
        try:
            return pd.read_csv(paths["raw_csv"])
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _prepare_news_raw(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for c in ["market", "symbol", "provider", "title", "summary", "url", "dedup_group_id"]:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str)
    out["market"] = out["market"].map(normalize_market)
    out["symbol"] = out.apply(
        lambda r: canonical_symbol(r.get("symbol", ""), r.get("market", "")),
        axis=1,
    )
    for c in ["raw_sentiment", "sentiment_confidence", "relevance", "source_weight"]:
        if c not in out.columns:
            out[c] = float("nan")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "available_at_utc" not in out.columns:
        out["available_at_utc"] = pd.NaT
    out["available_at_utc"] = _to_dt_utc(out["available_at_utc"])
    if "published_at_utc" in out.columns:
        out["published_at_utc"] = _to_dt_utc(out["published_at_utc"])
    if "ingested_at_utc" in out.columns:
        out["ingested_at_utc"] = _to_dt_utc(out["ingested_at_utc"])
    out = out.dropna(subset=["available_at_utc"])
    out = out[out["market"].astype(str).str.len() > 0]
    out = out[out["symbol"].astype(str).str.len() > 0]
    out = out.sort_values(["market", "symbol", "available_at_utc"]).reset_index(drop=True)
    return out


def _calc_features_at_ts(
    *,
    event_ts_ns: np.ndarray,
    sentiment: np.ndarray,
    base_weight: np.ndarray,
    query_ts: pd.Timestamp,
    windows: List[int],
    tau_minutes: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if query_ts is pd.NaT or pd.isna(query_ts):
        for w in windows:
            suffix = f"{int(w)}m"
            out[f"news_score_{suffix}"] = 0.0
            out[f"news_count_{suffix}"] = 0
        out["news_burst_zscore"] = 0.0
        out["news_pos_neg_ratio"] = 1.0
        out["news_conflict_score"] = 0.0
        out["news_risk_level"] = "low"
        out["news_event_risk"] = False
        out["news_gate_pass"] = True
        out["news_reason_codes"] = ""
        return out

    ts_ns = int(query_ts.value)
    age_min = (ts_ns - event_ts_ns).astype(np.float64) / NS_PER_MIN
    max_window = int(max(windows))
    valid = (age_min >= 0.0) & (age_min <= float(max_window))
    if not np.any(valid):
        for w in windows:
            suffix = f"{int(w)}m"
            out[f"news_score_{suffix}"] = 0.0
            out[f"news_count_{suffix}"] = 0
        out["news_burst_zscore"] = 0.0
        out["news_pos_neg_ratio"] = 1.0
        out["news_conflict_score"] = 0.0
        out["news_risk_level"] = "low"
        out["news_event_risk"] = False
        out["news_gate_pass"] = True
        out["news_reason_codes"] = ""
        return out

    age_valid = age_min[valid]
    sent_valid = sentiment[valid]
    base_w_valid = base_weight[valid]
    decay = np.exp(-np.maximum(age_valid, 0.0) / max(float(tau_minutes), 1e-6))
    w_valid = base_w_valid * decay

    count_by_window: Dict[int, int] = {}
    score_by_window: Dict[int, float] = {}
    pos_2h = 0
    neg_2h = 0
    for w in windows:
        win_mask = age_valid <= float(w)
        count = int(np.sum(win_mask))
        count_by_window[int(w)] = count
        if count <= 0:
            score = 0.0
        else:
            ww = w_valid[win_mask]
            ss = sent_valid[win_mask]
            den = float(np.sum(ww))
            score = float(np.sum(ss * ww) / den) if den > 1e-12 else 0.0
        score_by_window[int(w)] = score
        out[f"news_score_{int(w)}m"] = score
        out[f"news_count_{int(w)}m"] = count
        if int(w) == 120 and count > 0:
            pos_2h = int(np.sum(sent_valid[win_mask] > 0))
            neg_2h = int(np.sum(sent_valid[win_mask] < 0))

    c30 = count_by_window.get(30, 0)
    c24 = count_by_window.get(1440, 0)
    exp_30 = float(c24) * (30.0 / 1440.0)
    burst = (float(c30) - exp_30) / np.sqrt(exp_30 + 1.0)
    pos_neg_ratio = float((pos_2h + 1.0) / (neg_2h + 1.0))
    if (pos_2h + neg_2h) > 0:
        conflict = float(min(pos_2h, neg_2h) / max(pos_2h, neg_2h))
    else:
        conflict = 0.0

    score_30 = score_by_window.get(30, 0.0)
    score_120 = score_by_window.get(120, score_by_window.get(30, 0.0))
    reasons: List[str] = []
    event_risk = False
    risk_level = "low"
    if abs(score_30) >= 0.40 and c30 >= 3:
        event_risk = True
        reasons.append("POS_EVENT" if score_30 > 0 else "NEG_EVENT")
    if abs(score_120) >= 0.25 and burst >= 1.5:
        event_risk = True
        reasons.append("POS_BURST" if score_120 > 0 else "NEG_BURST")
    if event_risk:
        risk_level = "high"
    elif abs(score_120) >= 0.15 or burst >= 1.0:
        risk_level = "medium"
        reasons.append("ELEVATED_NEWS_RISK")
    else:
        risk_level = "low"

    out["news_burst_zscore"] = float(burst)
    out["news_pos_neg_ratio"] = float(pos_neg_ratio)
    out["news_conflict_score"] = float(conflict)
    out["news_risk_level"] = risk_level
    out["news_event_risk"] = bool(event_risk)
    out["news_gate_pass"] = not bool(event_risk)
    out["news_reason_codes"] = ";".join(reasons)
    return out


def _group_news_features(
    *,
    market: str,
    symbol: str,
    group_df: pd.DataFrame,
    query_times: pd.Series,
    windows: List[int],
    tau_minutes: float,
) -> pd.DataFrame:
    g = group_df.sort_values("available_at_utc").copy()
    event_ts_ns = g["available_at_utc"].astype("int64").to_numpy(dtype=np.int64)
    sentiment = pd.to_numeric(g["raw_sentiment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    conf = pd.to_numeric(g["sentiment_confidence"], errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    rel = pd.to_numeric(g["relevance"], errors="coerce").fillna(0.7).clip(lower=0.0, upper=1.0)
    sw = pd.to_numeric(g["source_weight"], errors="coerce").fillna(0.7).clip(lower=0.0, upper=5.0)
    base_weight = (conf * rel * sw).to_numpy(dtype=np.float64)

    rows: List[Dict[str, Any]] = []
    for ts in query_times:
        f = _calc_features_at_ts(
            event_ts_ns=event_ts_ns,
            sentiment=sentiment,
            base_weight=base_weight,
            query_ts=ts,
            windows=windows,
            tau_minutes=tau_minutes,
        )
        latest = g[g["available_at_utc"] <= ts].tail(3)
        headlines = " || ".join([_safe_str(x) for x in latest["title"].tolist() if _safe_str(x)])
        providers = " || ".join([_safe_str(x) for x in latest["provider"].tolist() if _safe_str(x)])
        f.update(
            {
                "market": market,
                "symbol": symbol,
                "timestamp_utc": ts,
                "news_latest_headlines": headlines,
                "news_latest_providers": providers,
                "news_last_available_at_utc": latest["available_at_utc"].max() if not latest.empty else pd.NaT,
            }
        )
        rows.append(f)
    return pd.DataFrame(rows)


def _build_hourly_feature_grid(
    news_df: pd.DataFrame,
    *,
    windows: List[int],
    tau_minutes: float,
    lookback_days: int,
) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC").floor("h")
    start = now - pd.Timedelta(days=max(int(lookback_days), 1))
    all_rows: List[pd.DataFrame] = []
    for (market, symbol), g in news_df.groupby(["market", "symbol"], dropna=False):
        g = g.sort_values("available_at_utc")
        g = g[g["available_at_utc"] >= start - pd.Timedelta(minutes=max(windows))].copy()
        if g.empty:
            continue
        ts_start = max(start, g["available_at_utc"].min().floor("h"))
        query_times = pd.date_range(ts_start, now, freq="1h", tz="UTC")
        if len(query_times) == 0:
            continue
        all_rows.append(
            _group_news_features(
                market=str(market),
                symbol=str(symbol),
                group_df=g,
                query_times=pd.Series(query_times),
                windows=windows,
                tau_minutes=tau_minutes,
            )
        )
    if not all_rows:
        return pd.DataFrame()
    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["market", "symbol", "timestamp_utc"]).reset_index(drop=True)
    return out


def _build_latest_features(hourly_df: pd.DataFrame) -> pd.DataFrame:
    if hourly_df.empty:
        return pd.DataFrame()
    out = (
        hourly_df.sort_values("timestamp_utc")
        .groupby(["market", "symbol"], as_index=False, dropna=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return out


def run_news_features(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    cfg_news = _merge_news_cfg(cfg.get("news", {}) if isinstance(cfg, dict) else {})
    paths = _resolve_paths(cfg, cfg_news)

    raw_df = _load_news_raw(paths)
    raw_df = _prepare_news_raw(raw_df)
    if raw_df.empty:
        payload = {
            "config_path": config_path,
            "rows_raw": 0,
            "rows_features_hourly": 0,
            "rows_features_latest": 0,
            "message": "no raw news available",
            "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        save_json(payload, paths["latest_json"])
        write_csv(pd.DataFrame(), paths["feat_csv"])
        write_csv(pd.DataFrame(), paths["latest_csv"])
        print("[WARN] news features skipped: no raw news data.")
        return payload

    windows = sorted([int(x) for x in cfg_news.get("feature_windows_minutes", [30, 120, 1440]) if int(x) > 0])
    if not windows:
        windows = [30, 120, 1440]
    if 30 not in windows:
        windows.append(30)
    if 120 not in windows:
        windows.append(120)
    if 1440 not in windows:
        windows.append(1440)
    windows = sorted(set(windows))
    tau_minutes = float(cfg_news.get("decay_tau_minutes", 360.0))
    lookback_days = int(cfg_news.get("grid_lookback_days", 30))

    hourly_df = _build_hourly_feature_grid(
        raw_df,
        windows=windows,
        tau_minutes=tau_minutes,
        lookback_days=lookback_days,
    )
    latest_df = _build_latest_features(hourly_df)

    write_csv(hourly_df, paths["feat_csv"])
    write_csv(latest_df, paths["latest_csv"])
    try:
        hourly_df.to_parquet(paths["feat_parquet"], index=False)
    except Exception:
        pass

    payload = {
        "config_path": config_path,
        "rows_raw": int(len(raw_df)),
        "rows_features_hourly": int(len(hourly_df)),
        "rows_features_latest": int(len(latest_df)),
        "markets": sorted(hourly_df["market"].dropna().astype(str).unique().tolist()) if not hourly_df.empty else [],
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        "outputs": {
            "hourly_csv": str(paths["feat_csv"]),
            "latest_csv": str(paths["latest_csv"]),
            "hourly_parquet": str(paths["feat_parquet"]),
        },
    }
    save_json(payload, paths["latest_json"])
    print(
        "[OK] news features built: "
        f"raw={len(raw_df)} hourly={len(hourly_df)} latest={len(latest_df)}"
    )
    return payload


def load_latest_news_features(processed_dir: str = "data/processed") -> pd.DataFrame:
    root = Path(processed_dir) / "news"
    path = root / "news_features_latest.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if "market" in df.columns:
        df["market"] = df["market"].map(normalize_market)
    if "symbol" in df.columns:
        df["symbol"] = df.apply(lambda r: canonical_symbol(r.get("symbol", ""), r.get("market", "")), axis=1)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = _to_dt_utc(df["timestamp_utc"])
    return df


def merge_latest_news_features(
    df: pd.DataFrame,
    *,
    market_col: str = "market",
    symbol_col: str = "symbol",
    processed_dir: str = "data/processed",
) -> pd.DataFrame:
    if df.empty:
        return df
    latest = load_latest_news_features(processed_dir=processed_dir)
    if latest.empty:
        return df
    out = df.copy()
    out["_news_market"] = out[market_col].map(normalize_market)
    out["_news_symbol"] = out.apply(
        lambda r: canonical_symbol(r.get(symbol_col, ""), r.get(market_col, "")),
        axis=1,
    )
    join_cols = [c for c in ["market", "symbol"] if c in latest.columns]
    if len(join_cols) < 2:
        return df
    latest = latest.rename(columns={"market": "_news_market", "symbol": "_news_symbol"})
    merged = out.merge(latest, on=["_news_market", "_news_symbol"], how="left", suffixes=("", "_news"))
    merged = merged.drop(columns=["_news_market", "_news_symbol"], errors="ignore")
    return merged


def merge_news_features_asof(
    df: pd.DataFrame,
    *,
    ts_col: str,
    market_col: str,
    symbol_col: str,
    processed_dir: str = "data/processed",
) -> pd.DataFrame:
    if df.empty:
        return df
    feat_path = Path(processed_dir) / "news" / "news_features_hourly.csv"
    if not feat_path.exists():
        return df
    try:
        feat = pd.read_csv(feat_path)
    except Exception:
        return df
    if feat.empty:
        return df

    feat["timestamp_utc"] = _to_dt_utc(feat["timestamp_utc"])
    feat["market"] = feat["market"].map(normalize_market)
    feat["symbol"] = feat.apply(lambda r: canonical_symbol(r.get("symbol", ""), r.get("market", "")), axis=1)
    feat = feat.sort_values("timestamp_utc")

    out = df.copy()
    out["_ts_utc"] = _to_dt_utc(out[ts_col])
    out["_market"] = out[market_col].map(normalize_market)
    out["_symbol"] = out.apply(lambda r: canonical_symbol(r.get(symbol_col, ""), r.get(market_col, "")), axis=1)
    out = out.sort_values("_ts_utc")

    merged_parts: List[pd.DataFrame] = []
    for (mk, sym), sub in out.groupby(["_market", "_symbol"], dropna=False):
        feat_sub = feat[(feat["market"] == mk) & (feat["symbol"] == sym)].copy()
        if feat_sub.empty:
            merged_parts.append(sub)
            continue
        merged_sub = pd.merge_asof(
            sub.sort_values("_ts_utc"),
            feat_sub.sort_values("timestamp_utc"),
            left_on="_ts_utc",
            right_on="timestamp_utc",
            direction="backward",
        )
        merged_parts.append(merged_sub)
    merged = pd.concat(merged_parts, ignore_index=True) if merged_parts else out
    merged = merged.drop(columns=["_ts_utc", "_market", "_symbol"], errors="ignore")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rolling news sentiment feature tables.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_news_features(config_path=args.config)


if __name__ == "__main__":
    main()

