from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from src.markets.universe import load_universe
from src.utils.config import load_config, save_yaml
from src.utils.io import ensure_dir, save_json, write_csv


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _deep_copy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(cfg))


def _seeds(base_seed: int, step: int, n: int) -> List[int]:
    return [int(base_seed + i * step) for i in range(max(int(n), 0))]


@dataclass
class TrainingTask:
    task_id: str
    tier: str
    group_key: str
    run_index: int
    total_runs: int
    seed: int
    market: str = ""
    symbol: str = ""
    provider: str = ""
    pools: List[str] = field(default_factory=list)
    pipeline: List[str] = field(default_factory=list)


def _default_provider_for_market(market: str) -> str:
    if market == "crypto":
        return "binance"
    if market in {"cn_equity", "us_equity"}:
        return "yahoo"
    return "yahoo"


def _normalize_crypto_symbol(symbol: str, snapshot_symbol: str | None = None) -> str:
    s = str(symbol or "").upper().strip()
    ss = str(snapshot_symbol or "").upper().strip()
    if s.endswith("USDT"):
        return s
    if ss.endswith("USDT"):
        return ss
    if s:
        return f"{s}USDT"
    return "BTCUSDT"


def _load_ranked_universe(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "tracking" / "ranked_universe.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_symbols_for_market_pools(
    *,
    market: str,
    pools: List[str],
    processed_dir: Path,
    prefer_ranked_cache: bool = False,
) -> pd.DataFrame:
    ranked = _load_ranked_universe(processed_dir)
    rows: List[Dict[str, Any]] = []

    for pool in pools:
        from_ranked = pd.DataFrame()
        if not ranked.empty and {"market", "pool_key", "symbol"}.issubset(set(ranked.columns)):
            mask = (ranked["market"].astype(str) == str(market)) & (ranked["pool_key"].astype(str) == str(pool))
            from_ranked = ranked.loc[mask].copy()

        loaded_from_live = False
        if not prefer_ranked_cache:
            try:
                uni = load_universe(market, pool)
                provider_col = (
                    uni["provider"] if "provider" in uni.columns else _default_provider_for_market(market)
                )
                snapshot_col = uni["snapshot_symbol"] if "snapshot_symbol" in uni.columns else uni["symbol"]
                for idx in range(len(uni)):
                    rows.append(
                        {
                            "market": market,
                            "pool_key": pool,
                            "symbol": str(uni.iloc[idx]["symbol"]).strip(),
                            "snapshot_symbol": str(snapshot_col.iloc[idx]).strip(),
                            "provider": (
                                str(provider_col.iloc[idx]).strip()
                                if hasattr(provider_col, "iloc")
                                else str(provider_col)
                            ),
                        }
                    )
                loaded_from_live = True
            except Exception:
                loaded_from_live = False

        if loaded_from_live:
            continue

        if not from_ranked.empty:
            for _, row in from_ranked.iterrows():
                rows.append(
                    {
                        "market": market,
                        "pool_key": pool,
                        "symbol": str(row.get("symbol", "")).strip(),
                        "snapshot_symbol": str(row.get("snapshot_symbol", "")).strip(),
                        "provider": str(row.get("provider", "")).strip() or _default_provider_for_market(market),
                    }
                )
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[out["symbol"].astype(str).str.len() > 0].copy()
    out = out.drop_duplicates(subset=["market", "symbol"]).reset_index(drop=True)
    return out


def _build_tasks(cfg: Dict[str, Any], tiers: Iterable[str]) -> List[TrainingTask]:
    st_cfg = cfg.get("training_stability", {})
    if not bool(st_cfg.get("enabled", False)):
        raise ValueError("training_stability.enabled=false, cannot build stability tasks.")

    output_dir = Path(st_cfg.get("output_dir", "experiments/stability_training"))
    processed_dir = Path(cfg.get("paths", {}).get("processed_data_dir", "data/processed"))

    base_seed = int(st_cfg.get("base_seed", cfg.get("project", {}).get("seed", 42)))
    seed_step = int(st_cfg.get("seed_step", 17))
    run_profile = st_cfg.get("run_profile", {})
    n_full = int(run_profile.get("full_symbol_runs", 5))
    n_core_market = int(run_profile.get("core_market_runs", 50))
    n_core_symbol = int(run_profile.get("core_symbol_runs", 50))

    exec_cfg = st_cfg.get("execution", {})
    symbol_pipeline = [str(x) for x in exec_cfg.get("symbol_pipeline", [])]
    market_pipeline = [str(x) for x in exec_cfg.get("market_pipeline", [])]

    tasks: List[TrainingTask] = []
    tiers_set = set(str(x) for x in tiers)

    if "full_symbol" in tiers_set:
        full_cfg = st_cfg.get("full_universe", {})
        if bool(full_cfg.get("enabled", True)):
            markets = [str(x) for x in full_cfg.get("markets", ["crypto", "cn_equity", "us_equity"])]
            pools_map = full_cfg.get("pools", {})
            max_map = full_cfg.get("max_symbols_per_market", {})
            prefer_ranked_cache = bool(full_cfg.get("use_cached_ranked_universe", False))
            seeds = _seeds(base_seed, seed_step, n_full)
            for market in markets:
                pools = [str(x) for x in pools_map.get(market, [])]
                if not pools:
                    continue
                symbols_df = _load_symbols_for_market_pools(
                    market=market,
                    pools=pools,
                    processed_dir=processed_dir,
                    prefer_ranked_cache=prefer_ranked_cache,
                )
                max_n = int(max_map.get(market, 0))
                if max_n > 0:
                    symbols_df = symbols_df.head(max_n).copy()
                for _, row in symbols_df.iterrows():
                    raw_symbol = str(row.get("symbol", ""))
                    snapshot_symbol = str(row.get("snapshot_symbol", ""))
                    provider = str(row.get("provider", "")).lower() or _default_provider_for_market(market)
                    train_symbol = raw_symbol
                    if market == "crypto":
                        provider = "binance"
                        train_symbol = _normalize_crypto_symbol(raw_symbol, snapshot_symbol)
                    elif market in {"cn_equity", "us_equity"}:
                        provider = "yahoo"
                    group_key = f"full_symbol::{market}::{train_symbol}"
                    for i, seed in enumerate(seeds, start=1):
                        task_id = f"full_{market}_{train_symbol}_{i:03d}".replace("/", "_")
                        tasks.append(
                            TrainingTask(
                                task_id=task_id,
                                tier="full_symbol",
                                group_key=group_key,
                                run_index=i,
                                total_runs=n_full,
                                seed=seed,
                                market=market,
                                symbol=train_symbol,
                                provider=provider,
                                pools=pools,
                                pipeline=symbol_pipeline,
                            )
                        )

    if "core_market" in tiers_set:
        core_market_cfg = st_cfg.get("core_markets", {})
        if bool(core_market_cfg.get("enabled", True)):
            defs = core_market_cfg.get("definitions", [])
            seeds = _seeds(base_seed, seed_step, n_core_market)
            for d in defs:
                name = str(d.get("name", "")).strip()
                market = str(d.get("market", "")).strip()
                pools = [str(x) for x in d.get("pools", [])]
                if not name or not market:
                    continue
                group_key = f"core_market::{name}"
                for i, seed in enumerate(seeds, start=1):
                    task_id = f"core_market_{name}_{i:03d}"
                    tasks.append(
                        TrainingTask(
                            task_id=task_id,
                            tier="core_market",
                            group_key=group_key,
                            run_index=i,
                            total_runs=n_core_market,
                            seed=seed,
                            market=market,
                            symbol=name,
                            provider="",
                            pools=pools,
                            pipeline=market_pipeline,
                        )
                    )

    if "core_symbol" in tiers_set:
        core_symbol_cfg = st_cfg.get("core_symbols", {})
        if bool(core_symbol_cfg.get("enabled", True)):
            symbols = [str(x) for x in core_symbol_cfg.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])]
            seeds = _seeds(base_seed, seed_step, n_core_symbol)
            for symbol in symbols:
                train_symbol = _normalize_crypto_symbol(symbol)
                group_key = f"core_symbol::{train_symbol}"
                for i, seed in enumerate(seeds, start=1):
                    task_id = f"core_symbol_{train_symbol}_{i:03d}"
                    tasks.append(
                        TrainingTask(
                            task_id=task_id,
                            tier="core_symbol",
                            group_key=group_key,
                            run_index=i,
                            total_runs=n_core_symbol,
                            seed=seed,
                            market="crypto",
                            symbol=train_symbol,
                            provider="binance",
                            pools=["top3"],
                            pipeline=symbol_pipeline,
                        )
                    )

    if not tasks:
        raise RuntimeError("No stability tasks were generated from training_stability config.")

    # Sort for deterministic execution.
    tasks = sorted(tasks, key=lambda x: (x.tier, x.group_key, x.run_index))
    ensure_dir(output_dir)
    return tasks


def _prepare_task_config(
    base_cfg: Dict[str, Any],
    task: TrainingTask,
    st_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = _deep_copy_cfg(base_cfg)
    overrides = st_cfg.get("overrides", {})
    cfg.setdefault("project", {})
    cfg["project"]["seed"] = int(task.seed)

    if bool(overrides.get("disable_hpo_for_repeats", True)):
        cfg.setdefault("training", {})
        cfg["training"].setdefault("hpo", {})
        cfg["training"]["hpo"]["enabled"] = False
    if bool(overrides.get("disable_binance_us_for_history", False)):
        cfg.setdefault("data", {})
        cfg["data"]["disable_binance_us"] = True

    cfg.setdefault("data", {})
    cfg.setdefault("tracking", {})
    cfg.setdefault("backtest_multi_market", {})

    if task.tier in {"full_symbol", "core_symbol"}:
        cfg["data"]["symbol"] = str(task.symbol)
        cfg["data"]["source"] = str(task.provider)
        branches = cfg["data"].get("branches", {})
        if task.market == "us_equity":
            for _, branch_cfg in branches.items():
                branch_cfg["market_tz"] = "America/New_York"
        elif task.market == "cn_equity":
            for _, branch_cfg in branches.items():
                branch_cfg["market_tz"] = "Asia/Shanghai"
        elif task.market == "crypto":
            for _, branch_cfg in branches.items():
                branch_cfg["market_tz"] = "Asia/Shanghai"
    elif task.tier == "core_market":
        cfg["tracking"]["universes"] = {task.market: list(task.pools)}
        max_cfg = cfg["backtest_multi_market"].get("max_symbols_per_market", {})
        if not isinstance(max_cfg, dict):
            max_cfg = {}
        for m in ["crypto", "cn_equity", "us_equity"]:
            if m != task.market:
                max_cfg[m] = 0
        cfg["backtest_multi_market"]["max_symbols_per_market"] = max_cfg

    return cfg


def _run_module(module_name: str, config_path: Path, log_path: Path) -> None:
    cmd = [sys.executable, "-m", module_name, "--config", str(config_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n[stdout]\n{proc.stdout}\n\n[stderr]\n{proc.stderr}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{module_name} failed with code {proc.returncode}")


def _clear_runtime_outputs(processed_dir: Path) -> None:
    targets = [
        "metrics_walk_forward_summary.csv",
        "metrics_walk_forward.csv",
        "metrics_train_holdout_summary.csv",
        "metrics_train_holdout.csv",
        "backtest_metrics.csv",
        "threshold_report.csv",
        "predictions_hourly.csv",
        "predictions_daily.csv",
        "backtest/metrics_summary.csv",
        "backtest/metrics_by_fold.csv",
        "backtest/latest_signals.csv",
        "tracking/policy_signals_multi_market.csv",
    ]
    for rel in targets:
        p = processed_dir / rel
        if p.exists() and p.is_file():
            try:
                p.unlink()
            except Exception:
                pass


def _copy_artifacts(processed_dir: Path, dst_dir: Path) -> None:
    artifact_files = [
        "metrics_walk_forward_summary.csv",
        "metrics_train_holdout_summary.csv",
        "backtest_metrics.csv",
        "threshold_report.csv",
        "predictions_hourly.csv",
        "predictions_daily.csv",
        "backtest/metrics_summary.csv",
        "backtest/metrics_by_fold.csv",
        "backtest/latest_signals.csv",
        "tracking/ranked_universe.csv",
        "tracking/policy_signals_multi_market.csv",
        "tracking/tracking_snapshot.csv",
        "tracking/tracking_actions.csv",
    ]
    for rel in artifact_files:
        src = processed_dir / rel
        if not src.exists() or not src.is_file():
            continue
        out = dst_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)


def _series_cv_pct(values: List[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(_safe_float(v))]
    if len(vals) < 2:
        return float("nan")
    mean_abs = abs(float(np.mean(vals)))
    std = float(np.std(vals))
    if mean_abs < 1e-12:
        return float("inf") if std > 0 else 0.0
    return float((std / mean_abs) * 100.0)


def _read_predictions_flip_rate(processed_dir: Path) -> float:
    flip_rates: List[float] = []
    for branch in ["hourly", "daily"]:
        p = processed_dir / f"predictions_{branch}.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        p_cols = [c for c in df.columns if c.startswith("dir_h") and c.endswith("_p_up")]
        if not p_cols:
            continue
        p_col = "dir_h1_p_up" if "dir_h1_p_up" in p_cols else sorted(p_cols)[0]
        sig = (pd.to_numeric(df[p_col], errors="coerce") >= 0.5).astype(float)
        sig = sig.dropna().reset_index(drop=True)
        if len(sig) < 2:
            continue
        flips = (sig.diff().abs() > 0).sum()
        flip_rates.append(float(flips / (len(sig) - 1)))
    if not flip_rates:
        return float("nan")
    return float(np.mean(flip_rates))


def _extract_run_metrics(processed_dir: Path, tier: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "auc": float("nan"),
        "brier": float("nan"),
        "coverage": float("nan"),
        "signal_flip_rate": _read_predictions_flip_rate(processed_dir),
        "net_return": float("nan"),
        "max_drawdown": float("nan"),
    }

    wf_path = processed_dir / "metrics_walk_forward_summary.csv"
    if wf_path.exists():
        try:
            wf = pd.read_csv(wf_path)
            if {"task", "metric"}.issubset(wf.columns):
                auc_df = wf[
                    (wf["task"].astype(str) == "direction")
                    & (wf["metric"].astype(str).isin(["roc_auc", "auc"]))
                ]
                if not auc_df.empty:
                    if "mean" in auc_df.columns:
                        metrics["auc"] = _safe_float(auc_df["mean"].mean())
                    elif "value" in auc_df.columns:
                        metrics["auc"] = _safe_float(auc_df["value"].mean())
        except Exception:
            pass

    holdout_path = processed_dir / "metrics_train_holdout_summary.csv"
    if holdout_path.exists():
        try:
            hs = pd.read_csv(holdout_path)
            if "metric" in hs.columns:
                brier_df = hs[hs["metric"].astype(str) == "brier"]
                if not brier_df.empty:
                    col = "value" if "value" in brier_df.columns else ("mean" if "mean" in brier_df.columns else None)
                    if col:
                        metrics["brier"] = _safe_float(brier_df[col].mean())
                cov_df = hs[hs["metric"].astype(str) == "coverage"]
                if not cov_df.empty:
                    col = "value" if "value" in cov_df.columns else ("mean" if "mean" in cov_df.columns else None)
                    if col:
                        metrics["coverage"] = _safe_float(cov_df[col].mean())
        except Exception:
            pass

    # Performance metrics
    if tier == "core_market":
        mm_path = processed_dir / "backtest" / "metrics_summary.csv"
        if mm_path.exists():
            try:
                mm = pd.read_csv(mm_path)
                if not mm.empty:
                    pol = mm[mm["strategy"].astype(str) == "policy"] if "strategy" in mm.columns else mm
                    if not pol.empty:
                        metrics["net_return"] = _safe_float(pol["total_return"].mean())
                        metrics["max_drawdown"] = _safe_float(pol["max_drawdown"].mean())
            except Exception:
                pass
    else:
        bt_path = processed_dir / "backtest_metrics.csv"
        if bt_path.exists():
            try:
                bt = pd.read_csv(bt_path)
                if not bt.empty:
                    metrics["net_return"] = _safe_float(bt["total_return"].mean())
                    metrics["max_drawdown"] = _safe_float(bt["max_drawdown"].mean())
            except Exception:
                pass

    return metrics


def _evaluate_stop(
    history: List[Dict[str, Any]],
    stop_cfg: Dict[str, Any],
    run_profile_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    min_runs = int(run_profile_cfg.get("min_runs_before_stop", 10))
    window = int(stop_cfg.get("metric_window", 10))
    vol_thr = float(stop_cfg.get("volatility_threshold_pct", 5.0))
    max_flip = float(stop_cfg.get("max_flip_rate", 0.2))
    min_ret_delta = float(stop_cfg.get("min_net_return_delta", -0.02))
    max_dd_delta = float(stop_cfg.get("max_drawdown_delta", 0.02))

    if len(history) < max(min_runs, window):
        return {"eligible": False, "should_stop": False}

    tail = history[-window:]
    auc_cv = _series_cv_pct([_safe_float(r.get("auc")) for r in tail])
    brier_cv = _series_cv_pct([_safe_float(r.get("brier")) for r in tail])
    cov_cv = _series_cv_pct([_safe_float(r.get("coverage")) for r in tail])
    flip_mean = _safe_float(np.nanmean([_safe_float(r.get("signal_flip_rate")) for r in tail]))

    metrics_ok = all(
        np.isfinite(v) and float(v) <= vol_thr
        for v in [auc_cv, brier_cv, cov_cv]
    )
    flip_ok = np.isfinite(flip_mean) and float(flip_mean) <= max_flip

    perf_ok = True
    ret_delta = float("nan")
    dd_delta = float("nan")
    if len(history) >= window * 2:
        prev = history[-2 * window : -window]
        prev_ret = _safe_float(np.nanmean([_safe_float(r.get("net_return")) for r in prev]))
        prev_dd = _safe_float(np.nanmean([_safe_float(r.get("max_drawdown")) for r in prev]))
        cur_ret = _safe_float(np.nanmean([_safe_float(r.get("net_return")) for r in tail]))
        cur_dd = _safe_float(np.nanmean([_safe_float(r.get("max_drawdown")) for r in tail]))
        ret_delta = _safe_float(cur_ret - prev_ret)
        dd_delta = _safe_float(cur_dd - prev_dd)
        perf_ok = (
            np.isfinite(ret_delta)
            and np.isfinite(dd_delta)
            and ret_delta >= min_ret_delta
            and dd_delta <= max_dd_delta
        )

    should_stop = bool(metrics_ok and flip_ok and perf_ok)
    return {
        "eligible": True,
        "should_stop": should_stop,
        "auc_cv_pct": auc_cv,
        "brier_cv_pct": brier_cv,
        "coverage_cv_pct": cov_cv,
        "flip_mean": flip_mean,
        "ret_delta": ret_delta,
        "dd_delta": dd_delta,
        "metrics_ok": metrics_ok,
        "flip_ok": flip_ok,
        "perf_ok": perf_ok,
    }


def run_stability_batch(
    config_path: str,
    *,
    plan_only: bool = False,
    max_tasks: int = 0,
    tiers: List[str] | None = None,
) -> None:
    cfg = load_config(config_path)
    st_cfg = cfg.get("training_stability", {})
    if not bool(st_cfg.get("enabled", False)):
        raise RuntimeError("training_stability.enabled=false; enable it in config first.")

    selected_tiers = tiers or ["full_symbol", "core_market", "core_symbol"]
    tasks = _build_tasks(cfg, selected_tiers)
    if max_tasks > 0:
        tasks = tasks[: int(max_tasks)]

    out_root = ensure_dir(Path(st_cfg.get("output_dir", "experiments/stability_training")) / f"run_{_utc_now()}")
    plan_df = pd.DataFrame([asdict(t) for t in tasks])
    write_csv(plan_df, out_root / "seed_plan.csv")
    save_json({"rows": plan_df.to_dict(orient="records")}, out_root / "seed_plan.json")

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config_path": str(config_path),
        "tiers": selected_tiers,
        "tasks_count": int(len(tasks)),
        "plan_only": bool(plan_only),
    }
    save_json(meta, out_root / "run_meta.json")
    if plan_only:
        print(f"[OK] Plan generated only -> {out_root}")
        return

    processed_dir = Path(cfg.get("paths", {}).get("processed_data_dir", "data/processed"))
    run_rows: List[Dict[str, Any]] = []
    stop_rows: List[Dict[str, Any]] = []
    group_history: Dict[str, List[Dict[str, Any]]] = {}
    stopped_groups: set[str] = set()
    run_profile_cfg = st_cfg.get("run_profile", {})
    stop_cfg = st_cfg.get("stop_criteria", {})

    for task in tasks:
        run_dir = ensure_dir(out_root / "runs" / task.task_id)
        run_record: Dict[str, Any] = {
            "task_id": task.task_id,
            "tier": task.tier,
            "group_key": task.group_key,
            "run_index": task.run_index,
            "total_runs": task.total_runs,
            "seed": task.seed,
            "market": task.market,
            "symbol": task.symbol,
            "provider": task.provider,
            "status": "pending",
            "error": "",
        }

        if task.group_key in stopped_groups:
            run_record["status"] = "skipped_early_stop"
            run_rows.append(run_record)
            continue

        try:
            run_cfg = _prepare_task_config(cfg, task, st_cfg)
            cfg_path = run_dir / "config_runtime.yaml"
            save_yaml(run_cfg, cfg_path)
            _clear_runtime_outputs(processed_dir)

            for module_name in task.pipeline:
                module_log = run_dir / "logs" / f"{module_name.replace('.', '_')}.log"
                module_log.parent.mkdir(parents=True, exist_ok=True)
                _run_module(module_name, cfg_path, module_log)

            _copy_artifacts(processed_dir, run_dir / "artifacts")
            metrics = _extract_run_metrics(processed_dir, task.tier)
            run_record.update(metrics)
            run_record["status"] = "success"

            hist = group_history.setdefault(task.group_key, [])
            hist.append(metrics)
            if task.tier in {"core_market", "core_symbol"}:
                stop_eval = _evaluate_stop(hist, stop_cfg, run_profile_cfg)
                stop_eval.update(
                    {
                        "task_id": task.task_id,
                        "tier": task.tier,
                        "group_key": task.group_key,
                        "run_index": task.run_index,
                    }
                )
                stop_rows.append(stop_eval)
                if bool(stop_eval.get("should_stop", False)):
                    stopped_groups.add(task.group_key)
        except Exception as exc:
            run_record["status"] = "failed"
            run_record["error"] = f"{type(exc).__name__}: {exc}"

        run_rows.append(run_record)
        write_csv(pd.DataFrame(run_rows), out_root / "run_results.csv")
        save_json({"rows": run_rows}, out_root / "run_results.json")
        if stop_rows:
            write_csv(pd.DataFrame(stop_rows), out_root / "early_stop_checks.csv")
            save_json({"rows": stop_rows}, out_root / "early_stop_checks.json")

    summary = pd.DataFrame(run_rows)
    if not summary.empty:
        agg = summary.groupby(["tier", "status"], as_index=False).size()
        write_csv(agg, out_root / "run_status_summary.csv")
    print(f"[OK] Stability batch finished -> {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="3-tier stability batch training runner.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument(
        "--tiers",
        type=str,
        default="full_symbol,core_market,core_symbol",
        help="Comma-separated tiers: full_symbol,core_market,core_symbol",
    )
    parser.add_argument("--plan-only", action="store_true", help="Only generate plan (no training).")
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit task count for smoke tests.")
    args = parser.parse_args()

    tiers = [x.strip() for x in str(args.tiers).split(",") if x.strip()]
    run_stability_batch(
        args.config,
        plan_only=bool(args.plan_only),
        max_tasks=int(args.max_tasks),
        tiers=tiers,
    )


if __name__ == "__main__":
    main()
