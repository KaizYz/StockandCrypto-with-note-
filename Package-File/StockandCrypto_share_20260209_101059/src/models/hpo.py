from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.factory import build_mvp_classifier
from src.models.registry import append_experiment_record
from src.utils.artifacts import get_git_commit_short
from src.utils.config import load_config, save_yaml
from src.utils.io import write_csv


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _default_hpo_cfg() -> Dict[str, Any]:
    return {
        "enabled": True,
        "coarse_trials": 4,
        "fine_trials": 2,
        "random_state": 42,
        "owner": "youka",
        "space_version": "hpo_v1",
        "validation_ratio": 0.2,
        "max_horizons_per_branch": 1,
        "max_train_rows": 6000,
        "param_space": {
            "n_estimators": [120, 240, 360],
            "learning_rate": [0.01, 0.03, 0.05, 0.08],
            "num_leaves": [31, 63, 127],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "reg_lambda": [0.0, 0.1, 1.0],
        },
    }


def _split_train_valid(df: pd.DataFrame, validation_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    valid_n = max(30, int(n * validation_ratio))
    valid_n = min(valid_n, max(1, n - 50))
    split = n - valid_n
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _sample_param_set(rng: np.random.Generator, space: Dict[str, List[Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, values in space.items():
        if not values:
            continue
        idx = int(rng.integers(0, len(values)))
        out[key] = values[idx]
    return out


def _jitter_param_set(
    rng: np.random.Generator,
    best: Dict[str, Any],
    space: Dict[str, List[Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, values in space.items():
        if not values:
            continue
        values_list = list(values)
        if key not in best:
            out[key] = values_list[int(rng.integers(0, len(values_list)))]
            continue
        try:
            i = values_list.index(best[key])
            lo = max(i - 1, 0)
            hi = min(i + 1, len(values_list) - 1)
            out[key] = values_list[int(rng.integers(lo, hi + 1))]
        except Exception:
            out[key] = values_list[int(rng.integers(0, len(values_list)))]
    return out


def _try_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _score_trial(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_ret: np.ndarray,
    fee_bps: float,
    slippage_bps: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = float((y_pred == y_true).mean()) if len(y_true) > 0 else float("nan")
    brier = float(np.mean((y_prob - y_true) ** 2)) if len(y_true) > 0 else float("nan")
    auc = _try_auc(y_true, y_prob)
    up_thr = 0.55
    down_thr = 0.45
    position = np.where(y_prob >= up_thr, 1.0, np.where(y_prob <= down_thr, -1.0, 0.0))
    position_lag = np.roll(position, 1)
    position_lag[0] = 0.0
    turnover = np.abs(position_lag - np.roll(position_lag, 1))
    turnover[0] = 0.0
    cost = (max(fee_bps, 0.0) + max(slippage_bps, 0.0)) / 10000.0
    strat_ret = position_lag * y_ret - turnover * cost
    edge_after_cost = float(np.nanmean(strat_ret)) if len(strat_ret) > 0 else float("nan")
    trades = int(np.sum(np.abs(position_lag) > 0))
    # Primary objective prefers edge and robust probability.
    score = 0.0
    if np.isfinite(edge_after_cost):
        score += edge_after_cost * 1000.0
    if np.isfinite(accuracy):
        score += accuracy
    if np.isfinite(auc):
        score += 0.2 * auc
    if np.isfinite(brier):
        score -= brier
    return {
        "score": float(score),
        "accuracy": accuracy,
        "brier": brier,
        "auc": auc,
        "edge_after_cost": edge_after_cost,
        "trades": float(trades),
    }


def run_hpo(config_path: str) -> None:
    cfg = load_config(config_path)
    hpo_cfg = _default_hpo_cfg()
    hpo_cfg = {**hpo_cfg, **((cfg.get("training", {}) or {}).get("hpo", {}))}
    if not bool(hpo_cfg.get("enabled", True)):
        print("[WARN] HPO disabled.")
        return

    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    root_dir = Path(config_path).resolve().parent.parent
    rng = np.random.default_rng(int(hpo_cfg.get("random_state", 42)))

    coarse_trials = int(hpo_cfg.get("coarse_trials", 12))
    fine_trials = int(hpo_cfg.get("fine_trials", 6))
    validation_ratio = float(hpo_cfg.get("validation_ratio", 0.2))
    max_horizons = int(hpo_cfg.get("max_horizons_per_branch", 1))
    max_train_rows = int(hpo_cfg.get("max_train_rows", 6000))
    space = hpo_cfg.get("param_space", {})
    owner = str(hpo_cfg.get("owner", "owner"))

    fee_bps = _safe_float((cfg.get("policy", {}).get("execution", {}) or {}).get("fee_bps", 10))
    slippage_bps = _safe_float((cfg.get("policy", {}).get("execution", {}) or {}).get("slippage_bps", 10))

    trial_rows: List[Dict[str, Any]] = []
    best_cfg: Dict[str, Any] = {"best": {}}
    git_commit = get_git_commit_short("-")
    exp_id = f"hpo_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    for branch_name, branch_cfg in (data_cfg.get("branches", {}) or {}).items():
        feat_path = processed_dir / f"features_{branch_name}.csv"
        label_path = processed_dir / f"labels_{branch_name}.csv"
        if not feat_path.exists() or not label_path.exists():
            continue
        feat = pd.read_csv(feat_path)
        labels = pd.read_csv(label_path)
        df = feat.merge(labels, on="timestamp_utc", how="inner").sort_values("timestamp_utc").reset_index(drop=True)
        feature_cols = [
            c
            for c in df.columns
            if c not in {"timestamp_utc", "timestamp_market", "market_tz"}
            and not c.startswith("y_")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(feature_cols) < 5:
            continue

        # avoid leakage from frozen holdout: trim tail by holdout ratio
        holdout_cfg = (cfg.get("training", {}) or {}).get("holdout", {})
        holdout_ratio = float(holdout_cfg.get("ratio", 0.1))
        holdout_min = int(holdout_cfg.get("min_rows", 60))
        n = len(df)
        holdout_n = max(int(n * holdout_ratio), holdout_min)
        holdout_n = min(holdout_n, max(0, n - 80))
        trainval = df.iloc[: n - holdout_n].copy() if holdout_n > 0 else df.copy()
        if len(trainval) > max_train_rows > 0:
            trainval = trainval.tail(max_train_rows).reset_index(drop=True)
        if len(trainval) < 120:
            continue
        train_df, valid_df = _split_train_valid(trainval, validation_ratio=validation_ratio)

        horizons = [int(h) for h in branch_cfg.get("horizons", [1])]
        if max_horizons > 0:
            horizons = horizons[:max_horizons]
        for h in horizons:
            t_dir = f"y_dir_h{h}"
            t_ret = f"y_ret_h{h}"
            if t_dir not in train_df.columns or t_ret not in valid_df.columns:
                continue

            tr = train_df.dropna(subset=[t_dir]).copy()
            va = valid_df.dropna(subset=[t_dir, t_ret]).copy()
            if len(tr) < 80 or len(va) < 30:
                continue
            X_tr = tr[feature_cols].fillna(tr[feature_cols].median(numeric_only=True))
            y_tr = tr[t_dir].astype(int)
            X_va = va[feature_cols].fillna(tr[feature_cols].median(numeric_only=True))
            y_va = va[t_dir].astype(int).to_numpy()
            y_ret = pd.to_numeric(va[t_ret], errors="coerce").fillna(0.0).to_numpy()

            trial_id = 0
            best_score = -1e18
            best_params: Dict[str, Any] = {}

            def _run_one(params: Dict[str, Any], stage: str) -> None:
                nonlocal trial_id, best_score, best_params
                trial_id += 1
                cfg_trial = copy.deepcopy(cfg)
                cfg_trial.setdefault("models", {})
                cfg_trial["models"].setdefault("classifier", {})
                cfg_trial["models"]["classifier"].update(params)
                model, backend = build_mvp_classifier(cfg_trial, n_classes=2, seed=int(hpo_cfg.get("random_state", 42)))
                model.fit(X_tr, y_tr)
                if hasattr(model, "predict_proba"):
                    p = np.asarray(model.predict_proba(X_va))[:, 1]
                else:
                    p = np.asarray(model.predict(X_va), dtype=float)
                    p = np.clip(p, 0.0, 1.0)
                scored = _score_trial(
                    y_true=y_va,
                    y_prob=p,
                    y_ret=y_ret,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                )
                row = {
                    "exp_id": exp_id,
                    "branch": branch_name,
                    "horizon": h,
                    "trial_id": trial_id,
                    "stage": stage,
                    "backend": backend,
                    "params": json.dumps(params, ensure_ascii=False),
                }
                row.update(scored)
                trial_rows.append(row)
                if scored["score"] > best_score:
                    best_score = scored["score"]
                    best_params = dict(params)

            for _ in range(coarse_trials):
                _run_one(_sample_param_set(rng, space), stage="coarse")
            for _ in range(fine_trials):
                _run_one(_jitter_param_set(rng, best_params, space), stage="fine")

            best_cfg.setdefault("best", {}).setdefault(branch_name, {})[str(h)] = {
                "params": best_params,
                "score": float(best_score),
            }

    trial_df = pd.DataFrame(trial_rows)
    write_csv(trial_df, processed_dir / "hpo_trials.csv")
    save_yaml(best_cfg, processed_dir / "hpo_best_config.yaml")

    append_experiment_record(
        root_dir=root_dir,
        row={
            "exp_id": exp_id,
            "market": "multi_market",
            "branch": "all",
            "horizon": "all",
            "universe_version": "runtime_universe",
            "data_version": datetime.now(timezone.utc).strftime("dv_%Y%m%d"),
            "feature_set_version": "features_runtime",
            "label_schema": "ret_h_quantile",
            "split_schema": "purged_wf_gap_h",
            "model_family": "mvp_classifier",
            "hpo_space_version": str(hpo_cfg.get("space_version", "hpo_v1")),
            "best_params": json.dumps(best_cfg.get("best", {}), ensure_ascii=False),
            "commit_hash": git_commit,
            "result_summary": (
                f"trials={len(trial_df)};best_mean_score="
                f"{_safe_float(trial_df['score'].max() if not trial_df.empty else np.nan):.6f}"
            ),
            "decision": "promote_candidate",
            "owner": owner,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        },
    )
    print(f"[OK] saved HPO outputs -> {processed_dir / 'hpo_trials.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-stage HPO (coarse + fine).")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_hpo(args.config)


if __name__ == "__main__":
    main()
