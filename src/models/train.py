from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.models.factory import (
    build_logistic_binary,
    build_mvp_classifier,
    build_quantile_regressor,
)
from src.models.registry import append_experiment_record
from src.utils.artifacts import get_git_commit_short, hash_files, hash_json
from src.utils.common import set_global_seed, utc_now_timestamp
from src.utils.config import load_config, save_yaml
from src.utils.io import ensure_dir, load_json, save_json, save_model, write_csv
from src.utils.metrics import (
    classification_metrics,
    interval_metrics,
    multiclass_metrics,
    pinball_loss,
    regression_metrics,
    rows_from_metric_dict,
)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return float("nan")
    except Exception:
        return float("nan")


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def _ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < bins - 1 else y_prob <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += abs(acc - conf) * (np.sum(mask) / len(y_true))
    return float(ece)


def _load_branch_dataset(processed_dir: Path, branch_name: str) -> pd.DataFrame:
    feat_path = processed_dir / f"features_{branch_name}.csv"
    label_path = processed_dir / f"labels_{branch_name}.csv"
    if not feat_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Missing features/labels for branch={branch_name}. Run feature and label pipelines first."
        )
    feats = pd.read_csv(feat_path)
    labels = pd.read_csv(label_path)
    return feats.merge(labels, on="timestamp_utc", how="inner").sort_values("timestamp_utc").reset_index(drop=True)


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    ignore_prefix = ("y_",)
    ignore_cols = {"timestamp_utc", "timestamp_market", "market_tz"}
    return [
        c
        for c in df.columns
        if c not in ignore_cols
        and not c.startswith(ignore_prefix)
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _split_train_valid_holdout(
    df: pd.DataFrame,
    *,
    train_ratio: float,
    holdout_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    n = len(df)
    if n < 100:
        raise ValueError("Dataset too small for robust training (need >=100 rows).")
    holdout_enabled = bool(holdout_cfg.get("enabled", True))
    holdout_ratio = float(holdout_cfg.get("ratio", 0.1))
    holdout_min_rows = int(holdout_cfg.get("min_rows", 60))

    holdout_n = 0
    if holdout_enabled:
        holdout_n = max(int(n * holdout_ratio), holdout_min_rows)
        holdout_n = min(holdout_n, max(0, n - 80))
    trainval_n = n - holdout_n
    if trainval_n < 60:
        holdout_n = 0
        trainval_n = n

    trainval = df.iloc[:trainval_n].copy()
    holdout = df.iloc[trainval_n:].copy() if holdout_n > 0 else pd.DataFrame(columns=df.columns)

    split = int(trainval_n * train_ratio)
    split = max(40, min(split, trainval_n - 20))
    train = trainval.iloc[:split].copy()
    valid = trainval.iloc[split:].copy()
    return {
        "train": train,
        "valid": valid,
        "holdout": holdout,
        "meta": {
            "rows_total": int(n),
            "rows_train": int(len(train)),
            "rows_valid": int(len(valid)),
            "rows_holdout": int(len(holdout)),
            "train_end_ts": str(train["timestamp_utc"].iloc[-1]) if not train.empty else "",
            "valid_start_ts": str(valid["timestamp_utc"].iloc[0]) if not valid.empty else "",
            "holdout_start_ts": str(holdout["timestamp_utc"].iloc[0]) if not holdout.empty else "",
        },
    }


def _prepare_xy(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    tr = train_df.dropna(subset=[target_col]).copy()
    ev = eval_df.dropna(subset=[target_col]).copy()
    X_train = tr[feature_cols].copy()
    y_train = tr[target_col].copy()
    X_eval = ev[feature_cols].copy()
    y_eval = ev[target_col].copy()
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_eval = X_eval.fillna(med)
    return X_train, y_train, X_eval, y_eval, med


def _load_hpo_best(processed_dir: Path) -> Dict[str, Any]:
    hpo_path = processed_dir / "hpo_best_config.yaml"
    if not hpo_path.exists():
        return {}
    try:
        return load_config(hpo_path)
    except Exception:
        return {}


def _override_cfg_with_hpo(
    cfg: Dict[str, Any],
    *,
    branch: str,
    horizon: int,
    hpo_best: Dict[str, Any],
    enabled: bool,
) -> Dict[str, Any]:
    if not enabled:
        return cfg
    best_map = hpo_best.get("best", {}) if isinstance(hpo_best, dict) else {}
    params = (((best_map.get(branch) or {}).get(str(horizon)) or {}).get("params") or {})
    if not params:
        return cfg
    out = json.loads(json.dumps(cfg))
    out.setdefault("models", {})
    out["models"].setdefault("classifier", {})
    out["models"]["classifier"].update(params)
    return out


def _threshold_report_rows(
    *,
    branch: str,
    horizon: int,
    y_ret: np.ndarray,
    p_up: np.ndarray,
    fee_bps: float,
    slippage_bps: float,
) -> List[Dict[str, Any]]:
    cost = (max(fee_bps, 0.0) + max(slippage_bps, 0.0)) / 10000.0
    rows: List[Dict[str, Any]] = []
    threshold_grid = [
        (0.52, 0.48),
        (0.55, 0.45),
        (0.58, 0.42),
    ]
    for up_thr, down_thr in threshold_grid:
        signal = np.where(p_up >= up_thr, 1.0, np.where(p_up <= down_thr, -1.0, 0.0))
        pos = np.roll(signal, 1)
        pos[0] = 0.0
        turnover = np.abs(pos - np.roll(pos, 1))
        turnover[0] = 0.0
        strat = pos * y_ret - turnover * cost
        rows.append(
            {
                "branch": branch,
                "horizon": int(horizon),
                "p_up_long": up_thr,
                "p_up_short": down_thr,
                "trades": int(np.sum(np.abs(pos) > 0)),
                "mean_ret_per_bar": _safe_float(np.mean(strat)),
                "total_return_proxy": _safe_float(np.sum(strat)),
                "win_rate": _safe_float(np.mean(strat > 0)),
            }
        )
    return rows


def _write_model_card(
    *,
    version_dir: Path,
    branch: str,
    backend: str,
    split_meta: Dict[str, Any],
    notes: List[str],
) -> None:
    lines = [
        f"# Model Card ({branch})",
        "",
        f"- backend: `{backend}`",
        f"- generated_at_utc: `{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        "",
        "## Data Split",
        f"- rows_total: {split_meta.get('rows_total')}",
        f"- rows_train: {split_meta.get('rows_train')}",
        f"- rows_valid: {split_meta.get('rows_valid')}",
        f"- rows_holdout: {split_meta.get('rows_holdout')}",
        f"- train_end_ts: {split_meta.get('train_end_ts')}",
        f"- valid_start_ts: {split_meta.get('valid_start_ts')}",
        f"- holdout_start_ts: {split_meta.get('holdout_start_ts')}",
        "",
        "## Known Limits",
    ]
    if notes:
        lines.extend([f"- {x}" for x in notes])
    else:
        lines.append("- none")
    (version_dir / "model_card.md").write_text("\n".join(lines), encoding="utf-8")


def run_train(config_path: str) -> None:
    cfg = load_config(config_path)
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))

    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    models_cfg = cfg.get("models", {})
    training_cfg = cfg.get("training", {})
    holdout_cfg = training_cfg.get("holdout", {}) if isinstance(training_cfg, dict) else {}

    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    models_root = ensure_dir(paths.get("models_dir", "data/models"))
    train_ratio = float(models_cfg.get("train_ratio", 0.8))
    quantiles: List[float] = [float(x) for x in models_cfg.get("quantiles", [0.1, 0.5, 0.9])]
    calibrate_cfg = models_cfg.get("calibration", {})
    do_calibrate = bool(calibrate_cfg.get("enabled", True))
    cal_method = str(calibrate_cfg.get("method", "sigmoid"))
    cal_cv = int(calibrate_cfg.get("cv", 3))
    cal_ece_bins = int(calibrate_cfg.get("ece_bins", 10))
    seed = int(cfg.get("project", {}).get("seed", 42))
    git_commit = get_git_commit_short("-")

    symbol = str(data_cfg.get("symbol", "BTCUSDT")).lower()
    branches = data_cfg.get("branches", {})
    ts = utc_now_timestamp()

    hpo_best = _load_hpo_best(processed_dir)
    use_hpo_best = bool(training_cfg.get("use_hpo_best_for_training", True))
    fee_bps = _safe_float((cfg.get("policy", {}).get("execution", {}) or {}).get("fee_bps", 10))
    slippage_bps = _safe_float((cfg.get("policy", {}).get("execution", {}) or {}).get("slippage_bps", 10))

    latest_versions: Dict[str, str] = {}
    overall_rows: List[Dict[str, Any]] = []
    holdout_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []

    for branch_name, branch_cfg in branches.items():
        df = _load_branch_dataset(processed_dir, branch_name)
        split = _split_train_valid_holdout(df, train_ratio=train_ratio, holdout_cfg=holdout_cfg)
        train_df = split["train"]
        valid_df = split["valid"]
        holdout_df = split["holdout"]
        split_meta = split["meta"]

        feature_cols = _select_feature_columns(df)
        version_dir = ensure_dir(models_root / f"{ts}_{branch_name}")
        latest_versions[branch_name] = str(version_dir)
        medians_to_save: Dict[str, float] = {}
        branch_rows: List[Dict[str, Any]] = []
        branch_backend = "unknown"

        target_start = "y_start_window"
        X_train, y_train, X_valid, y_valid, med = _prepare_xy(train_df, valid_df, feature_cols, target_start)
        y_train_cls = y_train.astype(int)
        y_valid_cls = y_valid.astype(int)
        medians_to_save.update({k: float(v) for k, v in med.to_dict().items()})

        most_freq = int(y_train_cls.value_counts().index[0])
        pred_base = np.full(len(y_valid_cls), most_freq, dtype=int)
        n_class = max(2, int(y_train_cls.max()) + 1)
        proba_base = np.zeros((len(y_valid_cls), n_class))
        for i, p in enumerate(pred_base):
            p = int(min(max(p, 0), n_class - 1))
            proba_base[i, p] = 1.0
        m_base = multiclass_metrics(y_valid_cls.to_numpy(), pred_base, proba_base)
        branch_rows.extend(
            rows_from_metric_dict(
                m_base,
                {
                    "branch": branch_name,
                    "task": "start_window",
                    "model": "baseline_most_freq",
                    "horizon": "max",
                    "split": "valid",
                },
            )
        )

        start_model, branch_backend = build_mvp_classifier(cfg, n_classes=n_class, seed=seed)
        start_model.fit(X_train, y_train_cls)
        pred_mvp = start_model.predict(X_valid)
        proba_mvp = start_model.predict_proba(X_valid)
        m_mvp = multiclass_metrics(y_valid_cls.to_numpy(), pred_mvp, proba_mvp)
        branch_rows.extend(
            rows_from_metric_dict(
                m_mvp,
                {
                    "branch": branch_name,
                    "task": "start_window",
                    "model": f"mvp_{branch_backend}",
                    "horizon": "max",
                    "split": "valid",
                },
            )
        )
        save_model(start_model, version_dir / "start_window_mvp.pkl")

        if not holdout_df.empty and target_start in holdout_df.columns:
            hold_idx = holdout_df[target_start].notna()
            if hold_idx.sum() > 0:
                X_hold = holdout_df.loc[hold_idx, feature_cols].fillna(med)
                y_hold = holdout_df.loc[hold_idx, target_start].astype(int)
                pred_hold = start_model.predict(X_hold)
                proba_hold = start_model.predict_proba(X_hold)
                m_hold = multiclass_metrics(y_hold.to_numpy(), pred_hold, proba_hold)
                branch_rows.extend(
                    rows_from_metric_dict(
                        m_hold,
                        {
                            "branch": branch_name,
                            "task": "start_window",
                            "model": f"mvp_{branch_backend}",
                            "horizon": "max",
                            "split": "frozen_holdout",
                        },
                    )
                )

        horizons: List[int] = [int(h) for h in branch_cfg.get("horizons", [1, 2, 4])]
        for h in horizons:
            target_dir = f"y_dir_h{h}"
            if target_dir in df.columns:
                X_train, y_train, X_valid, y_valid, med = _prepare_xy(train_df, valid_df, feature_cols, target_dir)
                y_train_bin = y_train.astype(int)
                y_valid_bin = y_valid.astype(int)
                medians_to_save.update({k: float(v) for k, v in med.to_dict().items()})

                prev_ret = valid_df.loc[y_valid.index, "return_1"].fillna(0.0)
                pred_base = (prev_ret > 0).astype(int).to_numpy()
                proba_base = pred_base.astype(float)
                m_base = classification_metrics(y_valid_bin.to_numpy(), pred_base, proba_base)
                branch_rows.extend(
                    rows_from_metric_dict(
                        m_base,
                        {
                            "branch": branch_name,
                            "task": "direction",
                            "model": "baseline_prev_bar",
                            "horizon": str(h),
                            "split": "valid",
                        },
                    )
                )

                logistic = build_logistic_binary(seed=seed)
                logistic.fit(X_train, y_train_bin)
                pred_lr = logistic.predict(X_valid)
                proba_lr = logistic.predict_proba(X_valid)[:, 1]
                m_lr = classification_metrics(y_valid_bin.to_numpy(), pred_lr, proba_lr)
                branch_rows.extend(
                    rows_from_metric_dict(
                        m_lr,
                        {
                            "branch": branch_name,
                            "task": "direction",
                            "model": "baseline_logistic",
                            "horizon": str(h),
                            "split": "valid",
                        },
                    )
                )
                save_model(logistic, version_dir / f"direction_h{h}_baseline_logistic.pkl")

                cfg_dir = _override_cfg_with_hpo(
                    cfg,
                    branch=branch_name,
                    horizon=h,
                    hpo_best=hpo_best,
                    enabled=use_hpo_best,
                )
                mvp_dir, backend = build_mvp_classifier(cfg_dir, n_classes=2, seed=seed)
                branch_backend = backend
                mvp_dir.fit(X_train, y_train_bin)

                if do_calibrate:
                    calibrated = CalibratedClassifierCV(estimator=mvp_dir, method=cal_method, cv=cal_cv)
                    calibrated.fit(X_train, y_train_bin)
                    dir_model = calibrated
                    model_name = f"mvp_{backend}_calibrated_{cal_method}"
                else:
                    dir_model = mvp_dir
                    model_name = f"mvp_{backend}"

                pred_mvp = dir_model.predict(X_valid)
                proba_mvp = dir_model.predict_proba(X_valid)[:, 1]
                m_mvp = classification_metrics(y_valid_bin.to_numpy(), pred_mvp, proba_mvp)
                branch_rows.extend(
                    rows_from_metric_dict(
                        m_mvp,
                        {
                            "branch": branch_name,
                            "task": "direction",
                            "model": model_name,
                            "horizon": str(h),
                            "split": "valid",
                        },
                    )
                )
                branch_rows.append(
                    {
                        "branch": branch_name,
                        "task": "direction_calibration",
                        "model": model_name,
                        "horizon": str(h),
                        "split": "valid",
                        "metric": "brier",
                        "value": _brier_score(y_valid_bin.to_numpy(), proba_mvp),
                    }
                )
                branch_rows.append(
                    {
                        "branch": branch_name,
                        "task": "direction_calibration",
                        "model": model_name,
                        "horizon": str(h),
                        "split": "valid",
                        "metric": "ece",
                        "value": _ece_score(y_valid_bin.to_numpy(), proba_mvp, bins=cal_ece_bins),
                    }
                )
                save_model(dir_model, version_dir / f"direction_h{h}_mvp.pkl")

                target_ret_for_thr = f"y_ret_h{h}"
                if target_ret_for_thr in valid_df.columns:
                    y_ret_thr = pd.to_numeric(valid_df.loc[y_valid.index, target_ret_for_thr], errors="coerce").fillna(0.0).to_numpy()
                else:
                    y_ret_thr = np.zeros(len(y_valid_bin))
                threshold_rows.extend(
                    _threshold_report_rows(
                        branch=branch_name,
                        horizon=h,
                        y_ret=y_ret_thr,
                        p_up=proba_mvp,
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                    )
                )

                if not holdout_df.empty and target_dir in holdout_df.columns:
                    hold_idx = holdout_df[target_dir].notna()
                    if hold_idx.sum() >= 20:
                        X_hold = holdout_df.loc[hold_idx, feature_cols].fillna(med)
                        y_hold = holdout_df.loc[hold_idx, target_dir].astype(int)
                        pred_hold = dir_model.predict(X_hold)
                        proba_hold = dir_model.predict_proba(X_hold)[:, 1]
                        m_hold = classification_metrics(y_hold.to_numpy(), pred_hold, proba_hold)
                        branch_rows.extend(
                            rows_from_metric_dict(
                                m_hold,
                                {
                                    "branch": branch_name,
                                    "task": "direction",
                                    "model": model_name,
                                    "horizon": str(h),
                                    "split": "frozen_holdout",
                                },
                            )
                        )
                        holdout_rows.append(
                            {
                                "branch": branch_name,
                                "task": "direction",
                                "horizon": int(h),
                                "samples": int(len(y_hold)),
                                "accuracy": _safe_float(m_hold.get("accuracy")),
                                "brier": _brier_score(y_hold.to_numpy(), proba_hold),
                                "ece": _ece_score(y_hold.to_numpy(), proba_hold, bins=cal_ece_bins),
                            }
                        )

            target_ret = f"y_ret_h{h}"
            if target_ret in df.columns:
                X_train, y_train, X_valid, y_valid, med = _prepare_xy(train_df, valid_df, feature_cols, target_ret)
                y_train_reg = y_train.astype(float)
                y_valid_reg = y_valid.astype(float)
                medians_to_save.update({k: float(v) for k, v in med.to_dict().items()})

                pred_zero = np.zeros(len(y_valid_reg), dtype=float)
                m_zero = regression_metrics(y_valid_reg.to_numpy(), pred_zero)
                branch_rows.extend(
                    rows_from_metric_dict(
                        m_zero,
                        {
                            "branch": branch_name,
                            "task": "magnitude",
                            "model": "baseline_zero",
                            "horizon": str(h),
                            "split": "valid",
                        },
                    )
                )

                q_preds: Dict[float, np.ndarray] = {}
                quant_backend = ""
                for q in quantiles:
                    reg, quant_backend = build_quantile_regressor(cfg, quantile=q, seed=seed)
                    reg.fit(X_train, y_train_reg)
                    yq = reg.predict(X_valid)
                    q_preds[q] = yq
                    save_model(reg, version_dir / f"ret_h{h}_q{q:.1f}_mvp.pkl")
                    branch_rows.append(
                        {
                            "branch": branch_name,
                            "task": "magnitude_quantile",
                            "model": f"mvp_{quant_backend}",
                            "horizon": str(h),
                            "split": "valid",
                            "metric": f"pinball_q{q:.1f}",
                            "value": pinball_loss(y_valid_reg.to_numpy(), yq, q),
                        }
                    )

                if 0.5 in q_preds:
                    reg_m = regression_metrics(y_valid_reg.to_numpy(), q_preds[0.5])
                    branch_rows.extend(
                        rows_from_metric_dict(
                            reg_m,
                            {
                                "branch": branch_name,
                                "task": "magnitude",
                                "model": f"mvp_{quant_backend}_q50",
                                "horizon": str(h),
                                "split": "valid",
                            },
                        )
                    )

                if 0.1 in q_preds and 0.9 in q_preds:
                    im = interval_metrics(y_valid_reg.to_numpy(), q_preds[0.1], q_preds[0.9])
                    branch_rows.extend(
                        rows_from_metric_dict(
                            im,
                            {
                                "branch": branch_name,
                                "task": "interval",
                                "model": f"mvp_{quant_backend}",
                                "horizon": str(h),
                                "split": "valid",
                            },
                        )
                    )

                if not holdout_df.empty and target_ret in holdout_df.columns and len(holdout_df) >= 30:
                    hold_idx = holdout_df[target_ret].notna()
                    if hold_idx.sum() >= 20:
                        X_hold = holdout_df.loc[hold_idx, feature_cols].fillna(med)
                        y_hold = holdout_df.loc[hold_idx, target_ret].astype(float).to_numpy()
                        reg10, _ = build_quantile_regressor(cfg, quantile=0.1, seed=seed)
                        reg50, _ = build_quantile_regressor(cfg, quantile=0.5, seed=seed)
                        reg90, _ = build_quantile_regressor(cfg, quantile=0.9, seed=seed)
                        reg10.fit(X_train, y_train_reg)
                        reg50.fit(X_train, y_train_reg)
                        reg90.fit(X_train, y_train_reg)
                        q10_hold = reg10.predict(X_hold)
                        q50_hold = reg50.predict(X_hold)
                        q90_hold = reg90.predict(X_hold)
                        reg_h = regression_metrics(y_hold, q50_hold)
                        int_h = interval_metrics(y_hold, q10_hold, q90_hold)
                        branch_rows.extend(
                            rows_from_metric_dict(
                                reg_h,
                                {
                                    "branch": branch_name,
                                    "task": "magnitude",
                                    "model": f"mvp_{quant_backend}_q50",
                                    "horizon": str(h),
                                    "split": "frozen_holdout",
                                },
                            )
                        )
                        branch_rows.extend(
                            rows_from_metric_dict(
                                int_h,
                                {
                                    "branch": branch_name,
                                    "task": "interval",
                                    "model": f"mvp_{quant_backend}",
                                    "horizon": str(h),
                                    "split": "frozen_holdout",
                                },
                            )
                        )
                        holdout_rows.append(
                            {
                                "branch": branch_name,
                                "task": "interval",
                                "horizon": int(h),
                                "samples": int(len(y_hold)),
                                "coverage": _safe_float(int_h.get("coverage")),
                                "interval_width": _safe_float(int_h.get("interval_width")),
                            }
                        )

        artifact_meta = {
            "symbol": symbol,
            "branch": branch_name,
            "feature_columns": feature_cols,
            "feature_medians": medians_to_save,
            "quantiles": quantiles,
            "seed": seed,
            "split_meta": split_meta,
            "git_commit": git_commit,
        }
        save_json(artifact_meta, version_dir / "artifact_meta.json")
        save_yaml(cfg, version_dir / "config_snapshot.yaml")
        (version_dir / "git_commit.txt").write_text(f"{git_commit}\n", encoding="utf-8")

        raw_file = Path(paths.get("raw_data_dir", "data/raw")) / f"{symbol}_{branch_name}.csv"
        data_files = [
            processed_dir / f"features_{branch_name}.csv",
            processed_dir / f"labels_{branch_name}.csv",
            raw_file,
        ]
        code_files = [
            Path("src/models/train.py"),
            Path("src/models/factory.py"),
            Path("src/features/build_features.py"),
            Path("src/labels/build_labels.py"),
        ]
        hashes = {
            "config_hash": hash_json(cfg),
            "data_hash": hash_files([p for p in data_files if Path(p).exists()]),
            "code_hash": hash_files([p for p in code_files if Path(p).exists()]),
        }
        save_json(hashes, version_dir / "hashes.json")

        branch_metrics = pd.DataFrame(branch_rows)
        write_csv(branch_metrics, version_dir / "metrics.csv")
        _write_model_card(
            version_dir=version_dir,
            branch=branch_name,
            backend=branch_backend,
            split_meta=split_meta,
            notes=[
                "No guarantee of 100% correctness; use risk constraints.",
                "Use go-live gate before enabling any automated execution.",
            ],
        )
        print(f"[OK] Saved model artifacts -> {version_dir}")
        overall_rows.extend(branch_rows)

    save_json(latest_versions, models_root / "latest_versions.json")
    if overall_rows:
        df_all = pd.DataFrame(overall_rows)
        write_csv(df_all, processed_dir / "metrics_train_holdout.csv")
        summary = (
            df_all.groupby(["branch", "task", "model", "horizon", "metric", "split"], as_index=False)["value"]
            .mean()
        )
        write_csv(summary, processed_dir / "metrics_train_holdout_summary.csv")
    write_csv(pd.DataFrame(threshold_rows), processed_dir / "threshold_report.csv")
    save_json(
        {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "rows": holdout_rows,
        },
        processed_dir / "holdout_report.json",
    )

    append_experiment_record(
        root_dir=Path(config_path).resolve().parent.parent,
        row={
            "exp_id": f"train_{ts}",
            "market": "multi_market",
            "branch": "all",
            "horizon": "all",
            "universe_version": "runtime_universe",
            "data_version": datetime.now(timezone.utc).strftime("dv_%Y%m%d"),
            "feature_set_version": "features_runtime",
            "label_schema": "ret_h_quantile",
            "split_schema": "train_valid_holdout",
            "model_family": str(models_cfg.get("mvp_backend", "mvp")),
            "hpo_space_version": "hpo_best_config_yaml",
            "best_params": json.dumps(hpo_best.get("best", {}), ensure_ascii=False),
            "commit_hash": git_commit,
            "result_summary": (
                f"metrics_rows={len(overall_rows)};holdout_rows={len(holdout_rows)};"
                f"threshold_rows={len(threshold_rows)}"
            ),
            "decision": "trained",
            "owner": str((training_cfg.get("hpo", {}) or {}).get("owner", "owner")),
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline + MVP models with governance artifacts.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_train(args.config)


if __name__ == "__main__":
    main()
