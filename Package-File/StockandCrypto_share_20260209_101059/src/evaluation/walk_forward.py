from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.factory import (
    build_logistic_binary,
    build_mvp_classifier,
    build_quantile_regressor,
)
from src.utils.common import set_global_seed
from src.utils.config import load_config
from src.utils.io import write_csv
from src.utils.metrics import (
    classification_metrics,
    interval_metrics,
    multiclass_metrics,
    pinball_loss,
    regression_metrics,
    rows_from_metric_dict,
)


def _load_data(processed_dir: Path, branch: str) -> pd.DataFrame:
    feats = pd.read_csv(processed_dir / f"features_{branch}.csv")
    labels = pd.read_csv(processed_dir / f"labels_{branch}.csv")
    df = feats.merge(labels, on="timestamp_utc", how="inner").sort_values("timestamp_utc")
    if "missing_flag" in df.columns:
        df = df[df["missing_flag"] == 0]
    return df.reset_index(drop=True)


def _feature_cols(df: pd.DataFrame) -> List[str]:
    ignore = {"timestamp_utc", "timestamp_market", "market_tz"}
    return [
        c
        for c in df.columns
        if c not in ignore and not c.startswith("y_") and pd.api.types.is_numeric_dtype(df[c])
    ]


def _get_fold_ranges(
    n: int, *, min_train: int, test_size: int, gap: int, max_folds: int
) -> List[Dict[str, int]]:
    folds: List[Dict[str, int]] = []
    for i in range(max_folds):
        train_end = min_train + i * test_size
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n:
            break
        folds.append(
            {
                "fold": i,
                "train_start": 0,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    return folds


def _prep_xy(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], target: str
):
    tr = train_df.dropna(subset=[target]).copy()
    te = test_df.dropna(subset=[target]).copy()
    X_tr = tr[feature_cols].copy()
    y_tr = tr[target].copy()
    X_te = te[feature_cols].copy()
    y_te = te[target].copy()
    med = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(med)
    X_te = X_te.fillna(med)
    return X_tr, y_tr, X_te, y_te


def run_walk_forward(config_path: str) -> None:
    cfg = load_config(config_path)
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))

    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    val_cfg = cfg.get("validation", {})
    model_cfg = cfg.get("models", {})
    quantiles = [float(x) for x in model_cfg.get("quantiles", [0.1, 0.5, 0.9])]
    seed = int(cfg.get("project", {}).get("seed", 42))

    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    rows: List[Dict] = []

    for branch_name, branch_cfg in data_cfg.get("branches", {}).items():
        horizons = [int(h) for h in branch_cfg.get("horizons", [1, 2, 4])]
        gap = max(horizons)  # Purged gap = max_horizon
        test_size = int(val_cfg.get("test_size", {}).get(branch_name, 120 if branch_name == "hourly" else 30))
        min_train = int(
            val_cfg.get("min_train_size", {}).get(branch_name, 1500 if branch_name == "hourly" else 400)
        )
        max_folds = int(val_cfg.get("max_folds", 5))

        df = _load_data(processed_dir, branch_name)
        feature_cols = _feature_cols(df)
        folds = _get_fold_ranges(
            len(df), min_train=min_train, test_size=test_size, gap=gap, max_folds=max_folds
        )
        if not folds:
            print(f"[WARN] branch={branch_name} has insufficient data for walk-forward.")
            continue

        for fold_info in folds:
            fold = fold_info["fold"]
            train_df = df.iloc[fold_info["train_start"] : fold_info["train_end"]].copy()
            test_df = df.iloc[fold_info["test_start"] : fold_info["test_end"]].copy()

            # Start-window
            X_tr, y_tr, X_te, y_te = _prep_xy(train_df, test_df, feature_cols, "y_start_window")
            y_tr_cls = y_tr.astype(int)
            y_te_cls = y_te.astype(int)
            if len(y_tr_cls) > 0 and len(y_te_cls) > 0:
                # Baseline most frequent class in train
                mf = int(y_tr_cls.value_counts().index[0])
                pred_b = np.full(len(y_te_cls), mf, dtype=int)
                n_class = int(max(2, int(y_tr_cls.max()) + 1))
                proba_b = np.zeros((len(y_te_cls), n_class))
                for i, p in enumerate(pred_b):
                    p = int(min(max(p, 0), n_class - 1))
                    proba_b[i, p] = 1.0
                m = multiclass_metrics(y_te_cls.to_numpy(), pred_b, proba_b)
                rows.extend(
                    rows_from_metric_dict(
                        m,
                        {
                            "branch": branch_name,
                            "fold": str(fold),
                            "task": "start_window",
                            "model": "baseline_most_freq",
                            "horizon": "max",
                        },
                    )
                )

                mvp, backend = build_mvp_classifier(cfg, n_classes=n_class, seed=seed)
                mvp.fit(X_tr, y_tr_cls)
                pred = mvp.predict(X_te)
                proba = mvp.predict_proba(X_te)
                m = multiclass_metrics(y_te_cls.to_numpy(), pred, proba)
                rows.extend(
                    rows_from_metric_dict(
                        m,
                        {
                            "branch": branch_name,
                            "fold": str(fold),
                            "task": "start_window",
                            "model": f"mvp_{backend}",
                            "horizon": "max",
                        },
                    )
                )

            # Direction + magnitude per horizon
            for h in horizons:
                # Direction
                target_dir = f"y_dir_h{h}"
                X_tr, y_tr, X_te, y_te = _prep_xy(train_df, test_df, feature_cols, target_dir)
                if len(y_tr) > 10 and len(y_te) > 0:
                    y_tr_bin = y_tr.astype(int)
                    y_te_bin = y_te.astype(int)
                    prev_ret = test_df.loc[y_te.index, "return_1"].fillna(0.0)
                    pred_b = (prev_ret > 0).astype(int).to_numpy()
                    proba_b = pred_b.astype(float)
                    m = classification_metrics(y_te_bin.to_numpy(), pred_b, proba_b)
                    rows.extend(
                        rows_from_metric_dict(
                            m,
                            {
                                "branch": branch_name,
                                "fold": str(fold),
                                "task": "direction",
                                "model": "baseline_prev_bar",
                                "horizon": str(h),
                            },
                        )
                    )

                    lr = build_logistic_binary(seed=seed)
                    lr.fit(X_tr, y_tr_bin)
                    pred_lr = lr.predict(X_te)
                    proba_lr = lr.predict_proba(X_te)[:, 1]
                    m = classification_metrics(y_te_bin.to_numpy(), pred_lr, proba_lr)
                    rows.extend(
                        rows_from_metric_dict(
                            m,
                            {
                                "branch": branch_name,
                                "fold": str(fold),
                                "task": "direction",
                                "model": "baseline_logistic",
                                "horizon": str(h),
                            },
                        )
                    )

                    mvp, backend = build_mvp_classifier(cfg, n_classes=2, seed=seed)
                    mvp.fit(X_tr, y_tr_bin)
                    pred = mvp.predict(X_te)
                    proba = mvp.predict_proba(X_te)[:, 1]
                    m = classification_metrics(y_te_bin.to_numpy(), pred, proba)
                    rows.extend(
                        rows_from_metric_dict(
                            m,
                            {
                                "branch": branch_name,
                                "fold": str(fold),
                                "task": "direction",
                                "model": f"mvp_{backend}",
                                "horizon": str(h),
                            },
                        )
                    )

                # Magnitude / interval
                target_ret = f"y_ret_h{h}"
                X_tr, y_tr, X_te, y_te = _prep_xy(train_df, test_df, feature_cols, target_ret)
                if len(y_tr) > 10 and len(y_te) > 0:
                    y_tr_reg = y_tr.astype(float)
                    y_te_reg = y_te.astype(float)

                    pred_zero = np.zeros(len(y_te_reg))
                    m = regression_metrics(y_te_reg.to_numpy(), pred_zero)
                    rows.extend(
                        rows_from_metric_dict(
                            m,
                            {
                                "branch": branch_name,
                                "fold": str(fold),
                                "task": "magnitude",
                                "model": "baseline_zero",
                                "horizon": str(h),
                            },
                        )
                    )

                    q_preds: Dict[float, np.ndarray] = {}
                    backend_name = ""
                    for q in quantiles:
                        reg, backend_name = build_quantile_regressor(cfg, quantile=q, seed=seed)
                        reg.fit(X_tr, y_tr_reg)
                        yq = reg.predict(X_te)
                        q_preds[q] = yq
                        rows.append(
                            {
                                "branch": branch_name,
                                "fold": str(fold),
                                "task": "magnitude_quantile",
                                "model": f"mvp_{backend_name}",
                                "horizon": str(h),
                                "metric": f"pinball_q{q:.1f}",
                                "value": pinball_loss(y_te_reg.to_numpy(), yq, q),
                            }
                        )

                    if 0.5 in q_preds:
                        m = regression_metrics(y_te_reg.to_numpy(), q_preds[0.5])
                        rows.extend(
                            rows_from_metric_dict(
                                m,
                                {
                                    "branch": branch_name,
                                    "fold": str(fold),
                                    "task": "magnitude",
                                    "model": f"mvp_{backend_name}_q50",
                                    "horizon": str(h),
                                },
                            )
                        )
                    if 0.1 in q_preds and 0.9 in q_preds:
                        m = interval_metrics(y_te_reg.to_numpy(), q_preds[0.1], q_preds[0.9])
                        rows.extend(
                            rows_from_metric_dict(
                                m,
                                {
                                    "branch": branch_name,
                                    "fold": str(fold),
                                    "task": "interval",
                                    "model": f"mvp_{backend_name}",
                                    "horizon": str(h),
                                },
                            )
                        )

    if not rows:
        print("[WARN] No walk-forward metrics generated.")
        return

    metrics_df = pd.DataFrame(rows)
    write_csv(metrics_df, processed_dir / "metrics_walk_forward.csv")

    summary = (
        metrics_df.groupby(["branch", "task", "model", "horizon", "metric"], as_index=False)["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    write_csv(summary, processed_dir / "metrics_walk_forward_summary.csv")
    print(f"[OK] Saved walk-forward metrics -> {processed_dir / 'metrics_walk_forward.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation with purged gap.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_walk_forward(args.config)


if __name__ == "__main__":
    main()
