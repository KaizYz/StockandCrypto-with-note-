from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.utils.config import load_config
from src.utils.io import load_json, load_model, save_json, save_model, write_csv


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
    bins = max(int(bins), 2)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < bins - 1 else y_prob <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += abs(acc - conf) * (np.sum(mask) / len(y_true))
    return float(ece)


def _split_train_calib_holdout(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    training_cfg = cfg.get("training", {})
    holdout_cfg = training_cfg.get("holdout", {}) if isinstance(training_cfg, dict) else {}
    calib_cfg = training_cfg.get("calibration", {}) if isinstance(training_cfg, dict) else {}
    holdout_enabled = bool(holdout_cfg.get("enabled", True))
    holdout_ratio = float(holdout_cfg.get("ratio", 0.1))
    holdout_min_rows = int(holdout_cfg.get("min_rows", 60))
    calib_ratio = float(calib_cfg.get("ratio", 0.2))

    n = len(df)
    holdout_n = 0
    if holdout_enabled and n > 0:
        holdout_n = max(int(n * holdout_ratio), holdout_min_rows)
        holdout_n = min(holdout_n, max(0, n - 50))
    trainval_n = max(0, n - holdout_n)
    calib_n = int(trainval_n * calib_ratio)
    calib_n = min(max(calib_n, 20), max(trainval_n - 20, 0)) if trainval_n > 40 else 0

    train = df.iloc[: trainval_n - calib_n].copy() if calib_n > 0 else df.iloc[:trainval_n].copy()
    calib = df.iloc[trainval_n - calib_n : trainval_n].copy() if calib_n > 0 else pd.DataFrame(columns=df.columns)
    holdout = df.iloc[trainval_n:].copy() if holdout_n > 0 else pd.DataFrame(columns=df.columns)
    return {"train": train, "calib": calib, "holdout": holdout}


def run_calibrate(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    models_cfg = cfg.get("models", {})
    cal_cfg = models_cfg.get("calibration", {}) if isinstance(models_cfg, dict) else {}
    method = str(cal_cfg.get("method", "sigmoid"))
    ece_bins = int(cal_cfg.get("ece_bins", 10))

    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    models_dir = Path(paths.get("models_dir", "data/models"))
    latest_versions = load_json(models_dir / "latest_versions.json")

    all_rows: List[Dict[str, Any]] = []
    for branch_name, branch_cfg in (data_cfg.get("branches", {}) or {}).items():
        version_dir = Path(str(latest_versions.get(branch_name, "")))
        if not version_dir.exists():
            continue
        meta_path = version_dir / "artifact_meta.json"
        if not meta_path.exists():
            continue
        meta = load_json(meta_path)
        feature_cols = meta.get("feature_columns", [])
        medians = meta.get("feature_medians", {})

        feat_path = processed_dir / f"features_{branch_name}.csv"
        label_path = processed_dir / f"labels_{branch_name}.csv"
        if not feat_path.exists() or not label_path.exists():
            continue
        feat = pd.read_csv(feat_path)
        labels = pd.read_csv(label_path)
        df = feat.merge(labels, on="timestamp_utc", how="inner").sort_values("timestamp_utc").reset_index(drop=True)
        split = _split_train_calib_holdout(df, cfg)
        calib_df = split["calib"]
        holdout_df = split["holdout"]
        if calib_df.empty:
            continue

        X_cal = calib_df[feature_cols].fillna(medians)
        horizons = [int(h) for h in branch_cfg.get("horizons", [1])]
        for h in horizons:
            target = f"y_dir_h{h}"
            model_path = version_dir / f"direction_h{h}_mvp.pkl"
            if target not in calib_df.columns or not model_path.exists():
                continue

            calib_idx = calib_df[target].notna()
            if calib_idx.sum() < 20:
                continue
            X_cal_h = X_cal.loc[calib_idx]
            y_cal_h = calib_df.loc[calib_idx, target].astype(int)

            model = load_model(model_path)
            if not hasattr(model, "predict_proba"):
                continue

            p_before = np.asarray(pd.to_numeric(model.predict_proba(X_cal_h)[:, 1], errors="coerce"), dtype=float)
            brier_before = _brier_score(y_cal_h.to_numpy(), p_before)
            ece_before = _ece_score(y_cal_h.to_numpy(), p_before, bins=ece_bins)

            calibrated = CalibratedClassifierCV(estimator=model, cv=3, method=method)
            calibrated.fit(X_cal_h, y_cal_h)
            save_model(calibrated, version_dir / f"direction_h{h}_mvp_calibrated.pkl")

            p_after = np.asarray(pd.to_numeric(calibrated.predict_proba(X_cal_h)[:, 1], errors="coerce"), dtype=float)
            brier_after = _brier_score(y_cal_h.to_numpy(), p_after)
            ece_after = _ece_score(y_cal_h.to_numpy(), p_after, bins=ece_bins)

            row: Dict[str, Any] = {
                "branch": branch_name,
                "horizon": int(h),
                "calibration_method": method,
                "samples_calibration": int(len(y_cal_h)),
                "brier_before": brier_before,
                "brier_after": brier_after,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "delta_brier": _safe_float(brier_after - brier_before),
                "delta_ece": _safe_float(ece_after - ece_before),
            }

            if not holdout_df.empty and target in holdout_df.columns:
                hold_idx = holdout_df[target].notna()
                if hold_idx.sum() >= 20:
                    X_hold = holdout_df.loc[hold_idx, feature_cols].fillna(medians)
                    y_hold = holdout_df.loc[hold_idx, target].astype(int).to_numpy()
                    p_hold_before = np.asarray(pd.to_numeric(model.predict_proba(X_hold)[:, 1], errors="coerce"), dtype=float)
                    p_hold_after = np.asarray(pd.to_numeric(calibrated.predict_proba(X_hold)[:, 1], errors="coerce"), dtype=float)
                    row["holdout_samples"] = int(len(y_hold))
                    row["holdout_brier_before"] = _brier_score(y_hold, p_hold_before)
                    row["holdout_brier_after"] = _brier_score(y_hold, p_hold_after)
                    row["holdout_ece_before"] = _ece_score(y_hold, p_hold_before, bins=ece_bins)
                    row["holdout_ece_after"] = _ece_score(y_hold, p_hold_after, bins=ece_bins)
            all_rows.append(row)

    report_df = pd.DataFrame(all_rows)
    write_csv(report_df, processed_dir / "calibration_report.csv")
    save_json(
        {
            "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
            "config_path": config_path,
            "rows": report_df.to_dict(orient="records"),
        },
        processed_dir / "calibration_report.json",
    )
    print(f"[OK] saved calibration report -> {processed_dir / 'calibration_report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-train probability calibration report.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_calibrate(args.config)


if __name__ == "__main__":
    main()
