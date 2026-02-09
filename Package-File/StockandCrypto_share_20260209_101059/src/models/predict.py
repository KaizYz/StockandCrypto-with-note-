from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.io import load_json, load_model, save_json, write_csv


WINDOW_NAME_MAP = {0: "W0", 1: "W1", 2: "W2", 3: "W3"}


def _safe_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    pred = model.predict(X)
    if pred.ndim == 1:
        out = np.zeros((len(pred), 2), dtype=float)
        out[:, 1] = pred
        out[:, 0] = 1.0 - pred
        return out
    return pred


def run_predict(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("models", {})
    quantiles = [float(x) for x in model_cfg.get("quantiles", [0.1, 0.5, 0.9])]

    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    models_dir = Path(paths.get("models_dir", "data/models"))
    latest_versions = load_json(models_dir / "latest_versions.json")

    summary: Dict[str, Dict] = {}
    for branch_name, branch_cfg in data_cfg.get("branches", {}).items():
        version_dir = Path(latest_versions[branch_name])
        meta = load_json(version_dir / "artifact_meta.json")
        feature_cols: List[str] = meta["feature_columns"]
        medians: Dict[str, float] = meta.get("feature_medians", {})
        horizons = [int(h) for h in branch_cfg.get("horizons", [1, 2, 4])]

        feat_path = processed_dir / f"features_{branch_name}.csv"
        feats = pd.read_csv(feat_path).sort_values("timestamp_utc").reset_index(drop=True)
        X = feats[feature_cols].copy()
        X = X.fillna(medians)

        base_cols = ["timestamp_utc", "timestamp_market", "market_tz"]
        if "close" in feats.columns:
            base_cols.append("close")
        pred_df = feats[base_cols].copy()

        # Start-window model.
        start_model = load_model(version_dir / "start_window_mvp.pkl")
        start_proba = _safe_predict_proba(start_model, X)
        classes = getattr(start_model, "classes_", np.arange(start_proba.shape[1]))
        start_idx = np.argmax(start_proba, axis=1)
        start_class = np.array([int(classes[i]) for i in start_idx], dtype=int)
        pred_df["start_window_pred"] = start_class
        pred_df["start_window_name"] = pd.Series(start_class).map(WINDOW_NAME_MAP).fillna("W?")
        for w in range(4):
            pred_df[f"start_p_w{w}"] = 0.0
        for cls_idx, cls_val in enumerate(classes):
            c = int(cls_val)
            if 0 <= c <= 3:
                pred_df[f"start_p_w{c}"] = start_proba[:, cls_idx]

        for h in horizons:
            # Direction
            dir_model = load_model(version_dir / f"direction_h{h}_mvp.pkl")
            dir_proba = _safe_predict_proba(dir_model, X)
            if dir_proba.shape[1] == 1:
                p_up = dir_proba[:, 0]
            else:
                p_up = dir_proba[:, 1]
            pred_df[f"dir_h{h}_p_up"] = p_up
            pred_df[f"dir_h{h}_p_down"] = 1.0 - p_up
            pred_df[f"dir_h{h}_pred"] = (p_up >= 0.5).astype(int)

            # Quantiles
            q_values = []
            for q in quantiles:
                q_model = load_model(version_dir / f"ret_h{h}_q{q:.1f}_mvp.pkl")
                q_pred = q_model.predict(X)
                pred_df[f"ret_h{h}_q{q:.1f}"] = q_pred
                q_values.append(q_pred)

            # Enforce non-crossing by sorting each row.
            if len(q_values) >= 3:
                stack = np.vstack(q_values).T
                stack_sorted = np.sort(stack, axis=1)
                for idx, q in enumerate(sorted(quantiles)):
                    pred_df[f"ret_h{h}_q{q:.1f}"] = stack_sorted[:, idx]

        out_path = processed_dir / f"predictions_{branch_name}.csv"
        write_csv(pred_df, out_path)
        print(f"[OK] Saved predictions -> {out_path}")

        latest_row = pred_df.iloc[-1].to_dict()
        summary[branch_name] = {
            "model_version": str(version_dir),
            "timestamp_utc": latest_row.get("timestamp_utc"),
            "latest": latest_row,
        }

    save_json(summary, processed_dir / "predictions_latest_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model inference using latest trained models.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_predict(args.config)


if __name__ == "__main__":
    main()
