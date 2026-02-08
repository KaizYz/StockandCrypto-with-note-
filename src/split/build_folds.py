from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json, write_csv


def _build_purged_folds(
    *,
    n_rows: int,
    min_train_size: int,
    test_size: int,
    gap: int,
    max_folds: int,
) -> List[Tuple[int, int, int, int, int]]:
    # (fold, train_end, purge_start, test_start, test_end)
    rows: List[Tuple[int, int, int, int, int]] = []
    for i in range(max_folds):
        train_end = min_train_size + i * test_size
        purge_start = train_end
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n_rows:
            break
        rows.append((i, train_end, purge_start, test_start, test_end))
    return rows


def _safe_ts(series: pd.Series, idx: int) -> str:
    if idx < 0 or idx >= len(series):
        return ""
    return str(series.iloc[idx])


def run_build_folds(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    val_cfg = cfg.get("validation", {})
    training_cfg = cfg.get("training", {})

    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    ensure_dir(processed_dir)
    holdout_cfg = training_cfg.get("holdout", {}) if isinstance(training_cfg, dict) else {}
    holdout_enabled = bool(holdout_cfg.get("enabled", True))
    holdout_ratio = float(holdout_cfg.get("ratio", 0.1))
    holdout_min_rows = int(holdout_cfg.get("min_rows", 60))

    manifest_rows: List[Dict[str, Any]] = []
    holdout_rows: List[Dict[str, Any]] = []

    for branch_name, branch_cfg in (data_cfg.get("branches", {}) or {}).items():
        feat_path = processed_dir / f"features_{branch_name}.csv"
        if not feat_path.exists():
            print(f"[WARN] missing features for branch={branch_name}: {feat_path}")
            continue
        df = pd.read_csv(feat_path).sort_values("timestamp_utc").reset_index(drop=True)
        n = int(len(df))
        if n <= 0:
            continue
        ts = df["timestamp_utc"].astype(str)

        horizons = [int(h) for h in branch_cfg.get("horizons", [1])]
        gap = int(max(horizons)) if horizons else 1
        test_size = int(
            val_cfg.get("test_size", {}).get(branch_name, 120 if branch_name == "hourly" else 30)
        )
        min_train = int(
            val_cfg.get("min_train_size", {}).get(
                branch_name, 1500 if branch_name == "hourly" else 400
            )
        )
        max_folds = int(val_cfg.get("max_folds", 5))

        holdout_n = 0
        if holdout_enabled:
            holdout_n = max(int(n * holdout_ratio), holdout_min_rows)
            holdout_n = min(holdout_n, max(0, n - (min_train + gap + test_size)))
        trainval_n = n - holdout_n
        if trainval_n <= (min_train + gap + test_size):
            print(
                f"[WARN] insufficient rows for folds branch={branch_name}: n={n}, trainval={trainval_n}"
            )
            continue

        if holdout_n > 0:
            hs = trainval_n
            he = n
            holdout_rows.append(
                {
                    "branch": branch_name,
                    "total_rows": n,
                    "trainval_rows": trainval_n,
                    "holdout_rows": holdout_n,
                    "holdout_start_idx": hs,
                    "holdout_end_idx": he - 1,
                    "holdout_start_ts": _safe_ts(ts, hs),
                    "holdout_end_ts": _safe_ts(ts, he - 1),
                }
            )

        folds = _build_purged_folds(
            n_rows=trainval_n,
            min_train_size=min_train,
            test_size=test_size,
            gap=gap,
            max_folds=max_folds,
        )
        for fold, train_end, purge_start, test_start, test_end in folds:
            manifest_rows.append(
                {
                    "branch": branch_name,
                    "fold": int(fold),
                    "train_start_idx": 0,
                    "train_end_idx": train_end - 1,
                    "purge_start_idx": purge_start,
                    "purge_end_idx": test_start - 1,
                    "test_start_idx": test_start,
                    "test_end_idx": test_end - 1,
                    "gap": gap,
                    "test_size": test_size,
                    "train_start_ts": _safe_ts(ts, 0),
                    "train_end_ts": _safe_ts(ts, train_end - 1),
                    "test_start_ts": _safe_ts(ts, test_start),
                    "test_end_ts": _safe_ts(ts, test_end - 1),
                    "purge_start_ts": _safe_ts(ts, purge_start),
                    "purge_end_ts": _safe_ts(ts, test_start - 1),
                    "holdout_rows": holdout_n,
                }
            )

    folds_df = pd.DataFrame(manifest_rows)
    holdout_df = pd.DataFrame(holdout_rows)
    write_csv(folds_df, processed_dir / "folds_manifest.csv")
    write_csv(holdout_df, processed_dir / "holdout_manifest.csv")
    save_json(
        {
            "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
            "config_path": config_path,
            "folds_count": int(len(folds_df)),
            "branches": sorted(folds_df["branch"].dropna().unique().tolist()) if not folds_df.empty else [],
            "holdout_enabled": holdout_enabled,
            "holdout_rows": holdout_rows,
        },
        processed_dir / "split_manifest.json",
    )
    print(f"[OK] saved folds manifest -> {processed_dir / 'folds_manifest.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build purged walk-forward fold manifest.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_build_folds(args.config)


if __name__ == "__main__":
    main()
