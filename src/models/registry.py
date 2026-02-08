from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


DEFAULT_COLUMNS = [
    "exp_id",
    "market",
    "branch",
    "horizon",
    "universe_version",
    "data_version",
    "feature_set_version",
    "label_schema",
    "split_schema",
    "model_family",
    "hpo_space_version",
    "best_params",
    "commit_hash",
    "result_summary",
    "decision",
    "owner",
    "generated_at_utc",
]


def append_experiment_record(
    *,
    root_dir: Path,
    row: Dict[str, Any],
) -> None:
    experiments_dir = root_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    registry_path = experiments_dir / "registry.csv"

    if registry_path.exists():
        df = pd.read_csv(registry_path)
    else:
        df = pd.DataFrame(columns=DEFAULT_COLUMNS)
    for col in DEFAULT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    new_row = {c: row.get(c, "") for c in DEFAULT_COLUMNS}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(registry_path, index=False, encoding="utf-8-sig")

    exp_id = str(new_row.get("exp_id", "")).strip()
    if exp_id:
        notes_dir = experiments_dir / exp_id
        notes_dir.mkdir(parents=True, exist_ok=True)
        notes_path = notes_dir / "notes.md"
        if not notes_path.exists():
            notes_path.write_text(
                "\n".join(
                    [
                        f"# Experiment {exp_id}",
                        "",
                        "## Notes",
                        "",
                        "-",
                    ]
                ),
                encoding="utf-8",
            )
