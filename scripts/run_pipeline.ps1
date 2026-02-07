param(
  [string]$Config = "configs/config.yaml"
)

$ErrorActionPreference = "Stop"

python -m src.ingestion.update_data --config $Config
python -m src.features.build_features --config $Config
python -m src.labels.build_labels --config $Config
python -m src.models.train --config $Config
python -m src.models.predict --config $Config
python -m src.markets.snapshot --config $Config
python -m src.markets.tracking --config $Config
python -m src.markets.session_forecast --config $Config
python -m src.models.generate_policy_signals --config $Config
python -m src.evaluation.walk_forward --config $Config
python -m src.evaluation.backtest --config $Config
python -m src.evaluation.backtest_multi_market --config $Config

Write-Host "Pipeline complete." -ForegroundColor Green
