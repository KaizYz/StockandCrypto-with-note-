# StockandCrypto

End-to-end multi-market forecasting MVP aligned with `Project_plan.md`:
- BTC model outputs: direction (`P(up)/P(down)`), start window (`W0-W3`), interval (`q10/q50/q90`)
- Purged walk-forward evaluation (`gap = max_horizon`)
- Multi-market live snapshot cards (Crypto + A-share + US):
  - å½“å‰ä»·æ ¼
  - é¢„æµ‹ä»·æ ¼
  - é¢„æµ‹å¹…åº¦
  - é¢„æœŸæ—¥æœŸ
  - å¯é€‰è‚¡ç¥¨æ± ä¸Žæ ‡çš„ï¼š
    - Crypto: BTC / ETH / SOL + å¸‚å€¼å‰100ï¼ˆå‰”é™¤ç¨³å®šå¸ï¼Œå¦‚ USDT/USDCï¼‰
    - Aè‚¡: ä¸Šè¯æŒ‡æ•°æˆåˆ†è‚¡ã€æ²ªæ·±300æˆåˆ†è‚¡
    - ç¾Žè‚¡: é“ç¼æ–¯30ã€çº³æ–¯è¾¾å…‹100ã€æ ‡æ™®500 æˆåˆ†è‚¡

## Project Structure

```text
CryptoForecast/
â”œâ”€â”€ configs/config.yaml
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â”œâ”€â”€ processed/
â”‚   â””â”€â”€ models/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ ingestion/
â”‚   â”œâ”€â”€ preprocessing/
â”‚   â”œâ”€â”€ features/
â”‚   â”œâ”€â”€ labels/
â”‚   â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ markets/
â”‚   â”œâ”€â”€ evaluation/
â”‚   â””â”€â”€ utils/
â””â”€â”€ dashboard/app.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline Commands

```bash
python -m src.ingestion.update_data --config configs/config.yaml
python -m src.features.build_features --config configs/config.yaml
python -m src.labels.build_labels --config configs/config.yaml
python -m src.models.train --config configs/config.yaml
python -m src.models.predict --config configs/config.yaml
python -m src.markets.snapshot --config configs/config.yaml
python -m src.markets.tracking --config configs/config.yaml
python -m src.markets.session_forecast --config configs/config.yaml
python -m src.models.generate_policy_signals --config configs/config.yaml
python -m src.evaluation.walk_forward --config configs/config.yaml
python -m src.evaluation.backtest --config configs/config.yaml
python -m src.evaluation.backtest_multi_market --config configs/config.yaml
python -m streamlit run dashboard/app.py
```

Or run all in one shot:

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1
```

## Notes

- Data is stored in UTC (`timestamp_utc`) and displayed with market timezone (`timestamp_market`).
- If Binance API fetch fails, ingestion automatically falls back to synthetic data so the pipeline remains runnable.
- Model artifacts are versioned in `data/models/<timestamp>_<branch>/`.
- Reproducibility: `seed=42`, config snapshot saved with each model version.
- Crypto current price uses live ticker (not last close) in dashboard display.
- Aè‚¡/ç¾Žè‚¡ current price prioritizes Eastmoney live quote; falls back to Yahoo/Stooq latest available price.


- Snapshot output now includes six quant factors:
- risk: size_factor, value_factor, growth_factor
- behavior: momentum_factor, reversal_factor, low_vol_factor
- risk factor columns also include source metadata (fundamental-first with proxy fallback).

## Tracking Workflow

Run universe selection + scoring + tracking outputs:

```bash
python -m src.markets.tracking --config configs/config.yaml
```

Generated files are saved under `data/processed/tracking/`:
- `universe_crypto.json`, `universe_ashares.json`, `universe_us.json`
- `coverage_matrix.csv`
- `ranked_universe.csv`
- `tracking_actions.csv`
- `data_quality_report.md`

## Crypto Session Forecast

Build crypto time-session outputs (Beijing time, 24h):

```bash
python -m src.markets.session_forecast --config configs/config.yaml
```

Generated files under `data/processed/`:
- `session_forecast_hourly.csv`
- `session_forecast_blocks.csv`
- `session_forecast_daily.csv`

## Policy Signals

Generate policy-layer outputs (`Long/Short/Flat`, position sizing, expected edge):

```bash
python -m src.models.generate_policy_signals --config configs/config.yaml
```

Generated files:
- `data/processed/policy_signals_hourly.csv`
- `data/processed/policy_signals_daily.csv`
- `data/processed/tracking/policy_signals_multi_market.csv`
- `data/processed/policy_signals_summary.json`

## Multi-Market Backtest

Run unified walk-forward backtest for Crypto / A-share / US:

```bash
python -m src.evaluation.backtest_multi_market --config configs/config.yaml
```

Generated files under `data/processed/backtest/`:
- `trades.csv`
- `equity.csv`
- `metrics_by_fold.csv`
- `metrics_summary.csv`
- `compare_baselines.csv`
- `metrics.json`
- `config_snapshot.yaml`
- `data_integrity_checks.json`
- `survivorship_coverage.csv`

Dashboard now includes page: `交易时间段预测（Crypto）`.
