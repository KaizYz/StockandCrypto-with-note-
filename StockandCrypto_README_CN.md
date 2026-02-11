# StockandCrypto

一个端到端的多市场预测 MVP（与 `Project_plan.md` 对齐）。

## 概览

- **BTC 模型输出**
  - **方向：** `P(up)` / `P(down)`
  - **启动窗口：** `W0–W3`
  - **区间预测：** `q10 / q50 / q90`
- **评估**
  - **Purged Walk-Forward** 评估（`gap = max_horizon`）
- **多市场实时快照卡片**（加密 + 中国 A 股 + 美股）
  - 当前价格
  - 预测价格
  - 预测幅度（magnitude）
  - 预期日期
  - 支持的标的池 / 范围
    - **加密：** BTC / ETH / SOL + 市值前 100（剔除稳定币，如 USDT/USDC）
    - **中国 A 股：** 上证指数成分股、沪深 300 成分股
    - **美股：** 道琼斯 30、纳斯达克 100、标普 500 成分股

## 项目结构

```text
CryptoForecast/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── labels/
│   ├── split/
│   ├── models/
│   ├── markets/
│   ├── evaluation/
│   ├── monitoring/
│   ├── reporting/
│   └── utils/
└── dashboard/
    └── app.py
```

## 环境配置（Setup）

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 流水线命令（Pipeline Commands）

```bash
python -m src.ingestion.update_data --config configs/config.yaml
python -m src.preprocessing.validate_data --config configs/config.yaml
python -m src.features.build_features --config configs/config.yaml
python -m src.labels.build_labels --config configs/config.yaml
python -m src.split.build_folds --config configs/config.yaml
python -m src.models.hpo --config configs/config.yaml
python -m src.models.train --config configs/config.yaml
python -m src.models.calibrate --config configs/config.yaml
python -m src.models.predict --config configs/config.yaml
python -m src.markets.snapshot --config configs/config.yaml
python -m src.markets.tracking --config configs/config.yaml
python -m src.markets.session_forecast --config configs/config.yaml
python -m src.models.generate_policy_signals --config configs/config.yaml
python -m src.evaluation.walk_forward --config configs/config.yaml
python -m src.evaluation.backtest --config configs/config.yaml
python -m src.evaluation.backtest_multi_market --config configs/config.yaml
python -m src.monitoring.drift --config configs/config.yaml
python -m src.monitoring.retirement --config configs/config.yaml
python -m src.reporting.export_report --config configs/config.yaml
python -m streamlit run dashboard/app.py
```

## 一键运行（One-shot Run）

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1
```

## 备注（Notes）

- 数据以 UTC 存储（`timestamp_utc`），并按各市场本地时区展示（`timestamp_market`）。
- 若 Binance API 拉取失败，`ingestion` 会自动回退到合成数据，保证流水线仍可运行。
- 模型产物按版本存放于 `data/models/<timestamp>_<branch>/`。
- 可复现性：`seed=42`，每个模型版本都会保存一份配置快照（config snapshot）。
- Dashboard 中的加密“当前价格”使用实时 ticker（不是上一个收盘价）。
- 中国 A 股 / 美股“当前价格”优先使用东方财富实时行情；若不可用则回退到 Yahoo/Stooq 的最新可用价格。
- 快照输出包含 6 个量化因子：
  - **风险类：** `size_factor`, `value_factor`, `growth_factor`
  - **行为类：** `momentum_factor`, `reversal_factor`, `low_vol_factor`
- 风险因子列还包含来源元数据（基本面优先，缺失时使用代理指标回退）。

## 跟踪流程（Tracking Workflow）

运行标的池选择 + 打分 + 跟踪输出：

```bash
python -m src.markets.tracking --config configs/config.yaml
```

生成文件保存于 `data/processed/tracking/`：

- `universe_crypto.json`, `universe_ashares.json`, `universe_us.json`
- `coverage_matrix.csv`
- `ranked_universe.csv`
- `tracking_actions.csv`
- `data_quality_report.md`

## 加密交易时段预测（Crypto Session Forecast）

构建加密市场时段输出（北京时间，24 小时）：

```bash
python -m src.markets.session_forecast --config configs/config.yaml
```

生成文件位于 `data/processed/`：

- `session_forecast_hourly.csv`
- `session_forecast_blocks.csv`
- `session_forecast_daily.csv`

## 策略信号（Policy Signals）

生成策略层输出（`Long/Short/Flat`、仓位大小、期望优势/edge）：

```bash
python -m src.models.generate_policy_signals --config configs/config.yaml
```

生成文件：

- `data/processed/policy_signals_hourly.csv`
- `data/processed/policy_signals_daily.csv`
- `data/processed/tracking/policy_signals_multi_market.csv`
- `data/processed/policy_signals_summary.json`

## 多市场回测（Multi-Market Backtest）

对加密 / 中国 A 股 / 美股运行统一 Walk-Forward 回测：

```bash
python -m src.evaluation.backtest_multi_market --config configs/config.yaml
```

生成文件位于 `data/processed/backtest/`：

- `trades.csv`
- `equity.csv`
- `metrics_by_fold.csv`
- `metrics_summary.csv`
- `compare_baselines.csv`
- `metrics.json`
- `config_snapshot.yaml`
- `data_integrity_checks.json`
- `survivorship_coverage.csv`

Dashboard 包含页面：`Crypto Trading Session Forecast`（加密交易时段预测）。

## 治理输出（Governance Outputs）

流水线会输出与 `Pick_stock/model_training_playbook.md` 对齐的治理产物：

- `data/processed/data_integrity_checks.json`
- `data/processed/folds_manifest.csv`
- `data/processed/hpo_trials.csv`
- `data/processed/hpo_best_config.yaml`
- `data/processed/holdout_report.json`
- `data/processed/threshold_report.csv`
- `data/processed/backtest/cost_stress_matrix.csv`
- `data/processed/drift_monitor_daily.csv`
- `data/processed/model_status.csv`
- `data/processed/go_live_decision.json`
