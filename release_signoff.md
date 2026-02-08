# Release Signoff

- Decision: **NO-GO**
- Generated at (UTC): 2026-02-07 21:21:33 UTC

## Release Manifest Snapshot
- release_name: paper_go_live_2026-02-07
- git_commit: 072df35
- config_hash: 1332dfca439eec8ccb5dbcd864dc4dd43a12072d07c7ebc9315052eac6b4deed
- data_hash: a8ef06459e7a234169b78be153cb8b6717f89efb3a582cf45f9058b8c85f4b3d
- model_version: baseline_momentum_quantile

## Threshold Results

| scope   | market    | metric                  |        value | op   |   threshold | result   | note   |
|:--------|:----------|:------------------------|-------------:|:-----|------------:|:---------|:-------|
| global  | all       | data_pass_rate          |   1          | >=   |        0.98 | PASS     |        |
| global  | all       | drift_red_count         |  47          | <=   |        0    | FAIL     |        |
| global  | all       | execution_fail_rate     |   0          | <=   |        0.01 | PASS     |        |
| global  | all       | calendar_alignment_pass |   1          | >=   |        1    | PASS     |        |
| market  | crypto    | sharpe                  |  -0.361956   | >=   |        0.8  | FAIL     |        |
| market  | crypto    | max_drawdown            |  -0.0219699  | >=   |       -0.15 | PASS     |        |
| market  | crypto    | profit_factor           |   1.1389     | >=   |        1.05 | PASS     |        |
| market  | crypto    | min_trades_in_window    | 698          | >=   |      120    | PASS     |        |
| market  | crypto    | sharpe_std              |   2.29863    | <=   |        0.35 | FAIL     |        |
| market  | crypto    | total_return_std        |   0.0366741  | <=   |        0.08 | PASS     |        |
| market  | crypto    | max_drawdown_std        |   0.0121268  | <=   |        0.05 | PASS     |        |
| market  | cn_equity | win_rate                |   0.399956   | >=   |        0.5  | FAIL     |        |
| market  | cn_equity | max_drawdown            |  -0.00529038 | >=   |       -0.12 | PASS     |        |
| market  | cn_equity | profit_factor           |   1.70039    | >=   |        1    | PASS     |        |
| market  | cn_equity | min_trades_in_window    | 340          | >=   |       60    | PASS     |        |
| market  | cn_equity | sharpe_std              |   1.88569    | <=   |        0.35 | FAIL     |        |
| market  | cn_equity | total_return_std        |   0.010437   | <=   |        0.08 | PASS     |        |
| market  | cn_equity | max_drawdown_std        |   0.00439944 | <=   |        0.05 | PASS     |        |
| market  | us_equity | sharpe                  |  -1.06172    | >=   |        0.8  | FAIL     |        |
| market  | us_equity | max_drawdown            |  -0.0101042  | >=   |       -0.15 | PASS     |        |
| market  | us_equity | profit_factor           |   0.941163   | >=   |        1.1  | FAIL     |        |
| market  | us_equity | min_trades_in_window    | 830          | >=   |       80    | PASS     |        |
| market  | us_equity | sharpe_std              |   2.05457    | <=   |        0.35 | FAIL     |        |
| market  | us_equity | total_return_std        |   0.0107934  | <=   |        0.08 | PASS     |        |
| market  | us_equity | max_drawdown_std        |   0.00431437 | <=   |        0.05 | PASS     |        |