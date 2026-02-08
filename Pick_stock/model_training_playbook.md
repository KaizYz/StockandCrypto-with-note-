# 模型训练与正确性保障手册（Multi-Market）

> 目标：不是“每次都猜对”，而是“长期在真实成本与真实约束下，稳定获得正的风险调整收益”。

## 0. 适用范围与原则

- 适用市场：Crypto / A股 / 美股。
- 底层时间：统一使用 `UTC` 存储。
- 展示时区：
- Crypto / A股：`Asia/Shanghai`
- 美股：`America/New_York`（含 DST）
- 硬原则：
- 先防错，再提收益。
- 先可复现，再追复杂模型。
- 不通过门槛，禁止上线。

---

## 1. 成功标准（先定义“正确”）

不要把“正确”定义为单次命中，而是以下四条都成立：

- 统计优势：`edge_after_cost > 0`（扣费后仍有优势）。
- 概率可信：Brier / ECE 可接受，且近期不恶化。
- 区间可信：`q10~q90` 覆盖率接近目标（如 80%±5%）。
- 交易可活：回撤、胜率、PF、Sharpe 在风险预算内。

---

## 2. 标准训练流水线（固定顺序，不可跳步）

> 下列脚本路径以当前项目规划为准，若尚未实现请标记为“待实现”。

### 2.1 建议命令

```bash
python -m src.ingestion.update_data --config configs/config.yaml
python -m src.preprocessing.validate_data --config configs/config.yaml
python -m src.features.build_features --config configs/config.yaml
python -m src.labels.build_labels --config configs/config.yaml
python -m src.split.build_folds --config configs/config.yaml
python -m src.models.train --config configs/config.yaml
python -m src.models.calibrate --config configs/config.yaml
python -m src.evaluation.walk_forward --config configs/config.yaml
python -m src.evaluation.backtest_multi_market --config configs/config.yaml
python -m src.reporting.export_report --config configs/config.yaml
```

### 2.2 每次训练必须产出的 Artifacts

- `config_snapshot.yaml`
- `data_integrity_checks.json`
- `universe_snapshot.json`
- `folds_manifest.csv`
- `metrics_summary.csv`
- `metrics_by_fold.csv`
- `compare_baselines.csv`
- `calibration_report.json`
- `interval_report.json`
- `threshold_report.csv`
- `model_card.md`
- `git_commit.txt`
- `hashes.json`（`config_hash/data_hash/code_hash`）

---

## 3. 数据层门槛（Data Gate）

不满足任一条：`fail-fast`，训练直接停止。

### 3.1 数据质量硬校验

- 缺失率 <= `max_missing_rate`（示例：5%）。
- 历史长度 >= `min_history_bars`（日线建议至少 3~5 年）。
- 时间戳严格单调，无重复/倒序/未来时间。
- OHLC 合法：`low <= open/close <= high`，`volume >= 0`。
- 交易日历对齐（尤其 A股午休、美股 DST）。

### 3.2 市场约束入模

- Crypto：
- 区分 `spot/perp`。
- `spot` 默认不做空。
- 费用拆分 `maker/taker`，可选资金费率。
- A股：
- `long-only`，处理涨跌停、停牌、复牌、午休。
- 美股：
- 处理盘前盘后策略边界。
- 卖空能力受券源约束（无券时降级为 `WAIT`）。

### 3.3 Universe 版本化（防幸存者偏差）

- 每次训练固定当期标的池并留档：`universe_snapshot.json`。
- 不允许“用今天成分股回测过去”。

---

## 4. 标签与特征（No Leakage）

- 标签仅来自未来窗口（如 `h=4h/1d/7d`）。
- 特征只能使用 `t` 及以前数据。
- 标准化必须 `fit(train)` 后 `transform(test)`。
- 建议加泄漏检测：抽样核验特征索引是否越界。

---

## 5. 切分协议（Purged Walk-Forward）

- 严禁随机切分时序数据。
- 使用 `purged walk-forward`：
- `gap >= max_horizon`
- 训练集严格早于测试集
- purge 掉标签窗口重叠样本
- 审计文件必须包含：
- `train_start/train_end/test_start/test_end/purge_range`

---

## 6. 固定冻结测试集（Frozen Holdout，新增）

> 用于最终验收，避免“调参调到测试集”。

- 划分三层：
- `train`：训练参数
- `validation`：调参选择
- `frozen_holdout`：只在候选模型最终确定后评一次
- 规则：
- Holdout 期间禁止任何调参回流。
- 若 Holdout 失败，回到 train/validation 重做，不得“补丁式”解释。
- 输出：
- `holdout_report.json`（含方向、区间、交易、成本后指标）

---

## 7. 超参数搜索协议（HPO，新增）

- 搜索范围必须预先写入配置，禁止临时加范围“救结果”。
- 建议协议：
- 第一层：粗搜（随机/贝叶斯，较宽范围）
- 第二层：精搜（围绕前 10% 方案）
- 评分函数建议：
- `primary`: cost-adjusted edge / PF / Sharpe（按任务）
- `secondary`: Brier / coverage 偏离 / turnover
- 防过拟合要求：
- 以 `walk-forward` 均值 + 方差联合打分，不只看均值。
- 输出：
- `hpo_trials.csv`、`hpo_best_config.yaml`

---

## 8. 模型任务拆分与融合

- 方向模型：`P(up)/P(down)`。
- 幅度模型：`q10/q50/q90`。
- 启动窗口模型：`W0..W3/no_start`。
- 决策融合：
- 先过方向门槛，再过成本门槛，再过风险门槛。
- 只要任一关键门槛失败，动作降级 `WAIT`。

---

## 9. 校准与纠偏

### 9.1 概率校准

- 方法：`Platt` 或 `Isotonic`。
- 输出：校准前后 Brier/ECE 对比。

### 9.2 区间校准

- 目标：`coverage(q10~q90)` 接近目标（如 80%）。
- 偏差大时可用缩放或 conformal 纠偏。

### 9.3 分位数交叉修复

- 发生 `q10 > q50` 或 `q50 > q90` 时强制排序修复。
- 记录交叉频率；频率高说明模型不稳定。

---

## 10. 成本压力测试矩阵（新增）

> 不能只看“默认手续费”。

每次回测必须跑成本网格：

- 手续费：低 / 中 / 高（示例：`2/5/10 bps`）
- 滑点：低 / 中 / 高（示例：`1/3/8 bps`）
- 延迟：`0/1/2` bar

输出矩阵：

- `cost_stress_matrix.csv`
- 字段示例：`fee_bps, slippage_bps, delay_bar, pnl, sharpe, max_dd, trades`

验收要求：

- 至少在“中成本”场景下保持策略可用。
- 若仅在“低成本理想场景”盈利，不可上线。

---

## 11. 分市场上线阈值（Go/No-Go，新增）

> 阈值示例可按你项目再调，关键是“先写死规则再训练”。

| 市场 | 指标 | 建议阈值（示例） |
|---|---|---|
| Crypto | Brier | <= 0.23 |
| Crypto | Coverage(q10~q90) | 75%~85% |
| Crypto | Cost-adjusted edge | > 0 |
| A股 | 胜率 | >= 50%（long-only口径） |
| A股 | Max Drawdown | >= -12% |
| 美股 | Sharpe | >= 0.8 |
| 美股 | Profit Factor | >= 1.1 |
| 全市场 | 数据完整率 | >= 98% |
| 全市场 | 近7天退化 | 不得显著劣于近30天阈值 |

说明：

- 所有阈值写入配置：`configs/go_live_thresholds.yaml`。
- 未达标自动降级：`NO-GO` 或 `WAIT-ONLY`。

---

## 12. 漂移检测与在线监控（新增）

### 12.1 监控维度

- 数据漂移：PSI / KS（特征分布变化）
- 标签漂移：上涨率、波动率结构变化
- 模型漂移：Brier、coverage、PF、Sharpe 滚动恶化

### 12.2 告警分级

- 绿：正常
- 黄：轻微漂移（仅告警，不停机）
- 橙：中度漂移（降仓位/提高触发阈值）
- 红：严重漂移（禁开新单，仅平仓或 WAIT）

输出：

- `drift_monitor_daily.csv`
- `drift_alerts.log`

---

## 13. 模型退役规则（新增）

满足任一条触发退役/重训：

- 连续 `N` 天 cost-adjusted edge <= 0。
- 近30天 Brier 明显劣化且校准无法修复。
- 区间覆盖率长期偏离目标（如连续两周低于下限）。
- 回测与线上偏差超阈值（执行一致性失真）。
- 数据质量长期不达标（源头问题未修复）。

退役动作：

- 将模型状态置为 `retired`。
- 自动切换到 `baseline` 或 `WAIT`。
- 生成 `retirement_report.md`。

---

## 14. 实验注册模板（Experiment Registry，新增）

每次实验必须登记，建议字段：

| 字段 | 示例 |
|---|---|
| exp_id | `exp_2026_02_07_001` |
| market | `crypto/us_equity/cn_equity` |
| universe_version | `sp500_2026w06` |
| data_version | `dv_20260207` |
| feature_set_version | `feat_v3` |
| label_schema | `ret_h4_quantile_v2` |
| split_schema | `purged_wf_gap4_v1` |
| model_family | `lgbm_quantile` |
| hpo_space_version | `hpo_v2` |
| best_params | `...json...` |
| commit_hash | `abc1234` |
| result_summary | `sharpe=1.02,pf=1.18` |
| decision | `promote/reject` |
| owner | `youka` |

建议文件：

- `experiments/registry.csv`
- `experiments/<exp_id>/notes.md`

---

## 15. 上线签署与回滚 SOP（新增）

### 15.1 上线签署（Signoff）

至少包含：

- 数据负责人签署（数据完整、时区、日历）
- 研究负责人签署（指标达标、可解释）
- 风控签署（回撤、风险预算、降级逻辑）

文件：

- `release_signoff.md`

### 15.2 回滚策略（Rollback）

- 触发条件：红色告警 / 实盘偏差超阈值 / 数据源异常。
- 回滚步骤：
- 切换上一个稳定模型版本目录
- 清理当前会话缓存
- 强制进入 `WAIT` 直到监控恢复
- 记录：
- `rollback_event.log`（时间、原因、操作者、恢复结果）

---

## 16. 决策输出统一协议（DecisionPacket）

训练、回测、模拟执行、页面展示都读取同一结构，避免“多口径”：

- `symbol`
- `asof_time_utc`
- `market`
- `current_price`
- `p_up / p_down`
- `q10 / q50 / q90`
- `edge_score / edge_risk`
- `final_action`（long/short/wait）
- `entry / stop_loss / take_profit / rr`
- `confidence / risk_level`
- `reason_codes`（例如 `edge_positive`, `low_confidence`, `short_disallowed`）

---

## 17. 最小落地顺序（你现在最该先做）

1. 固化 Go/No-Go 阈值到配置与代码（先挡住错误上线）。
2. 落地 Frozen Holdout + HPO 协议（防过拟合）。
3. 落地成本压力矩阵（防“纸上盈利”）。
4. 落地漂移监控与自动降级（防上线失控）。
5. 落地 Experiment Registry + Signoff/Rollback（防不可追溯）。

---

## 18. 一句话版本（可放 README）

模型训练的核心不是“预测一次对”，而是：
在真实市场约束下，可复现、可审计、可回滚地持续输出正的风险调整收益。
