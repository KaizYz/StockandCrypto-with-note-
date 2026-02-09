# 日内 + 日线训练手册（过去5年版）— 终稿版

> 目标：统一日内预测与日线预测的训练口径，降低“早上看空、盘中突然翻多但无法解释”的割裂感；  
> 用过去 **5 年** 数据训练与验证，覆盖牛熊切换、震荡区间与事件冲击，提高稳定性与可复现性。  
> 适用：Crypto / A股 / 美股（Multi-Market 架构）。  
> 说明：任何模型都不能保证 100% 正确，本手册目标是**更稳定、更可解释、更可上线**。  
> 时区约定：Crypto + 亚盘 = 北京时间（Asia/Shanghai）；美股 = 美东时间（America/New_York）。底层统一 UTC 存储。

---

## 1. 统一训练口径（先把“什么叫日内/日线”写死）

### 1.1 分支定义（主分支写死，避免口径混乱）
- **日内主分支：4H**（与 Session/时段预测页一致，噪声更低，适合执行层）
- **日线分支：1D**
- **1H 为增强分支（可选）**：只用于更细的热力图/时段分析，不作为核心执行信号

> 默认：系统的“可执行信号”以 4H + 1D 为主；1H 只做辅助观察或增强模块。

### 1.2 训练频率 vs 预测频率（必须区分）
- **预测频率（执行信号生成）**
  - 4H：每根 4H K 线收盘生成一次执行信号
  - 1D：每日收盘生成一次执行信号
- **训练频率（模型重训/发布）**
  - 建议：每周或每月重训一次（按资源决定）
  - 发布采用“版本冻结”：未发布前不影响执行层（见第 10 章）

---

## 2. 训练窗口（统一 5 年）与降级规则

### 2.1 训练窗口统一改为过去 5 年
- 训练窗口：`T - 1825d ~ T`（近似 5 年，工程可接受）
- 冻结测试集（最终验收专用）：最近 `3~6` 个月（**只用于最终验收，不参与调参**）

### 2.2 历史不足的降级规则（必须写清）
如果标的历史不足 5 年：
1. 尝试至少 **3 年**（`>= 1095d`）
2. 仍不足：标记 `insufficient_history = True`  
   - **不进入正式上线信号池**
   - 只能作为观察/研究

---

## 3. 数据要求（按市场口径）

### 3.1 Crypto（仅 Binance / Bybit，其他不考虑）
- 交易所：**Binance（spot/perp）**、**Bybit（spot/perp）**
- 建议优先标的：BTC / ETH / SOL + 高流动性币种
- 数据频率：
  - 日线：1D
  - 日内：4H（主） / 1H（可选增强）
- 交易约束：
  - 区分 `spot` / `perp`
  - `spot` 默认不做空；`perp` 才允许做空

### 3.2 A股
- 日线/日内都必须使用交易日历（午休、节假日、停牌）
- 禁止生成“虚假交易日/虚假时段 K 线”
- 默认 **long-only**

### 3.3 美股
- 明确是否仅 RTH（常规盘）还是含盘前盘后（推荐先只做 RTH）
- 训练与执行必须同一口径（都用 RTH 或都用扩展时段）
- 注意夏令时（DST）切换，**底层 UTC + 展示 ET**

---

## 4. 数据质量与异常处理（训练正确性的第一道门）

### 4.1 硬校验门槛（不通过就不训练/不发信号）
- 时间戳严格单调：无重复、无倒序
- 缺失率不超过阈值（如 `max_missing_rate <= 0.05`）
- 历史长度足够（满足 5y/3y 门槛）
- 市场时区展示一致，底层 timestamp 全部 UTC

### 4.2 异常 K 线（插针/跳点）处理规则（推荐保守）
- **不修改 raw OHLC**（原始价格不动）
- 对异常样本做：
  - 标记：`outlier_bar = True`
  - 训练时降权（sample weight）或在特征上 winsorize/clip
- 异常检测建议：
  - `abs(return)` 超过历史 `99.9%` 分位 → outlier
  - high/low 与 close 的比值出现极端跳点 → outlier

---

## 5. 任务定义与标签（统一三任务口径）

> 同一套任务定义适用于 4H 与 1D，区别只在 time unit 与 horizon。

### 5.1 方向任务（Direction Classification）
- 标签：未来 `h` 窗口收益是否 > 0
- 输出：`P(up)` / `P(down)`

### 5.2 幅度任务（Magnitude via Quantile Regression）
- 标签：未来 `h` 收益率（return）
- 输出：`q10 / q50 / q90`（区间预测优先于点预测）

### 5.3 启动窗口任务（可选，但强烈建议作为增强）
- 输出：`W0/W1/W2/W3/no_start`
- 回答：“更可能什么时候开始明显波动/趋势启动”

---

## 6. Horizon 口径（推荐默认值）

### 6.1 日内主分支（4H）
- `bar_size = 4H`
- 推荐 horizons（以 bar 为单位）：
  - `h = 1, 2, 3` → 4h / 8h / 12h
- 默认执行：**4h（h=1）**  
  其他 horizons 用于一致性/强度参考

### 6.2 日线分支（1D）
- 推荐 horizons：
  - `h = 1, 3, 7` → 1d / 3d / 7d
- 可选扩展：30d（如果你希望更偏中长线）

---

## 7. 切分与验证（必须时序化 + 防泄漏）

### 7.1 禁止随机切分
- 不允许 random split
- 不允许 shuffle

### 7.2 必须用 Purged Walk-Forward（含 gap）
- `split = purged_walk_forward`
- 每个 fold：训练集严格早于测试集
- purge 掉标签重叠样本
- `gap_bars` **必须自动计算**（避免写死导致浪费/泄露）

#### gap_bars 自动计算规则（建议）
- `gap_bars = max(horizon_bars)`  
  - 4H：若 max horizon=3 → gap=3  
  - 1D：若 max horizon=7 → gap=7  
- 若你做 high/low 或更复杂标签，可加 buffer：
  - `gap_bars = max_horizon_bars + extra_buffer`

### 7.3 建议折数（可配置）
- 4H（日内）：`5~8` 折（样本多）
- 1D（日线）：`4~6` 折

### 7.4 最小测试窗口（保证指标可信）
- 1D：每折测试集建议 `>= 60` 根（日线约 3 个月交易日）
- 4H：每折测试集建议 `>= 200` 根（约 1~2 个月）

---

## 8. Regime 分层评估（强烈建议，提升稳定性）

> 很多策略“总体还行”，但在高波动/震荡时完全失效。必须分层评估。

### 8.1 Regime 划分（推荐最小版）
可用 rolling volatility / ATR / ADX 之一：
- `trend`：趋势明显（如 ADX 高）
- `range`：震荡（ADX 低）
- `high_vol`：高波动（rolling vol 高）

### 8.2 输出指标必须同时包含
- overall metrics（总体）
- metrics_by_regime（按状态分层）

---

## 9. 配置示例（5年 + 4H主分支 + gap自动 + regime评估）

```yaml
training:
  lookback_days: 1825
  fallback_min_days: 1095
  freeze_test_months: 6
  retrain_schedule: "weekly"   # weekly/monthly
  publish_mode: "frozen_version"  # 发布冻结模型版本

data_quality:
  max_missing_rate: 0.05
  outlier_detection:
    enabled: true
    return_abs_quantile: 0.999
    action: "downweight"   # downweight/clip_features/drop

branches:
  intraday_4h:
    bar_size: "4h"
    horizon_bars: [1, 2, 3]     # 4h/8h/12h
    min_test_bars_per_fold: 200
  daily_1d:
    bar_size: "1d"
    horizon_bars: [1, 3, 7]
    min_test_bars_per_fold: 60

split:
  method: "purged_walk_forward"
  gap_bars: "auto"              # auto = max(horizon_bars)
  n_splits_intraday: 6
  n_splits_daily: 5

regime:
  enabled: true
  method: "rolling_vol"         # rolling_vol / atr / adx
  windows:
    intraday_4h: 60
    daily_1d: 30

gate:
  min_history_days: 1825
  allow_fallback_3y: true

10. 训练命令（标准流水线）
python -m src.ingestion.update_data --config configs/config.yaml
python -m src.preprocessing.validate_data --config configs/config.yaml
python -m src.features.build_features --config configs/config.yaml
python -m src.labels.build_labels --config configs/config.yaml
python -m src.split.build_folds --config configs/config.yaml
python -m src.models.train --config configs/config.yaml
python -m src.models.calibrate --config configs/config.yaml
python -m src.evaluation.walk_forward --config configs/config.yaml
python -m src.evaluation.backtest_multi_market --config configs/config.yaml

11. 验收门槛（上线前必须达标）
11.1 概率质量（Direction）

AUC / F1 至少优于 baseline

概率校准指标达标：

Brier / ECE（或 Reliability）

11.2 区间质量（Quantile）

q10~q90 覆盖率接近目标（例如 80% ± 5%）

区间宽度不过度膨胀（coverage 与 width 同时看）

11.3 交易质量（含成本）

成本后 edge_after_cost > 0（按策略定义）

PF / Sharpe / MaxDD 在阈值内

样本量足够（交易次数建议 >= 100 才认为指标可信）

11.4 稳定性

walk-forward 各 fold 波动不过大

regime 分层下不出现“某一类状态完全失效”的极端情况

最近 30 天不出现明显退化（漂移/健康监控可接入）

12. 一致性建议（针对“同日翻向”你最关心的问题）

训练口径统一只是基础，真正减少翻向需要执行层机制配合：

信号快照锁定（snapshot lock）

重算触发门槛（只有新 bar 或跨阈值才允许更新）

概率平滑（EWMA）

迟滞阈值（hysteresis）

多周期一致性门控（intraday vs daily 冲突则降级 WAIT）

变更日志（每次翻向必须可解释）

结论：训练更稳 + 执行更稳，才会“看起来不跳、用起来能下单”。

13. 版本冻结与可复现（最终一定要写进代码）

每次训练/发布必须落盘：

model_version

data_version

config_hash

git_commit

metrics_summary.csv

metrics_by_fold.csv

metrics_by_regime.csv（若启用）

data_integrity_checks.json

执行层只使用“已发布版本”，直到下一次发布切换：

避免盘中“偷偷换模型”导致用户感知为无理由跳变

14. 结论

将训练窗口统一为 过去 5 年，能显著提升对不同市场状态的覆盖与鲁棒性；
但“更长窗口”不等于“自动更准”，必须配合：

Purged walk-forward（含 gap）

数据质量硬校验 + 异常处理

Regime 分层评估

成本后回测与稳定性验收

版本冻结与可复现

这样才能把系统从“研究可看”推进到“可执行、可解释、可持续”。

::contentReference[oaicite:0]{index=0}