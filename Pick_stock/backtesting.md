# 量化模型在市面/机构里如何回测“胜率、盈亏比、收益曲线”（完整版指南 · 可直接用于你的项目）

> 目标：建立一套**可复现**、**可比较**的回测框架，用来评估模型产生的买卖信号在真实约束（手续费/滑点/延迟/交易制度）下的表现，并输出行业通用指标：胜率、盈亏比、收益率、回撤、夏普、基准对比等。  
> 适用市场：Crypto / A股 / 美股（通过 Market Adapter 处理制度差异）。  
> 注意：以下是研究与工程实现，不构成投资建议。

---

## 1. 行业回测的核心思路（你需要理解的“标准答案”）
市面上多数量化团队不会只看“预测准确率”，而是看：
- **信号是否能转换为可交易利润（Net PnL）**
- **在成本与约束后是否仍有边际优势（Edge）**
- **风险/收益是否合理（回撤、波动、暴露）**
- **稳定性是否足够（不同时间段、不同市场状态）**

因此，回测一般分两层：
1) **信号层评估（Model Metrics）**：AUC/F1、MAE、Coverage...
2) **交易层评估（Trading Metrics）**：胜率、盈亏比、收益率、回撤、夏普...

你的项目要做的是第 2 层，并且必须能对比 Baseline。

---

## 2. 回测输入输出（你需要哪些数据）

### 2.1 输入（最小集合）
- 市场数据：OHLCV（或 microbars/ticks）
- 预测输出：
  - `p_up`（方向概率）
  - `q10/q50/q90`（幅度区间）
  - `volatility_score`
  - `confidence_score`
- 交易规则（策略/Policy）参数：
  - 信号阈值（如 0.55/0.45）
  - 仓位规则（固定仓位/按风险缩放）
  - 交易成本（fee/slippage）
  - 延迟（t 发信号，t+1 执行）

### 2.2 输出（回测结果）
- 交易记录（trade log）：每笔开仓/平仓的时间、价格、方向、成本、盈亏
- 资金曲线（equity curve）：每个时间点的账户净值
- 绩效指标表（metrics）：胜率、盈亏比、收益、回撤、夏普等
- 对比报告：与基准策略（buy&hold、MA交叉、随机信号等）对比

---

## 3. 市场制度差异（Crypto / A股 / 美股怎么统一）
你需要 Market Adapter 层，回测逻辑尽量统一。

### 3.1 Crypto（24/7）
- 可以 long/short（perp），spot 常为 long/flat
- 需要模拟：taker/maker fee（可先统一一个 fee）

### 3.2 A股（制度限制大）
- 默认 long/flat（做空/融券一般不做 MVP）
- 有午休、涨跌停、停牌
- 必须使用交易日历，不能生成虚假 bars

### 3.3 美股
- 可以 long/short（若假设允许卖空）
- 有 DST（建议存 UTC、展示美东）
- 日线需要考虑复权（split/dividend）

---

## 4. 策略设计：从预测到买卖动作（行业常见）

### 4.1 最常见阈值信号（MVP 推荐）
- `Long` if `p_up > 0.55`
- `Short` if `p_up < 0.45`（A股可关闭）
- else `Flat`

### 4.2 加入“期望收益 - 成本”（更像机构）
只有当“预测优势足够覆盖成本”才交易：
- 预期收益：`edge = q50_change_pct`
- 成本：`cost = fee_bps + slippage_bps`
- 交易条件：
  - `Long` if `p_up > p_long AND edge > cost`
  - `Short` if `p_up < p_short AND (-edge) > cost`

### 4.3 仓位大小（Position Sizing）
机构常用风控缩放：
- **按置信度**：`position ∝ |p_up - 0.5|`
- **按不确定性**：区间宽 `q90-q10` 越大 → 仓位越小
- **按波动目标**：`position ∝ 1 / realized_vol`

MVP 可先用：
- 固定仓位：`position = 1.0`
- 或简单：`position = clip(2*|p_up-0.5|, 0, 1)`

---

## 5. 回测执行：订单如何成交（行业最容易出问题的点）

### 5.1 价格使用（bar级回测最常见）
- 信号在 `t` 产生
- **执行在 `t+1` 的 open**（避免偷看未来 close）
- 或执行在 `t+1` 的 VWAP（若可用，需注明 VWAP 计算区间，例如 `t+1` bar 的前 N 分钟）

说明（细化执行语义）：
- 若下一个可交易 bar（次一开盘）受停牌/涨跌停影响，使用 `next_tradable_bar` 的 open
- 若使用 limit/IOC 等订单类型，需在回测里明确 fill probability 与最大等待时间（例：IOC 尝试当期 partial fill，剩余撤单）

### 5.2 成本模型（必须）
最简成本：
- `fee_bps`：手续费（双边：开仓+平仓都扣）
- `slippage_bps`：基线滑点（模拟成交劣化）

基于流动性/规模的增强模型：
- 额外市场冲击（Impact）示例：
  - `impact_bps = lambda * (order_size / ADV)^beta`（常见取 beta≈0.5，lambda 可调）
  - 总滑点 = `slippage_bps + impact_bps`
- 部分成交：若当期可用量 < order_size，则按比例成交并记录 `partial_fill=true`

成本计算示例：
- `cost_open = exec_price * (fee_bps + total_slippage_bps)`
- `cost_close = exec_price * (fee_bps + total_slippage_bps)`

### 5.3 延迟（必须）
- `delay = 1 bar`（MVP）
- Tick 级回测可模拟固定秒数延迟或基于 order book 的消息延迟分布（增强项）

> 行业内通常要求：**信号→执行必须有延迟**，否则回测不可信。

### 5.4 可复现性（Reproducibility，高优先）
- 在回测与训练里必须固定随机种子并记录：`numpy.random.seed`, `random.seed`, ML 库（LightGBM/TF/PyTorch）的 seed。将 seed 写入 `config_snapshot.yaml` 与 `metrics.csv` 的 header。
- 保存运行环境快照：`requirements.txt`（或 `pip freeze`）、Python 版本、操作系统、`git commit hash`（若有代码仓库）。
- 推荐在训练/预测脚本开头加入：

```python
import os, random, numpy as np
seed = cfg.get('seed', 42)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
# LightGBM: pass seed parameter to training/predict functions
```

- 将 `seed` 与 `config_snapshot.yaml` 一起存储到 `backtest/config_snapshot.yaml`，并把 `model_version` 或 `git hash` 写入 `backtest/metrics.json`。

### 5.5 部分成交与市场冲击（Partial fills & Market Impact，高优先）
- 理由：大订单在实际市场上会造成价格移动或只被部分撮合，简单把全部按 open 成交会显著高估可实现收益。

两种可实现的简化策略：
1) 固定滑点 + 容量上限（MVP）
   - 先计算当期可用量指标（例如 `available_volume = ADV * liquid_ratio`），若 `order_size > available_volume`，按 `available_volume` 成交，剩余撤单。
2) 影响模型（更真实）
   - 先计算 `impact_bps = lambda * (order_size / ADV)^beta`，再把 impact 加入滑点并按此价格成交；若 order_size 超过深度档位，再根据深度分层计算分段成交价格。

回测实现建议：
- 每笔成交记录 `exec_price`, `exec_size`, `partial_fill`（bool），并在 trade log 中保留 `market_impact_bps` 字段。
- 提供可配置参数：`adv_window_days`, `lambda`, `beta`, `liquid_ratio`，并在 sensitivity 分析里测试不同参数对最终 PnL 的影响。


---

## 6. 交易记录（Trade Log）定义（你必须输出这个）
每笔交易至少包含：
- `trade_id`
- `symbol`
- `side`（long/short）
- `entry_time` / `entry_price`
- `exit_time` / `exit_price`
- `size`（仓位）
- `gross_pnl`（不含成本）
- `cost`（手续费+滑点）
- `net_pnl = gross_pnl - cost`
- `return_pct`（单笔收益率）
- `holding_period`（持仓时长）
- `reason`（触发原因：阈值/止损/反向信号等）

---

## 7. 行业通用指标：胜率、盈亏比、收益、风险（你要实现的核心）

### 7.1 胜率（Win Rate）
- 定义：盈利交易占比
- `win_rate = (# net_pnl > 0) / total_trades`

### 7.2 盈亏比（Profit/Loss Ratio, Avg Win / Avg Loss）
常见两种口径：

**(1) 平均盈亏比（常用）**
- `avg_win = mean(net_pnl | net_pnl > 0)`
- `avg_loss = abs(mean(net_pnl | net_pnl < 0))`
- `avg_win_loss_ratio = avg_win / avg_loss`

**(2) 总盈亏比（Profit Factor，更像机构报告）**
- `gross_profit = sum(net_pnl | net_pnl > 0)`
- `gross_loss = abs(sum(net_pnl | net_pnl < 0))`
- `profit_factor = gross_profit / gross_loss`

> 盈亏比不是越高越好，要结合胜率一起看：  
> 高胜率低盈亏比 vs 低胜率高盈亏比 都可能赚钱。

### 7.3 期望值（Expectancy，行业常用“单笔期望”）
- `expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss`
若 > 0，说明长期有正期望（在该回测假设下）。

### 7.4 总收益与年化收益（Return / CAGR）
- 总收益：
  - `total_return = final_equity / initial_equity - 1`
- 年化（按交易日/小时换算）：
  - `CAGR = (final_equity / initial_equity)^(1/years) - 1`

### 7.5 最大回撤（Max Drawdown，必须）
- 资金曲线 `equity[t]`
- 峰值 `peak[t] = max(equity[0..t])`
- 回撤 `dd[t] = 1 - equity[t] / peak[t]`
- `max_drawdown = max(dd)`

### 7.6 夏普比率（Sharpe，常用）
- 取周期收益 `r_t`
- `Sharpe = mean(r_t - r_f) / std(r_t)`（MVP 可忽略 r_f）
- 年化：乘以 `sqrt(periods_per_year)`

### 7.7 波动率与下行风险（可选但加分）
- 年化波动：`vol = std(r_t) * sqrt(periods_per_year)`
- Sortino：用下行 std 替代 std

### 7.8 曝光与交易频率（很关键）
- `exposure = 时间在仓位中 / 总时间`
- `trades_per_day` 或 `trades_per_month`
- 平均持仓时间（holding period）

---

## 8. 行业内“必须做”的对比（否则指标没有意义）

### 8.1 基准策略（Baselines）
至少三类：
1) **Buy & Hold**（现货可用）
2) **MA crossover**（简单均线交叉）
3) **Naive threshold**（例如随机信号或固定方向）

### 8.2 同一假设下对比
- 同样的 fee/slippage/delay
- 同样的时间切分（walk-forward 的测试段）
- 同样的仓位上限与制度限制（A股不做空）

---

## 9. Walk-forward 回测（行业标准）
不要只回测全历史一次；要按时间分段滚动验证：

- Fold 1：Train(过去) → Test(未来一段)
- Fold 2：窗口向前滚
- 记录每个 fold 的交易指标（胜率、PF、回撤、Sharpe）
- 最终输出：
  - 平均表现
  - 方差（稳定性）
  - 最差 fold（风险）

> 机构特别看“稳定性”：平均赚但某段爆炸的策略通常会被否掉。

---

## 10. 交易规则增强（可选但常见）

### 10.1 止损/止盈（Risk Controls）
- 固定止损：例如 -1%（短周期）
- 波动止损：`k * ATR`
- 时间止损：持仓超过 H bars 自动平仓

### 10.2 冷却期（Cooldown）
- 频繁反转会被成本吃掉
- 增加 `cooldown_bars` 限制连续交易

### 10.3 过滤条件（Regime Filter）
例如：
- 波动极高时减少交易
- 低流动性时段不交易（尤其逐笔策略）

---

## 11. 回测产物（你项目里应该输出哪些文件）
建议固定输出目录结构：

- `backtest/trades.csv`（交易记录）
- `backtest/equity.csv`（资金曲线）
- `backtest/metrics.json`（指标汇总）
- `backtest/metrics_by_fold.csv`（逐 fold）
- `backtest/compare_baselines.csv`（对比表）
- `backtest/config_snapshot.yaml`（回测配置快照）

---

## 12. 推荐“最小可行回测（MVP）”配置（你可以直接实现）
### 12.1 默认参数
- 执行价：`next_open`
- 延迟：`1 bar`
- 成本：
  - `fee_bps = 4`
  - `slippage_bps = 2`
  - 双边成本（开仓+平仓）
- 信号阈值：
  - `p_long = 0.55`
  - `p_short = 0.45`
- 仓位：固定 `1.0`
- A股：`allow_short = false`
- Crypto perp / 美股：`allow_short = true`（可配置）

### 12.2 必须输出指标
- win_rate
- avg_win_loss_ratio
- profit_factor
- expectancy
- total_return
- max_drawdown
- sharpe
- exposure
- trades_count

---

## 13. 常见回测陷阱（答辩/面试经常问）
- **偷看未来**：用 close 生成信号又用同一根 close 成交（必须用 next_open/延迟）
- **忽略成本**：导致“纸面盈利”
- **忽略制度差异**：A股不能随便做空、午休/停牌
- **过拟合参数**：阈值/止损调到完美但 out-of-sample 崩
- **只看平均**：不看最差区间与回撤

---

## 14. 你项目的落地建议（与你现有输出无缝对接）
你已有：
- `p_up`
- `q10/q50/q90`
- `volatility_score`
- `confidence_score`

直接用它做交易层：
- 动作：按 `p_up` 阈值决定 long/short/flat
- 过滤：`q50` 必须覆盖成本
- 仓位：按 `confidence_score` 或 `1/(q90-q10)` 缩放
- 风险：`volatility_score` 过高降仓或减少交易

---

## 15. 最终你要展示的“行业式报告”（建议 Dashboard 里做）
- 资金曲线（策略 vs baselines）
- 指标表（胜率、PF、回撤、Sharpe）
- 交易分布图（单笔收益直方图、持仓时间分布）
- 分段表现（按 fold 或按年份/季度）
- 结论：
  - 是否显著优于基准
  - 成本敏感性（成本加倍是否还能赚钱）
  - 哪些市场状态最有效（趋势/震荡/高波动）

---

## 16. 样本偏差控制（建议补为“硬规则”）
这部分直接决定回测是否可信，建议写成强制检查项：

- **幸存者偏差（Survivorship Bias）**：股票池必须按历史成分回放，不能只用当前还活着的股票。
- **前视偏差（Look-ahead Bias）**：财报/因子只能在“可获取时间”后生效，不能提前使用。
- **复权一致性**：训练、预测、回测必须使用同一复权口径（前复权或后复权），并记录到配置快照。
- **停牌与不可交易处理**：不可交易 bar 不能按理想价格成交，必须顺延到 `next_tradable_bar`。

建议新增输出：
- `backtest/data_integrity_checks.json`
- `backtest/survivorship_coverage.csv`

---

## 17. 空头真实成本（Short Side Realism）
如果允许做空，建议单独建成本项，否则会高估策略：

- **美股**：`borrow_fee_bps`（融券费）+ locate 失败概率。
- **Crypto perp**：`funding_rate`（多空资金费）按持仓时间累计。
- **Crypto spot 借币**：`borrow_interest_bps`（借币利率）。

净收益建议改为：
- `net_pnl = gross_pnl - fee - slippage - impact - funding - borrow_interest`

---

## 18. 组合级回测（不要只做单标的）
你现在多市场/多标的是组合问题，建议加组合约束：

- `max_positions`：最大同时持仓数
- `max_weight_per_symbol`：单标的权重上限
- `max_weight_per_market`：单市场暴露上限（Crypto/A股/美股）
- `gross_exposure_cap` 与 `net_exposure_cap`

组合级输出建议：
- `portfolio_equity.csv`
- `portfolio_exposure.csv`（按市场/行业/方向）
- `portfolio_turnover.csv`

---

## 19. 统计显著性（判断“不是运气”）
建议在报告中增加显著性检查，不只看点估计：

- **Bootstrap 置信区间**：对 Sharpe、Profit Factor、Expectancy 给 95% CI。
- **Deflated Sharpe Ratio（DSR）**：校正多次试参造成的虚高夏普。
- **PBO（Probability of Backtest Overfitting）**：估计过拟合概率。

最小要求：
- 指标表除了 `mean`，还要有 `ci_low` / `ci_high`。

---

## 20. 参数稳健性与敏感性分析（必须做）
避免“单点参数神话”，建议至少做二维热力图：

- 方向阈值：`p_long` / `p_short`
- 成本参数：`fee_bps` / `slippage_bps`
- 风险参数：`max_position` / `cooldown_bars`

输出建议：
- `backtest/sensitivity_grid.csv`
- `backtest/sensitivity_heatmap.png`（或在 Dashboard 中可视化）

验收标准：
- 在合理参数邻域内，策略仍保持正期望（而不是只有一个点赚钱）。

---

## 21. 换手与容量（Capacity）评估
若未来需要更大资金规模，必须看容量：

- **换手率**：`turnover_daily` / `turnover_monthly`
- **容量上限**：给定 ADV 约束下可承载资金规模（AUM）
- **冲击成本曲线**：AUM 放大后收益衰减曲线

建议加一个图：
- `AUM vs Net Return / Sharpe / Max Drawdown`

---

## 22. 生产守护规则（从研究到实盘前的桥）
即使目前只做研究，也建议预先定义“失效保护”：

- 连续 `N` 天回撤超过阈值 → 自动降仓
- 策略 30 日滚动 Sharpe < 阈值 → 进入观察/停机
- 数据质量告警（缺失率、延迟、价格跳变）→ 当天禁开新仓

建议落地为配置：
- `kill_switch.max_dd_rolling`
- `kill_switch.min_sharpe_rolling`
- `kill_switch.max_data_missing_rate`

---

## 23. 统一结果 Schema（便于审计与 Dashboard）
建议统一两层结果：`trade-level` + `run-level`。

### 23.1 Trade-level（每笔）
- `run_id, trade_id, symbol, market`
- `signal_time, exec_time, side, size`
- `entry_price, exit_price, gross_pnl, net_pnl`
- `fee, slippage, impact, funding, borrow_interest`
- `holding_period, reason, partial_fill`

### 23.2 Run-level（每次回测）
- `run_id, model_version, config_hash, seed`
- `period_start, period_end, universe_name`
- `total_return, sharpe, max_drawdown, win_rate, profit_factor`
- `baseline_name, baseline_delta`
- `git_commit, python_version, requirements_hash`

这样后续你做 Dashboard、周报、审计追踪会非常顺畅。

---
