# 跨市场“学K线→决定买卖”训练方案（Crypto + A股 + 美股）完整版（Markdown）

> 目标：用**同一套框架**在 Crypto / A股 / 美股上训练模型，输入过去 K 线与特征，输出可用于**买/卖/空仓**（以及仓位大小）的决策信号。  
> 原则：**不做实盘交易**；所有“买卖”仅用于研究、paper trading / 回测评估。  
> 时区规则：Crypto & A股 = 北京时间（Asia/Shanghai）；美股 = 美东时间（America/New_York）。工程上建议存 UTC，展示按市场时区。

---

## 1）行业/机构常见架构（不是“模型直接下单”，而是分层）
市面上专业团队通常拆成四层（跨市场复用最稳）：

1. **数据层 Data**
   - 拉取 OHLCV（+ 可选衍生数据/事件数据）
   - 清洗、对齐、时区转换、交易日历处理
2. **Alpha 层（预测/信号）**
   - 预测未来收益方向/幅度/区间/波动
   - 输出概率与置信区间（而不是一句“买/卖”）
3. **决策与风控层 Policy/Risk**
   - 将预测映射成 buy/sell/hold（或仓位比例）
   - 加入成本、风险预算、最大回撤/止损规则
4. **执行层 Execution（回测层模拟即可）**
   - 模拟手续费、滑点、延迟、交易时段、成交限制
   - 输出真实可评估的 PnL 与风险指标

> 你想要的“学习K线决定买卖”，在机构里通常是：**Alpha（预测） + Policy（映射动作）**一起实现。

---

## 2）两种主流训练范式（都能“决定买卖”）

### 2.1 范式 A：监督学习（Supervised）——最主流、最好跨市场复用（推荐主线）
**思路**：模型先学“未来会涨/跌多少、波动多大”，再由策略规则决定买卖。

#### A1. 模型学习目标（标签/Label）
跨市场通用的标签（推荐组合）：

- **方向分类 Direction**
  - `y_dir(t,h) = 1 if ret(t,h) > 0 else 0`
  - 输出：`P(up)`，用于决定 Long/Short/Flat
- **幅度/区间 Quantile Regression**
  - 预测 `q10/q50/q90` 的未来收益率
  - 输出：预期收益与不确定性（区间宽度）
- **波动预测 Volatility（可选）**
  - 预测未来 realized volatility 或高波动标签
  - 用于风险等级与仓位缩放
- **启动窗口 Start Window（可选）**
  - 预测显著走势更可能在哪个时间窗开始
  - 用于“择时”更精确的解释与展示

#### A2. 常见可落地模型（业内最实用）
- Baseline：Logistic Regression / Linear Regression（强对照）
- MVP：LightGBM / XGBoost（训练快、稳定、可解释）
- Advanced：TCN / Transformer / TFT（增强项，用于对比提升）

#### A3. 训练与验证（行业最重视的三点）
- **时间序列切分**：Walk-forward（滚动/扩展窗口）
- **防泄露**：gap/purge（标签依赖未来窗口，必须隔离重叠）
- **现实成本**：回测必须加入 fee/slippage/delay，否则“纸面过拟合”

#### A4. 从预测到买卖（Policy：把预测映射成动作）
机构常见做法：模型不直接输出“买卖”，而是规则把概率/区间映射成动作（更稳、可解释）。

- **阈值动作（最常见）**
  - `P(up) > 0.55` → `Long`
  - `P(up) < 0.45` → `Short`（A股可关掉）
  - 否则 → `Flat`
- **期望收益减成本（更专业）**
  - 若 `q50 - cost > 0` 才开仓
  - `cost = fee + slippage + latency_impact`
- **仓位大小（Position Sizing）**
  - 用不确定性缩放：区间越宽，仓位越小  
  - 示例：`position ∝ |P(up)-0.5| / (q90-q10)`
- **风险控制（必要的最小规则）**
  - 最大仓位上限
  - 波动过高时降仓
  - 触发极端风险时强制 flat（风控开关）

> 这套“预测→动作”在 Crypto/A股/美股都通用，只需要在市场层处理做空限制和交易时段差异。

---

### 2.2 范式 B：强化学习（RL）——端到端输出买卖（难但酷，建议当可选增强）
**思路**：模型直接输出动作 `buy/sell/hold` 或连续仓位比例，用回测收益当奖励训练。

- 状态：过去 N 根 K 线 + 指标 + 当前仓位
- 动作：Long/Short/Flat 或仓位比例
- 奖励：PnL（扣手续费/滑点）- 回撤惩罚 - 风险惩罚

**现实难点**
- 很容易过拟合回测环境
- 对成本、延迟、成交规则极敏感
- 跨市场迁移难（A股制度差异大）

> 建议：RL 只作为“展示高级能力”的对照实验（比如仅 BTC），不要作为主线。

---

## 3）如何真正跨市场复用：统一 Schema + 市场适配器（Adapters）

### 3.1 统一数据 Schema（强烈建议）
无论 Crypto/A股/美股，统一输出字段：
- `timestamp_utc`
- `timestamp_market`（按市场时区展示）
- `open, high, low, close, volume`
- `symbol`
- `market`（crypto / ashares / us_equities）
- `exchange/source`（可选）
- `calendar_id`（交易日历标识）

### 3.2 三个市场适配器（Market Adapters）
把差异关在 adapter 内，模型与训练代码保持一致。

#### Crypto Adapter（24/7）
- 连续小时索引、无休市
- 可选扩展：funding/OI（如果你后面要更强）

#### A股 Adapter（交易日历 + 午休）
- 仅保留交易时段 bars（午休不造假 K 线）
- 处理停牌/涨跌停导致的异常分布与缺失

#### 美股 Adapter（交易日历 + DST）
- 存 UTC，展示美东（避免 DST 边界错误）
- 处理复权（split/dividend），至少日线要考虑

---

## 4）训练策略：三种常见“行业做法”（从稳到强）

### 4.1 最稳（推荐，MVP）
- **每个市场/标的独立训练**（同一代码框架）
- 优点：最可控、最容易按时交付
- 缺点：每个标的模型多、训练次数多

### 4.2 更强（进阶）
- **同市场内多标的联合训练（Panel 数据）**
- 输入多标的样本，让模型学“共性规律”
- 输出：每个标的单独预测

### 4.3 最强但更难（研究级加分项）
- **跨市场联合训练 + market embedding**
- 在输入加入市场类别/交易制度信息（告诉模型 crypto/ashares/us）
- 目标：迁移学习、提高泛化  
- 风险：工程复杂度明显上升

---

## 5）模型输入（Features）：K线派最常用、跨市场通用
跨市场通用特征组（MVP 推荐）：

- **Returns & Momentum**
  - log return
  - lag returns（1/2/4/12/24…）
  - rolling mean returns
- **Trend**
  - EMA/MA（多窗口）
  - MACD
- **Volatility**
  - rolling std
  - ATR
  - Bollinger Band width
- **Volume**
  - volume change rate
  - volume moving average
- **Time Features**
  - hour_of_day / day_of_week（Crypto尤其重要）
  - month（可选）

---

## 6）决策层（Policy Layer）：统一动作空间 + 市场开关

### 6.1 动作空间（Action Space）
- `Long`
- `Short`
- `Flat`

可扩展为连续仓位：
- `position ∈ [-1, +1]`（-1 全空，+1 全多）

### 6.2 市场制度开关（非常重要）
- Crypto：允许 Long/Short（如果做 perp），Spot 只能 Long/Flat
- A股：默认 Long/Flat（除非你明确支持融券/做空）
- 美股：可 Long/Short（若假设允许卖空），也可配置为 Long/Flat

建议 config：
- `allow_short: true/false`
- `market_type: spot/perp`
- `max_position: 1.0`

---

## 7）回测与评估（让“决定买卖”可被验证）

### 7.1 现实成本（必须）
- 手续费 `fee_bps`
- 滑点 `slippage_bps`
- 延迟：信号在 `t` 产生，`t+1` 执行（bar delay）

### 7.2 核心指标
- 预测指标（模型层）：
  - 方向：AUC/F1/Accuracy + Brier（校准）
  - 区间：Coverage + Width + Pinball loss
- 策略指标（决策层）：
  - total return、max drawdown、Sharpe
  - win rate、profit factor
  - 与 baseline（buy&hold、MA crossover）对比

---

## 8）推荐落地路线（最像市面上可交付方案）

### 8.1 MVP 主线（强烈推荐）
- **监督学习 Supervised**
  - 方向：LightGBMClassifier（+ 概率校准）
  - 幅度：Quantile LightGBM（q10/q50/q90）
- **Policy（映射动作）**
  - 阈值动作 + 成本过滤 + 风险缩放（用区间宽度/波动降仓）
- **验证**
  - walk-forward + gap/purge
  - 加成本回测

### 8.2 增强（可选）
- 加一个深度模型做对照（TCN 或 TFT 二选一，优先 daily 分支）
- 做 2–3 个 case studies（解释模型对/错）

### 8.3 高级展示（可选）
- RL demo（建议仅 BTC，避免范围爆炸）

---

## 9）可复现与版本化（机构化必备）
每次训练/预测都保存：
- `model_version`（时间戳或 git hash）
- `config_snapshot.yaml`
- `metrics.csv`（逐 fold）
- `predictions.csv`（样例/全量）
- `data_quality_report.md`

---

## 10）你可以直接放进系统的“统一实现接口”（建议）
为了跨市场一致，建议所有市场最终输出一个统一的预测结果结构：

- 输入：`symbol, market, timeframe, horizon, timestamp`
- 输出（Alpha）：
  - `p_up, p_down`
  - `q10/q50/q90`（收益率或目标价区间）
  - `volatility_score`
  - `confidence_score`
- 输出（Policy）：
  - `action`（Long/Short/Flat）
  - `position_size`（可选）
  - `expected_edge = q50 - cost`

---

> 如果你愿意，我可以再给你一个“更工程化版本”的附录：
> - `config.yaml`（三市场制度开关、时区、交易日历、成本参数）
> - Policy 计算公式（从 p_up + q10/q50/q90 到 action/position）
> - walk-forward 训练脚本清单（你照着写就能跑）

---

## 附录：工程化示例（Config / Policy / 运行）

下面给出可以直接放进 `configs/` 的示例 `forecast_config.yaml`、一套可直接实现的 Policy 映射伪码、数据质量与监控建议，以及最小化运行命令，方便工程落地。

### A. 示例配置：`configs/forecast_config.yaml`
```yaml
forecast_config:
  general:
    timezone: "Asia/Shanghai"
    default_symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    data_dir: "data/processed"
    model_dir: "models"

  data_source:
    exchanges: ["binance", "bybit"]
    default_exchange: "binance"
    default_market_type: "perp"

  hourly:
    enabled: true
    horizon: "4h"
    refresh_times_bj: [0,4,8,12,16,20]
    lookforward_bars: 6

  daily:
    enabled: true
    horizon: "1d"
    lookforward_days: 14
    refresh_time_bj: "00:05"

  thresholds:
    p_bull: 0.55
    p_bear: 0.45
    ret_threshold: 0.002   # 0.2%

  risk:
    risk_window_days: 180
    risk_quantiles: [0.5, 0.75, 0.9]

  confidence:
    w_prob: 0.6
    w_width: 0.4
    w_recent_perf: 0.0

  execution:
    allow_short: true
    fee_bps: 0.02
    slippage_bps: 0.05
    max_position: 1.0

  monitoring:
    weekly_check_day: "monday"
    weekly_check_hour_bj: 8
    alert_accuracy_threshold: 0.45
    alert_coverage_threshold: 0.7

```

### B. Policy：从预测到动作的映射（伪码与公式）

目标：把 `p_up, q10,q50,q90, volatility_score, confidence_score` 映射到 `action` 与 `position_size`。

伪码实现（可放入 `src/models/policy.py`）：

```
def policy_from_forecast(row, cfg):
    # 输入：一行 forecast（p_up, q10, q50, q90, vol, conf, current_price）
    p_up = row['p_up']
    q50 = row['q50_change_pct']
    q10 = row['q10_change_pct']
    q90 = row['q90_change_pct']
    vol = row['volatility_score']
    conf = row.get('confidence_score', 0.5)

    # 成本过滤（fee + slippage，以收益率表示）
    cost = cfg['execution']['fee_bps']/10000 + cfg['execution']['slippage_bps']/10000

    # 基础动作逻辑（阈值）
    if p_up >= cfg['thresholds']['p_bull'] and q50 - cost > 0:
        base_action = 'long'
    elif p_up <= cfg['thresholds']['p_bear'] and cfg['execution']['allow_short']:
        base_action = 'short'
    else:
        base_action = 'flat'

    # 仓位大小：概率偏离与不确定性缩放
    # position_raw ∈ [0,1]
    pos_from_prob = max(0, 2 * abs(p_up - 0.5))
    pos_from_uncert = 1.0 / (1.0 + (q90 - q10))   # 不确定性越大，分母越大
    position_size = pos_from_prob * pos_from_uncert

    # 依据风险/volatility 缩放
    # risk_scale ∈ (0,1], 例如 vol 分位处于 P75 以上则降仓到 0.5
    risk_scale = get_risk_scale(vol, cfg)
    position_size *= risk_scale

    # 用 confidence_score 最终修正
    position_size *= conf

    # 限制最大仓位
    position_size = min(position_size, cfg['execution']['max_position'])

    # 最后决定 action
    if base_action == 'long' and position_size >= 0.01:
        action = 'long'
    elif base_action == 'short' and position_size >= 0.01:
        action = 'short'
    else:
        action = 'flat'

    return { 'action': action, 'position_size': position_size }

```

说明：
- `get_risk_scale` 可以基于历史 vol 分位数返回缩放系数，例如：vol<P50→1.0, P50–P75→0.8, P75–P90→0.6, >=P90→0.4。
- `position_size` 只是研究信号；真实执行需要额外考虑资金/仓位管理/风控。

### C. 数据质量与预处理检查清单（最小化）
- 缺失值处理：若缺失连续超过 N bars（例如 3），标记该区间数据质量 bad 并跳过训练/预测。
- 复权与停牌：A股需完成复权；Crypto 无需复权，但需处理 funding、rollover（若使用 perp）。
- 非交易日/午休处理：A股午休与美股交易日历需用 Adapter 层剔除或填充。
- 异常值检测：当单日/单 bar |return| > 20%（可配置），标记为异常并审查/过滤。
- 数据回溯修正：当历史数据被修正（exchange API 补数据），保留 data_quality_report.md 并重新跑 affected windows。
- 时区统一：存储使用 UTC，展示/刷新以 `forecast_config.timezone` 为准。

### D. 评估阈值与监控计划（最小可运行）
- 周期性检查：每周运行一次自动回测，输出 `metrics.csv`（方向准确率、coverage、profit_factor、sharpe）
- 告警规则：
  - 若方向准确率 < `monitoring.alert_accuracy_threshold` → 触发告警并自动切换到 Seasonality（历史统计）模式
  - 若区间覆盖率 < `monitoring.alert_coverage_threshold` → 扩大 q10/q90 或审查模型
- 保存 artifact：每次训练/预测保存 `config_snapshot.yaml`, `model_version`, `metrics.csv`, `predictions.csv`。

### E. 最小可复现运行示例（命令）
下面给出最小化的训练/预测/评估脚本示例（假设项目使用 `python`）：

训练（小时级示例）：
```bash
python src/models/train_hourly.py --config configs/forecast_config.yaml --symbol BTCUSDT
```

生成预测（预测/调度脚本）：
```bash
python src/models/predict_hourly.py --config configs/forecast_config.yaml --symbol BTCUSDT --out data/processed/session_forecast_hourly.csv
python src/models/predict_daily.py --config configs/forecast_config.yaml --symbol BTCUSDT --out data/processed/session_forecast_daily.csv
```

评估（回测）：
```bash
python src/evaluation/backtest.py --predictions data/processed/predictions_sample.csv --config configs/forecast_config.yaml
```

建议把这些命令写入 `scripts/` 目录下的简短 ps1/bash wrapper，方便定时任务调用。

### F. 输出 / API JSON Schema（示例）
建议统一输出字段（JSON / CSV 均支持）：

Example (daily JSON record):
```json
{
  "forecast_id": "20260206_0000_BTCUSDT_binance_1d",
  "symbol": "BTCUSDT",
  "exchange": "binance",
  "market_type": "perp",
  "mode": "forecast",
  "horizon": "1d",
  "date_bj": "2026-02-07",
  "day_of_week": "Sunday",
  "p_up": 0.62,
  "p_down": 0.38,
  "q10_change_pct": -0.004,
  "q50_change_pct": 0.008,
  "q90_change_pct": 0.02,
  "volatility_score": 0.024,
  "target_price_q50": 46200.12,
  "trend_label": "bull",
  "risk_level": "medium",
  "confidence_score": 0.78,
  "forecast_generated_at_bj": "2026-02-06T00:05:00+08:00"
}
```

API 返回示例（概览）已在 PRD 主文档中有示例，前端可以直接消费该 JSON 数组。

---

如果你希望，我可以：
- 把 `configs/forecast_config.yaml` 新建到 `configs/`（需要你确认是否要我直接写文件）；
- 生成 `src/models/policy.py` 的实际 Python 实现并运行简单静态测试（需要创建/编辑文件）。

以上先完成 `config.yaml` 与 Policy 伪码与补充说明，下一步我可以把 `policy.py` 与脚本样例落到工程里。

