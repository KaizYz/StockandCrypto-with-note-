# Crypto 交易时间段预测页面（北京时区）PRD · 完整版（Default: 4H）
> 数据源限定：**Binance（Perp 或 Spot） / Bybit（Perp 或 Spot）**  
> 其他交易所：**不考虑**  
> 页面默认展示：**4H 级别（horizon=4h）**  
> 时区：**北京时间（Asia/Shanghai）**，24 小时制

---

## 1. 目标（Goal）
在现有多市场预测系统中，新增一个**独立页面**：`交易时间段预测（Crypto）`，专门回答以下时间决策问题：

- **什么时候更可能上涨？**
- **什么时候更可能下跌？**
- **什么时候波动最大（风险最大）？**
- **未来 N 天哪几天最强趋势/最危险？**

该页面输出采用统一口径：**方向概率 + 启动窗口概率 + 幅度区间（q10/q50/q90）**，并通过可视化与榜单方式帮助用户做“择时”决策。

> 注：本页面用于研究与辅助决策，不构成投资建议，不执行真实交易。

---

## 2. 页面定位（Positioning）
- 页面名称：`交易时间段预测（Crypto）`
- 适用市场：Crypto（第一阶段）
- 默认币种：`BTC / ETH / SOL`
- 数据源范围：`Binance`（Perp/Spot）或 `Bybit`（Perp/Spot）
- 时区标准：北京时间（Asia/Shanghai）
- 时间显示：24 小时制（`00:00` - `23:59`）
- 默认预测周期：**小时级 4H**（horizon=4h）
- 页面模式（Mode）：
  - **预测模式 Forecast（默认）**：来自模型对未来的预测输出
  - **统计模式 Seasonality（可选）**：历史按小时/时段的经验统计（对照用）

---

## 3. 核心功能（MVP）

### 3.1 三大交易时段概率看板（默认 4H）
按北京时间固定划分（可配置）：
- 亚盘：`08:00 - 15:59`
- 欧盘：`16:00 - 23:59`
- 美盘：`00:00 - 07:59`

每个时段展示（默认聚合 4H 预测）：
- 上涨概率：`P(up)`
- 下跌概率：`P(down)`
- 预期涨跌幅（中位数）：`q50_change_pct`
- 目标价格（q50）：`target_price_q50`
- 区间目标价：`target_price_q10 / target_price_q90`
- 波动强度：`volatility_score = q90 - q10`
- 趋势标签：偏多 / 偏空 / 震荡（见第 6 章）
- 可信度：`confidence_score`（见第 6 章）

> 聚合方式说明：同一时段内包含多个小时点（00-23），对这些小时点的预测结果做加权/平均汇总（详见第 7 章）。

---

### 3.2 关键小时涨跌概率（00–23）
按小时输出（默认 4H 预测）：
- 每小时上涨概率 `p_up`
- 每小时下跌概率 `p_down`
- 每小时预期涨跌幅（q50）
- 每小时区间（q10/q90）
- 波动强度（q90-q10）
- 可信度（confidence_score）

并给出：
- **最可能上涨的时间段 Top N**
- **最可能下跌的时间段 Top N**
- **最有可能大波动的时间段 Top N**

---

### 3.3 波动时段识别（Risk/Volatility）
按小时排序输出：
- 小时（北京时区）
- 预期波动幅度：`volatility_score = q90 - q10`
- 目标价格区间：`target_price_q10 ~ target_price_q90`
- 趋势方向：`sign(q50_change_pct)`（正/负）
- 风险等级：低/中/高/极高（见第 6 章）

并给出：
- **最有可能出现大波动的时间段（Top N）**

---

### 3.4 日线级别预测（未来 N 天）——**独立预测**

> **关键理念**：日线预测与小时级（4H）预测是**两个独立的预测模块**，相互不影响。  
> 即：日线偏多不代表当日每个 4H 都是偏多；小时级偏空也不代表日线预测下跌。  
> **使用场景**：帮助用户做"多时间框架"决策（周期交易 vs 日内交易）。

按日期输出（北京时间）：
- 日期（YYYY-MM-DD）+ 星期
- `P(up)` / `P(down)`（基于 1 天收益率）
- 预期涨跌幅（q50）
- 预期波动幅度（q90-q10）
- 目标价格（q10/q50/q90）
- 趋势标签（偏多/偏空/震荡）
- 风险等级（低/中/高/极高）
- 可信度（confidence_score）
- 启动窗口预测（基于日线）：
  - `P(W0_1day/W1_2-3days/W2_3-7days)` 或 Top-1 启动窗口

并给出榜单：
- **未来 N 天涨幅最大的日期（Top N）**
- **未来 N 天跌幅最大的日期（Top N）**
- **未来 N 天波动最大的日期（Top N）**

---

## 4. 页面布局（UI Layout）

### 区块 A：筛选区（Filters）
- 币种选择：默认 `BTC / ETH / SOL`
- 数据源选择：`Binance` / `Bybit`
- 市场类型：`Perp` / `Spot`
- 页面模式：`Forecast（默认）` / `Seasonality（对照）`
- 预测周期：`4h（默认）`（可选扩展 1h/2h）
- 显示天数：未来 `N 天`（默认 7–14）
- 最新更新时间（北京时间）：`data_updated_at_bj`
- 预测生成时间（北京时间）：`forecast_generated_at_bj`

---

### 区块 B：三大时段总览卡（Sessions Overview）
三张卡：亚盘/欧盘/美盘
- `P(up)` / `P(down)`
- `q50_change_pct`
- `target_price_q50`
- `q10/q90 区间`
- `volatility_score`
- `trend_label`
- `confidence_score`

---

### 区块 B-2：日线级别总览卡（Daily Summary）
- 未来 N 天平均上涨概率 / 下跌概率
- 未来 N 天平均波动强度
- 近期最强趋势（偏多/偏空/震荡）
- 风险指数（综合 volatility_score）

---

### 区块 C：24 小时热力图（默认 4H）
- X 轴：小时（00–23，北京时间）
- 指标层（可切换 Tab）：
  - 上涨概率 `p_up`
  - 下跌概率 `p_down`
  - 波动强度 `volatility_score`
  - 可信度 `confidence_score`
- 色深：数值大小
- 默认展示：`p_up`（4H）

---

### 区块 D：未来日线预测表（N 天）
字段：
- 日期 + 星期
- `P(up)` / `P(down)`
- `q50_change_pct`
- `target_price_q50`
- `q10/q90 区间`
- `volatility_score`
- `trend_label`
- `risk_level`
- `start_window_top1`（或 `P(W0..W3)`）

支持：
- 按 `P(up)` / `q50_change_pct` / `volatility_score` 排序
- 支持搜索/筛选（可选）

---

### 区块 E：关键榜单（Leaderboards）
#### 小时级（当日 00–23）
- 最可能上涨时段 Top N
- 最可能下跌时段 Top N
- 最可能大波动时段 Top N

#### 日线级（未来 N 天）
- 最可能上涨日期 Top N
- 最可能下跌日期 Top N
- 波动最大日期 Top N

榜单每条至少包含：
- 时间/日期
- `P(up)` / `P(down)`
- `q50_change_pct`
- `target_price_q50`
- `q10/q90 区间`
- `trend_label / risk_level / confidence_score`

---

### 区块 F：可读性说明（Explainability）
用通俗语言解释：
- 为什么某时段“上涨概率高”
- 为什么“上涨概率高”不等于“涨幅最大”
- 为什么方向概率 vs 幅度预测可能不一致
- 为什么波动越大代表风险越高（但也可能机会更大）
- Forecast 模式 vs Seasonality 模式的差异

---

## 5. 指标与口径定义（统一口径）

### 5.1 概率类（Direction）
- `P(up) = P(ret_h > 0)`（h 默认 4h）
- `P(down) = 1 - P(up)`

> Forecast 模式：来自模型输出概率（可校准）  
> Seasonality 模式：历史分组统计概率（见第 7 章）

---

### 5.2 幅度与价格（Magnitude & Target）
- 预期涨跌幅（中位）：`q50_change_pct`
- 下沿涨跌幅：`q10_change_pct`
- 上沿涨跌幅：`q90_change_pct`

目标价：
- `target_price_q50 = current_price * (1 + q50_change_pct)`
- `target_price_q10 = current_price * (1 + q10_change_pct)`
- `target_price_q90 = current_price * (1 + q90_change_pct)`

---

### 5.3 波动强度（Volatility）
- `volatility_score = q90_change_pct - q10_change_pct`

含义：
- 数值越大，该时段潜在波动越大（风险越高）

---

### 5.4 启动窗口（Start Window）——与主系统一致（建议保留）
对日线（也可对小时）输出启动窗口概率：
- `W0: no_start`
- `W1: 0–1`
- `W2: 1–3`
- `W3: 3–7`
（小时可用 0–1h/1–2h/2–4h）

输出：
- `P(W0..W3)` 或 `start_window_top1`

---

## 6. 标签体系（Trend / Risk / Confidence）

### 6.1 趋势标签（Trend Label，推荐升级版）
使用“三维逻辑”更稳：概率 + 幅度 + 波动

可配置阈值：
- `p_bull = 0.55`
- `p_bear = 0.45`
- `ret_thr = 0.20%`（0.002）

规则：
- **偏多**：`p_up >= p_bull AND q50_change_pct >= +ret_thr`
- **偏空**：`p_up <= p_bear AND q50_change_pct <= -ret_thr`
- **震荡**：其它情况

---

### 6.2 风险等级（Risk Level）
基于 `volatility_score` 分位数分箱（更自适应）
- 低：< P50
- 中：P50–P75
- 高：P75–P90
- 极高：≥ P90

> 分位数参考窗口可配置：近 90 天/180 天（默认 180 天）

---

### 6.3 可信度（Confidence Score，0–100）
目的：避免用户把 Top N 当成“必涨必跌”，提供“预测可信程度”提示。

一个简单、可落地的 MVP 公式（可配置权重）：
- `conf_prob = 2 * |p_up - 0.5|`（0~1）
- `conf_width = 1 - normalize(volatility_score)`（波动越大区间越宽，可信度越低）
- `conf_recent_perf`（可选）：最近 K 天该模型的方向命中率/区间覆盖率

组合：
- `confidence_score = 100 * ( w1*conf_prob + w2*conf_width + w3*conf_recent_perf )`
- 默认权重：`w1=0.6, w2=0.4, w3=0`（MVP 先不依赖历史表现也能跑）

---

## 7. 数据来源、预测逻辑与模式说明（关键）

### 7.0 日线 vs 小时级独立性说明（核心）

**问题**：「为什么日线偏多，但小时级可能偏空？」

**答案**：两者基于不同的预测目标：

| 维度 | 小时级（4H） | 日线（1D） |
|------|------------|----------|
| **预测目标** | 未来 4 小时的收益率 | 未来 1 天（24h）的收益率 |
| **数据周期** | 4h K 线 | 1d K 线 |
| **预测模型** | 独立分类/分位数模型（小时数据训练） | 独立分类/分位数模型（日数据训练） |
| **典型用途** | 日内短线择时 | 周期/多日趋势择时 |
| **聚合关系** | 无（不聚合到日线） | 无（不从小时聚合） |

**典型案例**：
- BTC 在 2 月 6 日的**日线**预测：`P(up)=0.65, q50=+0.8%`（偏多）
- 但当日的**4H 时段**预测可能：
  - 08:00-12:00（亚盘）：`P(up)=0.48`（偏空）
  - 12:00-16:00：`P(up)=0.52`（震荡）
  - 16:00-20:00（欧盘）：`P(up)=0.62`（偏多）  
  - 20:00-00:00（美盘）：`P(up)=0.55`（偏多）  
  - **4H 平均**：`P(up)≈0.54`（与日线 0.65 不同）

**原因**：
1. **模型训练集不同**：4H 模型在 4H K 线上训练，1D 模型在 1D K 线上训练，特征工程/策略差异导致预测不一致
2. **信息粒度不同**：1D K 线看不到 4H 的振荡细节，只关注日终收盘；4H 看得更细致
3. **时间框架偏差**：4H 的「未来 4h 涨」≠ 1D 的「整天涨」（4H 可能 up，但后续 4H down，日终 down）

**对用户的意义**：
- **多时间框架决策**：日线多头+小时空头 → 可能是"好的买点"（小周期回调）
- **风险识别**：日线多头但小时极度波动 → 谨慎；小时多头但日线非多 → 超短线机会
- **避免过度拟合**：两个独立模型互为对比，增加系统鲁棒性

---

### 7.1 数据源限制（硬约束）
仅使用：
- Binance（Perp 或 Spot）
- Bybit（Perp 或 Spot）

字段需保留：
- `exchange`（binance/bybit）
- `market_type`（perp/spot）
- `symbol`

---

### 7.2 Forecast 模式（默认）
含义：所有概率/区间来自模型预测输出。

- `p_up`：分类模型输出（建议做概率校准）
- `q10/q50/q90`：分位数回归模型输出（Quantile LightGBM 等）

---

### 7.3 Seasonality 模式（对照用）
含义：按小时/时段做历史统计，回答“历史上这个小时更容易涨吗”。

建议统计窗口：
- 过去 `90天` 或 `180天`（可配置）

统计方法（示例）：
- 对每个 `hour_bj`，统计在该小时开盘后未来 `4h` 的收益率：
  - `p_up_stat = mean( ret_4h > 0 )`
  - `q50_stat = median(ret_4h)`
  - `q10/q90_stat = quantile(ret_4h, 0.1/0.9)`

用途：
- 给用户一个“季节性/日内节奏”对照
- 给你一个 sanity check（模型预测与统计是否严重矛盾）

---

### 7.4 时段聚合逻辑（Session Aggregation）
对每个 session（亚/欧/美），将 session 内的小时结果聚合成一条 session 结果：

MVP 聚合推荐：
- `p_up_mean = mean(p_up_hour)`
- `q50_change_mean = mean(q50_change_pct_hour)`（或 median）
- `volatility_score_mean = mean(volatility_score_hour)`
- `target_price_q50 = current_price * (1 + q50_change_mean)`

可选加权（更合理）：
- 按 `confidence_score` 加权平均：
  - `weighted_mean(x) = sum(conf*x)/sum(conf)`

---

### 7.5 日线预测逻辑（Daily Prediction）

完全独立的预测流程，不涉及小时数据的汇总。

**输入**：
- 1D K 线数据（open/high/low/close/volume）
- 技术指标（基于 1D）
- 可选：事件特征（如即将有 CPI/FOMC 等）

**模型**：
- 分类模型（预测 `P(up)`）：LightGBM Classifier（1D 特征训练）
- 分位数回归（预测 `q10/q50/q90`）：Quantile Regression（1D 特征训练）

**输出**：
- `date_bj`, `p_up`, `p_down`, `q10_change_pct`, `q50_change_pct`, `q90_change_pct`
- `volatility_score`, `target_price_q10/q50/q90`
- `trend_label`, `risk_level`, `confidence_score`

**注意**：
- 日线预测窗口为「当前日期至未来 N 日」（通常 7-14 天）
- 不依赖小时级的 4H 结果，反之亦然

---

## 8. 数据输出结构（后端建议）

### 8.1 小时级结果表（核心）
文件：`data/processed/session_forecast_hourly.csv`

建议字段：
- `forecast_id`（唯一键：timestamp + symbol + exchange + market_type + horizon + mode）
- `symbol`
- `exchange`（binance/bybit）
- `market_type`（perp/spot）
- `mode`（forecast/seasonality）
- `horizon`（默认 `4h`）
- `hour_bj`（0-23）
- `p_up`
- `p_down`
- `q10_change_pct`
- `q50_change_pct`
- `q90_change_pct`
- `volatility_score`
- `target_price_q10`
- `target_price_q50`
- `target_price_q90`
- `trend_label`
- `risk_level`
- `confidence_score`
- `current_price`
- `forecast_generated_at_bj`
- `data_updated_at_bj`
- `model_version`（时间戳或 git hash）
- `timestamp_market_bj`（该小时对应的时间戳）

---

### 8.2 时段聚合结果表
文件：`data/processed/session_forecast_blocks.csv`

字段：
- `forecast_id`
- `symbol`
- `exchange`
- `market_type`
- `mode`
- `horizon`（4h）
- `session_name`（亚盘/欧盘/美盘）
- `p_up_mean`
- `p_down_mean`
- `q10_change_mean`（可选）
- `q50_change_mean`
- `q90_change_mean`（可选）
- `volatility_score_mean`
- `target_price_q50`
- `trend_label`
- `risk_level`
- `confidence_score_mean`
- `forecast_generated_at_bj`
- `data_updated_at_bj`
- `model_version`

---

### 8.3 日线级结果表
文件：`data/processed/session_forecast_daily.csv`

字段：
- `forecast_id`
- `symbol`
- `exchange`
- `market_type`
- `mode`
- `horizon`（默认 1d，可扩展 3d/7d）
- `date_bj`（YYYY-MM-DD）
- `day_of_week`
- `p_up`
- `p_down`
- `q10_change_pct`
- `q50_change_pct`
- `q90_change_pct`
- `volatility_score`
- `target_price_q10`
- `target_price_q50`
- `target_price_q90`
- `trend_label`
- `risk_level`
- `confidence_score`
- `start_window_top1`（或 `p_w0/p_w1/p_w2/p_w3`）
- `forecast_generated_at_bj`
- `data_updated_at_bj`
- `model_version`

---

## 9. 数据刷新策略

### 9.1 小时级（4H）刷新
- **刷新频率**：每 4 小时触发一次（00:00, 04:00, 08:00, ..., 20:00 北京时间）
- **数据延迟**：应在当前时段开始后 5-10 分钟内更新（如 04:05 前完成 0-4h 的预测）
- **数据源**：Binance/Bybit 当日最新数据
- **预测覆盖**：0-23 点共 6 个 4H 时段（若要细化可扩展到 1H、2H：Phase 2）

### 9.2 日线级（1D）刷新
- **刷新频率**：每天 1 次，北京时间 00:05（零点后 5 分钟）更新前一天的行情闭合
- **预测范围**：更新后的当日 + 未来 N-1 天（共 N 天）
- **数据源**：Binance/Bybit 1D K 线收盘数据

### 9.3 历史统计（Seasonality 模式）刷新
- **刷新频率**：每周一次（如周日晚 23:00）
- **统计窗口**：过去 180 天的数据（可配置）
- **目的**：对照 Forecast 模式，检查模型是否与历史规律严重偏离

---

## 10. 参数配置管理

所有 MVP 参数集中管理在 `configs/config.yaml`（或 `configs/forecast_config.yaml`）：

```yaml
forecast_config:
  # 趋势标签阈值（6.1）
  trend_labels:
    p_bull: 0.55        # 偏多概率阈值
    p_bear: 0.45        # 偏空概率阈值
    ret_threshold: 0.002 # 涨跌幅阈值（0.2%）
  
  # 风险等级分位数（6.2）
  risk_levels:
    window_days: 180    # 分位数参考窗口（天）
    # 分箱：low < P50, P50 <= medium < P75, P75 <= high < P90, P90 <= extreme
  
  # 可信度权重（6.3）
  confidence_score:
    w_prob: 0.6         # 概率偏离度权重
    w_width: 0.4        # 区间宽度权重
    w_recent_perf: 0.0  # 历史表现权重（MVP 先不用）
  
  # 聚合方式（7.4）
  session_aggregation:
    method: "weighted_mean"  # simple_mean / weighted_mean
    weight_field: "confidence_score"
  
  # 小时级预测
  hourly:
    horizon: "4h"                # 默认 4H（Phase 2 可支持 1h/2h）
    refresh_freq_hours: 4
    refresh_time_bj: [0, 4, 8, 12, 16, 20]  # 北京时间刷新点
  
  # 日线预测
  daily:
    horizon: "1d"
    lookforward_days: 14        # 预测未来天数
    refresh_freq_hours: 24
    refresh_time_bj: "00:05"    # 每天零点后 5 分钟
  
  # Seasonality 模式
  seasonality:
    window_days: 180
    refresh_freq_days: 7
    refresh_day: "sunday"
  
  # 数据源
  data_source:
    exchanges: ["binance", "bybit"]
    market_types: ["perp", "spot"]
    default_exchange: "binance"
    default_market_type: "perp"
  
  # 币种
  symbols:
    default: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    extended: []  # Phase 2 支持 Top 100
```

---

## 11. 模型监控与迭代机制

### 11.1 关键指标
- **方向准确率（Accuracy）**：预测方向（up/down）与实际方向一致的比例
- **盈利因子（Profit Factor）**：用预测目标价 q50 模拟交易，盈利总额 / 亏损总额
- **区间覆盖率（Coverage）**：实际价格落在 [q10, q90] 区间的比例（建议 >= 80%）
- **拆分精准度（Calibration）**：预测 P(up)=0.6 的样本中，实际有 ~60% 上涨（概率校准）

### 11.2 监控频率
- **每周回测一次**（如周一 08:00）：评估过去 7 天的模型表现
- **每月论证一次**（如月初）：决定是否更新模型权重或重训练

### 11.3 性能阈值与告警
- 若方向准确率 < 45%，使用 Seasonality 模式代替 Forecast（降级保护）
- 若区间覆盖率 < 70%，放宽 q10/q90 分位数（如改为 5/95）
- 若盈利因子持续 < 1.0，通知开发人员重新训练

### 11.4 模型版本管理
- 每次重训练生成新的 `model_version`（格式：`YYYYMMDD_HHMMSS` 或 git commit hash）
- 存储在数据表 `model_version` 字段，支持版本对比与回溯

---

## 12. 容错与降级机制

### 12.1 数据源不可用处理
**场景**：Binance 数据延迟或不可用

**方案**：
- 优先使用 Binance；若不可用，自动切换到 Bybit
- 页面筛选区显示「当前数据源：Binance（正常）」或「当前数据源：Bybit（备用）」
- 数据表新增字段 `data_source_actual`，记录实际使用的源

### 12.2 模型预测失效处理
**场景**：模型决策能力严重下降（如方向准确率 < 45%）

**方案**：
- 切换到 Seasonality 模式（降级）
- 页面显示警告：「当前为历史统计模式（Seasonality），Forecast 模式暂时不可用」
- 用户仍能看到榜单和热力图（基于历史统计）

### 12.3 实时数据延迟处理
**场景**：新的 4H 时段应该刷新但数据延迟

**方案**：
- 若 5 分钟内未到新数据，使用上一期的预测结果
- 页面显示：「数据更新于 XX 时间（可能存在延迟）」

### 12.4 极端行情保护
**场景**：市场波动超出历史范围（如日线波动 > 20%）

**方案**：
- 模型输出的置信度 `confidence_score` 应自动大幅降低
- 页面榜单显示「当前行情波动异常，模型表现可能不稳定」

---

## 13. 系统集成与接口规范

### 13.1 与现有多市场预测系统的集成
**共享模块**：
- 数据获取层（`src/ingestion/`）
- 特征工程层（`src/features/`）
- 标签生成层（`src/labels/`）
- 指标计算层（`src/utils/metrics.py`）

**独立模块**：
- 小时级模型训练与预测（新建 `src/models/hourly_forecast.py`）
- 日线级模型训练与预测（新建 `src/models/daily_forecast.py`）
- 聚合逻辑（新建 `src/models/aggregation.py`）

### 13.2 API 接口（Dashboard 调用）
**后端端点**：

```
GET /api/forecast/session
  ?symbol=BTCUSDT
  &mode=forecast|seasonality
  &horizon=4h
  &exchange=binance
  &market_type=perp
返回：session 级别的亚/欧/美三时段卡片数据

GET /api/forecast/hourly
  ?symbol=BTCUSDT&mode=forecast&horizon=4h&...
返回：24 小时热力图 + 榜单数据

GET /api/forecast/daily
  ?symbol=BTCUSDT&lookforward_days=14&...
返回：日线表格 + 榜单数据

GET /api/forecast/metadata
返回：最新刷新时间、模型版本、可支持的币种/交易所等
```

### 13.3 数据输出格式（JSON）
```json
{
  "status": "success",
  "metadata": {
    "symbol": "BTCUSDT",
    "exchange": "binance",
    "market_type": "perp",
    "mode": "forecast",
    "horizon": "4h",
    "forecast_generated_at_bj": "2026-02-06T12:00:00+08:00",
    "data_updated_at_bj": "2026-02-06T11:55:00+08:00",
    "model_version": "20260206_033542",
    "current_price": 45820.50
  },
  "hourly": [
    {
      "hour_bj": 0,
      "p_up": 0.52,
      "p_down": 0.48,
      "q50_change_pct": 0.0015,
      "target_price_q50": 45869.28,
      ...
    }
  ],
  "sessions": [
    {
      "session_name": "亚盘",
      "session_hours": "08:00-15:59",
      "p_up_mean": 0.55,
      ...
    }
  ],
  "daily": [
    {
      "date_bj": "2026-02-07",
      "day_of_week": "Saturday",
      "p_up": 0.62,
      ...
    }
  ]
}
```

---

## 14. 交互与可用性要求
- 全部时间显示北京时间（24小时制），不截断
- 日期完整显示（YYYY-MM-DD + 星期）
- 关键字段提供中文解释（小白可读）
- 表格支持按：
  - `P(up)` / `P(down)` / `q50_change_pct` / `volatility_score` / `confidence_score` 排序
- 默认 4H 视图（用户可切换 1H/2H：Phase 2）

---

## 15. 验收标准（MVP）
完成后应满足：
1. 有独立页面：`交易时间段预测（Crypto）`
2. 能在 `BTC/ETH/SOL` 切换并刷新
3. 数据源可切换：`Binance/Bybit` 与 `Perp/Spot`
4. 默认展示 **4H** 预测（热力图、榜单、时段卡）
5. 有亚盘/欧盘/美盘三时段概率与趋势卡片
6. 有 24 小时热力图（默认 4H）
7. 有日线预测表（未来 7–14 天）并含趋势/风险/目标价
8. 有 6 个榜单（小时级+日线级）：
   - 小时：最可能上涨/下跌/大波动 Top N
   - 日线：最可能上涨/下跌/大波动 Top N
9. 每条榜单项至少包含：
   - 时间/日期 + `q50_change_pct` + `target_price_q50` + `q10/q90` + `trend_label` + `risk_level` + `confidence_score`
10. 所有输出表可落地为 CSV（第 8 章格式）
11. **核心验证**：日线和小时级预测相互独立，页面清晰展示两者不相互影响的逻辑
12. 支持 Forecast 与 Seasonality 两个模式的快速切换

---

## 16. 使用场景示例（可选展示给团队）

### 场景 1：短线交易员（关注小时级）
**用户**：日内交易 BTC，追求 4H-1D 的快速盈利

**决策流程**：
1. 打开页面，选择 BTC，确保 mode=**Forecast**（预测模式）
2. 查看「24 小时热力图」，发现 16:00-20:00（欧盘）的 4H 时段 `P(up)=0.68`，高于平均
3. 查看「最可能上涨时段 Top 3」，确认 16:00-20:00 在列表中
4. 检查「波动强度」：`volatility_score=0.035`（中等）→ 风险可控
5. **决策**：在 16:00 附近关注 BTC，如果技术面配合，考虑做多 4H
6. **目标价**参考表格的 `target_price_q50 = 45870`（基于当前 45800 的预测）
7. **止损**参考 `target_price_q10 = 45600`（下沿保护）

---

### 场景 2：中线持仓者（关注日线级）
**用户**：做多 BTC，计划持有 7-14 天

**决策流程**：
1. 打开页面，选择 BTC，查看「日线预测表」
2. 发现未来 14 天中，2 月 10 日-12 日连续 3 天 `P(up)` 都 > 0.6（偏多趋势）
3. 查看「最可能上涨日期 Top 5」，2 月 10-12 日确实在列表，且 `q50_change_pct` 高（如 +1.2% / +0.8% / +0.9%）
4. 但同时发现 2 月 8 日 `volatility_score=0.082`（极高）→ `risk_level=high`
5. **决策**：
   - 不建议在 2 月 8 日前加仓（波动太大）
   - 在 2 月 9 日-10 日波动下降后，考虑加仓（迎接 10-12 的上涨窗口）
6. **风险管理**：设置止损在 2 月 7 日收盘跌破某价位时执行

---

### 场景 3：多时间框架组合（小时+日线）
**用户**：同时参考小时级和日线级，做精细化决策

**场景**：日线偏多（`P(up)=0.65`），但今日小时级偏空

**决策流程**：
1. 查看日线：2 月 6 日 `P(up)=0.65`，趋势偏多 → 中期看多
2. 查看小时级：但 08:00-12:00 的 4H `P(up)=0.45`，偏空 → 短期做空
3. **决策**：
   - **短期做空**（同日 4H 做空）：在 45870-45875 做空，目标 45650，止损 45950
   - **中期看多**：做空本质是"低价加多单"的准备
   - 如果短期做空成功，在 45650-45750 加多单，持仓至 2 月 7-8 日（等待日线的多头启动）
4. **风险控制**：确保两个方向都在止损范围内

---

### 场景 4：历史对照（Seasonality 模式）
**用户**：想检验 Forecast 模式是否靠谱

**决策流程**：
1. 打开页面，同时查看两个 mode：
   - **Forecast 模式**：8:00-12:00 的 4H `P(up)=0.48`
   - **Seasonality 模式**（切换 Tab）：历史同时段平均 `P(up)=0.52`
2. **对比**：Forecast 略空，但历史平均偏多 → 说明模型在"消化"最近的空头新闻
3. **决策**：
   - 如果你信任历史节奏，Seasonality 提示做多
   - 如果你信任模型权重，Forecast 提示谨慎
   - **建议**：Forecast + Seasonality 的分歧本身就是信息，值得深入研究

---

## 17. 迭代路线（Phase 2）
- 支持 Crypto Top100（排除稳定币）
- 增加“工作日/周末”分层统计（forecast + seasonality）
- 增加事件窗口过滤（CPI/FOMC/ETF 流入等）
- 增加择时策略模拟（按时段/日期择时）
- 日线增强：
  - 周线/日线联动（识别周级趋势）
  - 历史同日期回测（节奏统计）
  - 趋势反转预警、极端波动预警
- 小时维度增强：
  - 支持 1H/2H 切换（当前默认 4H）

---

## 18. 风险提示
本页面用于研究与辅助决策，不构成投资建议。加密市场波动大，模型结果可能在极端行情失效，建议配合风险控制与仓位管理。

---

## 19. 内容要求补充（执行版）

> 本章节用于把“想法”转成“可交付要求”，开发、测试、验收统一按本节执行。

### 19.1 需求分级（MoSCoW）

#### P0（MVP 必须有）
- 独立页面：`交易时间段预测（Crypto）`
- 北京时间 24 小时制，时间完整显示（禁止截断）
- 默认 `4H`，默认币种 `BTC/ETH/SOL`
- 三大时段卡：亚盘/欧盘/美盘（P(up)/P(down)/q50/目标价/趋势）
- 24 小时热力图（至少支持 `p_up`、`volatility_score` 两层）
- 小时榜单 Top N：最可能上涨 / 下跌 / 大波动
- 日线预测表（未来 N 天，默认 14 天）
- 日线榜单 Top N：最可能上涨 / 下跌 / 大波动
- Forecast / Seasonality 双模式切换
- 输出落盘 CSV（hourly / blocks / daily）

#### P1（建议本期完成）
- 可信度 `confidence_score` 与风险分层 `risk_level`
- Session 聚合支持 `weighted_mean(confidence_score)`
- 数据源自动降级（Binance -> Bybit）
- 页面内解释文案（方向概率 vs 幅度预测为何可能不一致）
- 支持 `Perp/Spot` 切换

#### P2（后续增强）
- 1H / 2H 切换
- Top100 币种扩展（排除稳定币）
- 事件窗口过滤（CPI/FOMC/ETF 流入）
- 择时策略回测模拟

---

### 19.2 非功能要求（NFR）

- **刷新时效**
  - 4H 页面数据更新时间延迟不超过 10 分钟
  - 日线页面每天 00:05（北京时间）后完成更新
- **性能**
  - 在缓存命中情况下，页面主视图渲染 <= 2 秒
  - 榜单排序交互响应 <= 500ms
- **可用性**
  - 关键字段均有中文解释
  - 日期、时区、币种、数据源必须可见
- **可追溯**
  - 每条数据必须带 `model_version`、`forecast_generated_at_bj`、`data_updated_at_bj`

---

### 19.3 数据质量门槛（上线前）

- 小时级缺失率（近 30 天） <= 1%
- 时间戳重复率 == 0
- 未来目标字段完整率（q10/q50/q90） >= 99%
- `p_up + p_down` 误差 <= 1e-6
- `q10 <= q50 <= q90` 违规率 == 0

未满足任一条件：页面显示“数据质量告警”，并禁用“可执行建议”标识。

---

### 19.4 页面文案要求（小白可读）

每个核心区块下方必须提供一句“怎么用”的解释：
- 时段卡：  
  “看 `P(up)` 判断方向概率，看 `q50` 判断预期幅度，看 `volatility_score` 判断风险。”
- 小时榜单：  
  “榜单用于找可能的高胜率时间段，不代表必然上涨/下跌。”
- 日线榜单：  
  “用于中短期择时，不应替代仓位与止损管理。”

---

### 19.5 验收测试用例（最少 10 条）

1. 切换 BTC/ETH/SOL，页面指标均变化且无报错。  
2. 切换 Binance/Bybit，数据源标识同步变化。  
3. 切换 Perp/Spot，`current_price` 与目标价重新计算。  
4. 时间显示为北京时间 24h 且无省略号。  
5. 三时段卡片的 `p_up + p_down = 1`。  
6. 任意小时记录满足 `q10 <= q50 <= q90`。  
7. 热力图数值与小时表同一字段一致。  
8. 小时榜单 Top N 按对应指标正确降序。  
9. 日线榜单 Top N 排序与表格一致。  
10. Forecast 与 Seasonality 切换后，结果来源和口径标识正确。  

---

### 19.6 交付清单（Done Definition）

- 文档：本 PRD 更新完成并冻结版本号
- 代码：
  - 页面代码（dashboard）
  - 小时/时段/日线数据生成代码（pipeline）
  - 配置项（config）
- 数据：
  - `session_forecast_hourly.csv`
  - `session_forecast_blocks.csv`
  - `session_forecast_daily.csv`
- 验收：
  - 测试记录（至少 10 条）
  - 一张全页截图（包含时段卡、热力图、榜单、日线表）
