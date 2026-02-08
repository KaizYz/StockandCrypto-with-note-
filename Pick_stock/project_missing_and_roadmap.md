# Multi-Market 项目缺口与落地路线图（最终稿）

> **目标**：把当前系统从“可看/可分析”升级到“可执行/可跟踪/可验证”的产品级形态。  
> **范围**：Crypto / A股 / 美股 + Selection/Research/Tracking + 回测 + Paper Trading。  
> **说明**：以下为工程规划与系统设计，不构成投资建议。  
> **时区约定**：Crypto & 亚盘统一 **北京时间（Asia/Shanghai）**；美股统一 **美东时间（America/New_York）**；底层统一 **UTC 存储**。

---

## 1. 目前项目完成度（简评）

你现在已经有：
- 多市场页面（Crypto / A股 / 美股）+ 核心字段展示（当前价/预测价/预期涨跌幅/预测区间）
- 开单信号展示（建议动作、TP/SL、RR）+ 回测结果可视化 + Walk-forward 曲线
- Crypto 交易时间段预测页面（时段概率/关键小时热力图/TopN榜单）
- Selection / Research / Tracking 页面（筛选、policy layer 状态分布、watchlist、变化提醒、动作建议明细）

当前主要短板不是“有没有页面”，而是 **实盘闭环、稳定性、可追溯性与可观测性**：  
缺少“从信号到执行再到复盘”的完整链路，以及跨市场统一口径与可验证的工程机制。

---

## 2. 核心缺口（按重要性分层）

### P0（必须先做：把系统变“可执行”）

#### 2.1 统一决策输出协议（DecisionPacket / TradeInstruction）
- **缺口**：页面与模块输出分散（信号、区间、风险、回测各一套），执行层缺少稳定输入契约，后续容易反复改接口。
- **要做**：定义统一决策包 `DecisionPacket`（可序列化、可审计、可复用），供：页面展示 / paper trading / 回测 / 复盘共用。
  - 必备字段（示例）：
    - `market, symbol, timeframe, horizon`
    - `action: long/short/wait`
    - `signal_strength, confidence_score, risk_level`
    - `entry, sl, tp1, tp2, rr`
    - `expected_edge (q50 - cost), edge_risk ((q50-cost)/(q90-q10))`
    - `start_window_top1 / P(W0..W3)`
    - `cost_assumption (bps, single/double, maker/taker)`
    - `valid_until, generated_at_market_tz, generated_at_utc`
    - `reasons[] (reason codes)`, `notes`
    - `model_version, data_version, config_hash, git_commit`
- **验收标准**：
  - 任意一个市场页顶部“决策卡”能输出一份完整 `DecisionPacket`
  - 执行/回测/复盘模块都只消费 `DecisionPacket`（单一真相）

---

#### 2.2 真实执行闭环（Paper Trading 最小闭环）
- **缺口**：目前主要是信号与回测，尚未形成稳定的 `信号 -> 下单 -> 持仓 -> 平仓 -> 复盘` 流程。
- **要做**：
  - 建立统一订单/成交/持仓模型：
    - `Order`：entry/exit/side/qty/type/status/created_at
    - `Fill`：price/qty/fee/slippage/time
    - `Position`：avg_price/pnl/unrealized/realized/status
  - 执行日志与成交回写（paper trading：模拟撮合与成本）
  - 异常处理：下单失败、重试、断线恢复、重复订单去重（idempotent）
- **验收标准**：
  - 任意一个市场可完成“模拟下单到平仓”的全流程记录
  - 每笔交易有唯一 ID + 全生命周期状态（created/open/partial/closed/canceled/failed）
  - 页面能展示：当前持仓、历史交易、执行失败原因

---

#### 2.3 Fail-safe（失效规则：数据/模型/市场）
- **缺口**：缺少“什么时候必须禁止交易/强制降级”的规则，容易在异常期输出误导性信号。
- **要做**：
  - **Data fail-safe**：缺失率超阈值、时间戳对齐失败、重复K线/跳点异常 → 禁止生成信号
  - **Model fail-safe**：健康状态=异常（Brier/Coverage/AUC明显劣化）→ 自动降级 baseline 或强制 `WAIT`
  - **Market fail-safe**：
    - A股：涨跌停/停牌/集合竞价/午休 → 禁止开仓或强制规则
    - 美股：财报/停牌/无法卖空 → 禁止或降仓
    - Crypto：极端点差/流动性枯竭 → 降仓或禁止
- **验收标准**：
  - 任意“禁止交易”必须输出 reason code（用于页面显示 & 审计）
  - 模型异常可自动降级（预测→baseline）并记录降级事件

---

#### 2.4 组合层风控（Portfolio Risk）
- **缺口**：当前偏单标的视角，缺少组合约束（真实场景里“组合风险预算”比单标更重要）。
- **要做**：
  - 总仓位上限、单标仓位上限
  - 同类资产暴露限制（行业/高相关标的/主题组）
  - 组合级止损与熔断（当回撤/波动超过阈值，禁止新增仓）
- **验收标准**：
  - 新开仓前能显示“组合剩余风险预算”
  - 超限时系统自动拒单/降仓，并输出原因

---

#### 2.5 成本与交易约束分市场建模（Execution Assumptions）
- **缺口**：成本模型还不够贴近不同市场规则，回测/信号的真实性不足。
- **要做**：
  - **Crypto（仅 Binance/Bybit，spot/perp）**：
    - maker/taker、perp/spot 分开
    - 可选：资金费率（perp）
  - **A股**：
    - long-only、交易时段、午休、涨跌停、停牌/复牌
  - **美股**：
    - 交易时段 & DST、卖空/借券约束（MVP 可先规则化）
    - 最小滑点/最小成交单位（简化）
- **验收标准**：
  - 回测与信号页都显示当前市场适用的成本/约束口径
  - 回测报告打印“执行假设摘要”（fee/slippage/delay/fill）

---

### P1（强烈建议：把系统变“可验证、可持续”）

#### 2.6 模型监控与漂移告警（Model Health）
- **缺口**：缺少“模型最近是否失效”的自动判断与可视化。
- **要做**：
  - 滚动监控：Brier / Coverage / AUC / Accuracy（按 hourly/daily 分开）
  - 漂移检测：近7天 vs 近30天（趋势性劣化识别）
  - 告警分级：正常 / 观察 / 异常（并产生告警记录）
- **验收标准**：
  - 页面可见“模型健康状态”（良/中/差 + 指标）
  - 指标异常自动记录并提示降级建议（或触发 fail-safe）

---

#### 2.7 概率校准与区间质量纠偏（Calibration）
- **缺口**：只监控指标不够，缺少“校准/纠偏”机制。
- **要做**：
  - 方向/启动窗口概率校准：Platt（sigmoid）/ Isotonic
  - 区间覆盖偏差纠正：coverage 目标 80%（±5%），必要时做简单校准（或 conformal 作为增强）
- **验收标准**：
  - 校准前/后 Brier 对比可见
  - 区间 coverage 长期维持在目标范围内

---

#### 2.8 回测一致性与防泄漏审计（Reproducible Backtest Audit）
- **缺口**：需要更系统地证明“没有未来函数、可复现”。
- **要做**：
  - walk-forward 审计模板（fold-by-fold、gap/purge）
  - 特征可用时点校验（timestamp alignment + no leakage）
  - 回测版本快照：配置/数据/模型版本（hash + 文件快照）
- **验收标准**：
  - 任意一份回测结果可追溯到 config/data/model 版本
  - 能复现同一版本下的同一回测结果（可比对 checksum）

---

#### 2.9 数据工程稳定性（ETL 任务编排 + 数据质量报告）
- **缺口**：缺自动补数、重试、任务编排的统一策略。
- **要做**：
  - 定时任务：拉数、特征、预测、回测、报告（cron/APScheduler，后续可 Airflow）
  - 失败重试与降级策略（fail-fast + retry + fallback）
  - 数据完整性检查：缺失率、历史长度、重复/跳点、交易日历对齐
  - 每日输出 snapshot（数据/预测/指标）
- **验收标准**：
  - 任务失败有明确告警并可自动重试
  - 每日可产出完整 snapshot + quality report

---

#### 2.10 Selection/Tracking 的去重与相关性控制（Screener Quality）
- **缺口**：Selection 页能筛，但缺“相关性去重/主题聚类”，容易出现同涨同跌一堆重复标的。
- **要做**：
  - correlation filter（rolling corr 30d > 0.8 视为同组）
  - 每组只保留 top edge_risk 的 N 个
  - 记录去重原因：`high_corr_group=X`
- **验收标准**：
  - Selection 页可解释：为何被筛掉/为何保留（reason code + group id）

---

### P2（增强项：工程成熟度 & 体验一致性）

#### 2.11 解释层统一化（UX Consistency）
- **缺口**：不同页面解释口径仍有分散，用户要重新理解。
- **要做**：
  - 统一顶部“决策卡”模板（LONG/SHORT/WAIT + Entry/SL/TP/RR + 强度/置信度/风险）
  - 统一术语解释（小白友好，tooltip + FAQ）
- **验收标准**：
  - 三个市场页信息结构一致、解释一致

---

#### 2.12 测试与发布流程（CI / Regression）
- **缺口**：缺系统化保障，越做越大容易引入回归问题。
- **要做**：
  - 单元测试：指标/信号/回测核心函数
  - 回归测试：关键页面数据断言（schema + 数值范围）
  - 发布前检查清单（data ok / model ok / health ok / calendar ok）
- **验收标准**：
  - 核心模块有基础测试覆盖
  - 发布前自动跑校验并阻断明显错误

---

#### 2.13 可观测性（Observability）
- **缺口**：告警只是结果，缺“日志/指标/追踪”体系支撑定位问题。
- **要做**：
  - Structured logs（JSON）
  - Metrics：任务成功率、耗时、缺失率、信号数量、拒单数量、降级次数
  - Trace：预测→信号→执行→回写全链路 `trace_id`
- **验收标准**：
  - 任意一笔交易能查到全链路 trace

---

#### 2.14 安全与密钥管理（Security Basics）
- **缺口**：即便先 paper trading，也要提前规划密钥与权限。
- **要做**：
  - API key 使用 `.env`/secrets，不入库
  - 权限最小化（只读/仅下单）
  - 发布前敏感信息扫描（pre-commit/CI）
- **验收标准**：
  - repo 中不包含敏感信息，发布流水线通过安全检查

---

## 3. 两周落地排期（W1–W2，可验收版本）

> 原则：**先小闭环（单市场跑通）→ 再扩市场**。  
> 建议先以 Crypto 或美股其中一个作为主战场跑通执行闭环，再推广到另外市场。

### W1：把系统变“可执行”（Paper Trading + 风控 + 成本口径）

**Day 1（可验收）**
- 定义 `DecisionPacket` schema + reason codes
- 页面顶部决策卡统一使用 DecisionPacket
- ✅ 输出一份可落地的 JSON（带版本信息）

**Day 2（可验收）**
- 订单/成交/持仓模型（Order/Fill/Position）
- paper trading 执行器 v1（delay + fixed bps 成本）
- ✅ 能产生一笔完整交易生命周期记录

**Day 3（可验收）**
- 组合风控 v1：总仓位/单标仓位/拒单原因
- ✅ 开仓前能返回 risk_check pass/fail + reason

**Day 4（可验收）**
- 分市场成本/约束 v1（crypto spot/perp、A股规则、美股时段）
- ✅ 回测与信号页能展示执行假设摘要

**Day 5（可验收）**
- 页面接入执行状态：持仓/历史交易/失败原因
- ✅ 前端可见“信号→执行→回写”闭环

**W1 交付物**
- Paper trading 最小闭环可跑
- 组合级风险控制生效
- 分市场成本与约束口径落地

---

### W2：把系统变“可验证、可持续”（健康监控 + 审计 + 自动化）

**Day 6（可验收）**
- 模型健康面板 v1（Brier/Coverage/AUC，滚动30天）
- ✅ health_grade（良/中/差）可见

**Day 7（可验收）**
- 漂移检测（7d vs 30d）+ 告警记录
- ✅ 异常自动标红 + 记录事件（可触发降级建议）

**Day 8（可验收）**
- 回测审计模板：walk-forward + gap/purge + no-leakage checks
- ✅ 任意结果可追溯 config/data/model hash

**Day 9（可验收）**
- 报告导出（策略表现 + 执行假设 + 版本信息）
- ✅ 一键导出报告（md/pdf/表格皆可）

**Day 10（可验收）**
- 自动任务编排：拉数→预测→回测→报告→告警
- ✅ 每日 snapshot + 数据质量报告稳定生成

**W2 交付物**
- 可靠性监控 + 告警
- 回测可追溯与基础自动化
- 最小运维闭环

---

## 4. 里程碑验收（可直接用于项目评审）

### M1（执行闭环）
- 能稳定产出 paper trading 交易记录（开仓/平仓/成本/原因）
- 页面能看到当前持仓、历史交易、执行失败原因
- 任意交易可追溯 `DecisionPacket`（版本信息齐全）

### M2（风险闭环）
- 新开仓前有风险预算检查
- 超限自动阻断并提示原因（reason code）
- 组合层风险预算可视化

### M3（可信闭环）
- 能展示近30天模型健康分数（Brier/Coverage/AUC）
- 回测结果可追溯到模型与数据版本（hash + 快照）
- 有防泄漏审计（walk-forward + gap/purge + timestamp check）

### M4（运维闭环）
- 每日自动运行任务（ETL/预测/回测/报告）
- 异常有告警与可恢复策略（retry + fallback）
- 可观测性基础可用（logs/metrics/trace）

---

## 5. 你现在最该先做的 5 件事（高性价比）

1. 定义并落地 `DecisionPacket`（统一决策输出，后续不返工）  
2. 跑通 paper trading 的交易流水（先选一个市场跑通）  
3. 上组合风控（总仓位/单标/相关性暴露）  
4. 三市场成本与交易约束分开建模 + 回测执行假设摘要  
5. 做回测版本追溯（model/data/config hash + 可复现）

---

## 6. 备注与建议

- 如果目标是“研究展示”，做到 **P1** 已经非常强（可解释 + 可验证 + 可追溯）。  
- 如果目标是“接近实盘”，**P0** 必须优先完成（执行闭环 + 风控 + fail-safe）。  
- 建议保持节奏：**先小闭环（单市场）→ 再扩市场**：  
  - 先 Crypto 或美股跑通完整链路，再推广到 A股。  
- 系统所有关键输出必须可追溯（版本/配置/数据），否则评审与复现会被质疑。  

---
