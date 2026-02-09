# 预测统一性与稳定性方案（Prediction Consistency Playbook）— 终稿版

> 目标：把“会变的预测”升级为“可交易、可解释、可追踪、可复现”的信号系统。  
> 适用：Crypto / A股 / 美股（Multi-Market 架构）。  
> 说明：预测不保证 100% 正确，本方案的目标是**减少抖动、提高一致性、降低错误成本**。  
> 时区约定：Crypto + 亚盘 = 北京时间（Asia/Shanghai）；美股 = 美东时间（America/New_York）。底层统一 UTC 存储。

---

## 1. 你遇到的问题（现象定义）

典型场景：  
- 08:00：日内、日线都偏空  
- 10:53：全部转为看涨  

这不一定意味着系统错了，更常见原因是：**新数据到来后模型再定价**。  
但如果产品没有统一规则，用户会感知为：**信号跳变、不可用、无法执行**。

---

## 2. 为什么会发生“同一天方向翻转”

### 2.1 数据层原因
- 价格源更新频率不同（ticker / 1m / 1h / 4h 不同步）
- 市场时区与交易时段对齐不一致（A股午休、美股 DST）
- 新闻/事件特征在盘中新增或突发，导致 gate/score 突变
- 数据延迟/缺失导致“短时误判”（stale / missing）

### 2.2 模型层原因
- 方向概率接近 50%（本来就不稳定）
- q50 幅度很小，轻微波动就会改符号（+0.01% ↔ -0.01%）
- 日内模型与日线模型目标不同（短期噪声 vs 中期结构）
- 概率未经校准 / 未平滑，导致对边界非常敏感

### 2.3 策略层原因
- 无“冻结窗口”（每次刷新都重算并立即替换）
- 无“迟滞阈值”（刚过阈值就翻多/翻空）
- 无“冷却时间/最小持有期”（刚翻就又翻回）
- 无“状态机”（只看到最终动作，看不到执行阶段）

---

## 3. 统一性的目标（不是永远不变）

我们要的是：
- **同一信号周期内可解释、可追踪**
- **允许变化，但变化有门槛、有记录**
- **执行动作稳定，减少抖动与噪声交易**

不是：
- 强行让预测一整天不变（会丢失有效信息）

---

## 4. 核心机制（必须同时具备）

> 核心组合：**快照锁定 + 重算触发 + 迟滞阈值 + 概率平滑 + 冷却期 + 多周期一致性 + 变更日志**  
> 目标：把“模型输出”变成“可执行信号”。

### 4.1 信号快照锁定（Signal Snapshot Lock）

每次生成执行信号时固定以下字段，**在 valid_until 之前不随页面刷新改变**：

- `signal_id`（唯一）
- `signal_time_utc`
- `signal_time_market`（Crypto/A股：BJ；美股：ET）
- `horizon`（默认 4h；日线可选 1d/3d/7d）
- `valid_until`（信号有效期，见 4.2）
- `entry_price`（固定，不随 current_price 漂移）
- `sl / tp1 / tp2 / rr`
- `p_up / q10 / q50 / q90`（以及 `edge_after_cost`）
- `consensus_state / consensus_score`
- `news_gate_pass / risk_gate_pass`（若启用）
- `model_version / data_version / config_hash / git_commit`

> 页面默认展示 Execution View（快照）  
> 同时可切换 Model View（实时最新概率，仅用于观察，不直接触发执行）。

---

### 4.2 信号周期与 valid_until 口径（必须写死）

默认规则：**信号只在“新 bar 收盘”时允许更新**，盘中 tick 刷新不改执行信号。

- Crypto 默认 `4h`：  
  - `valid_until = next_4h_bar_close`（推荐）  
  - 或 `signal_time + 4h`（可选但更粗）
- A股：按交易日历 + 午休规则生成 bar，`valid_until = next_bar_close`  
- 美股：按交易日历（含 DST），`valid_until = next_bar_close`

同时记录数据同步状态：
- `data_stale = (now - last_kline_close_ts) > stale_threshold`
- 若 `data_stale = True`：执行层强制降级 `WAIT_RISK_BLOCKED`（见 4.8）

---

### 4.3 允许重算的触发条件（Recompute Triggers）

盘中允许方向切换必须满足**至少一类硬触发**，否则不更新执行信号：

**触发 A：新 bar 收盘**
- `NEW_BAR_CLOSE`（最常见、最稳）

**触发 B：跨越幅度阈值（大变化）**
- `|p_up_new - p_up_old| >= delta_p_threshold`（建议 0.05）
- 或 `|edge_new - edge_old| >= delta_edge_threshold`（建议 3~5 bps）

**触发 C：成本后优势翻转且幅度足够**
- `edge_after_cost` 从负到正（或正到负），且  
  `|edge_after_cost| >= edge_min_threshold`

**触发 D：重大风险状态变化**
- `news_risk_level` 或 `risk_gate` 从 green→red / red→green

---

### 4.4 概率平滑（Anti-Noise Smoothing）

为减少 50% 附近抖动，引入平滑概率：

- `p_up_raw`：模型实时输出
- `p_up_smooth = EWMA(p_up_raw, span=3~6 bars)`（默认 span=4）

规则：
- **阈值判断使用 `p_up_smooth`**
- Model View 展示 raw + smooth；Execution View 只用 smooth

---

### 4.5 迟滞阈值（Hysteresis）防抖（执行层用）

建议默认（可配置）：

- 开多阈值：`p_up_smooth >= 0.56`
- 平多/观望阈值：`p_up_smooth <= 0.52`
- 开空阈值：`p_up_smooth <= 0.44`
- 平空/观望阈值：`p_up_smooth >= 0.48`

> 迟滞的意义：在 0.50 附近不来回翻转。

---

### 4.6 冷却期 & 最小持有期（Cooldown / Min-Hold）

用于防止“刚翻多→又翻空”的震荡噪声交易。

- `cooldown_bars`：动作翻转后，在 N 根 bar 内禁止反向再次翻转  
  - 建议：4h 分支 `cooldown_bars=1~2`；日线 `cooldown_bars=1`
- `min_hold_bars`：开仓后最少持有 N 根 bar 才允许策略主动反向  
  - 除非触发 SL/熔断/风控红线

---

### 4.7 多周期一致性门控（Multi-Horizon Consensus）

开仓前增加一致性检查，避免短期噪声与中期结构冲突导致频繁反复。

输出字段：
- `consensus_state`: `aligned` / `mixed` / `conflict`
- `consensus_score`: 0~100
- `conflict_reason`

推荐：加权一致性（比纯投票更稳）
- `score = w4h*sign(4h) + w1d*sign(1d) + w3d*sign(3d)`  
- 默认权重：`w4h > w1d > w3d`（例：0.5 / 0.3 / 0.2）
- 若 `consensus_state = conflict`：执行层降级为 `WAIT_NEUTRAL` 或降低仓位（视策略）

---

### 4.8 执行层状态机（Execution State Machine）

> 解决“到价但没反应”的关键：页面必须显示**执行状态**，而不是只显示预测数值。

建议枚举：

- `WAIT_NEUTRAL`：概率在中性区/无明显优势
- `WAIT_NO_EDGE`：成本后 edge ≤ 0
- `WAIT_NOT_TOUCHED`：未到入场价（entry_touched = False）
- `WAIT_RISK_BLOCKED`：风控/数据/模型健康拦截
- `READY_TO_LONG` / `READY_TO_SHORT`：可执行
- `IN_POSITION_LONG` / `IN_POSITION_SHORT`：持仓中
- `EXITED_TP` / `EXITED_SL` / `EXITED_TIME` / `EXITED_MANUAL`：已退出

必备字段：
- `entry_touched`（是否到达入场触发价）
- `gate_status`（总门控状态：pass/block）
- `blocked_reason`（若 block，原因码）

---

## 5. 变更日志与原因码（必须可追踪）

### 5.1 变更日志表（Change Log）

每次执行信号发生变化，记录：

- `change_time_utc`
- `market_time`
- `symbol`
- `old_action -> new_action`
- `old_state -> new_state`
- `reason_code`
- `key_metric_delta`（如 `p_up: 0.49→0.57, edge: -3.2bps→+5.1bps`）
- `snapshot_id_old / snapshot_id_new`

落盘建议：
- `data/processed/signals/signal_change_log.csv`

### 5.2 reason_code 枚举（机器可读）

建议标准化以下原因码：

- `NEW_BAR_CLOSE`
- `PUP_CROSSED_THRESHOLD`
- `EDGE_FLIPPED_AFTER_COST`
- `CONSENSUS_CONFLICT`
- `NEWS_RISK_CHANGED`
- `RISK_GATE_BLOCKED`
- `DATA_STALE`
- `DATA_QUALITY_FAIL`
- `MODEL_HEALTH_DEGRADED`
- `MANUAL_OVERRIDE`

---

## 6. 页面层必须可见的解释（避免“为什么变了”）

在决策卡（顶部）至少展示：

- **执行状态**（state）+ **最终动作**（action）
- `signal_id / signal_time / valid_until`
- `entry_price / entry_touched`
- `gate_status`（通过/拦截）+ `blocked_reason`（可展开解释）
- “上次信号”摘要：时间 + 动作 + 分数
- “本次变化原因”摘要（从 reason_code + key_metric_delta 生成中文解释）

新增“变更日志”模块：
- 最近 N 次变更（可下载 CSV）

---

## 7. 双视图输出（减少误解）

同页同时给两层：

1) **Model View（实时）**  
- 展示 `p_up_raw`、最新 q10/q50/q90、最新特征/事件状态  
- 用于观察“最新模型怎么看”

2) **Execution View（执行）**  
- 展示锁定快照（snapshot lock）  
- 只在触发条件满足时才更新  
- 用户下单只参考 Execution View

---

## 8. 你的案例怎么解释（08:00 空 → 10:53 多）

最常见链路：

1. 08:00：`p_up_smooth` 在弱空/中性附近（接近阈值）  
2. 09:00-10:53：新 bar 收盘 + 价格反弹 + 特征更新  
3. `p_up_smooth` 跨越开多阈值，且 `edge_after_cost` 从负转正  
4. 若系统没快照锁定/迟滞/冷却，就会直接显示“全面转多”  
5. 终稿方案下：你会看到  
   - 触发原因：`NEW_BAR_CLOSE` + `PUP_CROSSED_THRESHOLD`  
   - 执行层可能是 `READY_TO_LONG` 或 `WAIT_NO_EDGE`（取决于成本后 edge）

---

## 9. 落地实施（按优先级）

### P0（本周必须完成）
- [ ] 信号快照锁定：`signal_id + valid_until + entry_price 固化 + 版本哈希`
- [ ] 只允许新 bar 或硬触发才更新执行信号（Recompute Triggers）
- [ ] 迟滞阈值（Hysteresis）落地
- [ ] 概率平滑（EWMA）落地 seesion/branch 可配置
- [ ] 顶部决策卡展示：执行状态 + 变化原因 + 上次信号摘要
- [ ] 变更日志落盘（signal_change_log）

### P1（下周强烈建议）
- [ ] 多周期一致性：`consensus_state/score`
- [ ] 冷却期 & 最小持有期
- [ ] Model View / Execution View 切换
- [ ] 数据 stale / 质量 fail 统一闸门（DATA_STALE / DATA_QUALITY_FAIL）

### P2（后续增强）
- [ ] 在线校准与健康监控（Brier/Coverage/Calibration drift）
- [ ] 分市场自适应阈值（crypto/cn/us 独立）
- [ ] 根据波动 regime（趋势/震荡）自动调阈值与冷却期

---

## 10. 可量化验收指标（KPI）

改造后应看到：

- 同一信号周期内翻转次数下降 ≥ 30%
- “无原因翻转”比例接近 0（全部有 reason_code）
- Action churn rate（单位时间动作变化次数）显著下降
- Execution View 相对稳定，Model View 允许实时波动
- 用户对“为什么变化”的解释覆盖率 = 100%

---

## 11. 一句话结论

“预测会变”是正常的；  
“变化没规则、没解释、没执行分层”才是问题。  

用 **快照锁定 + 重算触发 + 概率平滑 + 迟滞阈值 + 冷却期 + 一致性门控 + 变更日志**，就能把“会变的模型”变成“可交易的系统”。
