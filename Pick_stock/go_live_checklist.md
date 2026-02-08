# Go-Live Checklist（交易系统上线前清单）【终稿版｜Paper Trading Only】

> **Go-Live 定义（本阶段）**：仅上线到 **Paper Trading（前瞻、真实时间顺序）**，不连接真实下单。  
> **目标**：把当前项目从“功能完整的研究平台”推进到“可交易、可风控、可复盘、可持续运行”的产品级系统（纸上交易版本）。  
> **适用范围**：Crypto / A股 / 美股，覆盖预测、策略、回测、执行（模拟）、监控全链路。  
> **时区约定**：底层统一 **UTC 存储**；Crypto & 亚洲时段显示 **北京时间 Asia/Shanghai**；美股显示 **美东时间 America/New_York**（注意 DST）。  
> **重要声明**：本系统用于研究与模拟执行，不构成投资建议。

---

## 0. 上线对象与上线边界（Scope Lock）

### 0.1 本次 Go-Live 包含
- 数据增量更新（按计划频率）
- 预测产出（方向/启动窗口/幅度区间）
- 信号生成与融合（输出单一最终动作）
- 回测（用于对照，不参与实时决策回填）
- **Paper Trading 前瞻执行**（真实时间顺序、不可回填）
- 监控与告警（数据/模型/执行）

### 0.2 本次 Go-Live 不包含
- 真实交易所下单
- 资金管理与实盘对账
- 自动化实盘风控（仅模拟风控与闸门）

---

## 1. 当前缺口（先对齐共识）

1. 还缺少 **前瞻验证闭环**（真实时间顺序 paper trading，禁止回填、禁止重算历史信号）。  
2. 成本模型仍偏静态（固定 bps），与真实成交差距较大。  
3. 风控多在单标的层，组合级暴露约束不足。  
4. 模型“健康度”有展示，但自动降级机制不够刚性（闸门不够硬）。  
5. 数据质量告警有基础能力，但缺统一 SLA 与故障降级策略。  
6. 多模型（policy/session/Lux MS）尚未形成统一融合决策层与回退策略。  
7. 回归测试和 CI 不够完整，后续改动有回归风险。  
8. **时区/交易日历对齐**（DST、A股午休、Crypto 24/7）若无硬测试，会导致回测/执行错位。  
9. 缺少 **Release Freeze**（冻结 data/config/model/code hash），上线后难以复现与审计。  
10. 缺少 **Kill Switch（紧急停止）**，无法在异常期快速阻断新交易。

---

## 2. 上线优先级（P0 / P1 / P2）

> 规则：**P0 未完成 = 不允许 Go-Live**（即便只是 paper trading 也不例外）。

---

## P0（必须先做，未完成不上线）

### P0-1 前瞻 Paper Trading（30–60 天）
- **要求**
  - 按真实时间滚动执行（例如每小时/每日固定时刻触发）
  - 禁止回填、禁止重算历史信号（信号生成当下即冻结）
  - 每次运行必须产出“本次决策包 + 本次执行结果 + 本次持仓状态”
- **产物**
  - `paper_orders.csv`
  - `paper_positions.csv`
  - `paper_fills.csv`（可选但强烈建议；当前已存在）
  - `paper_daily_pnl.csv`（**P0-1 新增产物**，当前代码未稳定产出）
  - `paper_run_log.jsonl`（每次运行一条记录，含 trace_id）
- **通过标准**
  - 最小样本：交易次数 ≥ 100（或按策略设定最低交易频率）
  - 成本后核心指标不劣化到不可接受（见第 4 节阈值）
  - 执行失败率 < 1%（含异常自动恢复）

---

### P0-2 执行成本真实化（动态 fee/slippage/impact + 延迟）
- **要求**
  - 支持**双边成本**（开仓/平仓）
  - 滑点与冲击：随波动/成交量/点差动态变化（MVP 可先用分段函数）
  - 可配置执行延迟（bars）：信号在 t 生成，**最早 t+1 执行**
  - 分市场成本模板：Crypto / A股 / 美股分别定义
- **产物**
  - 回测与执行日志中必须记录成本分解：
    - `fee_bps`, `slippage_bps`, `impact_bps`, `delay_bars`
  - `cost_profile_snapshot.yaml`
- **通过标准**
  - 固定成本 vs 动态成本差异可解释
  - 策略/信号排名在小幅成本扰动下保持相对稳定（鲁棒性可接受）

---

### P0-3 撮合/成交模型契约（Fill Model Contract）
- **要求**
  - 明确成交价取法（可配置）：
    - `next_open` / `vwap` / `mid` ± `slippage`
  - 回测与 paper trading 必须使用同一套撮合配置（避免线上线下不一致）
- **产物**
  - `fill_model_spec.yaml`
  - `execution_assumptions.md`（自动输出到报告）
- **通过标准**
  - 任意一次 paper trading 运行可复现成交假设（可审计）

---

### P0-4 硬性风控闸门（交易前 Gate，Fail-safe）
- **要求**
  - 任一红色告警触发时，信号必须强制降级为 `WAIT`，且不得开新仓
  - Gate 覆盖三类异常：
    - **Data Gate**：缺失率/时间戳错位/重复/延迟超 SLA
    - **Model Gate**：健康度异常（Brier/Coverage/AUC劣化）
    - **Risk Gate**：组合风险超限（仓位/暴露/熔断）
- **产物**
  - 决策包字段：`gate_status`, `blocked_reason[]`, `health_grade`, `risk_budget_left`
  - `gates_audit_log.jsonl`
- **通过标准**
  - 任一红色告警触发时，系统不会开新仓
  - 所有阻断都输出 reason code（可追溯）

---

### P0-5 Release Freeze（上线冻结与复现）
- **要求**
  - 每次 Go-Live 评审必须冻结并记录：
    - `data_hash`, `universe_version`, `config_hash`, `model_version`, `git_commit`
  - 任何一次 `release_signoff.md` 必须可复现
- **产物**
  - `release_manifest.json`（模板见附录）
  - `release_signoff.md`（自动生成）
- **通过标准**
  - 给定 `release_manifest.json`，可 100% 复现当次评审结果与关键指标

---

### P0-6 交易日历/时区对齐硬测试（Calendar Alignment Tests）
- **要求（必须自动测试）**
  - Crypto：24/7 连续索引完整，无缺口伪造
  - A股：午休不生成假 bar；节假日不伪造交易日
  - 美股：DST 周不重复/不丢 bar；交易日历对齐
- **产物**
  - `calendar_alignment_report.json`
  - 单元测试：`test_calendar_alignment.py`
- **通过标准**
  - 测试全绿；否则 Gate=BLOCK

---

### P0-7 Kill Switch（紧急停止开新仓）
- **要求**
  - 一键停机：`DISABLE_TRADING=true` 或系统开关
  - 触发条件（至少包含）：
    - 数据红警 / 模型红警 / 执行失败率超阈值 / 回撤超阈值
  - **恢复条件（必须显式）**：
    - 仅 `ops_admin` 角色可解除
    - 最近连续 `N=3` 次健康检查全部通过（Data/Model/Risk）
    - 解除操作必须写审计日志并附原因
- **产物**
  - `kill_switch_events.jsonl`
  - `kill_switch_recovery_log.jsonl`
- **通过标准**
  - 触发后不再产生新订单，且原因可追溯
  - 解除后首个执行窗口仅允许小仓位试运行（建议 0.25x）

---

### P0-8 Go-Live 阈值自动判定（自动签核）
- **要求**
  - 统一读取 `configs/go_live_thresholds.yaml` 做 pass/fail 判断
  - 自动生成 `release_signoff.md`（含原因与建议）
- **产物**
  - `release_signoff.md`
  - `threshold_report.csv`
- **通过标准**
  - 全局规则 + 市场规则全部通过（无红项）

---

## P1（强烈建议：Go-Live 后 1–2 周内完成）

1. **组合级风控（Portfolio Risk）**
   - 总仓位、单标风险、相关性上限、同主题暴露、单日亏损熔断。
2. **模型可信度面板（滚动）**
   - Brier、Coverage、Calibration 漂移，7D vs 30D 对比；并联动 Gate。
3. **信号融合层（统一最终动作）**
   - policy + session forecast + Lux MS 统一投票/加权；
   - 必须有回退策略（融合层不可用 → baseline 或 WAIT）。
4. **故障降级策略（SLA + Fallback）**
   - 数据源超时退避、主备切换、缺字段回退逻辑、告警分级。
5. **容量/流动性约束（Capacity）**
   - `max_position_notional <= k * ADV`（或小时成交量），并将 impact 纳入滑点。

---

## P2（增强项：机构化能力）

1. 事件风险（财报/宏观/政策）注入到仓位与风控。  
2. 组合优化（风险预算、相关性约束、行业中性）。  
3. 自动化参数巡检与模型退休机制（含回滚策略）。  
4. 更完善 CI/CD（回归测试、数据契约测试、性能基准测试）。  

---

## 3. 指标门槛（建议版｜以配置为准）

> 以 `configs/go_live_thresholds.yaml` 为准；下面是执行时必须展示的最小集合。  
> 建议同时加入“稳定性门槛”（fold 方差、最小交易数），避免偶然表现。

### 3.1 全局（必须）
- `data_pass_rate >= 0.98`
- `drift_red_count == 0`
- `execution_fail_rate < 0.01`
- `calendar_alignment_pass == true`
- `min_trades_in_window` 按市场阈值：
  - `crypto >= 120`
  - `cn_equity >= 60`
  - `us_equity >= 80`
- `fold_variance` 使用明确阈值：
  - `sharpe_std <= 0.35`
  - `total_return_std <= 0.08`
  - `max_drawdown_std <= 0.05`

### 3.2 Crypto（示例）
- `sharpe >= 0.8`
- `max_drawdown >= -0.15`
- `profit_factor >= 1.05`

### 3.3 A股（示例）
- `win_rate >= 0.50`
- `max_drawdown >= -0.12`
- `profit_factor >= 1.00`

### 3.4 美股（示例）
- `sharpe >= 0.8`
- `max_drawdown >= -0.15`
- `profit_factor >= 1.10`

---

## 4. 最终上线判定（必须全部满足｜Paper Trading）

1. **P0 项全部完成**。  
2. **Go-Live 阈值全部通过**（无红项）。  
3. 最近一个前瞻窗口（至少 30 天）中：
   - 成本后收益为正（或达到策略定义目标）
   - 最大回撤在阈值内
   - 执行失败率 < 1%
4. 有可回滚方案：
   - 模型版本、配置版本、数据版本可追溯
   - 一键切回上一个稳定版本（paper trading 规则回滚）
5. Kill Switch 验证通过（触发后禁止新开仓）

---

## 5. 建议的两周落地节奏（面向 P0）

### 第 1 周（P0 核心链路）
- 完成：P0-1 前瞻执行 + P0-2 动态成本 + P0-3 撮合契约  
- 同步：P0-6 日历/时区硬测试（越早越好）

### 第 2 周（上线评审与自动签核）
- 完成：P0-4 硬闸门 + P0-5 Release Freeze + P0-7 Kill Switch + P0-8 自动 signoff
- 周末验收：生成 `release_signoff.md`，开一次“是否可 Go-Live（paper trading）”评审

---

## 6. 交付物清单（上线前必须存在）

1. `data/processed/backtest/metrics_summary.csv`
2. `data/processed/execution/paper_orders.csv`
3. `data/processed/execution/paper_positions.csv`
4. `data/processed/execution/paper_fills.csv`（当前已存在）
5. `data/processed/execution/paper_daily_pnl.csv`（P0-1 新增）
6. `data/processed/drift_monitor_daily.csv`（与当前实现对齐）
7. `data/processed/calendar_alignment_report.json`（P0-6 新增）
8. `release_manifest.json`
9. `release_signoff.md`（含 pass/fail 明细）
10. 当前线上版本的：
   - `model_version`
   - `config_hash`
   - `data_hash`
   - `git_commit`

---

## 附录 A：release_manifest.json（建议模板）

```json
{
  "release_name": "paper_go_live_2026-02-07",
  "scope": "paper_trading_only",
  "generated_at_utc": "2026-02-07T18:00:00Z",
  "markets": ["crypto", "ashares", "us_equities"],
  "timezones": {
    "storage": "UTC",
    "crypto_display": "Asia/Shanghai",
    "ashares_display": "Asia/Shanghai",
    "us_display": "America/New_York"
  },
  "versions": {
    "git_commit": "abc123",
    "config_hash": "sha256:...",
    "data_hash": "sha256:...",
    "model_version": "2026-02-07_crypto_hourly_v3",
    "universe_version": "sp500_2026-01-01"
  },
  "assumptions": {
    "delay_bars": 1,
    "fill_price_mode": "next_open",
    "cost_profile": "dynamic_v1"
  }
}
