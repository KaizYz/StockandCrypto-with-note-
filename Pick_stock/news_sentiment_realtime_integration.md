# 实时新闻情绪流接入方案（Crypto / A股 / 美股）【终稿版】

> 目标：把“新闻/事件”从手工判断升级为 **可量化、可回测、可监控、可复盘** 的实时特征流，并在 **Paper Trading** 中验证其增益与风险控制价值。  
> 适用：Crypto / A股 / 美股（Multi-Market 架构）。  
> 时区：底层统一 **UTC 存储**；Crypto & 亚洲时段显示 **北京时间 Asia/Shanghai**；美股显示 **美东 America/New_York（DST）**。  
> 声明：本方案用于研究与模拟执行，不构成投资建议。

---

## 1. 目标（What & Why）

把新闻情绪做成“可工程化特征流”，让系统能回答三件事：

1. **现在有没有新闻风险（Event Risk）？**
2. **新闻偏利好还是利空（Direction Bias）？**
3. **新闻是否足以改变当前做多/做空/观望决策（Decision Impact）？**

最终落地形态：  
- 新闻流 → 情绪特征 → **Gate（门控）** / **Position Adjust（仓位调节，后续）** → 回测对照 → Paper Trading 前瞻验证 → 监控告警。

---

## 2. 接入范围与优先级（Scope & Phases）

### P0（先做｜上线到 Paper Trading）
- 市场：**Crypto + 美股**
- 数据源（优先级）：
  1) Alpha Vantage `NEWS_SENTIMENT`（接入快，统一英文）
  2) NewsAPI（补充覆盖）
  3)（可选增强）官方/权威源 RSS：交易所公告/SEC/主流媒体（白名单）
- 输出（必须）：
  - 标准化新闻表（去重后）
  - 分钟级/小时级情绪聚合特征
  - 策略可用字段：
    - `news_score_30m / 2h / 24h`
    - `news_count_30m / 2h / 24h`
    - `news_burst_zscore`
    - `news_pos_neg_ratio`
    - `news_risk_level`, `news_gate_pass`, `news_reason_codes[]`

### P1（第二阶段）
- 加入：A股中文源（TuShare 新闻/公告）
- 中文情绪：本地模型或 API（先可解释版本）
- 行业/主题映射（半导体、AI、地产等）

### P2（增强）
- 多源置信度融合（源权重动态学习）
- 事件类型识别（财报、监管、黑天鹅、并购）
- 新闻到价格冲击的因果回归（事件 alpha）
- 新闻驱动的仓位调节（从 gate 升级为 size 调节）

---

## 3. 系统架构（Engineering）

### 3.1 模块划分
1) `src/ingestion/news_sentiment.py`
   - 拉取多源新闻（带限流/缓存/断点续拉）
   - 标准化字段、语言标注
   - symbol/实体映射（ticker/entity disambiguation）
   - 去重（URL + 相似去重）
   - 写入 `news_raw.parquet`

2) `src/features/news_features.py`
   - 计算滚动情绪特征（30m / 2h / 24h）
   - 计算新闻突发指数（burst）
   - 输出 `news_features.parquet`

3) `src/features/build_features.py`
   - 将新闻特征 merge 到现有特征管道（按 `available_at_utc` 对齐）
   - 确保无未来函数（只用 decision_time 之前可见的新闻）

4) `src/models/generate_policy_signals.py`
   - 在策略决策层接入新闻门控（news risk gate）
   - 输出标准 `DecisionPacket.news` 字段（用于复盘与前端展示）

5) `dashboard/app.py`
   - 展示：新闻分值、突发状态、关键新闻列表、门控结果与解释

### 3.2 拉取策略与限流（P0 必做）
- **限流**：为每个 provider 设置 `rate_limit_per_minute / daily_quota`
- **缓存**：
  - 若 provider 支持 `ETag/Last-Modified` → 使用
  - 否则基于 `dedup_key` 与 `last_ingested_at_utc` 增量拉取
- **断点续拉**：记录 provider cursor 或 `since_time`
- **重试策略**：指数退避 + 最大重试次数 + 熔断（连续失败暂停 X 分钟）
- **降级**：源不可用 → `news_unavailable=true`，策略回退为“无新闻版本”

---

## 4. 数据规范（统一 Schema + 存储分区）

### 4.1 统一字段（Raw）
- `published_at_utc`（新闻发布时间，若源缺失则置空并标记）
- `ingested_at_utc`（系统抓取时间）
- `available_at_utc`（**可用时间**：`max(published_at_utc, ingested_at_utc)`，回测/线上统一使用）
- `provider`（alpha_vantage/newsapi/tushare/...）
- `market`（crypto/cn_equity/us_equity）
- `symbol`（BTCUSDT/AAPL/600000 等，统一规范）
- `entity_name`（可选：公司/币名）
- `title`
- `summary`（**短摘要**，建议限制长度，避免版权风险）
- `url`
- `language`
- `raw_sentiment`（统一后情绪，范围 [-1,1]）
- `sentiment_confidence`（来源或模型置信度，可选）
- `relevance`（与 symbol 相关性，0~1）
- `source_tier`（tier1/tier2/tier3）
- `source_weight`（来源可信度权重）
- `match_type`（provider_ticker | regex | ner | manual_map）
- `dedup_key_url`（hash(url)）
- `dedup_key_simhash`（simhash(title+summary)）
- `dedup_group_id`（去重归并后的组 id）

> **合规建议**：不落正文，只落 `title + short_summary + url`。

### 4.2 存储位置与分区（建议）
- Raw：
  - `data/processed/news/news_raw.parquet`
  - 分区：`dt=YYYY-MM-DD/provider/market`
- Features：
  - `data/processed/news/news_features.parquet`
  - 分区：`dt=YYYY-MM-DD/market/symbol`

---

## 5. Symbol 映射与实体消歧（P0 必做）

> 新闻“归属哪个标的”是决定特征质量的核心。

### 5.1 映射优先级
1) provider 自带 ticker 映射（若有）→ `match_type=provider_ticker`
2) 规则映射（关键词/正则）→ `match_type=regex`
3) NER（可选）→ `match_type=ner`
4) 手工白名单映射（兜底）→ `match_type=manual_map`

### 5.2 Crypto 规范（仅 Binance/Bybit）
- 统一为交易对：如 `BTCUSDT`, `ETHUSDT`, `SOLUSDT`
- 处理别名：`Bitcoin/BTC` → `BTCUSDT`
- 对 spot/perp：可在下游用 `venue_type` 区分，不在新闻层重复写

### 5.3 审计要求
- 每条新闻必须能解释“为什么归到这个 symbol”（match_type + evidence）

---

## 6. 去重策略（Dedup）【必须加强】

> 解决同稿多源、标题轻微改写、转载重复导致“情绪被放大”。

### 6.1 多级去重规则
- Level 1：URL 去重（最强）
  - `dedup_key_url = hash(url)`
- Level 2：相似去重（标题+摘要）
  - `dedup_key_simhash = simhash(normalize(title+summary))`
  - 相似阈值：例如 Hamming Distance ≤ 3（可配置）
- Level 3：兜底（标题归一化+日期）
  - `hash(normalize(title)+provider+dt)`

### 6.2 输出
- `dedup_group_id`：同一新闻组归并后共享 group_id
- 聚合时按 group_id 只计一次（避免重复放大）

---

## 7. 情绪评分方法（可解释版本：先稳再强）

### 7.1 分值标准化
- 统一到 `[-1, 1]`
  - 强利好：`+1`
  - 中性：`0`
  - 强利空：`-1`

### 7.2 时间衰减加权（核心）
对每条新闻计算权重：
- `w = relevance * source_weight * sentiment_confidence * exp(-age_minutes / tau)`
- `tau` 默认 `360`（6小时，可配置）

聚合分数（窗口内）：
- `news_score = sum(raw_sentiment * w) / sum(w)`

### 7.3 关键特征（P0 必出）
- 分数：
  - `news_score_30m`
  - `news_score_2h`
  - `news_score_24h`
- 计数：
  - `news_count_30m`
  - `news_count_2h`
  - `news_count_24h`
- 突发：
  - `news_burst_zscore`（基于计数的 z-score）
- 结构：
  - `news_pos_neg_ratio`
  - `news_conflict_score`（同窗正负冲突程度；冲突高→风险升高）

---

## 8. 策略接入规则（可直接落地：Gate）

> P0 只做 **门控/降级**，不做复杂仓位调节（避免范围爆炸）。

### 8.1 Gate 规则（示例，可配置）
1) 做多信号降级（利空突发）
- 条件：
  - `news_score_2h < -0.25` 且 `news_burst_zscore > 1.5`
- 动作：
  - `LONG -> WATCH/WAIT`（或降仓到 0）

2) 做空信号降级（利好突发）
- 条件：
  - `news_score_2h > +0.25` 且 `news_burst_zscore > 1.5`
- 动作：
  - `SHORT -> WATCH/WAIT`（或降仓到 0）

3) 事件风险提示（短期极端）
- 条件：
  - `abs(news_score_30m) > 0.4` 且 `news_count_30m >= 3`
- 标记：
  - `event_risk = True`

### 8.2 输出字段（必须写入 DecisionPacket）
新增标准结构（建议固定 schema）：

- `news_gate_pass: bool`
- `news_risk_level: low/medium/high`
- `news_scores: {score_30m, score_2h, score_24h}`
- `news_counts: {count_30m, count_2h, count_24h}`
- `news_burst_zscore`
- `event_risk: bool`
- `news_reason_codes[]`（例如：`NEG_BURST`, `POS_BURST`, `NEWS_UNAVAILABLE`）
- `top_headlines[]`（最多 3 条：title/provider/available_at/url）

---

## 9. 仪表盘展示建议（可解释 + 可筛选）

### 9.1 市场页面“新闻模块”
- 当前新闻情绪：
  - `news_score_2h`、`news_score_24h`
- 突发状态：
  - `normal / burst`
- 最近 3 条关键新闻：
  - 标题 + 时间 + 来源 + 链接
- 决策影响说明：
  - “新闻门控通过：是/否”
  - “原因：reason codes + 可读描述”

### 9.2 Selection / Tracking 页
- 新增列：`news_risk_level`
- 筛选器：
  - `仅看 news_gate_pass=True`
  - `仅看 event_risk=True`

---

## 10. 回测与验证（必须做：对照 + 稳健性）

### 10.1 三组核心对照（必须）
1) 基线策略（无新闻）
2) 新闻只做过滤（gate）
3) 新闻过滤 + 仓位调整（P1 再做；P0 可先留参数位）

### 10.2 增强验证（强烈建议）
- Regime 分层：趋势/震荡/高波动分别评估（新闻 gate 是否只在某一类行情有效）
- Ablation：仅 burst、仅 score、score+burst 三种组合对比

### 10.3 指标（必须）
- 年化收益、Sharpe、最大回撤
- 胜率、Profit Factor
- 交易次数变化
- `exposure_time`（市场暴露时间占比，防“靠少交易赢”）
- `turnover`（换手，影响成本）

### 10.4 验收标准（保守）
- Sharpe 不下降（或下降可解释且回撤显著改善）
- 最大回撤改善 ≥ 5%
- 交易次数不异常骤降（避免“过拟合过滤”）

---

## 11. 风险控制与防泄漏（Hard Rules）

1) 严禁使用 `decision_time` 之后才 **可见** 的新闻  
   - 回测/线上统一使用：`available_at_utc <= decision_time`
2) 同一新闻去重必须开启（避免放大）
3) 数据源异常必须降级：
   - `news_unavailable=True`
   - 策略回退到“无新闻版本”（且记录 reason）
4) 监控数据延迟：
   - 输出 `availability_lag_seconds`，超过 SLA 触发 `Data Gate`

### 11.1 迟到数据（Late Arrival）处理（新增）

- 新增配置：
  - `lateness_grace_minutes`（例：30）
  - `allow_late_backfill`（true/false）
- 统一规则：
  - 线上决策只用当下可见数据，不因迟到新闻重写已发出的信号。
  - 回测使用同一 `available_at_utc` 规则，保证线上/回测一致。
- 审计字段：
  - `is_late_arrival`
  - `late_minutes`

### 11.2 幂等与重跑规则（新增）

- ingestion 必须幂等（idempotent）：
  - 同一批次重跑不重复入库、不重复计数。
- 新增审计字段：
  - `run_id`
  - `source_batch_id`
  - `upsert_key`
- 建议写入方式：
  - 先去重，再 upsert（按 `dedup_group_id + symbol + available_at_utc`）

### 11.3 无新闻默认行为（新增）

- 必须区分两种状态：
  - `neutral_no_news`：窗口内确实无新闻（`count=0`）
  - `unavailable`：数据源异常/延迟超标（不可用）
- 建议默认：
  - `neutral_no_news`：`news_score=0`，`news_gate_pass=true`
  - `unavailable`：`news_score=NaN`，回退无新闻策略并打标 `NEWS_UNAVAILABLE`

### 11.4 事件日历通道（新增）

- 新闻情绪与硬事件分离建模，新增：
  - `calendar_event_risk`（财报/CPI/FOMC/政策窗口）
  - `news_event_risk`（新闻突发驱动）
- 策略门控建议：
  - 任一风险为高时，降仓或观望。
  - 在前端分别展示“新闻风险”与“日历风险”。

---

## 12. 工程落地任务清单（按顺序）

1. 新增 `src/ingestion/news_sentiment.py`（含限流/缓存/断点续拉）
2. 新增 `src/ingestion/symbol_mapper.py`（entity/ticker 映射）
3. 新增 `src/ingestion/dedup.py`（URL + simhash 去重）
4. 新增 `src/features/news_features.py`（窗口聚合 + burst + conflict）
5. 在 `src/features/build_features.py` 合并新闻特征（按 `available_at_utc` 对齐）
6. 在 `src/models/generate_policy_signals.py` 加门控规则（DecisionPacket.news）
7. 在 `dashboard/app.py` 加新闻模块（分数 + burst + headlines + gate）
8. 新增回测参数：`--use-news-gate`
9. 新增测试：
   - 时间边界测试（防泄漏：available_at）
   - 去重测试（同稿多源不放大）
   - 映射测试（symbol 归属可解释）
   - 缺失降级测试（news_unavailable 回退）
   - 迟到数据测试（late arrival 不改写已决策）
   - 幂等测试（同 run 重跑不重复计数）

---

## 13. 配置样例（加入 `configs/config.yaml`）

```yaml
news:
  enabled: true
  providers:
    - alpha_vantage
    - newsapi

  # ingestion
  rate_limit:
    alpha_vantage_per_min: 5
    newsapi_per_min: 30
  retry:
    max_retries: 5
    backoff_seconds: 5
    circuit_breaker_minutes: 10
  lateness_grace_minutes: 30
  allow_late_backfill: false

  # mapping / dedup
  symbol_map_file: configs/news_symbol_map.yaml
  dedup:
    simhash_distance: 3

  # scoring
  tau_minutes: 360
  use_sentiment_confidence: true
  source_tier_weights:
    tier1: 1.0
    tier2: 0.7
    tier3: 0.4
  defaults:
    neutral_score_when_no_news: 0.0
    mark_unavailable_as_nan: true

  # gate
  gate:
    neg_threshold: -0.25
    pos_threshold: 0.25
    burst_zscore_threshold: 1.5
    event_abs_score_30m: 0.4
    event_min_count_30m: 3

  fallback_on_missing: true
  sla:
    max_data_lag_seconds: 900   # 15 min

calendar_events:
  enabled: true
  us_equity:
    earnings: true
    macro: true
  cn_equity:
    macro: true
  crypto:
    macro: true
```

---

## 14. 上线监控 KPI（新增）

P0 阶段建议至少日监控以下指标：

- 新闻覆盖率：`symbols_with_news / symbols_total`
- 平均延迟：`avg(availability_lag_seconds)` + P95
- gate 触发率：`gate_blocked / total_signals`
- 门控增益：开启 gate 与关闭 gate 的收益/回撤差
- 不可用率：`news_unavailable=true` 占比
- 映射错误率：人工抽检失败占比

告警建议：

- 覆盖率连续 3 天低于阈值（如 40%）报警
- 延迟 P95 超 SLA 连续 1 小时报警
- gate 触发率异常跳变（如 >2 倍）报警

---

## 15. 下一步建议（落地路线）

先用 P0 版本上线到 Paper Trading（仅 Crypto + 美股），稳定跑 2-4 周。

观察重点：

- 新闻 gate 是否降低回撤/避免踩雷
- 是否导致交易次数异常降低（过度过滤）
- 是否存在映射/去重造成的情绪放大或错配
- `unavailable` 和 `neutral_no_news` 是否被前端正确区分

通过后再进入 P1：

- 加 A股中文源 + 主题映射
- 引入仓位调节（size）替代纯 gate
