# CS 487 Senior Project 完整方案（Detailed Plan · v2 工程落地版）
**项目名称（建议）**  
**BTC 市场趋势与波动预测系统：小时级 + 日线级的方向、启动时间与幅度区间联合预测**

---

## 0. 项目一句话定义（Scope Lock）
构建一个端到端预测系统，面向 BTC（后续可扩展到美股等），在 **小时级（1–4h）** 与 **日线级（1–7d）** 两个时间尺度上，输出：
- **方向概率**（涨/跌）
- **启动时间窗口**（什么时候开始明显上涨/下跌）
- **幅度/目标区间**（能涨到多少/跌到多少，给区间而不是单点）
并通过 **Dashboard** 展示预测结果与严格的 **walk-forward** 评估表现（不进行真实交易执行）。

---

## 1. 项目背景与动机
加密市场高波动、快节奏。仅预测“明天涨还是跌”不足以支持分析与决策。交易者/分析者更在意：
- **未来会涨还是跌（Direction）**
- **什么时候开始涨/跌（Start Time）**
- **大概能涨到多少 / 跌到多少（Magnitude / Target Range）**

因此，本项目目标是开发一个可复现、可评估的预测系统，在多时间尺度上输出“方向 + 启动 + 幅度区间”，并提供工程化实现与可视化展示。  
> **免责声明**：本项目仅用于研究与分析，不构成投资建议，不执行自动交易。

---

## 2. 项目目标（What I will deliver）

### 2.1 核心预测能力（Two Timeframes）
#### A. 小时级（1H 数据）
- **预测 horizon**：`1h / 2h / 4h`
- **输出**：
  - 方向：`P(up)`、`P(down)`
  - 启动窗口：`0–1h / 1–2h / 2–4h`（输出 Top-1 或每窗概率）
  - 幅度：未来收益率的 `q10 / q50 / q90`（区间预测），可选 high/low 版本

#### B. 日线级（1D 数据）
- **预测 horizon**：`1d / 3d / 7d`（可选 30d）
- **输出**：
  - 方向：`P(up)`、`P(down)`
  - 启动窗口：`0–1d / 1–3d / 3–7d`
  - 幅度：未来收益率的 `q10 / q50 / q90`（区间预测）

### 2.2 系统交付（End-to-End Deliverables）
- 数据收集与增量更新脚本（可复现）
- 特征工程模块（可复现）
- 标签生成模块（方向 / 启动 / 幅度区间）
- 模型训练、调参、评估（walk-forward）、保存与加载
- 模型对比报告（Baseline / MVP / Advanced）
- Dashboard（展示预测、区间、指标、baseline 对比）
- （加分）轻量回测：加入手续费/滑点/延迟，检验预测是否有边际价值

---

## 3. 任务定义（把“想法”变成可训练目标）
本项目拆成 **三个任务**（在 Hourly 与 Daily 分支都做同样定义，只是时间单位不同）：

### 3.1 任务 1：方向预测（Direction Classification）
对每个 horizon `h`，定义未来收益率：
- `r(t,h) = (Close[t+h] - Close[t]) / Close[t]`
- `y_dir(t,h) = 1 if r(t,h) > 0 else 0`

**输出**：`P(up)` 与 `P(down)=1-P(up)`  
**horizon**：  
- Hourly：`{1h,2h,4h}`  
- Daily：`{1d,3d,7d}`

---

### 3.2 任务 2：趋势启动时间预测（Start Time Window Prediction）
目标是预测“**显著走势**”更可能在哪个时间窗启动。

#### 3.2.1 启动事件定义（数据驱动阈值）
定义“显著启动”触发条件：在未来的累计收益率首次达到阈值。
- 先计算未来 k 步的累计收益率 `r(t,k)`
- 定义阈值 `thr`（不要拍脑袋，使用历史分位数自动设定）：
  - Hourly：过去一年（或可用数据范围）1H `|r|` 的 `80% 分位数` → `thr_h`
  - Daily：过去五年（或可用范围）1D `|r|` 的 `80% 分位数` → `thr_d`
- 启动时刻：
  - `τ = min k such that |r(t,k)| >= thr`
  - 若在目标 horizon 内不存在该 k，则标记为 `no_start`

> 可选方案：用未来 realized volatility 触发（更稳但实现略复杂），本方案 MVP 用收益率阈值即可。

#### 3.2.2 时间窗划分
**Hourly 窗口**
- `W0: no_start`（在 4h 目标窗口内未触发）
- `W1: 0–1h`
- `W2: 1–2h`
- `W3: 2–4h`

**Daily 窗口**
- `W0: no_start`（在 7d 目标窗口内未触发）
- `W1: 0–1d`
- `W2: 1–3d`
- `W3: 3–7d`

**输出形式（MVP 推荐）**
- 多分类：输出最可能窗口 `argmax P(Wi)`  
- 或输出每个窗口概率 `P(W0),P(W1),P(W2),P(W3)`

---

### 3.3 任务 3：幅度/目标区间预测（Magnitude + Interval）
你要的是“涨到/跌到多少”，更适合用**区间预测**。

#### 3.3.1 MVP：收益率区间预测（Quantile Regression）
- 目标：`y_ret(t,h) = r(t,h)`
- 输出：`q10 / q50 / q90` 三个分位数  
示例输出：未来 `4h` 收益率中位数 `+0.6%`，区间 `[-0.2%, +1.4%]`

**训练方式（必须写清楚）**
- 对每个分位数 `q`，训练一个 Quantile Model（LightGBM quantile objective）
- 使用 **Pinball Loss（分位数损失）**：
  - `L_q(y,ŷ) = max(q*(y-ŷ), (q-1)*(y-ŷ))`

**区间评估指标（必须补充）**
- **Coverage**：实际值落在 `[q10,q90]` 的比例（目标接近 80%）
- **Interval Width**：区间宽度（越窄越好，但不能牺牲 coverage）
- （可选）Winkler Score / Pinball Loss 汇总

#### 3.3.2 增强：预测未来窗口 High/Low（更贴近“目标价”）
- `y_high(t,h) = (max_high_in_next_h - Close[t]) / Close[t]`
- `y_low(t,h)  = (min_low_in_next_h  - Close[t]) / Close[t]`
输出最大上冲与最大回撤，适合作为增强项。

---

## 4. 数据方案（Data）

### 4.1 数据粒度
- Hourly branch：`1H OHLCV`
- Daily branch：`1D OHLCV`

### 4.2 数据源（优先级）
1. Binance API（或 ccxt）
2. CoinGecko（备选）
3. Yahoo Finance（备选，日线更稳）

### 4.3 市场时区规则（你指定的标准）
> **统一规则（用于展示、对齐“市场日历”概念）**：
- **加密（Crypto）统一使用北京时间（Asia/Shanghai）**
- **亚盘（Asia session）统一使用北京时间（Asia/Shanghai）**
- **美股（US equities）统一使用美东时间（America/New_York）**

**工程建议（为了不出错）**
- 存储层建议保留 `timestamp_utc`（避免夏令时/跨时区混乱）
- 业务层再生成：
  - `timestamp_market`（BTC=北京时间，美股=美东时间）
  - 并在 Dashboard 明确标注时区

> 这样同时满足“统一市场时区”的需求，又避免数据工程层面踩坑。

### 4.4 数据质量与时间对齐（必须写清楚）
**(1) 时间索引补齐**
- Hourly：补齐连续小时索引（按市场时区定义的“小时边界”或 UTC 再转换）
- Daily：补齐连续日期（注意加密是 24/7，美股有交易日历）

**(2) 缺失值处理**
- 优先：重新拉取缺失区间数据
- 若仍缺失：
  - OHLC：不建议插值伪造；可标记 `missing_flag=1` 并在训练时过滤或作为特征
  - 技术指标特征：滚动窗口前期自然 NaN → 丢弃前 N 行

**(3) 极端值/跳点**
- 不修改原始价格（避免污染）
- 对“特征”可做稳健处理：
  - winsorize（如 1%–99% 裁剪）
  - 或对 returns 做 z-score 异常标记 `outlier_flag`

**(4) 版本与日志**
- 每次更新生成 `data_quality_report`：
  - 最新时间戳、缺失比例、重复比例、异常比例

---

## 5. 特征工程（Feature Engineering）

### 5.1 特征设计原则（答辩可讲）
- 用“过去信息”预测“未来”（无泄露）
- 同一套特征结构用 config 控制窗口（hourly/daily 共享框架）
- 兼顾可解释性（树模型特征重要性）与预测性能

### 5.2 共用特征类别（两边都做）
#### A. Return & Momentum
- log return
- lag returns：
  - Hourly：`1,2,4,12,24`（可加 48/72）
  - Daily：`1,3,7,14`
- rolling mean return：
  - Hourly：`24,72,168`
  - Daily：`7,30,90`

#### B. Trend
- EMA（你给的固定组合也可以）：
  - Hourly：`8/20/55/144/233`
  - Daily：`8/20/55/144/233`
- MACD（快慢线 + histogram）
- （可选）MA 5/10/20/50/200（更传统）

#### C. Volatility
- rolling std of returns（同样分 hourly/daily 窗口）
- ATR
- Bollinger Band width

#### D. Volume
- volume change rate
- volume MA
- OBV（可选）

#### E. Time Features
- Hourly：hour_of_day、day_of_week（非常有用）
- Daily：day_of_week、month（可选）

### 5.3 防止数据泄露（必须写明）
- 所有 rolling/指标只使用 `t` 及之前的数据
- 标准化/归一化：
  - 每个 walk-forward 训练窗口 **单独 fit scaler**
  - 测试窗口只做 transform
- 时间序列 split 保证训练集时间 < 测试集时间

---

## 6. 模型清单分层（Baseline / MVP / Advanced）
> 这是你补充点里最关键的部分：让项目“像研究也像工程”。

### 6.1 Baseline（必须实现）
**方向（分类）**
- Naive：方向=上一根方向
- **LogisticRegression（baseline 强推荐）**
  - `class_weight="balanced"`（处理不平衡）

**启动窗口（多分类）**
- 最频繁窗口 baseline（总预测最常见的 Wi）
- 多项 Logistic Regression（softmax）作为 baseline

**幅度（回归/区间）**
- Naive：收益率预测=0
- Linear Regression 作为传统 baseline（可选）

**波动/高波动窗口（可选 baseline）**
- rolling volatility + 阈值判别（非常轻量）

### 6.2 MVP（主力模型）
**主力：LightGBM / XGBoost**
- 方向预测：LGBMClassifier
- 启动窗口：LGBMClassifier（多分类）
- 幅度区间：**Quantile LightGBM（主力）**
  - 分别训练 q=0.1/0.5/0.9（pinball loss）

> 选择 LightGBM 作为 MVP 主力：快、稳、可解释、实现成本低，最适合 CS 487 落地。

### 6.3 Advanced（高级模型，二选一：我全权选择 TFT）
**Advanced：Temporal Fusion Transformer（TFT）**
- 理由：
  - 擅长 multi-horizon forecasting（天然匹配“多个 horizon 同时预测”）
  - 可解释性更好（变量选择/attention）
- 用法：
  - 框架固定：`PyTorch Forecasting + PyTorch Lightning`（不使用 TensorFlow 分支）
  - Daily 分支优先（噪声较小）
  - 输出：multi-horizon 的方向概率/收益率（或收益率区间）

> Advanced 只做一个（TFT），避免范围爆炸。若时间不足，可用 TCN 作为 fallback，但不写进主计划。

---

## 7. 多任务建模策略（MTL vs 分开训练）
为避免实现阶段反复改动，方案写死：

### 7.1 MVP：分任务训练（Recommended）
- 方向模型：单独训练（每个 horizon 可独立或共享一个模型输出多个 horizon）
- 启动窗口模型：单独训练
- 幅度区间模型：分位数模型（q10/q50/q90）单独训练

**优点**
- Debug 容易、指标清晰、时间可控、可交付性强

### 7.2 Advanced：可选多任务学习（仅在 TFT 阶段探索）
- 共享 encoder/backbone
- 3 个 head：direction / start_window / quantile_return
- 只做对比实验，不作为 MVP 必需

---

## 8. 超参数搜索方案（Optuna + Early Stopping）
> 你补充点非常对：需要写清楚“怎么调参”和“预算”。

### 8.1 搜索工具与策略
- **Optuna（推荐）**：更高效
- 若时间紧：Random Search（备用）

### 8.2 调参预算（建议写死）
- Daily：`30–50 trials`
- Hourly：`20–30 trials`
- 每个 trial 使用“简化版验证”：
  - 用一个固定 walk-forward fold（或 TimeSeriesSplit 的小子集）快速打分
- 最优参数再跑完整 walk-forward 输出最终成绩

### 8.3 典型搜索空间（LightGBM）
- `num_leaves: [16, 256]`
- `max_depth: [-1, 16]`
- `learning_rate: [1e-3, 0.2] (log)`
- `min_data_in_leaf: [20, 500]`
- `feature_fraction: [0.5, 1.0]`
- `bagging_fraction: [0.5, 1.0]`
- `lambda_l1: [0, 5]`
- `lambda_l2: [0, 5]`

### 8.4 早停策略
- `early_stopping_rounds = 100`
- `num_boost_round` 给足上限（如 5000），由 early stopping 自动截断

---

## 9. 不平衡处理与概率校准（Imbalance + Calibration）
### 9.1 类别不平衡
方向分类/启动窗口可能存在不平衡：
- LogisticRegression：`class_weight="balanced"`
- LightGBM：
  - 二分类：`scale_pos_weight`
  - 多分类：自定义 class weights（或 sample weights）

> 时间序列下不建议随便 SMOTE（可能破坏时间结构），优先用权重法。

### 9.2 概率校准（推荐对最终模型做）
- Platt scaling（sigmoid）或 Isotonic regression
- 输出校准后概率，用于更可靠的阈值决策

**报告指标**
- **Brier Score**（概率质量）
- （可选）reliability diagram / ECE

---

## 10. 区间预测细节（Quantile Training + Metrics）
### 10.1 训练方式（明确写 pinball loss）
- 训练三个模型：
  - `Model_q10, Model_q50, Model_q90`
- Loss：Pinball Loss（quantile regression objective）

### 10.2 保证分位数不交叉（实践细节）
若出现 `q10 > q50` 或 `q50 > q90`：
- 简单策略：对预测分位数排序 `sort(q10,q50,q90)`
- 或做 post-processing 单调修正（可选）

### 10.3 区间评估（必须）
- Coverage：`P( y ∈ [q10,q90] )`（目标接近 0.8）
- Interval Width：`mean(q90 - q10)`
- 结合评价：Coverage 不够 → 区间偏窄；Width 太大 → 区间太保守

---

## 11. 训练与验证（Walk-forward 评估协议）
### 11.1 Walk-forward 设置
- expanding window 或 rolling window
- 每个 fold：
  - 用过去训练
  - 训练集与测试集之间设置 `purge/gap`，防止标签窗口重叠导致泄露
  - `gap = max_horizon`（Hourly=`4 bars`，Daily=`7 bars`）
  - 若训练样本的标签窗口与测试区间重叠，则从训练样本中剔除
  - 用未来测试
  - 记录 metrics

**输出**
- `metrics.csv`（每个 fold、每个 horizon、每个任务）
- 汇总均值/方差（稳定性）

### 11.2 指标体系（最终报告必须有）
**方向（分类）**
- Accuracy, Precision, Recall, F1, ROC-AUC

**启动窗口（多分类）**
- Top-1 accuracy
- Top-2 accuracy
- Macro-F1（可选）

**幅度（回归）**
- MAE / RMSE / Pinball Loss
- sign accuracy（方向命中）

**区间（quantile）**
- Coverage（如 q10–q90）
- Interval Width

---

## 12. 交易可用性评估（轻量回测 / Paper Trading）
> 即使你不做交易执行，这一部分会非常加分。

### 12.1 信号生成（最小可行）
- 当 `P(up) > 0.55` → 做多
- 当 `P(up) < 0.45` → 做空
- 否则空仓
- 可对不同 horizon 分别做（hourly/daily）

### 12.2 成本与延迟建模
- 手续费：固定 bps（例如 5–20 bps）
- 滑点：固定 bps（例如 5–20 bps）
- 延迟：信号在 `t` 产生，**在 t+1 执行**（避免 look-ahead）

### 12.3 输出指标
- 总收益率、年化收益率（daily 更适合）
- 最大回撤、Sharpe
- 胜率、盈亏比
- 与 baseline（buy&hold、均线策略）对比

---

## 13. 系统工程落地（Pipeline / API / 调度 / 版本）
### 13.1 项目结构（建议）
CryptoForecast/
├── data/
│ ├── raw/
│ ├── processed/
│ └── models/
├── src/
│ ├── ingestion/ # 拉数据 + 更新
│ ├── preprocessing/ # 清洗、对齐、质量检查
│ ├── features/ # 特征工程
│ ├── labels/ # label 生成
│ ├── models/ # 训练/预测/校准/分位数
│ ├── evaluation/ # walk-forward + backtest
│ └── utils/ # config、logging、helpers
├── dashboard/ # streamlit
├── configs/ # config.yaml
└── README.md


### 13.2 Pipeline（建议的可执行命令）
- `python -m src.ingestion.update_data --config configs/config.yaml`
- `python -m src.features.build_features --config ...`
- `python -m src.labels.build_labels --config ...`
- `python -m src.models.train --config ...`
- `python -m src.models.predict --config ...`
- `python -m src.evaluation.walk_forward --config ...`
- `streamlit run dashboard/app.py`

### 13.3 调度（MVP）
- 使用 cron 或 APScheduler（轻量、够用）
- 每天更新数据 & 生成最新预测

### 13.4 模型版本管理（MVP）
- 目录按时间戳：
  - `data/models/2026-02-05_hourly/`
  - `data/models/2026-02-05_daily/`
- 同时保存：
  - `model.pkl`
  - `config_snapshot.yaml`
  - `metrics.json`

> MLflow 作为加分项，非必需。

---

## 14. Dashboard 设计（单页两栏 · 可演示）
### 左栏：Hourly（1h/2h/4h）
- Up/Down 概率卡片
- Start window（Top-1 或每窗概率）
- Return interval（q10–q90）
- 图：价格曲线 + 预测区间带

### 右栏：Daily（1d/3d/7d）
- 同样三类输出
- 图：日线价格 + 预测区间带

### 底部：模型表现与对比
- Baseline vs MVP 指标表（hourly/daily 分开）
- 区间 coverage/width 报告
- （可选）特征重要性/SHAP

---

## 15. 里程碑与验收标准（W1–W8 可验收）
> 每周必须有“可验收产物”，保证项目按时收敛。

### W1：数据与时区对齐
- 完成 1H/1D 数据拉取与存储
- 生成 `data_quality_report`
- **验收**：时间戳对齐正确（BTC=北京时间展示；存储保留 UTC）

### W2：特征工程 v1 + 标签 v1
- features_hourly / features_daily 生成
- labels（direction/start/return）生成
- **验收**：无泄露（只用过去），前 N 行 NaN 处理明确

### W3：Baseline 全部跑通 + walk-forward v1
- Naive + LogisticRegression（方向/启动）
- 输出 `metrics.csv`
- **验收**：walk-forward 流程可复现（命令一键跑）

### W4：MVP 分类模型（LightGBM）
- 方向分类 + 启动窗口分类
- 与 baseline 对比（AUC/F1/Top-1/Top-2）
- **验收**：指标优于 baseline（至少在多个 fold 上稳定不差）

### W5：Quantile LightGBM（区间预测）
- q10/q50/q90 完成
- coverage/width 指标输出
- **验收**：coverage 接近目标区间（如 0.75–0.85）且区间宽度合理

### W6：Dashboard v1
- 展示 hourly/daily 预测 + 指标 + baseline 对比
- **验收**：演示流程顺畅，能解释每个输出含义

### W7：轻量回测 + case studies
- 引入成本与延迟的 paper backtest
- 2–3 段行情案例分析（预测对/错原因）
- **验收**：策略结果与 baseline 对照完整（不追求赚钱，追求严谨）

### W8：Final polish
- 文档、可复现说明、演示脚本、最终结果表
- **验收**：从零开始按 README 一键复现核心结果

---

## 16. 风险与应对（Risk Mitigation）
- **小时级噪声大**：先做日线稳健 MVP，再扩展小时级；小时级优先树模型
- **标签阈值不合理**：使用历史分位数自动设定，并做阈值敏感性测试
- **数据泄露风险**：强制 walk-forward + scaler 每 fold 独立 fit
- **范围过大**：Advanced（TFT）严格作为加分项，不影响 MVP 交付

---

## 17. 技术栈总结（Tools & Tech）
- Python（核心）
- pandas, numpy
- requests / ccxt
- ta / pandas_ta
- scikit-learn
- lightgbm / xgboost
- （Advanced 可选）TFT（固定 PyTorch：PyTorch Forecasting + Lightning）
- streamlit
- plotly/matplotlib
- git + config.yaml
- cron / APScheduler

### 17.1 可复现性设置（必须）
- 固定随机种子：`seed=42`（python / numpy / lightgbm / torch）
- 依赖版本固定到 `requirements.txt`（避免环境漂移）
- 训练产物必须保存：`model.pkl`、`config_snapshot.yaml`、`metrics.json`
- 额外记录：训练时对应的 git commit hash（便于复现实验）
