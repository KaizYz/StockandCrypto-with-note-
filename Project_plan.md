# Multi-Market Trend & Volatility Forecasting System (BTC + China A-shares + US Equities)
**Multi-timeframe outputs**: Direction probability + Start-window probability + Magnitude/return interval

---

## 1) One-Sentence Scope (Scope Lock)
Build an end-to-end forecasting system focused primarily on crypto markets (Top 3 first: BTC, ETH, SOL), with an extensible architecture to support China A-shares and US equities.  
At both the **hourly level (1–4h)** and **daily level (1–7d)**, the system outputs **direction probability**, **start-window probability**, and **magnitude/return interval**, and presents results in a dashboard.  
The full pipeline uses strict **walk-forward** time-series evaluation and provides reproducible data and training workflows.

---

## 2) Motivation (Why this project)
In fast-changing markets, answering only “up or down tomorrow” is not enough. Traders and analysts care more about:

- **Direction**: Probability of up vs down
- **Timing**: When a significant move is most likely to start
- **Magnitude**: How far price may move (intervals are more practical than point estimates)

This project builds a reproducible system that answers all three questions across multiple horizons, with rigorous evaluation to ensure trustworthy conclusions.

---

## 3) Deliverables
### End-to-end, reproducible system
- Data ingestion + incremental updates (re-runnable)
- Data validation + data quality report (`data_quality_report`)
- Feature engineering module (config-driven)
- Label generation (direction / start window / magnitude)
- Model training + tuning + walk-forward evaluation
- Saved model artifacts (with config snapshot + metrics + version/timestamp)
- Dashboard (forecasts + intervals + performance + baseline comparison)

---

## 4) Problem Formulation (Three Tasks)
Hourly and daily branches use the **same three task types**, with different time units.

### Task 1: Direction Classification
For forecast horizon `h`, define future return:
- `r(t,h) = (Close[t+h] - Close[t]) / Close[t]`

Labels:
- `y_dir(t,h) = 1` (if `r(t,h) > 0`)
- `y_dir(t,h) = 0` (otherwise)

Outputs:
- `P(up)`
- `P(down) = 1 - P(up)`

Horizons:
- **Hourly**: 1h / 2h / 4h
- **Daily**: 1d / 3d / 7d (optional 30d)

---

### Task 2: Start Time Window Prediction (Multiclass)
Goal: Predict which future time window is most likely to contain the **first significant volatility/trend start**.

#### 2.1 Start Event Definition (Data-driven threshold)
- Compute future cumulative returns `r(t,k)` (`k` is future steps in hours/days)
- Define threshold `thr` automatically from historical quantiles (avoid guesswork):
  - **Hourly**: 80th percentile of historical 1H `|r|` → `thr_h`
  - **Daily**: 80th percentile of historical 1D `|r|` → `thr_d`
- Start time:
  - `τ = min { k : |r(t,k)| ≥ thr }`
- If not triggered within the target horizon, label as `no_start`

#### 2.2 Window bins
**Hourly (within next 4 hours)**
- `W0: no_start`
- `W1: 0–1h`
- `W2: 1–2h`
- `W3: 2–4h`

**Daily (within next 7 days)**
- `W0: no_start`
- `W1: 0–1d`
- `W2: 1–3d`
- `W3: 3–7d`

Outputs (MVP):
- Probabilities `P(W0..W3)`
- Top-1 window (highest probability)

---

### Task 3: Magnitude / Interval Prediction (Quantile Regression)
Compared with single-point prediction, interval outputs are more practical in real usage.

#### 3.1 MVP: Return quantile interval
Target:
- `y_ret(t,h) = r(t,h)`

Predictions:
- `q10 / q50 / q90`

Training:
- Quantile regression (LightGBM quantile objective)
- Use **Pinball Loss**:
  - `L_q(y, ŷ) = max( q*(y-ŷ), (q-1)*(y-ŷ) )`

Interval evaluation (required):
- **Coverage**: Proportion of true values inside `[q10, q90]` (target ≈ 80%)
- **Width**: `mean(q90 - q10)` (smaller is better, while keeping good coverage)

#### 3.2 Optional enhancement (closer to “target price” usage)
Predict future-window high/low relative to current close:
- `y_high(t,h) = (max High(next h) - Close[t]) / Close[t]`
- `y_low(t,h)  = (min Low(next h)  - Close[t]) / Close[t]`

---

## 5) Data Plan (Crypto + China A-shares + US Equities)
### 5.1 Data granularity
- Hourly branch: 1H OHLCV
- Daily branch: 1D OHLCV

### 5.2 Data sources (priority)
- **Crypto (BTC/ETH/SOL)**: Binance/Bybit API (or ccxt), fallback: CoinGecko
- **US equities**: Yahoo Finance (stable for daily), optional: Polygon/Alpaca
- **China A-shares**: AkShare / TuShare

### 5.3 Timezone & trading calendar rules (very important)
#### Timezone rules (display/market convention)
- **Crypto**: `Asia/Shanghai` (Beijing Time)
- **China A-shares**: `Asia/Shanghai` (Beijing Time)
- **US equities**: `America/New_York` (US Eastern Time)

#### Storage vs display (engineering practice)
- Unified storage in: `timestamp_utc` (avoids DST/cross-timezone bugs)
- UI display generated as: `timestamp_market`
  - Crypto + A-shares → Beijing Time
  - US equities → Eastern Time
- Dashboard must clearly label timezone.

#### Trading calendar alignment (critical differences)
- **Crypto**: 24/7 continuous; hourly data should fill continuous hourly index
- **A-shares**: Exchange calendar + lunch break (09:30–11:30, 13:00–15:00 Beijing Time)
  - Hourly: keep only trading-session bars (avoid fabricated lunch-break candles)
  - Daily: trading days only (no weekend/holiday filling)
- **US equities**: Exchange calendar + DST
  - Daily: trading days only
  - Store in UTC, display in Eastern to avoid DST boundary errors
  - Use a reliable exchange calendar library for alignment

### 5.4 Data quality strategy
- Fill missing timestamps; prioritize refetching missing ranges
- Avoid interpolating fake OHLC; use `missing_flag` or drop affected samples
- Do not alter raw prices; if needed, apply winsorization only on features
- Generate `data_quality_report` for each update:
  - latest timestamp, missing%, duplicates%, anomaly flags

---

## 6) Feature Engineering
### 6.1 Principles
- Use only information observable at time `t` (no leakage)
- Shared feature framework; window lengths controlled by config (hourly/daily differ)
- MVP prioritizes interpretability (tree-based models)

### 6.2 Feature groups
#### Returns & Momentum
- log returns
- lagged returns
  - Hourly: 1/2/4/12/24
  - Daily: 1/3/7/14
- rolling mean returns
  - Hourly: 24/72/168
  - Daily: 7/30/90

#### Trend
- EMA: 8/20/55/144/233
- MACD (line/signal/histogram)

#### Volatility
- rolling std of returns
- ATR
- Bollinger Band width

#### Volume
- volume change rate
- volume MA
- (optional) OBV

#### Time features
- Hourly: `hour_of_day`, `day_of_week`
- Daily: `day_of_week`, `month` (optional)

### 6.3 Leakage prevention
- Indicators/rolling stats computed using data `≤ t` only
- Scaling:
  - fit on training fold only
  - transform on test fold only
- Time-series splits only (no random shuffle)

---

## 7) Model Stack (Baseline / MVP / Advanced)
### 7.1 Baselines (required)
**Direction**
- Naive: same direction as previous bar
- Logistic Regression (`class_weight="balanced"`)

**Start window**
- Most-frequent-class baseline
- Multinomial Logistic Regression (softmax)

**Magnitude**
- Naive: return = 0
- (optional) Linear Regression

### 7.2 MVP (primary models)
**LightGBM / XGBoost**
- Direction: `LGBMClassifier`
- Start window: `LGBMClassifier` (multiclass)
- Magnitude interval: Quantile LightGBM (train q10/q50/q90 separately)

**Why LightGBM for MVP**: fast training, stable, interpretable, lower engineering risk.

### 7.3 Advanced (optional)
**Temporal Fusion Transformer (TFT)**
- Strong for multi-horizon forecasting
- Better interpretability (variable selection / attention)
- Prefer starting from daily branch (less noise)
- Recommended stack: PyTorch Forecasting + PyTorch Lightning

---

## 8) Multi-Task Strategy
**MVP decision: train per task (hard requirement)**
- Train direction model separately
- Train start-window model separately
- Train quantiles q10/q50/q90 separately

Advantages: easier debugging, clearer metrics, more controllable delivery.

**Advanced (optional): TFT multi-task comparison**
- Shared encoder + separate heads (direction/start/quantiles), used only as comparison experiments.

---

## 9) Hyperparameter Tuning (Optuna + Early Stopping)
- Tool: Optuna (fallback: random search)
- Fixed budget:
  - Daily: 30–50 trials
  - Hourly: 20–30 trials
- First score quickly on small time-split subsets, then run full walk-forward with best params

Example LightGBM search space:
- `num_leaves`: 16–256
- `max_depth`: -1–16
- `learning_rate`: 1e-3–0.2 (log-scale)
- `min_data_in_leaf`: 20–500
- `feature_fraction`: 0.5–1.0
- `bagging_fraction`: 0.5–1.0
- `lambda_l1 / lambda_l2`: 0–5

Early stopping:
- `early_stopping_rounds = 100`
- `num_boost_round ≤ 5000`

---

## 10) Imbalance + Probability Calibration
**Imbalance handling**
- Logistic: `class_weight="balanced"`
- LightGBM:
  - binary: `scale_pos_weight`
  - multiclass: sample weights / class weights

**Calibration (on final model)**
- Platt scaling (sigmoid) or Isotonic
- Report: Brier score
- (optional) reliability diagram / ECE

---

## 11) Interval Prediction Details
- Train q10/q50/q90 separately
- If quantile crossing occurs (`q10 > q50`, etc.):
  - Simple post-processing: sort `(q10, q50, q90)`

Report:
- Coverage (`[q10,q90]`)
- Width (`mean(q90-q10)`)

---

## 12) Evaluation Protocol (Walk-Forward Evaluation)
### 12.1 Walk-forward setup
- expanding or rolling window
- add gap/purge to prevent horizon-overlap leakage:
  - `gap = max horizon` (Hourly: 4 bars; Daily: 7 bars)
- save fold-level metrics to `metrics.csv`

### 12.2 Required metrics
**Direction**
- Accuracy, Precision, Recall, F1, ROC-AUC

**Start window**
- Top-1 accuracy
- Top-2 accuracy
- (optional) Macro-F1

**Magnitude**
- MAE / RMSE / Pinball loss
- (optional) sign accuracy

**Intervals**
- Coverage
- Width

---

## 13) Bonus: Lightweight Backtest (No live trading)
Minimum signal rules:
- `P(up) > 0.55` → Long
- `P(up) < 0.45` → Short
- otherwise Flat

Realism settings:
- transaction fee (bps), slippage (bps)
- execution delay: signal at `t`, execution at `t+1`

Report:
- total return, max drawdown, Sharpe
- win rate, profit factor
- baseline comparison (buy-and-hold, moving-average strategy)

---

## 14) Engineering Implementation
### 14.1 Repository structure
```text
Forecast/
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── labels/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── dashboard/Streamlit
├── configs/config.yaml
└── README.md
```

### 14.2 Pipeline commands
- `python -m src.ingestion.update_data --config configs/config.yaml`
- `python -m src.features.build_features --config ...`
- `python -m src.labels.build_labels --config ...`
- `python -m src.models.train --config ...`
- `python -m src.models.predict --config ...`
- `python -m src.markets.snapshot --config ...`
- `python -m src.evaluation.walk_forward --config ...`
- `streamlit run dashboard/app.py`

### 14.3 Scheduling (MVP)
- cron or APScheduler
- daily update + latest forecast generation

### 14.4 Model/version management
- Timestamped saving:
  - `data/models/2026-02-05_hourly/`
  - `data/models/2026-02-05_daily/`
- Saved artifacts:
  - `model.pkl`
  - `config_snapshot.yaml`
  - `metrics.json`
  - git commit hash

---

## 15) Dashboard (Demo Structure)
**Top: Multi-market snapshot (Crypto / A-shares / US equities)**
- For each market, display at least:
  - Current Price
  - Predicted Price
  - Predicted Change
  - Expected Date
- Support optional universes and symbols:
  - Crypto: `BTC / ETH / SOL` + `Top 100 by market cap (excluding stablecoins like USDT/USDC)`
  - A-shares: `SSE Composite constituents`, `CSI 300 constituents`
  - US equities: `Dow Jones 30`, `Nasdaq 100`, `S&P 500`
- Price rules:
  - **Crypto current price must use real-time ticker (not close price)**
  - A-shares/US equities should use latest API price when possible; if real-time is unavailable, clearly label as latest available price

**Left column: Hourly (1h/2h/4h)**
- `P(up)/P(down)` cards
- start-window probabilities
- q10–q90 interval
- chart: price + interval band

**Right column: Daily (1d/3d/7d)**
- same outputs

**Bottom: Performance**
- baseline vs MVP metrics table
- interval coverage/width
- (optional) feature importance / SHAP

---

## 16) Weekly Plan (W1–W8)
- **W1**: Data ingestion + timezone alignment + quality report
- **W2**: Features v1 + Labels v1 (no leakage)
- **W3**: Baselines + walk-forward v1 (generate reproducible `metrics.csv`)
- **W4**: MVP classification (LightGBM) + baseline comparison
- **W5**: Quantile LightGBM (q10/q50/q90) + coverage/width
- **W6**: Dashboard v1 (demo-ready)
- **W7**: Backtest + 2–3 case studies (explain why predictions were right/wrong)
- **W8**: Final polish + one-command reproducibility in README

---

## 17) Risks & Mitigation
- High hourly noise: finish Daily MVP first, then extend to Hourly
- Threshold sensitivity: quantile-based thresholds + sensitivity checks
- Leakage risk: strict walk-forward + gap/purge + fold-wise independent scaler fit
- Scope creep: TFT is optional; MVP delivery is mandatory

---

## 18) Tech Stack + Reproducibility
**Core**
- Python, pandas, numpy
- requests / ccxt
- ta / pandas_ta
- scikit-learn
- lightgbm / xgboost
- streamlit + plotly/matplotlib
- git + config.yaml
- cron / APScheduler

**Advanced (optional)**
- TFT: PyTorch Forecasting + PyTorch Lightning

## 19) Quant Factors (Implemented in MVP)
- Risk factors: market cap (size), value (earnings/book proxy), growth (fundamental growth or 90d return proxy).
- Market behavior factors: momentum (20d return), reversal (-5d return), low volatility (-20d return std).
- For symbols without fundamentals (common in crypto), factors fall back to transparent price-based proxies, and source labels are displayed in the dashboard.
