# Analysis Methodology — Stockpy

This document covers (1) equity Pros/Cons evaluation and (2) F&O pre-market scoring.

---

# Part A — Equity Scanner

Stockpy’s equity pipeline produces **Pros** and **Cons** from:

1. **Trend** — SMA50 vs SMA200  
2. **Momentum** — RSI (14)  
3. **Volume** — vs 20-day average  
4. **Sentiment** — recent headlines (VADER)

Each rule is deterministic and independent. The overview signal is a simple pro/con count (not a price prediction).

## Technical Indicators

### SMA50 / SMA200

```
SMA50  = mean(Close[t-49] … Close[t])
SMA200 = mean(Close[t-199] … Close[t])
```

| Condition | Signal |
|-----------|--------|
| SMA50 > SMA200 | Pro — long-term uptrend (golden-cross style) |
| SMA50 < SMA200 | Con — long-term downtrend |

### RSI (14)

```
RS  = avg_gain / avg_loss
RSI = 100 - (100 / (1 + RS))
```

| Range | Signal |
|-------|--------|
| > 70 | Con — overbought |
| < 30 | Pro — potentially oversold |
| Mid bands | Neutral / weak momentum messaging per `evaluator.py` |

### Volume vs 20-day average

| Condition | Signal |
|-----------|--------|
| Volume > 1.5 × Volume_Avg_20 | Pro — volume spike |

## News Sentiment

Headlines from `yfinance` are scored with **VADER** (`scanner/news.py`). Positive → Pro, Negative → Con. (Older docs called this a stub; the live path is VADER + Yahoo news.)

## Equity Signal Aggregation

| Condition | UI signal |
|-----------|-----------|
| More Pros than Cons | ▲ Bullish |
| More Cons than Pros | ▼ Bearish |
| Equal | ◆ Neutral |

This does **not** weight signal importance — use judgment.

Top-N ranking (`screener.py`) applies numeric weights on top of the same features for `--top50` / Top 10 menu.

---

# Part B — F&O Pre-Market Analysis

Goal: a **pre-market directional bias** for index F&O research before 9:15, then validate it after open. Not a trade signal.

## Inputs (when available)

| Input | Notes |
|-------|------|
| NIFTY / BANK NIFTY prior session OHLC + SMA/RSI | yfinance |
| Expected open / indication | Last price or configured GIFT symbol; else may equal prior close |
| Gap amount & % | `(expected − prev close) / prev close` |
| India VIX | Level + short-term trend |
| US / Asia indices, USD/INR, crude, gold | Direction aggregates |
| FII / DII cash | Buy / sell / net; optional 5-day trend |
| Option chain | Max call/put OI, OI change, PCR (OI/volume), dynamic strikes |
| Events | Expiry proximity, known calendar flags (no random scraping) |

Unavailable fields are labeled explicitly; they do not invent values.

## Gap Classification (configurable)

Thresholds: `GAP_FLAT_PCT`, `GAP_SMALL_PCT`, `GAP_MODERATE_PCT` (percent of previous close).

| \|gap %\| band | Label |
|----------------|-------|
| < flat | Flat |
| < small | Small Gap Up/Down |
| < moderate | Moderate Gap Up/Down |
| ≥ moderate | Strong Gap Up/Down |

## Option Chain Heuristics

- **Put support** — highest put OI strikes at/below spot  
- **Call resistance** — highest call OI strikes at/above spot  
- **PCR (OI)** — `total put OI / total call OI`  
- **Important strikes** — ATM ± radius × strike interval (`NIFTY_STRIKE_INTERVAL` default 50, BANKNIFTY 100)  
- **Expiry day** — PCR / OI signals are dampened (`expiry_type == expiry_day`)

## Key Levels

Immediate/major support & resistance from (sources labeled in report):

- Previous day high / low / close  
- Classic floor pivots (S1/S2/R1/R2)  
- Max put / call OI when chain is available  

## Market Regime

One of: Trending Bullish, Trending Bearish, Range Bound, High Volatility, Low Volatility, Unclear — from SMA relationship, closeness of MAs, and VIX thresholds (`VIX_ELEVATED`, `VIX_HIGH`).

## Checklist

Configurable boolean items across Trend, Global, Volatility, F&O, Institutional, Events. Each item is ✓ / ✗ / — (unavailable).

## Scoring Engine

Each category emits signal ∈ {−2, −1, 0, +1, +2} × configurable **weight** (`WEIGHT_*` in `.env`).

```
total        = Σ (signal × weight)   for available categories
max_score    = Σ (weight × 2)        for available categories
normalized   = map total from [-max, +max] → [0, 100]
```

Bias bands (normalized %):

| Range | Label |
|-------|-------|
| 0–20 | Strong Bearish |
| 20–35 | Bearish |
| 35–45 | Mild Bearish |
| 45–55 | Neutral |
| 55–65 | Mild Bullish |
| 65–80 | Bullish |
| 80–100 | Strong Bullish |

## Confidence

Derived from data coverage, agreement vs conflict among signals, signal strength, and checklist completeness. Coverage &lt; 40% caps confidence. Missing major inputs → lower confidence by design.

## 9:15 / 9:30 Confirmation

| Time | Question |
|------|----------|
| 9:15 | Did actual open align with expected open / bias? |
| 9:30 | Is bias Confirmed, Partially confirmed, or Invalidated? |

The 9:00 report is a **hypothesis**; confirmation is required before treating it as useful context.

## Historical Accuracy

Stored fields (expected/actual opens, scores, bias, confidence, VIX, FII/DII, PCR, confirmations, later outcome) feed `premarket-accuracy`. Segment stats (expiry, high VIX, gap days, regimes) appear only when outcomes exist. **Do not claim edge until sample size supports it.**

---

## Limitations

| Limitation | Detail |
|------------|--------|
| Lagging equity indicators | SMA/RSI confirm, don’t predict |
| No full fundamentals | No P/E, debt, earnings model |
| yfinance delays / rate limits | Suitable for research, not HFT |
| NSE option-chain / GIFT | May be unavailable from some networks; report continues |
| Pre-market bias | Decision support only — not an order signal |

## Recommended Workflow

1. Run **pre-market** near 9:00 IST (or `--force` for a dry run).  
2. Read bias + confidence + risk flags; note unavailable data.  
3. At/after open, run **9:15** and **9:30** confirmation.  
4. Separately use equity scan for stock ideas.  
5. Decide manually — Stockpy does not place trades.
