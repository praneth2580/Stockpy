# Analysis Methodology — Stockpy

This document explains every individual value that Stockpy evaluates, how it is calculated, and what it tells you about a stock.

---

## Overview

Stockpy's evaluation pipeline produces **Pros** and **Cons** for each stock by examining:

1. **Trend** — Is the stock trending up or down over time?
2. **Momentum** — Is the stock overbought, oversold, or neutral?
3. **Volume** — Is there unusual trading activity?
4. **Sentiment** — What does the recent news say?

Each indicator is evaluated independently using simple, deterministic rules. The results are combined into a human-readable report.

---

## Technical Indicators

### 1. SMA50 — 50-Day Simple Moving Average

**What it is:**
The average closing price over the last 50 trading days.

**Calculation:**
```
SMA50 = mean(Close[t-49] ... Close[t])
```

**What it tells you:**
Represents the **short-to-medium term trend**. When the current price is above SMA50, the stock is generally considered to be in a short-term uptrend. When below, a downtrend.

**Used in evaluation:** Compared against SMA200 (see Golden/Death Cross below).

---

### 2. SMA200 — 200-Day Simple Moving Average

**What it is:**
The average closing price over the last 200 trading days.

**Calculation:**
```
SMA200 = mean(Close[t-199] ... Close[t])
```

**What it tells you:**
Represents the **long-term trend**. Institutional investors often use the 200-day MA as a benchmark. A stock trading above its SMA200 is generally considered to be in a healthy long-term trend.

---

### 3. SMA50 vs SMA200 — Trend Signal (Golden Cross / Death Cross)

**Evaluation rule:**

| Condition | Signal | Meaning |
|-----------|--------|---------|
| SMA50 **>** SMA200 | ✔ Pro: Long-term uptrend | The shorter-term average has crossed above the longer-term average. This is known as a **Golden Cross** and generally indicates bullish momentum. The stock's recent performance is outpacing its historical average. |
| SMA50 **<** SMA200 | ✖ Con: Long-term downtrend | The shorter-term average is below the longer-term average. This is known as a **Death Cross** and generally indicates bearish momentum. The stock's recent performance is weaker than its historical average. |

**Why it matters:**
- A Golden Cross is one of the most widely followed technical signals. It suggests that buying pressure is building.
- A Death Cross suggests that selling pressure is dominant and the trend may continue downward.
- These signals are lagging indicators — they confirm trends rather than predict them.

---

### 4. RSI — Relative Strength Index (14-day)

**What it is:**
A momentum oscillator that measures the speed and magnitude of recent price changes. It ranges from 0 to 100.

**Calculation:**
```
delta     = Close[t] - Close[t-1]
avg_gain  = rolling_mean(positive deltas, window=14)
avg_loss  = rolling_mean(negative deltas, window=14)
RS        = avg_gain / avg_loss
RSI       = 100 - (100 / (1 + RS))
```

**Evaluation rules:**

| RSI Range | Signal | Meaning |
|-----------|--------|---------|
| **> 70** | ✖ Con: Overbought | The stock has gained value rapidly and may be due for a pullback or correction. Buyers may be exhausted. |
| **< 30** | ✔ Pro: Potentially oversold | The stock has lost value rapidly and may be undervalued. Could represent a buying opportunity if fundamentals are sound. |
| **30 – 70** | ✔ Pro: Neutral RSI | The stock is trading in a normal momentum range. No extreme buying or selling pressure detected. |

**Why it matters:**
- RSI helps identify when a stock might be stretched too far in either direction.
- An overbought RSI doesn't mean the stock will fall immediately, but the risk of a reversal increases.
- An oversold RSI doesn't guarantee a bounce, but combined with other positive signals, it can indicate value.
- RSI is most useful when combined with trend analysis (SMA signals).

---

### 5. Volume vs 20-Day Average Volume

**What it is:**
A comparison of the current day's trading volume against the 20-day rolling average volume.

**Calculation:**
```
Volume_Avg_20 = rolling_mean(Volume, window=20)
```

**Evaluation rule:**

| Condition | Signal | Meaning |
|-----------|--------|---------|
| Volume **> 1.5×** Volume_Avg_20 | ✔ Pro: Volume spike | Significantly higher trading activity than usual. This often precedes or confirms a meaningful price move. |
| Volume **≤ 1.5×** Volume_Avg_20 | _(not reported)_ | Volume is within normal range. No signal generated. |

**Why it matters:**
- Volume is the **fuel behind price moves**. A price increase on high volume is more sustainable than one on low volume.
- Volume spikes can indicate institutional buying/selling, breakout confirmation, or increased market interest.
- A breakout above resistance on high volume is more trustworthy than one on average volume.
- Conversely, a price drop on very high volume may indicate panic selling.

---

## News Sentiment

### 6. News Sentiment Analysis

**What it is:**
A classification of recent news headlines as Positive, Negative, or Neutral.

> ⚠️ **Note:** The current implementation is a **mock/stub**. It returns deterministic sentiment based on the ticker hash. This is designed to be replaced with a real news API (e.g., NewsAPI, Google News, or Yahoo Finance headlines with NLP sentiment analysis).

**Evaluation rules:**

| Sentiment | Signal | Meaning |
|-----------|--------|---------|
| **Positive** | ✔ Pro | Recent news trends are optimistic. Positive media coverage, earnings beats, favorable analyst ratings, or sector tailwinds. |
| **Negative** | ✖ Con | Recent news trends are pessimistic. Negative headlines, earnings misses, regulatory concerns, or management issues. |
| **Neutral** | _(not reported)_ | News is mixed or absent. No signal generated. |

**Why it matters:**
- News sentiment captures information that price data alone cannot — regulatory changes, product launches, management scandals, sector trends.
- Positive sentiment combined with positive technical signals creates a stronger case for the stock.
- Negative sentiment can serve as a warning even when technical indicators look healthy.

---

## Signal Aggregation

After all individual evaluations are complete, the results are combined:

### Overall Signal

| Condition | Signal | Badge |
|-----------|--------|-------|
| More Pros than Cons | **▲ Bullish** | 🟢 |
| More Cons than Pros | **▼ Bearish** | 🔴 |
| Equal Pros and Cons | **◆ Neutral** | 🟡 |

This is a simple heuristic count — it does **not** weight the importance of individual signals. A single strong con (e.g., overbought RSI at 85 during a death cross) may outweigh two mild pros. **Use human judgment to weigh the signals.**

---

## Limitations

| Limitation | Detail |
|------------|--------|
| **Lagging indicators** | SMA and RSI are based on historical data. They confirm trends, they don't predict them. |
| **No fundamental analysis** | Stockpy does not evaluate earnings, P/E ratios, debt levels, or company financials. |
| **Mock news sentiment** | Real news analysis requires API integration and NLP — currently stubbed. |
| **No position sizing** | The tool does not suggest how much to invest. |
| **Indian market hours** | Data is fetched from Yahoo Finance, which may have slight delays for NSE/BSE data. |

---

## Recommended Workflow

1. **Run a scan** to identify interesting candidates.
2. **Review the pros and cons** — are the signals aligned?
3. **Cross-reference** with fundamental data (earnings, P/E, sector performance).
4. **Check the charts** manually for patterns the scanner doesn't detect (support/resistance, chart patterns).
5. **Make your decision** — Stockpy surfaces signals, but the final call is yours.
