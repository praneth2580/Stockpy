# Project Flow — Stockpy

This document explains the complete execution flow of Stockpy, step by step, from launch to final report output.

---

## 1. Entry Point — `main.py`

When the user runs `python3 main.py`, the `main()` function is invoked. `main.py` is now focused purely on the **User Interface** and **CLI coordination**, while the **Execution Engine** has been moved to `scanner/runner.py`.

### 1.1 Argument Parsing

```
python3 main.py [tickers...] [--workers N] [--interactive/-i] [--dev]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `tickers` | _(none)_ | Space-separated stock tickers (e.g. `RELIANCE.NS TCS.NS`) |
| `--workers` | `3` | Number of concurrent threads |
| `--interactive` / `-i` | `false` | Force interactive menu mode |
| `--dev` | `false` | Enable verbose debug logging |

### 1.2 Logging Initialization

`setup_logging(dev)` is called:
- **Normal mode**: Sets console handler to `WARNING`. File handler is set to `INFO`.
- **Dev mode**: Sets both console and file handlers to `DEBUG`.
- **File Persistence**: All logs are appended to `stockpy.log` in the project root.
- **Suppression**: Noisy third-party loggers (`yfinance`, `peewee`, `urllib3`, `curl_cffi`) are suppressed at `WARNING` level across all handlers.

### 1.3 Mode Selection

| Condition | Mode |
|-----------|------|
| No tickers provided, or `--interactive` flag | **Interactive mode** |
| Tickers provided on command line | **CLI mode** |

---

## 2. Interactive Mode

### 2.1 Banner Display

`show_banner()` clears the terminal and prints the ASCII art logo, subtitle, and a dev mode indicator if active.

### 2.2 Main Menu Loop

An arrow-key navigable menu is displayed using `simple-term-menu`:

```
❯ 🔍  Scan Stocks
  📋  Quick Scan  (Popular NSE Stocks)
  ⚙️   Settings
  ℹ️   Help
  🚪  Exit
```

The user navigates with `↑ ↓` and selects with `Enter`. The loop continues until the user selects Exit or presses `Esc`.

### 2.3 Scan Stocks

1. User is prompted to type ticker symbols separated by spaces.
2. Input is split into a list.
3. `run_scan(tickers)` is called (see Section 4).

### 2.4 Quick Scan

1. A list of 20 popular NSE stocks is displayed as a multi-select menu.
2. User selects stocks with `Space` and confirms with `Enter`.
3. Selected tickers are passed to `run_scan()`.

### 2.5 Settings

An arrow-key sub-menu allows changing:

| Setting | Options | Default |
|---------|---------|---------|
| Workers | 1–10 | 3 |
| Data Period | 1mo, 3mo, 6mo, 1y, 2y, 5y | 1y |
| Dev Mode | Toggle ON/OFF | OFF |

Changes take effect immediately for subsequent scans.

### 2.6 Help

Displays a `rich` panel with pipeline explanation, ticker format guide, CLI usage examples, and keyboard shortcuts.

---

## 3. CLI Mode

When tickers are provided directly:
1. `settings["workers"]` is set from `--workers`.
2. The banner is displayed.
3. `run_scan(tickers)` is called directly.

---

## 4. Scan Pipeline — `runner.py`

The core scanning logic resides in `scanner/runner.py`.

### 4.1 Configuration Display

A panel is printed showing the tickers being scanned, the number of workers, and the data period.

### 4.2 Thread Pool Execution

Each ticker is submitted as an independent task to the thread pool (`ThreadPoolExecutor`).
- Tasks execute **concurrently**.
- A `rich` progress bar updates as each ticker completes.
- Results are gathered and returned to `main.py` for rendering.

### 4.3 Individual Ticker Pipeline — `process_ticker(ticker, period)`

For each ticker, these four steps execute sequentially:

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: FETCH DATA        (data_fetcher.py)            │
│  Step 2: COMPUTE INDICATORS (indicators.py)             │
│  Step 3: FETCH NEWS         (news.py)                   │
│  Step 4: EVALUATE           (evaluator.py)              │
└─────────────────────────────────────────────────────────┘
```

#### Step 1 — Fetch Data (`scanner/data_fetcher.py`)

- Calls `yf.Ticker(ticker).history(period=period)`.
- Returns a pandas DataFrame with columns: `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, `Stock Splits`.
- Returns `None` if the ticker is invalid or the API fails.

#### Step 2 — Compute Indicators (`scanner/indicators.py`)

- Takes the raw DataFrame and adds new columns:
  - `SMA50` — 50-day Simple Moving Average of `Close`
  - `SMA200` — 200-day Simple Moving Average of `Close`
  - `RSI` — 14-day Relative Strength Index
  - `Volume_Avg_20` — 20-day rolling average of `Volume`
- Returns the enriched DataFrame.

#### Step 3 — Fetch News (`scanner/news.py`)

- Currently a **mock implementation** that returns a deterministic sentiment.
- Returns a dict: `{"sentiment": "Positive"|"Negative"|"Neutral", "summary": "..."}`.
- Designed to be replaced with a real news API integration.

#### Step 4 — Evaluate (`scanner/evaluator.py`)

- Reads the latest row of the indicator-enriched DataFrame.
- Applies heuristic rules to generate a list of `pros` and `cons`.
- Incorporates the news sentiment result.
- Returns: `{"pros": [...], "cons": [...]}`.

### 4.4 Result Aggregation

Results are collected as they complete and stored in a list. Each result contains:

```python
{
    "ticker": "RELIANCE.NS",
    "status": "success" | "failed",
    "evaluation": {"pros": [...], "cons": [...]},  # if success
    "error": "..."                                   # if failed
}
```

---

## 5. Report Rendering — `render_report(results)`

### 5.1 Overview Table

A summary table is printed with columns:

| Column | Description |
|--------|-------------|
| Ticker | Stock symbol |
| Signal | ▲ Bullish (more pros) / ▼ Bearish (more cons) / ◆ Neutral (equal) |
| Pros | Count of pro signals |
| Cons | Count of con signals |
| RSI | Current RSI value |

### 5.2 Detailed Stock Cards

For each stock, a bordered panel is printed:
- **Title**: badge (🟢/🔴/🟡) + ticker name
- **Body**: lists each Pro (✔) and Con (✖)
- **Subtitle**: data period used
- **Border color**: green (bullish), red (bearish), yellow (neutral)

---

## 6. Error Handling

| Layer | Handling |
|-------|----------|
| `data_fetcher` | Returns `None` on API failure; logged as warning |
| `process_ticker` | Wraps entire pipeline in try/except; returns `status: "failed"` |
| `run_scan` | Catches exceptions from `future.result()`; adds error to results |
| `render_report` | Displays error panel for failed tickers |

---

## 7. Dev Mode Logging

When `--dev` is active (or toggled from Settings), the console displays timestamped logs for every pipeline action:

```
16:57:02 │ DEBUG    │ __main__             │ [pipeline] Start → RELIANCE.NS
16:57:02 │ DEBUG    │ __main__             │ [fetch]     Downloading 1y of data
16:57:03 │ DEBUG    │ __main__             │ [indicators] Computing SMA / RSI
16:57:03 │ INFO     │ scanner.news         │ Fetching news sentiment
16:57:03 │ DEBUG    │ __main__             │ [evaluate]  Generating pros/cons
16:57:03 │ DEBUG    │ __main__             │ [pipeline] Done ✔ RELIANCE.NS
```

Third-party library logs are suppressed to keep output clean.
