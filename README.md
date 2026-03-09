# Stockpy

**Indian Stock Market Scanner & Research Assistant**

Stockpy is a lightweight, parallelized command-line tool that scans Indian equities (NSE/BSE) and surfaces actionable **Pros** and **Cons** for each stock. It fetches historical price data, computes technical indicators, gathers news sentiment, and presents a structured report — so you can make informed decisions faster.

> **Stockpy is a research assistant, not a trading bot.** Human judgment remains the final step.

---

## Features

- **Parallel scanning** — Analyzes multiple stocks concurrently via `ThreadPoolExecutor`
- **Technical indicators** — SMA50, SMA200, RSI (14-day), Volume analysis
- **News sentiment** — Lightweight sentiment classification per ticker
- **Interactive CLI** — Arrow-key menus, multi-select, progress bars (powered by `rich` + `simple-term-menu`)
- **Dev mode** — Verbose pipeline logging for debugging (`--dev`)
- **Quick Scan** — One-click analysis of 20 popular NSE stocks

---

## Installation

```bash
git clone https://github.com/your-username/Stockpy.git
cd Stockpy
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | Historical price data from Yahoo Finance |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `rich` | Styled terminal output |
| `simple-term-menu` | Arrow-key menu navigation |

---

## Usage

### Interactive mode (default)

```bash
python3 main.py
```

Launches the interactive menu where you can navigate with arrow keys:

| Option | Description |
|--------|-------------|
| 🔍 Scan Stocks | Enter any tickers manually |
| 📋 Quick Scan | Select from 20 popular NSE stocks |
| ⚙️ Settings | Adjust workers, data period, toggle dev mode |
| ℹ️ Help | Usage guide and keyboard shortcuts |
| 🚪 Exit | Quit the application |

### Direct CLI mode

```bash
python3 main.py RELIANCE.NS TCS.NS INFY.NS --workers 5
```

### Dev mode (verbose logging)

```bash
python3 main.py --dev
python3 main.py RELIANCE.NS --dev
```

### Logs

Stockpy now maintains a persistent log file:
- **File**: `stockpy.log` (located in the project root)
- **Level**: Automatically matches the selected mode (INFO for normal, DEBUG for Dev Mode).
- **Use Case**: Review historical scan timestamps, indicator calculations, and sentiment scores.

---

## Ticker Format

| Exchange | Format | Example |
|----------|--------|---------|
| NSE | `SYMBOL.NS` | `RELIANCE.NS` |
| BSE | `BSECODE.BO` | `500325.BO` |

---

## Output

Stockpy produces a two-part report:

1. **Overview table** — Ticker, signal (▲ Bullish / ▼ Bearish / ◆ Neutral), pros count, cons count, RSI
2. **Detailed cards** — Color-coded panels (🟢 / 🔴 / 🟡) listing individual Pros and Cons

---

## Project Structure

```
Stockpy/
├── main.py                  # CLI entry point, menus, parallelism
├── requirements.txt         # Python dependencies
├── CLAUDE.md                # AI assistant context
├── project.md               # Detailed project flow documentation
├── analysis.md              # Technical analysis methodology
└── scanner/
    ├── __init__.py
    ├── data_fetcher.py      # Fetches historical data via yfinance
    ├── indicators.py        # Computes SMA, RSI, Volume averages
    ├── news.py              # News sentiment (stub)
    └── evaluator.py         # Generates Pros/Cons from indicators + news
```

---

## Disclaimer

This tool is for **educational and research purposes only**. It does not constitute financial advice. Always do your own research before making investment decisions.
