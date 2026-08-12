# Stockpy

**Indian Stock Market Scanner & F&O Pre-Market Research Assistant**

Stockpy is a lightweight command-line tool for Indian markets (NSE/BSE). It has two complementary modes:

1. **Equity scanner** — parallel stock scans with technicals, news sentiment, and Pros/Cons
2. **F&O pre-market analysis** — automated ~9:00 AM IST directional-bias report for NIFTY / BANK NIFTY, plus 9:15 / 9:30 confirmation and historical tracking

> **Stockpy is a research assistant, not a trading bot.** It produces a *pre-market directional bias* with confidence — never a guaranteed CALL/PUT or profit claim. Human judgment remains the final step.

---

## Features

### Equity scanner
- **Parallel scanning** — multiple tickers via `ThreadPoolExecutor`
- **Technical indicators** — SMA50, SMA200, RSI (14), volume vs 20-day average
- **News sentiment** — Yahoo Finance headlines + VADER
- **Top-N screening** — rank NSE Top 50 candidates
- **Telegram alerts** — optional HTML summaries after scans

### F&O pre-market (new)
- **9:00 IST report** — NIFTY / BANK NIFTY gaps, India VIX, globals, FII/DII, option-chain OI/PCR (when available), checklist, bias + confidence
- **9:15 open snapshot** — expected vs actual open vs pre-market bias
- **9:30 confirmation** — Confirmed / Partially confirmed / Invalidated
- **SQLite history** — every report stored for accuracy review
- **Accuracy dashboard** — performance by bias, expiry, VIX, gap, regime (once enough data exists)
- **Trading-day guards** — `Asia/Kolkata`, weekends, NSE holiday list, duplicate-run lock
- **Resilient collectors** — retries + per-source failure isolation (`DATA UNAVAILABLE` instead of crashing)

### CLI UX
- Interactive menus (`rich` + `simple-term-menu`)
- Dev mode logging (`--dev` → `stockpy.log`)

---

## Installation

```bash
git clone https://github.com/your-username/Stockpy.git
cd Stockpy
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # optional: Telegram, thresholds, symbols
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | Price / index history |
| `pandas` / `numpy` | Data & indicators |
| `rich` / `simple-term-menu` | Terminal UI |
| `vaderSentiment` | Headline sentiment |
| `requests` | Telegram + NSE HTTP |
| `python-dotenv` | `.env` configuration |
| `pytest` | Unit tests |

---

## Usage

### Interactive mode (default)

```bash
python3 main.py
```

| Option | Description |
|--------|-------------|
| Scan Stocks | Enter tickers manually |
| Quick Scan | Multi-select popular NSE names |
| Top 10 from Top 50 | Rank liquid large-caps |
| F&O Pre-Market Report | Run the 9:00-style analysis now |
| Pre-Market Accuracy | Historical bias performance |
| Settings | Workers, period, Telegram, dev mode |
| Help | Shortcuts and CLI flags |

### Equity CLI

```bash
python3 main.py RELIANCE.NS TCS.NS INFY.NS --workers 5
python3 main.py --top50          # NSE Top 50 → top 10 + Telegram
python3 main.py --dev
```

### F&O pre-market CLI

```bash
# Full 9:00-style report (force = ignore holiday/duplicate guards for testing)
python3 main.py --premarket --force --no-notify

# After market open
python3 main.py --premarket-open --force --no-notify
python3 main.py --premarket-confirm --force --no-notify

# Historical dashboard
python3 main.py --premarket-accuracy
```

| Flag | Meaning |
|------|---------|
| `--premarket` | Build & print F&O pre-market report; save to DB |
| `--premarket-open` | 9:15 open snapshot vs bias |
| `--premarket-confirm` | 9:30 confirmation |
| `--premarket-accuracy` | Accuracy dashboard from stored reports |
| `--force` | Skip weekend / holiday / duplicate locks (manual testing) |
| `--no-notify` | Do not send Telegram |

Without `--force`, jobs only run on NSE trading days and at most once per job type per date.

### Tests

```bash
python3 -m pytest tests/ -q
```

### Logs

- File: `stockpy.log` (project root)
- Level: INFO normally, DEBUG with `--dev`

---

## Configuration

Copy `.env.example` → `.env`. Important variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | — | Notifications |
| `PREMARKET_NOTIFY` | `true` | Enable Telegram for pre-market jobs |
| `PREMARKET_TIMEZONE` | `Asia/Kolkata` | Schedule timezone |
| `PREMARKET_REPORT_TIME` | `09:00` | Logical report label |
| `MARKET_OPEN_TIME` / `CONFIRMATION_TIME` | `09:15` / `09:30` | Confirmation slots |
| `NIFTY_SYMBOL` / `BANKNIFTY_SYMBOL` | `^NSEI` / `^NSEBANK` | Yahoo symbols |
| `GIFT_NIFTY_SYMBOL` | _(empty)_ | Set only if you have a reliable Yahoo/NSE symbol |
| `PREMARKET_DB_PATH` | `data/premarket.db` | History DB |
| `GAP_*_PCT` | 0.10 / 0.30 / 0.70 | Gap classification bands |
| `WEIGHT_*` | see `.env.example` | Scoring weights |

---

## Scheduling (GitHub Actions)

See [`.github/README.md`](.github/README.md) for secrets and manual dispatch.

Crons are started **~30 minutes early** because GitHub Actions is often late.

| Workflow | Cron (aims for) | Command |
|----------|-----------------|---------|
| `ci.yml` | Push / PR | `pytest` + import smoke |
| `stockpy-daily.yml` | Mon–Fri ≈ 15:30 IST | `--top50` (skips NSE holidays) |
| `stockpy-premarket.yml` | Mon–Fri ≈ 09:00 / 09:15 / 09:30 IST | `--premarket` / `--premarket-open` / `--premarket-confirm` |

Set repository secrets `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` for Actions notifications. Scheduled runs send **only** the glance/summary (no connectivity ping). Premarket DB is artifact-persisted between runs (best-effort).

---

## Ticker Format (equity)

| Exchange | Format | Example |
|----------|--------|---------|
| NSE | `SYMBOL.NS` | `RELIANCE.NS` |
| BSE | `CODE.BO` | `500325.BO` |

---

## Output

### Equity
1. Overview table — signal, price, SMA, RSI, volume ratio  
2. Detail cards — Pros / Cons + technicals + news  

### F&O pre-market
Structured text report: indices, globals, VIX, FII/DII, option chain, key levels, checklist, **pre-market bias**, confidence, scenarios, risk flags. Missing sources are labeled `DATA UNAVAILABLE` (confidence reduced).

Reports are stored in SQLite for later accuracy review. Do **not** treat early sample accuracy as proof of edge.

---

## Project Structure

```
Stockpy/
├── main.py                      # CLI, menus, pre-market flags
├── requirements.txt
├── .env.example
├── CLAUDE.md / PROJECT.md / ANALYSIS.md
├── .github/workflows/
│   ├── stockpy-daily.yml        # Equity top-50 cron
│   └── stockpy-premarket.yml    # F&O 9:00 / 9:15 / 9:30 IST
├── scanner/
│   ├── config.py                # Env-backed settings
│   ├── runner.py                # Equity scan orchestration
│   ├── data_fetcher.py
│   ├── indicators.py
│   ├── news.py                  # yfinance + VADER
│   ├── evaluator.py
│   ├── screener.py
│   ├── notifier.py              # Telegram
│   └── premarket/               # F&O pre-market package
│       ├── pipeline.py          # 9:00 orchestration
│       ├── confirm.py           # 9:15 / 9:30
│       ├── db.py / accuracy.py
│       ├── report.py
│       ├── calendar.py / retry.py
│       ├── collectors/          # Indices, globals, FII/DII, OC, events
│       └── analysis/            # Gap, OI, levels, regime, scoring, checklist
└── tests/                       # pytest suite for pre-market logic
```

---

## Disclaimer

This tool is for **educational and research purposes only**. It does not constitute financial advice. Pre-market bias is a hypothesis that requires open confirmation. Always do your own research before making investment decisions.
