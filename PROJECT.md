# Project Flow — Stockpy

This document explains Stockpy’s execution flow: equity scanning and F&O pre-market analysis.

---

## 1. Entry Point — `main.py`

`main.py` owns the **CLI / UI**. Equity execution lives in `scanner/runner.py`. F&O pre-market execution lives in `scanner/premarket/`.

### 1.1 Argument Parsing

```
python3 main.py [tickers...] [options]
```

| Argument | Description |
|----------|-------------|
| `tickers` | Equity symbols (e.g. `RELIANCE.NS TCS.NS`) |
| `--workers N` | Thread pool size (equity) |
| `--interactive` / `-i` | Force interactive menu |
| `--dev` | Verbose logging → console + `stockpy.log` |
| `--top50` | Non-interactive NSE Top 50 → top 10 + Telegram |
| `--premarket` | F&O ~9:00 pre-market report |
| `--premarket-open` | 9:15 open snapshot |
| `--premarket-confirm` | 9:30 confirmation |
| `--premarket-accuracy` | Historical accuracy dashboard |
| `--force` | Ignore weekend / holiday / duplicate guards |
| `--no-notify` | Skip Telegram |

### 1.2 Logging

Same as before: root logger + console + `stockpy.log`; noisy libs suppressed.

### 1.3 Mode Selection

| Condition | Mode |
|-----------|------|
| `--premarket*` flags | Pre-market / confirmation / accuracy |
| `--top50` | Equity daily screen |
| No tickers or `--interactive` | Interactive menu |
| Tickers on CLI | Equity CLI scan |

---

## 2. Interactive Mode

Menu options:

```
❯ Scan Stocks
  Quick Scan
  Top 10 from Top 50
  F&O Pre-Market Report
  Pre-Market Accuracy
  Settings
  Help
  Exit
```

Settings still adjust workers, period, Telegram token/chat id, and dev mode (backed by `scanner/config.py` / env).

---

## 3. Equity Scan Pipeline — `scanner/runner.py`

Unchanged high-level flow:

```
fetch (data_fetcher) → indicators → news (VADER) → evaluate → optional screener
```

Parallelism via `ThreadPoolExecutor`. Results rendered in `main.render_report` and optionally sent with `TelegramNotifier`.

News is **live** (`yf.Ticker.news` + VADER), not a stub.

---

## 4. F&O Pre-Market Pipeline — `scanner/premarket/`

### 4.1 Trading-day & duplicate guards

`calendar.is_trading_day` — weekends + static NSE holiday set (`Asia/Kolkata`).  
`db.try_acquire_job(job_type, date)` — SQLite unique lock prevents double runs.  
`--force` bypasses both for local testing.

### 4.2 09:00 — `pipeline.generate_premarket_report`

Collectors run independently (failure → unavailable, pipeline continues):

| Collector | Module | Source |
|-----------|--------|--------|
| NIFTY / BANK NIFTY | `collectors/indices.py` | yfinance |
| India VIX | `collectors/indices.py` | yfinance `^INDIAVIX` |
| Globals / FX / commodities | `collectors/global_markets.py` | yfinance |
| GIFT Nifty | same | only if `GIFT_NIFTY_SYMBOL` set |
| FII / DII | `collectors/fii_dii.py` | NSE API |
| Option chain | `collectors/option_chain.py` | NSE API (best-effort) |
| Events / expiry flags | `collectors/events.py` | calendar + chain expiry |

Then analysis:

| Step | Module |
|------|--------|
| Gap class | `analysis/gap.py` |
| Levels (PDH/L, pivots, OI) | `analysis/levels.py` |
| Regime | `analysis/regime.py` |
| Checklist | `analysis/checklist.py` |
| Weighted score + bias label | `analysis/scoring.py` |
| Confidence | `analysis/confidence.py` |
| Text report | `report.py` |

Persist via `db.save_report` → `data/premarket.db` (configurable).  
Optional Telegram via existing notifier + `format_premarket_telegram`.

### 4.3 09:15 — `confirm.run_open_snapshot`

Fetches actual open/last for index symbols, compares to expected open and pre-market bias → updates report row (`open_915_result`).

### 4.4 09:30 — `confirm.run_confirmation`

Combines open + post-open move vs bias → `Confirmed` / `Partially confirmed` / `Invalidated` / `Insufficient data`.

### 4.5 Accuracy — `accuracy.py`

Aggregates stored premarket rows: bias counts, 9:30 confirmation rate, directional accuracy, segments (expiry, high VIX, gap up/down, trending, range). Only meaningful after enough confirmed outcomes exist.

---

## 5. Configuration — `scanner/config.py`

Loads `.env` (via `python-dotenv`) into:

- `DEFAULT_SETTINGS` — equity workers / period / Telegram (also mirrored into `runner.settings`)
- `PREMARKET` — symbols, times, gap bands, VIX/PCR thresholds, score weights, DB path, retries

See `.env.example`.

---

## 6. Scheduling

| Workflow | Role |
|----------|------|
| `.github/workflows/ci.yml` | Pytest on push/PR |
| `.github/workflows/stockpy-daily.yml` | Mon–Fri equity `--top50` (cron early for ~15:30 IST) |
| `.github/workflows/stockpy-premarket.yml` | Mon–Fri F&O (cron ~08:00/08:15/08:30 IST → aim ~09:xx); job from `github.event.schedule`; no Telegram ping on schedule |

Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Details: `.github/README.md`.

---

## 7. Error Handling (pre-market)

| Layer | Behavior |
|-------|----------|
| `retry.retry_call` | Transient retries with backoff; returns `None` on exhaustion |
| Individual collectors | Isolated try/except in pipeline |
| Missing data | Explicit `DATA UNAVAILABLE`; confidence reduced |
| Duplicate job | Skipped with reason logged |

---

## 8. Manual Test Recipe

```bash
python3 main.py --premarket --force --no-notify
python3 main.py --premarket-open --force --no-notify
python3 main.py --premarket-confirm --force --no-notify
python3 main.py --premarket-accuracy
python3 -m pytest tests/ -q
```
