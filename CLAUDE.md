# CLAUDE.md

This repository contains a Python-based research tool for Indian markets (NSE/BSE). It has two layers:

1. **Equity scanner** — filter stocks and present Pros/Cons for manual decisions  
2. **F&O pre-market assistant** — deterministic ~9:00 AM IST directional-bias report for index F&O research, with 9:15 / 9:30 confirmation and historical tracking  

The system does **not** automate trading or guarantee predictions. Human judgment remains the final step.

## Core Objective

### Equity

1. Fetch historical stock data for Indian equities.  
2. Compute technical indicators and momentum signals.  
3. Identify interesting candidates (including top-N screening).  
4. Gather recent news and lightweight sentiment.  
5. Output clear **pros and cons** per ticker.  

### F&O pre-market

1. Collect latest available pre-open context (indices, VIX, globals, FII/DII, option chain when available).  
2. Evaluate a configurable checklist.  
3. Produce a **transparent weighted score** → pre-market directional bias + confidence.  
4. Persist every report; compare against 9:15 / 9:30 reality for accuracy study.  
5. Never claim “market will definitely rise/fall” or “100% CALL/PUT”.  

Use terminology such as *pre-market directional bias*, *confidence*, *confirmation required*, *risk flags*.

## Data Sources

| Source | Used for |
|--------|----------|
| Yahoo Finance (`yfinance`) | Equity OHLCV, NIFTY / BANK NIFTY / India VIX, US/Asia indices, USDINR, crude, gold |
| NSE public HTTP APIs | FII/DII cash activity; index option chain (best-effort; may be blocked) |
| Optional `GIFT_NIFTY_SYMBOL` | Overnight indication **only if configured** — never invent values |
| Telegram Bot API | Existing notifier for equity + pre-market reports |

If a source fails: mark `DATA UNAVAILABLE`, continue the report, reduce confidence. Do not invent numbers.

## Analysis Philosophy

Prefer **deterministic, explainable heuristics** over black-box ML.

Equity signals: SMA50/200, RSI, volume spikes, news sentiment.  

Pre-market signals: gap class, index trend, globals, GIFT (if any), VIX, FII/DII, OI / PCR / OI change, support-resistance context, event/expiry flags — each with configurable weights in `scanner/config.py` / `.env`.

## Architecture Conventions

- Keep the equity pipeline in `scanner/{data_fetcher,indicators,news,evaluator,screener,runner,notifier}.py`.  
- Put F&O pre-market code under `scanner/premarket/`; do not rewrite unrelated equity modules.  
- Config via env / `.env` (`scanner/config.py`) — no new hardcoded secrets.  
- Persistence: SQLite at `PREMARKET_DB_PATH` (default `data/premarket.db`).  
- Scheduling: extend GitHub Actions (existing pattern); trading-day + duplicate guards in `premarket/calendar.py` + `premarket/db.py`.  
- Notifications: reuse `TelegramNotifier` — do not add a second framework.  

## Output Expectations

Equity: Pros / Cons cards + overview table.  

Pre-market: structured text report (bias, confidence, checklist, levels, scenarios, disclaimer). Accuracy dashboard only from **stored** outcomes — do not claim predictive accuracy without sample size.

## Development Guidelines

- Prefer deterministic rules over complex ML.  
- Keep computations explainable.  
- Avoid heavy dependencies unless clearly beneficial.  
- One failed API must not crash the whole report.  
- Add/extend tests under `tests/` for pure logic (gap, PCR, scoring, calendar, duplicates).  

## Scope

**In scope:** scanning equities, F&O pre-market research, confirmation tracking, local/CI automation.  

**Out of scope:** automated order placement, guaranteed outcomes, definitive price predictions.
