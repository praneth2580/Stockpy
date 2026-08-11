# Real-Life Quality Evaluation — Stockpy

How Stockpy stacks up for real research use after the equity scanner **and** F&O pre-market extension.

---

## Technical Architecture — Grade: A-

### Strengths
- Modular equity pipeline (`data_fetcher` → `indicators` → `news` → `evaluator` / `screener`)
- Parallel scans via `ThreadPoolExecutor`
- F&O logic isolated under `scanner/premarket/` without rewriting the equity path
- Env-backed config (`.env` / `scanner/config.py`), SQLite history, retry helpers, per-source failure isolation
- GitHub Actions for equity daily + IST pre-market / open / confirm slots
- pytest coverage for gap, OI/PCR, scoring, calendar, duplicates, accuracy helpers

### Remaining concerns
- `yfinance` can rate-limit or lag (fine for research, not for low-latency trading)
- NSE option-chain endpoints are fragile from some IPs/clouds — design correctly degrades to `DATA UNAVAILABLE`
- Shipping Telegram credentials should stay in env/secrets only (prefer `.env` / GitHub Secrets over defaults)

---

## Analysis Methodology — Grade: B+

### Equity
- Classic SMA50/200 + RSI + volume spike heuristics remain clear and explainable
- Live VADER news (not a stub)

### F&O pre-market (addresses earlier gaps)
| Earlier gap | Status |
|-------------|--------|
| NIFTY / market context | Covered — index snapshot + globals + VIX |
| Support / resistance | Covered — PDH/L, pivots, OI levels with sources |
| Volatility | Covered — India VIX + regime |
| Backtesting / tracking | Partial — reports + 9:30 confirmation stored; accuracy dashboard needs time to accumulate samples |

Scoring is weighted and transparent; language is bias/confidence — not “guaranteed CALL”.

---

## User Experience — Grade: A

- Rich interactive menus including F&O report + accuracy
- CLI flags for cron/Actions (`--premarket`, `--premarket-open`, `--premarket-confirm`)
- Telegram reuse for both equity and pre-market

No web dashboard yet — accuracy is CLI/`--premarket-accuracy` (fits the current CLI-first architecture).

---

## Final Verdict

| Purpose | Ready? | Why |
| :--- | :--- | :--- |
| Learning / practice | ✅ Excellent | Clean pipelines + tests |
| Daily equity research | ✅ Good (with chart confirmation) | Pros/Cons + top-N screen |
| F&O pre-open context | ✅ Useful hypothesis tool | Bias + checklist + confirmations |
| Automated trading | ❌ Not in scope | No execution, sizing, or guaranteed outcomes |

---

## Next upgrades (optional)

1. Reliable GIFT Nifty symbol/source when you have one (`GIFT_NIFTY_SYMBOL`)
2. Persist Actions `data/premarket.db` across runs (artifact or remote store) for long-run accuracy
3. Enrich later-day outcomes automatically (EOD close vs bias) for stronger backtests
4. Keep NSE holiday list updated each calendar year

> **Conclusion:** Stockpy is a solid Indian-market **research assistant**. The pre-market module fills the index/OI/VIX gap called out earlier, as long as you treat 9:00 output as a hypothesis and wait for open confirmation.
