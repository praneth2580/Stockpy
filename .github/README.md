# GitHub Actions for Stockpy
#
# Workflows
# ---------
# ci.yml                 — pytest + import smoke on push/PR
# stockpy-daily.yml      — Mon–Fri ~15:30 IST equity Top-50 scan
# stockpy-premarket.yml  — Mon–Fri 09:00 / 09:15 / 09:30 IST F&O jobs
#
# Required secrets (Settings → Secrets and variables → Actions)
# ------------------------------------------------------------
# TELEGRAM_BOT_TOKEN   Bot token from @BotFather
# TELEGRAM_CHAT_ID     Destination chat / channel id
#
# Optional: leave secrets empty to run jobs without Telegram
# (notifier skips when unset).
#
# Manual runs
# -----------
# Actions → StockPy F&O Pre-Market → Run workflow
#   job: premarket | open | confirm | accuracy
#   force: true  → ignore holiday / duplicate locks (testing)
#
# Notes
# -----
# - "Telegram ping" step actually sends a short test message. If that step
#   is green, you should see "Stockpy Telegram ping" in Telegram.
#   If it fails, secrets/bot/chat are wrong — fix before expecting reports.
# - Look for lines starting with `[telegram]` in the job log for send status.
# - Manual workflow_dispatch always uses --force so holiday/duplicate locks
#   do not silently skip the report.
# - Premarket DB is uploaded as artifact `premarket-db` and restored
#   on the next run (best-effort) so accuracy history can accumulate.
# - NSE holidays use scanner/premarket/calendar.py — keep that list current.
# - Cron times are UTC; jobs set TZ=Asia/Kolkata for calendar checks.
