# GitHub Actions for Stockpy
#
# Workflows
# ---------
# ci.yml                 — pytest + import smoke on push/PR
# stockpy-daily.yml      — Mon–Fri equity Top-50 (cron ~15:00 IST, aims ~15:30)
# stockpy-premarket.yml  — Mon–Fri F&O (cron ~08:00/08:15/08:30 IST, aims ~09:xx)
#
# Crons are scheduled ~30 minutes early because GitHub Actions is often late.
#
# Required secrets (Settings → Secrets and variables → Actions)
# ------------------------------------------------------------
# TELEGRAM_BOT_TOKEN   Bot token from @BotFather
# TELEGRAM_CHAT_ID     @channelusername or numeric chat id
#
# Manual runs
# -----------
# Actions → StockPy F&O Pre-Market → Run workflow
#   job: premarket | open | confirm | accuracy
#   send_ping: true only if you want a connectivity test message
#
# Notes
# -----
# - Scheduled runs do NOT send Telegram pings (avoids chat clutter).
# - Only the glance / confirmation / equity summary is sent.
# - Job type is chosen from github.event.schedule, not wall-clock IST.
# - Look for `[telegram]` lines in the job log for send status.
# - Manual workflow_dispatch always uses --force.
# - Premarket DB artifact `premarket-db` is restored best-effort between runs.
# - Keep scanner/premarket/calendar.py holidays updated yearly.
