"""Telegram notification delivery with CI-friendly logging."""

from __future__ import annotations

import html
import logging
import os

import requests

logger = logging.getLogger(__name__)


def _mask_token(token: str | None) -> str:
    if not token:
        return "(empty)"
    t = token.strip()
    if len(t) <= 10:
        return "***"
    return f"{t[:6]}…{t[-4:]} (len={len(t)})"


def _mask_chat(chat_id: str | None) -> str:
    if not chat_id:
        return "(empty)"
    c = str(chat_id).strip()
    if c.startswith("@"):
        return c
    if len(c) <= 4:
        return "***"
    return f"{c[:2]}…{c[-2:]} (len={len(c)})"


def _ci_print(msg: str) -> None:
    """Always visible in GitHub Actions logs (bypasses log-level filters)."""
    print(f"[telegram] {msg}", flush=True)
    logger.info("%s", msg)


class TelegramNotifier:
    """Handles sending notifications to Telegram."""

    def __init__(self, token=None, chat_id=None):
        self.token = (token or "").strip() or None
        self.chat_id = str(chat_id).strip() if chat_id not in (None, "") else None
        self.api_url = (
            f"https://api.telegram.org/bot{self.token}/sendMessage" if self.token else None
        )
        self._log_config_once()

    def _log_config_once(self) -> None:
        token_env = bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip())
        chat_env = bool(os.getenv("TELEGRAM_CHAT_ID", "").strip())
        _ci_print(
            "config: "
            f"token_set={bool(self.token)} token={_mask_token(self.token)} "
            f"token_from_env={token_env} | "
            f"chat_id_set={bool(self.chat_id)} chat_id={_mask_chat(self.chat_id)} "
            f"chat_from_env={chat_env} | "
            f"GITHUB_ACTIONS={os.getenv('GITHUB_ACTIONS', 'false')}"
        )
        if self.chat_id and self.chat_id.startswith("@"):
            _ci_print(
                "hint: chat_id looks like @username — works for public channels/groups; "
                "private DMs need a numeric chat id. Bot must be added as admin for channels."
            )
        if not self.token or not self.chat_id:
            missing = []
            if not self.token:
                missing.append("TELEGRAM_BOT_TOKEN")
            if not self.chat_id:
                missing.append("TELEGRAM_CHAT_ID")
            _ci_print(
                f"WARNING: Telegram not fully configured (missing: {', '.join(missing)}). "
                "Messages will be skipped."
            )

    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id and self.api_url)

    def send_message(self, text, parse_mode="HTML") -> bool:
        """Sends a plain text or HTML message to Telegram."""
        if not self.is_configured():
            _ci_print("SKIP: not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID secrets.")
            return False

        if not text or not str(text).strip():
            _ci_print("SKIP: empty message body.")
            return False

        text = str(text)
        max_len = 3800
        chunks = self._chunk(text, max_len)
        total_parts = len(chunks)
        _ci_print(
            f"sending {total_parts} part(s), total_chars={len(text)}, "
            f"parse_mode={parse_mode}, chat_id={_mask_chat(self.chat_id)}"
        )

        success = True
        for idx, chunk in enumerate(chunks, start=1):
            if total_parts > 1:
                chunk_to_send = f"[{idx}/{total_parts}]\n{chunk}"
            else:
                chunk_to_send = chunk

            ok = self._post(chunk_to_send, parse_mode=parse_mode, part=idx, total=total_parts)
            if not ok and parse_mode:
                _ci_print(f"part {idx}: HTML/Markdown failed — retrying as plain text…")
                ok = self._post(chunk_to_send, parse_mode=None, part=idx, total=total_parts)
            if not ok:
                success = False

        if success:
            _ci_print("SUCCESS: all parts delivered.")
        else:
            _ci_print("FAILURE: one or more parts were not delivered. See errors above.")
            if os.getenv("GITHUB_ACTIONS"):
                chat = self.chat_id or ""
                if str(chat).startswith("@"):
                    hint = (
                        f"Chat id {chat} looks like a public channel/group. "
                        "In Telegram: open the channel → Administrators → Add Admin → "
                        "select your bot → enable 'Post Messages'. "
                        "Then re-run the workflow."
                    )
                else:
                    hint = (
                        "For a private chat: open the bot and press Start, then use the numeric chat id. "
                        "For a channel: use @channelusername and add the bot as admin with Post Messages."
                    )
                print(
                    f"::error title=Telegram notification failed::{hint}",
                    flush=True,
                )
        return success

    def _post(self, text: str, *, parse_mode: str | None, part: int, total: int) -> bool:
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            response = requests.post(self.api_url, json=payload, timeout=15)
            body = response.text
            try:
                data = response.json()
            except Exception:
                data = {"raw": body[:500]}

            if response.ok and data.get("ok", True):
                _ci_print(f"part {part}/{total}: OK (HTTP {response.status_code})")
                return True

            description = data.get("description") if isinstance(data, dict) else body
            error_code = data.get("error_code") if isinstance(data, dict) else None
            _ci_print(
                f"part {part}/{total}: FAILED HTTP {response.status_code} "
                f"error_code={error_code} description={description}"
            )
            logger.error("Telegram API full response: %s", body[:2000])
            return False
        except requests.Timeout:
            _ci_print(f"part {part}/{total}: FAILED — request timed out")
            return False
        except Exception as e:
            _ci_print(f"part {part}/{total}: FAILED — {type(e).__name__}: {e}")
            logger.exception("Telegram send exception")
            return False

    @staticmethod
    def _chunk(text: str, max_len: int) -> list[str]:
        if len(text) <= max_len:
            return [text]
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for line in text.split("\n"):
            line_len = len(line) + 1
            if current and current_len + line_len > max_len:
                chunks.append("\n".join(current))
                current = [line]
                current_len = line_len
            else:
                current.append(line)
                current_len += line_len
        if current:
            chunks.append("\n".join(current))
        return chunks

    def format_analysis_report(self, results):
        """Formats the scan results into an HTML message for Telegram."""
        total = len(results)
        succeeded = sum(1 for r in results if r["status"] == "success")

        counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        failures = []

        for r in results:
            if r["status"] != "success":
                failures.append(r)
                continue

            ev = r["evaluation"]
            pros = ev.get("pros", [])
            cons = ev.get("cons", [])

            if len(pros) > len(cons):
                counts["bullish"] += 1
            elif len(cons) > len(pros):
                counts["bearish"] += 1
            else:
                counts["neutral"] += 1

        lines = []
        lines.append("<b>📊 Stockpy Analysis Report</b>")
        lines.append(f"<i>{succeeded} of {total} stocks analyzed successfully</i>")
        lines.append(
            f"🟢 Bullish: {counts['bullish']}   "
            f"🔴 Bearish: {counts['bearish']}   "
            f"🟡 Neutral: {counts['neutral']}"
        )
        lines.append("")

        for r in results:
            ticker = r["ticker"]
            safe_ticker = html.escape(str(ticker), quote=False)

            if r["status"] != "success":
                continue

            ev = r["evaluation"]
            tech = ev.get("technicals", {})
            pros = ev.get("pros", [])
            cons = ev.get("cons", [])
            news = ev.get("news", {}) or {}
            sentiment = news.get("sentiment", "Unknown")
            summary = news.get("summary", "")

            if len(pros) > len(cons):
                signal = "🟢 Bullish"
            elif len(cons) > len(pros):
                signal = "🔴 Bearish"
            else:
                signal = "🟡 Neutral"

            close = tech.get("close")
            sma50 = tech.get("sma50")
            sma200 = tech.get("sma200")
            rsi = tech.get("rsi")
            vol = tech.get("volume")
            vol_ratio = tech.get("volume_ratio")

            lines.append(f"<b>{safe_ticker}</b> — {signal}")

            price_bits = []
            if close is not None:
                price_bits.append(f"₹{close:,.2f}")
            if sma50 is not None:
                price_bits.append(f"SMA50: ₹{sma50:,.2f}")
            if sma200 is not None:
                price_bits.append(f"SMA200: ₹{sma200:,.2f}")
            if rsi is not None:
                price_bits.append(f"RSI: {rsi:.1f}")

            if price_bits:
                lines.append("• " + " | ".join(price_bits))

            vol_bits = []
            if vol is not None:
                if vol >= 1_000_000:
                    vol_bits.append(f"Vol: {vol / 1_000_000:.2f}M")
                else:
                    vol_bits.append(f"Vol: {vol:,}")
            if vol_ratio is not None:
                vol_bits.append(f"Vol vs 20d: {vol_ratio:.2f}x")

            if vol_bits:
                lines.append("• " + " | ".join(vol_bits))

            if sentiment or summary:
                safe_summary = html.escape(str(summary), quote=False)
                lines.append(f"• News: {sentiment or 'Unknown'} — {safe_summary}")

            if pros:
                for p in [html.escape(str(p), quote=False) for p in pros[:2]]:
                    lines.append(f"✅ {p}")
            if cons:
                for c in [html.escape(str(c), quote=False) for c in cons[:2]]:
                    lines.append(f"⚠️ {c}")

            lines.append("")

        if failures:
            lines.append("<b>⚠️ Failed Analyses</b>")
            for r in failures:
                safe_ticker = html.escape(str(r["ticker"]), quote=False)
                safe_error = html.escape(str(r.get("error", "Unknown error")), quote=False)
                lines.append(f"❌ <b>{safe_ticker}</b> — {safe_error}")
            lines.append("")

        return "\n".join(lines)


def send_or_log(notifier: TelegramNotifier, text: str, *, label: str = "notification") -> bool:
    """Helper used by CLI/CI paths — always prints outcome."""
    _ci_print(f"--- starting {label} ---")
    ok = notifier.send_message(text)
    _ci_print(f"--- finished {label}: {'OK' if ok else 'FAILED'} ---")
    return ok


def ping_telegram(token: str | None = None, chat_id: str | None = None) -> bool:
    """
    Send a short connectivity test message.
    Used by GitHub Actions to prove secrets + bot access before the main job.
    """
    token = token if token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "")
    notifier = TelegramNotifier(token, chat_id)
    if not notifier.is_configured():
        return False
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return send_or_log(
        notifier,
        f"<b>Stockpy Telegram ping</b>\nConnectivity OK at {now}",
        label="ci ping",
    )
