import requests
import logging
import html

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Handles sending notifications to Telegram."""
    
    def __init__(self, token=None, chat_id=None):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage" if self.token else None

    def is_configured(self):
        """Checks if both token and chat_id are present."""
        return bool(self.token and self.chat_id)

    def send_message(self, text, parse_mode="HTML"):
        """Sends a plain text or HTML message to Telegram."""
        if not self.is_configured():
            logger.warning("Telegram not configured. Skipping notification.")
            return False
        # Telegram imposes a per-message length limit (~4096 chars for text).
        # To avoid "message is too long" errors, we chunk and send sequentially.
        max_len = 3800  # leave headroom for safety
        chunks = []
        if len(text) <= max_len:
            chunks = [text]
        else:
            current = []
            current_len = 0
            for line in text.split("\n"):
                # +1 for the newline that will be re-added
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

        success = True

        total_parts = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            # Prepend a lightweight part indicator to help visually separate messages.
            if total_parts > 1:
                chunk_to_send = f"[{idx}/{total_parts}]\\n{chunk}"
            else:
                chunk_to_send = chunk

            payload = {
                "chat_id": self.chat_id,
                "text": chunk_to_send,
                "parse_mode": parse_mode,
            }

            try:
                response = requests.post(self.api_url, json=payload, timeout=10)

                if not response.ok:
                    logger.error(
                        "Telegram API error %s (part %s/%s): %s",
                        response.status_code,
                        idx,
                        len(chunks),
                        response.text,
                    )
                    response.raise_for_status()

                logger.info("Telegram notification part %s/%s sent successfully.", idx, len(chunks))
            except Exception as e:
                logger.error(f"Failed to send Telegram notification part {idx}/{len(chunks)}: {e}")
                success = False

        return success

    def format_analysis_report(self, results):
        """Formats the scan results into an HTML message for Telegram."""
        total = len(results)
        succeeded = sum(1 for r in results if r["status"] == "success")
        failed = total - succeeded

        counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        failures = []

        # First pass: classify signals and collect failures
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

        # Header / summary
        lines.append("<b>📊 Stockpy Analysis Report</b>")
        lines.append(f"<i>{succeeded} of {total} stocks analyzed successfully</i>")
        lines.append(
            f"🟢 Bullish: {counts['bullish']}   "
            f"🔴 Bearish: {counts['bearish']}   "
            f"🟡 Neutral: {counts['neutral']}"
        )
        lines.append("")

        # Per-ticker details
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

            # Price / trend line
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

            # Volume line
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

            # News sentiment
            if sentiment or summary:
                safe_summary = html.escape(str(summary), quote=False)
                lines.append(f"• News: {sentiment or 'Unknown'} — {safe_summary}")

            # Pros / cons (top 2 each)
            if pros:
                safe_pros = [html.escape(str(p), quote=False) for p in pros[:2]]
                for p in safe_pros:
                    lines.append(f"✅ {p}")
            if cons:
                safe_cons = [html.escape(str(c), quote=False) for c in cons[:2]]
                for c in safe_cons:
                    lines.append(f"⚠️ {c}")

            lines.append("")

        # Failures section
        if failures:
            lines.append("<b>⚠️ Failed Analyses</b>")
            for r in failures:
                ticker = r["ticker"]
                safe_ticker = html.escape(str(ticker), quote=False)
                error = r.get("error", "Unknown error")
                safe_error = html.escape(str(error), quote=False)
                lines.append(f"❌ <b>{safe_ticker}</b> — {safe_error}")
            lines.append("")

        return "\n".join(lines)
