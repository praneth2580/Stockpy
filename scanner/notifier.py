import requests
import logging

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
            
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Telegram notification sent successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False

    def format_analysis_report(self, results):
        """Formats the scan results into an HTML message for Telegram."""
        message = "<b>📊 Stockpy Analysis Report</b>\n\n"
        
        for r in results:
            ticker = r["ticker"]
            if r["status"] == "failed":
                message += f"❌ <b>{ticker}</b>: Analysis Failed\n"
                continue
            
            ev = r["evaluation"]
            tech = ev.get("technicals", {})
            pros = ev.get("pros", [])
            cons = ev.get("cons", [])
            
            # Signal
            if len(pros) > len(cons):
                signal = "🟢 Bullish"
            elif len(cons) > len(pros):
                signal = "🔴 Bearish"
            else:
                signal = "🟡 Neutral"
                
            close = tech.get("close", 0)
            rsi = tech.get("rsi", 0)
            
            message += f"<b>{ticker}</b> — {signal}\n"
            message += f"Price: ₹{close:,.2f} | RSI: {rsi:.1f}\n"
            
            if pros:
                message += f"✅ {pros[0]}\n" # Just show top pro to keep it concise
            if cons:
                message += f"⚠️ {cons[0]}\n" # Just show top con
            
            message += "\n"
            
        return message
