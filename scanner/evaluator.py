import pandas as pd
import numpy as np

def evaluate_stock(ticker: str, df: pd.DataFrame, news: dict) -> dict:
    """
    Evaluates the stock based on calculated indicators and news sentiment.
    Returns a dictionary of pros, cons, and raw technical numbers.
    """
    pros = []
    cons = []
    technicals = {}

    if df is None or df.empty:
        return {"pros": [], "cons": ["Failed to fetch historical data."], "technicals": {}}

    latest = df.iloc[-1]

    # ─── Raw technical numbers ──────────────────────────
    technicals["close"] = round(float(latest["Close"]), 2)

    if "SMA50" in df.columns and not np.isnan(latest["SMA50"]):
        technicals["sma50"] = round(float(latest["SMA50"]), 2)
    if "SMA200" in df.columns and not np.isnan(latest["SMA200"]):
        technicals["sma200"] = round(float(latest["SMA200"]), 2)
    if "RSI" in df.columns and not np.isnan(latest["RSI"]):
        technicals["rsi"] = round(float(latest["RSI"]), 2)
    if "Volume" in df.columns:
        technicals["volume"] = int(latest["Volume"])
    if "Volume_Avg_20" in df.columns and not np.isnan(latest["Volume_Avg_20"]):
        technicals["volume_avg_20"] = int(latest["Volume_Avg_20"])
        if technicals["volume"] > 0 and technicals["volume_avg_20"] > 0:
            technicals["volume_ratio"] = round(
                technicals["volume"] / technicals["volume_avg_20"], 2
            )

    # ─── Trend evaluation ───────────────────────────────
    if "SMA50" in df.columns and "SMA200" in df.columns:
        if latest["SMA50"] > latest["SMA200"]:
            pros.append("Price is in a long-term uptrend (SMA50 > SMA200)")
        else:
            cons.append("Price is in a long-term downtrend (SMA50 < SMA200)")

    # ─── RSI evaluation ─────────────────────────────────
    if "RSI" in df.columns:
        rsi = latest["RSI"]
        if rsi > 70:
            cons.append(f"Stock is overbought (RSI {rsi:.2f})")
        elif rsi < 30:
            pros.append(f"Stock is potentially oversold (RSI {rsi:.2f})")
        elif 45 <= rsi <= 55:
            pros.append(f"RSI is neutral ({rsi:.2f})")
        elif rsi < 45:
            cons.append(f"RSI indicates weak momentum ({rsi:.2f})")
        else:
            pros.append(f"RSI is neutral ({rsi:.2f})")

    # ─── Volume evaluation ──────────────────────────────
    if "Volume" in df.columns and "Volume_Avg_20" in df.columns:
        if latest["Volume"] > latest["Volume_Avg_20"] * 1.5:
            pros.append("Recent trading volume is significantly above average")

    # ─── News sentiment ─────────────────────────────────
    sentiment = news.get("sentiment", "Unknown")
    summary = news.get("summary", "")

    if sentiment == "Positive":
        pros.append(f"Positive news sentiment: {summary}")
    elif sentiment == "Negative":
        cons.append(f"Negative news sentiment: {summary}")
    else:
        # We'll still return it for technicals/UI if needed, 
        # but only Pros/Cons affect the Signal
        pass

    return {
        "pros": pros,
        "cons": cons,
        "technicals": technicals,
        "news": {"sentiment": sentiment, "summary": summary}
    }
