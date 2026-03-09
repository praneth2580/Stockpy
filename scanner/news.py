import yfinance as yf
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_analyzer = SentimentIntensityAnalyzer()


def fetch_news_sentiment(ticker: str) -> dict:
    """
    Fetches real news for a ticker via yfinance and analyzes sentiment
    using VADER (Valence Aware Dictionary and sEntiment Reasoner).

    Returns:
        dict with keys:
            - sentiment: "Positive", "Negative", or "Neutral"
            - score: compound sentiment score (-1.0 to 1.0)
            - summary: brief summary text
            - headlines: list of (title, individual_score) tuples
    """
    logger.info(f"Fetching news for {ticker}")

    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news or []
    except Exception as e:
        logger.warning(f"Failed to fetch news for {ticker}: {e}")
        return {
            "sentiment": "Neutral",
            "score": 0.0,
            "summary": "Unable to fetch news.",
            "headlines": [],
        }

    if not news_items:
        logger.debug(f"No news articles found for {ticker}")
        return {
            "sentiment": "Neutral",
            "score": 0.0,
            "summary": "No recent news found.",
            "headlines": [],
        }

    # Extract titles from news items
    headlines = []
    scores = []

    for item in news_items:
        # yfinance news structure may vary across versions
        title = item.get("title", "") or item.get("content", {}).get("title", "")
        if not title:
            continue

        vs = _analyzer.polarity_scores(title)
        compound = vs["compound"]
        headlines.append((title, round(compound, 3)))
        scores.append(compound)
        logger.debug(f"  [{compound:+.3f}] {title}")

    if not scores:
        return {
            "sentiment": "Neutral",
            "score": 0.0,
            "summary": "No parseable headlines found.",
            "headlines": [],
        }

    # Aggregate: average compound score across all headlines
    avg_score = sum(scores) / len(scores)

    if avg_score >= 0.02:
        sentiment = "Positive"
        summary = f"News sentiment is positive ({avg_score:+.3f} avg)."
    elif avg_score <= -0.02:
        sentiment = "Negative"
        summary = f"News sentiment is negative ({avg_score:+.3f} avg)."
    else:
        sentiment = "Neutral"
        summary = f"News sentiment is mixed/neutral ({avg_score:+.3f} avg)."

    logger.info(f"{ticker}: {sentiment} ({avg_score:+.3f})")

    return {
        "sentiment": sentiment,
        "score": round(avg_score, 3),
        "summary": summary,
        "headlines": headlines,
    }
