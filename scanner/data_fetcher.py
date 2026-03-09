import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetches historical stock data for the given ticker.
    """
    try:
        logger.info(f"Fetching data for {ticker} over {period}")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return None
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None
