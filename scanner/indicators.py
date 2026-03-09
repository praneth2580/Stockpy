import pandas as pd
import numpy as np

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for the provided DataFrame.
    """
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # Calculate Simple Moving Averages
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate RSI (14 day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # Handle division by zero
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    # Fill remaining NaNs for exact 0 loss cases where calculation might be infinite
    df['RSI'] = df['RSI'].fillna(100)
    
    # Volume average
    df['Volume_Avg_20'] = df['Volume'].rolling(window=20).mean()
    
    return df
