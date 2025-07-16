import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import datetime
import streamlit as st

# Constants
STOCK_SYMBOL = "NIFTY"
EXPIRY_DAYS = 30  # Look for nearest expiry
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------- Fetch Option Chain Data ----------------------
def get_option_chain(stock_symbol):
    """Fetch option chain data from NSE/BSE"""
    url = f"https://www.nseindia.com/option-chain"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error("Failed to fetch option chain data")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = pd.read_html(str(soup))
    
    option_chain = tables[0]  # Assuming the first table is the option chain
    return option_chain

# ---------------------- Fetch Historical Stock Data ----------------------
def fetch_historical_data(stock_symbol, days=60):
    """Get past data for training"""
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=days)
    
    data = yf.download(stock_symbol, start=start_date.strftime('%Y-%m-%d'), 
                       end=end_date.strftime('%Y-%m-%d'))
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

# ---------------------- Preprocess Data ----------------------
def preprocess_data(data, time_steps=60):
    """Scale and reshape data for LSTM"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i-time_steps:i])
        y.append(data_scaled[i, 3])  # Predict Close Price
    
    return np.array(X), np.array(y), scaler

# ---------------------- Build LSTM Model ----------------------
def build_lstm_model(input_shape):
    """Create an LSTM model for predicting stock trends"""
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.1),
        LSTM(units=100, return_sequences=False),
        Dropout(0.1),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ---------------------- Train Model ----------------------
def train_model(stock_symbol):
    """Train and save an LSTM model"""
    data = fetch_historical_data(stock_symbol)
    if data is None or data.empty:
        print(f"No data available for {stock_symbol}")
        return
    
    X, y, scaler = preprocess_data(data)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    model.save(os.path.join(MODEL_DIR, f"{stock_symbol}_model.h5"))
    return model, scaler

# ---------------------- Predict Next Price ----------------------
def predict_next_price(stock_symbol):
    """Predict the next stock price and find best F&O"""
    model_path = os.path.join(MODEL_DIR, f"{stock_symbol}_model.h5")
    
    if not os.path.exists(model_path):
        print(f"Training new model for {stock_symbol}...")
        model, scaler = train_model(stock_symbol)
    else:
        model = load_model(model_path)
    
    data = fetch_historical_data(stock_symbol, days=60)
    X_live, _, scaler = preprocess_data(data)
    X_live = X_live[-1:].reshape(1, X_live.shape[1], X_live.shape[2])
    
    predicted_price = model.predict(X_live)
    predicted_price_rescaled = scaler.inverse_transform(
        np.column_stack((np.zeros((1, 4)), predicted_price)))[:, 4][0]
    
    return predicted_price_rescaled

# ---------------------- Generate F&O Recommendations ----------------------
def suggest_fno_trades(stock_symbol):
    """Suggest best Futures & Options to Buy/Sell"""
    option_chain = get_option_chain(stock_symbol)
    if option_chain is None:
        return []
    
    predicted_price = predict_next_price(stock_symbol)
    current_price = yf.Ticker(stock_symbol).history(period="1d")['Close'].iloc[-1]
    
    print(f"📊 Current Price: {current_price} | 📈 Predicted: {predicted_price}")
    
    if predicted_price > current_price * 1.02:
        # Suggest Call Options (Bullish)
        options = option_chain[(option_chain['Strike Price'] > current_price) & 
                               (option_chain['CE Open Interest'] > 1000)]
        return options[['Strike Price', 'CE Last Price', 'CE Open Interest']].head(3).to_dict(orient='records')
    
    elif predicted_price < current_price * 0.98:
        # Suggest Put Options (Bearish)
        options = option_chain[(option_chain['Strike Price'] < current_price) & 
                               (option_chain['PE Open Interest'] > 1000)]
        return options[['Strike Price', 'PE Last Price', 'PE Open Interest']].head(3).to_dict(orient='records')
    
    else:
        return "No strong BUY/SELL signal detected."

# ---------------------- Streamlit UI ----------------------
st.title("📊 Futures & Options Predictor")
stock_name = st.sidebar.text_input("Enter Stock Symbol", "NIFTY")

if st.button("Predict Best F&O Trades"):
    recommendations = suggest_fno_trades(stock_name)
    st.write(recommendations)

def main():
    """Main function to run the Streamlit app."""
    st.title("📊 Futures & Options Predictor")
    stock_name = st.sidebar.text_input("Enter Stock Symbol", "NIFTY", key="fando_stock_input")

    if st.button("Predict Best F&O Trades", key="fando_predict_button"):
        recommendations = suggest_fno_trades(stock_name)
        st.write(recommendations)

if __name__ == "__main__":
    main()

