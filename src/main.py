import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
import pickle
import streamlit as st
import time
import threading
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

# Directory for saved models
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Global storage for live predictions
live_predictions = []

def fetch_stock_data(ticker, interval='1m', days=1):
    """Fetch latest stock data"""
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

def preprocess_data(data, time_steps=20):
    """Scale and prepare data for training/prediction"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i-time_steps:i])
        y.append(data_scaled[i, :])
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape, output_dim):
    """Build LSTM model"""
    model = Sequential([
        Bidirectional(LSTM(units=128, return_sequences=True, input_shape=input_shape)),
        Dropout(0.1),
        Bidirectional(LSTM(units=128, return_sequences=True)),
        Dropout(0.1),
        Bidirectional(LSTM(units=128)),
        Dropout(0.1),
        Dense(units=output_dim)
    ])
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_absolute_error')
    return model

def train_and_save_model(model, X_train, y_train, stock_name, scaler):
    """Train and save model"""
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, shuffle=False)
    model.save(os.path.join(MODEL_DIR, f"{stock_name}_model.h5"))
    with open(os.path.join(MODEL_DIR, f"{stock_name}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

def predict_next_10_mins(model, scaler, stock_name):
    """Predict stock price for the next 10 minutes"""
    data = fetch_stock_data(stock_name, interval='1m', days=1)
    X_live, _, _ = preprocess_data(data)
    X_live = X_live[-1:].reshape(1, X_live.shape[1], X_live.shape[2])

    predictions = []
    for _ in range(10):  # Predict for next 10 minutes
        pred = model.predict(X_live)
        predictions.append(pred[0])
        X_live = np.roll(X_live, -1, axis=1)  # Shift window
        X_live[:, -1, :] = pred  # Append latest prediction

    return scaler.inverse_transform(np.array(predictions))

def update_predictions(stock_name):
    """Thread function to update predictions every 5 minutes"""
    global live_predictions
    model_path = os.path.join(MODEL_DIR, f"{stock_name}_model.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{stock_name}_scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        while True:
            predictions = predict_next_10_mins(model, scaler, stock_name)
            timestamp = datetime.now().strftime("%H:%M:%S")
            live_predictions.append((timestamp, predictions[:, 3]))  # Store Close prices
            time.sleep(300)  # Wait 5 minutes

def start_prediction_thread(stock_name):
    """Start the background prediction thread"""
    thread = threading.Thread(target=update_predictions, args=(stock_name,), daemon=True)
    thread.start()

def plot_live_predictions():
    """Plot live predictions vs real prices"""
    if not live_predictions:
        st.write("No live data yet...")
        return

    timestamps, predicted_prices = zip(*live_predictions)

    # Fetch real stock data
    real_data = fetch_stock_data(stock_name, interval='1m', days=1)
    real_prices = real_data['Close'][-len(timestamps):]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, real_prices, label="Real Prices", marker='o')
    plt.plot(timestamps, predicted_prices, label="Predicted Prices", linestyle='dashed', marker='x')
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Live Stock Prediction")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

# Streamlit UI
st.title("Stock Price Prediction App")
st.sidebar.header("Select a Stock")
stock_name = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY.NS)", "AAPL")

data = fetch_stock_data(stock_name)
st.write(f"Showing latest data for: {stock_name}")
st.dataframe(data.tail())

if st.button("Train Model"):
    st.write("Fetching Data & Training Model...")
    X, y, scaler = preprocess_data(data)
    X_train, y_train = X[:-10], y[:-10]
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), output_dim=y_train.shape[1])
    train_and_save_model(model, X_train, y_train, stock_name, scaler)
    st.success(f"Model trained and saved for {stock_name}!")

saved_models = [f.split("_model.h5")[0] for f in os.listdir(MODEL_DIR) if f.endswith("_model.h5")]

if saved_models:
    st.sidebar.subheader("Previously Trained Models")
    selected_model = st.sidebar.selectbox("Select a model", saved_models)

    if st.button("Start Live Prediction"):
        start_prediction_thread(selected_model)

    if st.button("Show Live Graph"):
        plot_live_predictions()
