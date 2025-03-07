import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
import pickle
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

def fetch_stock_data(ticker, interval='5m', days=59):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

def preprocess_data(data, time_steps=20):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i-time_steps:i])
        y.append(data_scaled[i, :])
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape, output_dim):
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

def train_and_save_model(model, X_train, y_train, stock_name):
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, shuffle=False)
    model.save(f"{stock_name}_model.h5")

def predict_live(model, scaler, stock_name):
    data = fetch_stock_data(stock_name, interval='1m', days=2)
    X_live, _, _ = preprocess_data(data)
    X_live = X_live[-1:].reshape(1, X_live.shape[1], X_live.shape[2])
    prediction = model.predict(X_live)
    return scaler.inverse_transform(prediction)[0]

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
    train_and_save_model(model, X_train, y_train, stock_name)
    with open(f"{stock_name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    st.success(f"Model trained and saved for {stock_name}!")

if os.path.exists(f"{stock_name}_model.h5") and os.path.exists(f"{stock_name}_scaler.pkl"):
    if st.button("Predict Live Price"):
        st.write("Loading Model & Predicting Live Data...")
        model = load_model(f"{stock_name}_model.h5")
        with open(f"{stock_name}_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        prediction = predict_live(model, scaler, stock_name)
        st.write(f"Predicted Prices: Open={prediction[0]:.2f}, High={prediction[1]:.2f}, Low={prediction[2]:.2f}, Close={prediction[3]:.2f}")
