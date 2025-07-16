import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
import streamlit as st

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

def preprocess_data(data, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i-time_steps:i])
        y.append(data_scaled[i, 3])  # Predict Close price
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.05),
        LSTM(units=100, return_sequences=True),
        Dropout(0.05),
        LSTM(units=100),
        Dropout(0.05),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_improve(model, X_train, y_train, X_test, y_test, scaler, iterations=5):
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1, shuffle=False)
        y_pred = model.predict(X_test)
        
        # Correcting the scaling issue
        y_pred_rescaled = scaler.inverse_transform(np.column_stack((np.zeros((len(y_pred), 4)), y_pred)))[:, 4]
        y_test_rescaled = scaler.inverse_transform(np.column_stack((np.zeros((len(y_test), 4)), y_test)))[:, 4]
        
        error = np.mean(abs(y_test_rescaled - y_pred_rescaled))
        print(f"Iteration {i+1} Error: {error}")
    return model, y_pred_rescaled, y_test_rescaled

def main():
    ticker = "AAPL"
    start, end = "2023-01-01", "2024-01-01"
    data = fetch_stock_data(ticker, start, end)
    
    time_steps = 60
    X, y, scaler = preprocess_data(data, time_steps)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model, y_pred, y_real = train_and_improve(model, X_train, y_train, X_test, y_test, scaler, iterations=5)
    
    plt.figure(figsize=(14, 5))
    plt.plot(y_real, color='blue', label='Actual Stock Price')
    plt.plot(y_pred, color='red', linestyle='dashed', label='Predicted Stock Price')
    plt.title(f"Stock Price Prediction vs Real Data ({ticker})")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.legend()
    fig, ax = plt.subplots()
    ax.plot(y_real, color='blue', label='Actual Stock Price')
    ax.plot(y_pred, color='red', linestyle='dashed', label='Predicted Stock Price')
    st.pyplot(fig)
    # plt.show()

if __name__ == "__main__":
    main()
