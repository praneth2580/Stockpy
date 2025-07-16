import streamlit as st
import pandas as pd
import sqlite3
from app.model import load_model, predict_index
from app.strategy import suggest_strategy
from app.train import online_train
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="Index Predictor", layout="centered")
st.title("📈 Real-Time Index Predictor with Strategy Suggestions")

DB_PATH = "app/predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            predicted REAL,
            actual REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(symbol, predicted, actual):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (timestamp, symbol, predicted, actual)
        VALUES (?, ?, ?, ?)
    """, (datetime.datetime.now().isoformat(), symbol, predicted, actual))
    conn.commit()
    conn.close()

def get_prediction_history(symbol):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions WHERE symbol = ? ORDER BY timestamp", conn, params=(symbol,))
    conn.close()
    return df

init_db()

symbol = st.text_input("Enter Index Symbol (e.g., ^NSEI, ^BSESN):", value="^NSEI")

if st.button("Predict Now"):
    with st.spinner("Fetching data and predicting..."):
        df = yf.download(symbol, period="7d", interval="1h")
        df = df.dropna()
        recent_data = df[['Close']].tail(50)

        model = load_model()
        prediction = float(predict_index(model, recent_data))
        strategy, profit, loss = suggest_strategy(prediction)

        st.subheader("Prediction Result")
        st.write(f"Next predicted value: **{prediction:.2f}**")
        st.write(f"Strategy: **{strategy}**")
        st.write(f"Expected Profit: ₹{profit} | Expected Loss: ₹{loss}")

        if st.button("Update Model with Latest Data"):
            actual = float(df['Close'].iloc[-1])
            online_train(recent_data, actual)
            log_prediction(symbol, prediction, actual)
            st.success("Model updated with actual data!")

# Realtime graph display
st.subheader("📊 Prediction vs Actual Trend")
history = get_prediction_history(symbol)
if not history.empty:
    fig, ax = plt.subplots()
    ax.plot(history['timestamp'], history['predicted'], label='Predicted', color='blue', marker='o')
    ax.plot(history['timestamp'], history['actual'], label='Actual', color='green', linestyle='--', marker='x')
    ax.set_title("Predicted vs Actual Index Value")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.info("No historical prediction data yet. Run a prediction and update it.")