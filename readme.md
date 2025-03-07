# Stock Price Prediction App

## Overview
This is a **Stock Price Prediction App** that allows users to:
- Select a stock symbol (e.g., AAPL, TSLA, INFY.NS)
- Train an LSTM-based machine learning model on historical stock data
- Save the trained model for future use
- Predict live stock prices using the saved model

## Features
✅ Fetches **real-time stock market data** using Yahoo Finance
✅ Uses an **LSTM deep learning model** for predictions
✅ **Trains and saves** models for future use
✅ Provides **live stock price predictions**
✅ User-friendly **Streamlit UI**

## Installation
### **1️⃣ Install Dependencies**
Run the following command to install required packages:
```bash
pip install numpy pandas matplotlib yfinance tensorflow scikit-learn streamlit
```

### **2️⃣ Save the Script**
Save the updated Python script as `app.py`.

## Usage
### **Run the Streamlit App**
Navigate to the folder where `app.py` is saved and run:
```bash
streamlit run app.py
```

### **Using the Web UI**
1. **Enter a stock symbol** (e.g., AAPL for Apple, TSLA for Tesla)
2. Click **"Train Model"** to train and save a model for that stock
3. If a model is already trained, click **"Predict Live Price"** to fetch real-time predictions

## Dependencies
- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `tensorflow`
- `scikit-learn`
- `streamlit`

## Notes
- **Make sure you have a stable internet connection** to fetch stock data.
- The model is trained on the **last 60 days of 5-minute interval stock data**.
- The app automatically saves trained models for each stock separately.

## License
This project is open-source and available for personal and educational use. 🚀

