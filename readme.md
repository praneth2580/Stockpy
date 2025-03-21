# Stock Price Prediction App

## Overview

This is a **Stock Price Prediction App** that allows users to:

- Select a stock symbol (e.g., AAPL, TSLA, INFY.NS)
- Train an LSTM-based machine learning model on historical stock data
- Save the trained model for future use
- Predict live stock prices using the saved model
- Switch between different prediction models: `60_days.py`, `live.py`, and future additions

## Features

✅ Fetches **real-time stock market data** using Yahoo Finance
✅ Uses an **LSTM deep learning model** for predictions
✅ **Trains and saves** models for future use
✅ Provides **live stock price predictions**
✅ Supports **multiple prediction strategies**
✅ User-friendly **Streamlit UI**

## Installation

### **1️⃣ Install Dependencies**

Run the following command to install required packages:

```bash
pip install numpy pandas matplotlib yfinance tensorflow scikit-learn streamlit
```

### **2️⃣ Save the Scripts**

Ensure you have the following files in your project structure:

```
project-folder/
│── .venv/
│── saved_models/
│── src/
│   │── models/
│   │   │── 60_days.py
│   │   │── fando.py
│   │   │── live.py
│   │── main.py  # Handles switching between models
│── .gitignore
│── LICENSE
│── readme.md
```

## Usage

### **Run the Main Script**

Navigate to the folder where `main.py` is located and run:

```bash
python main.py
```

### **Using the Web UI for Individual Models**

To run individual models like `live.py` or `60_days.py`, use:

```bash
streamlit run src/models/60_days.py
streamlit run src/models/fando.py
streamlit run src/models/live.py
```

### **Workflow**

1. **Enter a stock symbol** (e.g., AAPL for Apple, TSLA for Tesla)
2. Select a **prediction model** (e.g., `60_days`, `live`)
3. Click **"Train Model"** to train and save a model for that stock
4. If a model is already trained, click **"Predict Live Price"** to fetch real-time predictions

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `tensorflow`
- `scikit-learn`
- `streamlit`
- `bs4`
- `requests`

## Notes

- **Ensure an internet connection** to fetch stock data.
- The `60_days` model predicts long-term trends, while `live` focuses on short-term real-time predictions.
- The app allows for easy **expansion** by adding more model files in `src/models/`.

## License

This project is open-source and available for personal and educational use. 🚀

