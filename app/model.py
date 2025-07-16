import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
import os

MODEL_PATH = "app/model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return LinearRegression()

def predict_index(model, recent_data):
    data = np.arange(len(recent_data)).reshape(-1, 1)
    target = recent_data['Close'].values
    model.fit(data, target)
    next_time = [[len(recent_data)]]
    return float(model.predict(next_time)[0])