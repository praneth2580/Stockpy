import joblib
import numpy as np
import os
from app.model import MODEL_PATH

def online_train(recent_data, actual):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    x = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data['Close'].values
    model.fit(x, y)

    x_new = np.append(x, [[len(x)]], axis=0)
    y_new = np.append(y, [actual])
    model.fit(x_new, y_new)

    joblib.dump(model, MODEL_PATH)