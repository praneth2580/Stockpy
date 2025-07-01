# python-ml-service/src/prediction_logic.py
import joblib # Or pickle, tensorflow.keras.models, etc.
import pandas as pd
import numpy as np
import os

# Define the path to your model file.
# When deployed via Docker, 'models' will be a subdirectory within /app
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'stock_predictor_model.pkl')
MODEL = None # Initialize model as None

def load_model():
    """Loads the pre-trained machine learning model."""
    global MODEL
    if MODEL is None:
        try:
            MODEL = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {MODEL_PATH}")
            # You might want to raise an exception or handle this more robustly
            MODEL = None
        except Exception as e:
            print(f"Error loading model: {e}")
            MODEL = None
    return MODEL

def preprocess_input(data: dict):
    """
    Preprocesses input data for the model.
    This is a placeholder; replace with your actual preprocessing.
    `data` might contain features like 'open_price', 'high_price', 'volume', etc.
    """
    # Example: Create a DataFrame from the input dictionary
    df = pd.DataFrame([data])
    # Ensure columns match your training features.
    # This is a very basic example; your actual preprocessing will be complex.
    # For example, you might need to handle dates, create lagged features, normalize, etc.
    required_features = ['feature1', 'feature2', 'feature3'] # Replace with your actual model features
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0 # Or some default/imputed value

    return df[required_features] # Return only the features your model expects

def predict_stock(input_data: dict):
    """
    Makes a prediction using the loaded model.
    """
    model = load_model()
    if model is None:
        return {"error": "Model not loaded. Cannot make prediction."}

    try:
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

# You might want to call load_model() once when the service starts
# For Uvicorn/Gunicorn, this might be handled by FastAPI's startup event,
# or simply by the first call if `MODEL` is initialized globally.