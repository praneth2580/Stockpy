# python-ml-service/src/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .prediction_logic import predict_stock, load_model # Note the relative import

app = FastAPI(title="Stock Prediction ML Service")

# Pydantic model for request body validation
class StockFeatures(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add all features your model expects
    # Example:
    # open_price: float
    # high_price: float
    # low_price: float
    # close_price: float
    # volume: float
    # Add date if your model uses it (e.g., date_str: str)


@app.on_event("startup")
async def startup_event():
    """Load the model when the FastAPI application starts."""
    print("Loading ML model on startup...")
    load_model()
    print("ML model loaded.")

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ML service is running!"}

@app.post("/predict")
async def get_prediction(features: StockFeatures):
    """
    Endpoint to get stock predictions.
    Accepts stock features and returns a prediction.
    """
    input_data = features.dict() # Convert Pydantic model to dictionary
    result = predict_stock(input_data)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result