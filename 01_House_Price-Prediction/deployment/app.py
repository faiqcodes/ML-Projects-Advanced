from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd 
import numpy as np 
from pathlib import Path
from typing import Dict, Any

app = FastAPI(title="House Price Prediction API")

current_dir = Path(__file__).resolve().parent

model = joblib.load(current_dir / "house_price_model.pkl")
scaler = joblib.load(current_dir / "scaler.pkl")
feature_columns = joblib.load(current_dir / "feature_columns.pkl")

# Dynamically create the model from feature_columns
field_definitions = {col: (float, 0) for col in feature_columns}
HouseInput = type('HouseInput', (BaseModel,), {
    '__annotations__': {col: float for col in feature_columns}
})

@app.get("/")
def home():
    return {"message": "House Price Prediction API is Live! ✅"}

@app.get("/features")
def get_features():
    """Get list of required features"""
    return {
        "features": list(feature_columns),
        "count": len(feature_columns)
    }

@app.post("/predict")
def predict_price(data: HouseInput):
    try:
        # Data dict mein convert kar
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        
        scaled_data = scaler.transform(input_df)
        log_prediction = model.predict(scaled_data)[0]
        predicted_price = float(np.expm1(log_prediction))
        rmse = 0.13
        lower_price = float(np.expm1(log_prediction - (1.96 * rmse)))
        upper_price = float(np.expm1(log_prediction + (1.96 * rmse)))
        
        return {
            "status": "success",
            "predicted_price_usd": round(predicted_price, 2),
            "confidence_interval_95": {
                "lower_bound": round(lower_price, 2),
                "upper_bound": round(upper_price, 2)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


