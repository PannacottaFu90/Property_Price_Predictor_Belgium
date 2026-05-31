from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.input_data_cleaning import preprocess
from src.prediction import predict_price
from typing import Optional, Literal
import json

app = FastAPI()

with open("model/metrics_h.json", "r") as f:
    MAE_HOUSE = json.load(f)["mae_h"]

with open("model/metrics_a.json", "r") as f:
    MAE_APART = json.load(f)["mae_a"]


class PropertyData(BaseModel):
    livable_surface_m2: int
    property_type: Literal["apartment", "house"]
    # region: Literal["Bruxelles", "Wallonia", "Flanders"]
    zip_code: int

    garages_final: Optional[int] = None  #
    terrace_area_m2: Optional[int] = None
    garden_area_m2: Optional[int] = None
    has_swimming_pool: Optional[bool] = None
    has_terrace: Optional[bool] = None
    has_garden: Optional[bool] = None
    furnished: Optional[bool] = None

    kitchen_equipped: Optional[bool] = None

    building_condition: Literal["To Rebuild", "To Renovate", "Good", "New"] = "Good"


class HouseInput(BaseModel):
    data: PropertyData


@app.get("/")
def read_root():
    return "alive"


@app.post("/predict")
def predict(input_data: HouseInput):
    try:
        processed_df = preprocess(input_data.data)

        # Prediction
        price = predict_price(processed_df)
        prop_type = input_data.data.property_type.lower()
        current_mae = MAE_HOUSE if prop_type == "house" else MAE_APART

        return {
            "prediction": round(price, 2),
            "lower_bound": round(price - current_mae, 2),
            "upper_bound": round(price + current_mae, 2),
            "mae": round(current_mae, 2),
        }
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict")
def predict_info():
    return {
        "message": "Do a POST to get the prevision",
        "format": "Look at /docs.",
    }
