import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.path.join(BASE_DIR, "model")

model_a = joblib.load(os.path.join(MODEL_DIR, "model_apartament.pkl"))
model_h = joblib.load(os.path.join(MODEL_DIR, "model_house.pkl"))


def predict_price(df: pd.DataFrame):
    prop_type = df["property_type"].iloc[0].upper()

    model = model_h if prop_type == "HOUSE" else model_a

    prediction_log = model.predict(df)

    price = np.expm1(prediction_log)[0]

    return round(float(price), 2)
