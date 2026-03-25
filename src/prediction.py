import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

# 1. Costruiamo il percorso assoluto alla cartella 'model'
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.path.join(BASE_DIR, "model")

# 2. Carichiamo i modelli (con gestione errore base)

model_a = joblib.load(os.path.join(MODEL_DIR, "model_apartament.pkl"))
model_h = joblib.load(os.path.join(MODEL_DIR, "model_house.pkl"))


def predict_price(df: pd.DataFrame):
    # 1. Identifica il tipo
    prop_type = df["property_type"].iloc[0].upper()

    # 2. Scegli il modello
    model = model_h if prop_type == "HOUSE" else model_a

    # 3. Predizione (il risultato è logaritmico)
    prediction_log = model.predict(df)

    # 4. Inversione logaritmica e arrotondamento
    price = np.expm1(prediction_log)[0]

    return round(float(price), 2)
