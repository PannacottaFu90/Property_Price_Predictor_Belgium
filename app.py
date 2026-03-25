from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Importiamo le tue funzioni dai file che abbiamo creato
from src.input_data_cleaning import preprocess
from src.prediction import predict_price
from typing import Optional, Literal

app = FastAPI()


class PropertyData(BaseModel):
    livable_surface_m2: int
    property_type: Literal["apartment", "house"]
    # region: Literal["Bruxelles", "Wallonia", "Flanders"]
    zip_code: int

    # Campi opzionali
    garages_final: Optional[int] = None  #
    terrace_area_m2: Optional[int] = None
    garden_area_m2: Optional[int] = None
    has_swimming_pool: Optional[bool] = None
    has_terrace: Optional[bool] = None
    has_garden: Optional[bool] = None
    furnished: Optional[bool] = None

    kitchen_equipped: Optional[bool] = None

    building_condition: Literal["To Rebuild", "To Renovate", "Good", "New"] = "Good"


# Definiamo il formato di input richiesto dalla sfida: {"data": {...}}
class HouseInput(BaseModel):
    data: PropertyData


@app.get("/")
def read_root():
    return "alive"


@app.post("/predict")
def predict(input_data: HouseInput):
    try:
        # 1. Preprocessing (trasforma Pydantic in DataFrame)
        # Passiamo input_data.data perché la classe HouseInput è un wrapper
        processed_df = preprocess(input_data.data)

        # 2. Prediction
        price = predict_price(processed_df)

        # 3. Output come richiesto dalla challenge
        return {"prediction": price, "status_code": 200}

    except Exception as e:
        import traceback

        print(traceback.format_exc())  # Questo stamperà l'errore REALE nel terminale
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict")
def predict_info():
    return {
        "message": "Invia un POST a questa rotta con i dati della casa in formato JSON per ottenere una previsione.",
        "format": "Vedi la documentazione a /docs per lo schema esatto.",
    }
