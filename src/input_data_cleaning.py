import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
zip_map_h = joblib.load(BASE_DIR / "model" / "zip_map_h.pkl")
zip_map_a = joblib.load(BASE_DIR / "model" / "zip_map_a.pkl")


def preprocess(property_data):
    data_dict = property_data.dict()
    df = pd.DataFrame([data_dict])

    numeric_features = [
        "livable_surface_m2",
        "garages_final",
        "terrace_area_m2",
        "garden_area_m2",
        "zip_code",
    ]
    binary_features = [
        "has_swimming_pool",
        "furnished",
        "has_terrace",
        "has_garden",
        "kitchen_equipped",
    ]

    categorical_features = ["property_type", "region", "building_condition"]

    # REGION & ZIP_CODE
    def get_region_from_zip(z):
        if 1000 <= z <= 1299:
            return "Bruxelles"
        elif (1300 <= z <= 1499) or (4000 <= z <= 7999):
            return "Wallonia"
        return "Flanders"

    df["region"] = df["zip_code"].apply(get_region_from_zip)

    current_zip = str(df["zip_code"].iloc[0])
    is_house = str(df["property_type"].iloc[0]).lower() == "house"

    if is_house:
        df["zip_code"] = float(zip_map_h.get(current_zip, zip_map_h.mean()))
    else:
        df["zip_code"] = float(zip_map_a.get(current_zip, zip_map_a.mean()))

    # for OrdinalEncoder - OneHot
    df["building_condition"] = (
        df["building_condition"].str.upper().str.replace(" ", "_")
    )
    df["property_type"] = df["property_type"].replace("apartment", "appartment")

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in binary_features:
        df[col] = df[col].astype(int)

    column_order = [
        "zip_code",
        "property_type",
        "livable_surface_m2",
        "furnished",
        "has_terrace",
        "terrace_area_m2",
        "has_garden",
        "garden_area_m2",
        "has_swimming_pool",
        "building_condition",
        "garages_final",
        "region",
        "kitchen_equipped",
    ]
    return df.reindex(columns=column_order)
