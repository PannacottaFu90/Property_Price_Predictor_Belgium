import pandas as pd
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.models import (
    update_leaderboard,
    linear_preprocessor,
    prep_impute_ordinal,
)
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import json

BASE_DIR = Path(__file__).resolve().parent

htot_path = BASE_DIR / "data" / "3_data_for_training" / "df_h_tot.csv"
clean_path = BASE_DIR / "data" / "3_data_for_training" / "df_clean.csv"
atot_path = BASE_DIR / "data" / "3_data_for_training" / "df_a_tot.csv"

df_clean = pd.read_csv(clean_path)
df_h_tot = pd.read_csv(htot_path)
df_a_tot = pd.read_csv(atot_path)

pipeline1 = Pipeline(
    [
        ("prep", prep_impute_ordinal),
        (
            "reg",
            XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
            ),
        ),
    ]
)

# case total
X = df_h_tot.drop("price", axis=1)
y = df_h_tot["price"]

df_temp = X.copy()
df_temp["price_m2"] = y / X["livable_surface_m2"]
final_zip_map_h = df_temp.groupby("zip_code")["price_m2"].mean()
X["zip_code"] = X["zip_code"].map(final_zip_map_h)
y_log = np.log1p(y)

final_model_h = clone(pipeline1)
final_model_h.fit(X, y_log)

y_pred_h = final_model_h.predict(X)
y_pred_h_euro = np.expm1(y_pred_h)
mae_value_h = mean_absolute_error(y, y_pred_h_euro)
r2_h = r2_score(y, y_pred_h_euro)
made_h = mean_absolute_percentage_error(y, y_pred_h_euro)
print(f"R2 House: {r2_h}")
print(f"made h: {made_h}")
hmae_path = BASE_DIR / "model" / "metrics_h.json"
with open(hmae_path, "w") as f:
    json.dump({"mae_h": float(mae_value_h)}, f)

house_path = BASE_DIR / "model" / "model_house.pkl"
h_zip_map_path = BASE_DIR / "model" / "zip_map_h.pkl"
joblib.dump(final_model_h, house_path)
joblib.dump(final_zip_map_h, h_zip_map_path)
print("Modello house allenato e salvato con successo in model/model.pkl")

# appartamenti total
X = df_a_tot.drop("price", axis=1)
y = df_a_tot["price"]

df_temp = X.copy()
df_temp["price_m2"] = y / X["livable_surface_m2"]
final_zip_map_a = df_temp.groupby("zip_code")["price_m2"].mean()
X["zip_code"] = X["zip_code"].map(final_zip_map_a)
y_log = np.log1p(y)
final_model_a = clone(pipeline1)
final_model_a.fit(X, y_log)

y_pred_a = final_model_a.predict(X)
y_pred_a_euro = np.expm1(y_pred_a)
mae_value_a = mean_absolute_error(y, y_pred_a_euro)
r2_a = r2_score(y, y_pred_a_euro)
made_a = mean_absolute_percentage_error(y, y_pred_a_euro)
print(f"made a: {made_a}")
print(f"R2 apartment: {r2_a}")
amae_path = BASE_DIR / "model" / "metrics_a.json"
with open(amae_path, "w") as f:
    json.dump({"mae_a": float(mae_value_a)}, f)

app_path = BASE_DIR / "model" / "model_apartament.pkl"
a_zip_map_path = BASE_DIR / "model" / "zip_map_a.pkl"
joblib.dump(final_zip_map_a, a_zip_map_path)
joblib.dump(final_model_a, app_path)
print("Modello appartamenti allenato e salvato con successo in model/model.pkl")
