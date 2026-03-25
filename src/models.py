import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error,
)
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# divisione delle colonne in categorie
numeric_median_features = [
    # "number_of_bedrooms",  # nans to manage
    "livable_surface_m2",
    # "land_area_m2",  #
    # "energy_consumption",  #
    "garages_final",
    "terrace_area_m2",  #
    "garden_area_m2",
    "zip_code",  #
]
# numeric_most_frequent_features = ["number_of_facades"]  #
binary_features = [
    # "energy_data_missing",
    "has_swimming_pool",
    # "has_cellar",
    # "has_elevator",
    # "has_access_for_disabled",
    # "has_solar_panels",
    # "has_floor_heating",
    # "has_fireplace",
    # "has_balcony",
    # "has_attic",
    "furnished",
    "has_terrace",
    "has_garden",
    "kitchen_equipped",
]
hot_encoder_features = [
    "property_type",
    # "property_subtype",
    "region",
    # "type_of_heating",
    # "type_of_glazing",
    # "build_year_group",
    "building_condition",
]
condition_order = [
    "TO_REBUILD",
    "TO_RENOVATE",
    "GOOD",
    "NEW",
]

# pipelines preparation
mode_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Ecco la Moda
        ("scaler", StandardScaler()),
    ]
)
median_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
# Pipeline per le categorie: trasforma il testo in numeri (OneHotEncoder)
# handle_unknown='ignore' è fondamentale per non far bloccare il modello se nel test appare un CAP mai visto
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# A. LINEAR PREPROCESSOR
linear_preprocessor = ColumnTransformer(
    transformers=[
        ("num_median", median_transformer, (numeric_median_features + binary_features)),
        ("cat", categorical_transformer, hot_encoder_features),
    ]
)

# 1. Rimuovi building_condition da hot_encoder_features per non duplicarla
hot_encoder_no_condition = [
    c for c in hot_encoder_features if c != "building_condition"
]

# A. PREPROCESSOR: Con Imputer + Ordinal (Scala numerica)
prep_impute_ordinal = ColumnTransformer(
    transformers=[
        (
            "num_med",
            median_transformer,  # USA IL TRANSFORMER CHE HA LO SCALER!
            (numeric_median_features + binary_features),
        ),
        (
            "ord_cond",
            OrdinalEncoder(
                categories=[condition_order],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            ["building_condition"],
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            hot_encoder_no_condition,
        ),  # Usa la lista pulita
    ]
)


def update_leaderboard(
    leaderboard, model_name, params, subset_name, r2, mae, rmse, mape, medae, max_err
):
    """
    Aggiunge una nuova riga con i risultati della simulazione alla tabella.
    """
    # Creiamo il dizionario con i dati della simulazione corrente
    new_entry = {
        "Algorithm": model_name,
        "Parameters": str(params),  # Convertiamo in stringa per leggibilità nel DF
        "Subset": subset_name,
        "R2_Score": round(r2, 4),
        "MAE": round(mae, 2),
        # All'interno del tuo ciclo di valutazione:
        "mape": round(mape, 2),
        "medae": round(medae, 2),
        "rmse": round(rmse, 2),
        "max_err": round(max_err, 2),
    }

    # Aggiungiamo la riga al DataFrame esistente
    # Se il DF è vuoto, lo inizializza con questa riga
    new_row = pd.DataFrame([new_entry])
    leaderboard = pd.concat([leaderboard, new_row], ignore_index=True)

    return leaderboard
