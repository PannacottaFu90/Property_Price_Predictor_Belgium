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
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
    ]
)
median_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

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

hot_encoder_no_condition = [
    c for c in hot_encoder_features if c != "building_condition"
]

# A. PREPROCESSOR: Imputer + Ordinal
prep_impute_ordinal = ColumnTransformer(
    transformers=[
        (
            "num_med",
            median_transformer,
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
        ),
    ]
)


def update_leaderboard(
    leaderboard, model_name, params, subset_name, r2, mae, rmse, mape, medae, max_err
):
    """
    Add a new line for every simulation
    """
    new_entry = {
        "Algorithm": model_name,
        "Parameters": str(params),
        "Subset": subset_name,
        "R2_Score": round(r2, 4),
        "MAE": round(mae, 2),
        "mape": round(mape, 2),
        "medae": round(medae, 2),
        "rmse": round(rmse, 2),
        "max_err": round(max_err, 2),
    }

    new_row = pd.DataFrame([new_entry])
    leaderboard = pd.concat([leaderboard, new_row], ignore_index=True)

    return leaderboard
