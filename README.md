# 🏠 Belgian Real Estate Price Estimator

An end-to-end Machine Learning ecosystem to estimate property prices in Belgium. This project features a dual-model approach (House/Apartment), a robust FastAPI backend, and an interactive Streamlit dashboard for end-users.

---

## 🚀 Live Demo
* **Interactive Dashboard:** [Streamlit App](https://propertypricepredictorbelgium-iz2jtxky8ubomuogzuvwos.streamlit.app/)
* **REST API Service:** [FastAPI on Render](https://belgian-real-estate-price-estimator.onrender.com/docs)

---

## 🧠 Model & Performance
The prediction engine is built on two optimized **XGBoost Regressor** models. To handle the right-skewed nature of real estate prices, a **logarithmic transformation** (`log1p`) was applied to the target variable.

### 📊 Performance Metrics
| Property Type | R² Score | MADE |
| :--- | :--- | :--- |
| **Houses** | 0.92 | 0.13 |
| **Apartments** | 0.95 | 0.08 |

### Key Features:
* **Custom Preprocessing:** Automated region assignment (Flanders, Wallonia, Brussels) and ZIP code mapping based on average price per m².
* **Robust Pipeline:** Scikit-learn pipelines with `ColumnTransformer` for seamless handling of numerical imputer, `OrdinalEncoding` (for building conditions), and `OneHotEncoding`.
* **Deal Checker:** A unique feature in the Streamlit app that compares the asking price with the model's estimate to identify potential market deals.

---

## 🛠️ Tech Stack
- **Backend:** FastAPI (Pydantic for data validation)
- **Frontend:** Streamlit
- **Machine Learning:** Scikit-Learn, XGBoost, Pandas, Joblib
- **DevOps:** Docker, Render

---

## 📂 Project Structure
```text
├── app.py                 # FastAPI Main Entry
├── streamlit_app.py       # Streamlit Dashboard
├── model_creator.py       # Training & Evaluation Script
├── src/
│   ├── input_data_cleaning.py # Preprocessing Pipeline
│   ├── prediction.py          # Prediction Logic
│   └── models.py              # ML Transformers
├── model/                 # Saved .pkl models and metrics
└── Dockerfile             # Containerization config
```
---

## 💻 Local Setup

### Using Docker (Recommended)
This is the fastest way to run the API locally without worrying about dependencies.

    docker build -t belgium-re-predictor .
    docker run -p 8000:8000 belgium-re-predictor

The API will be available at: http://localhost:8000/docs

### Manual Installation

    git clone [https://github.com/PannacottaFu90/Property_Price_Predictor_Belgium.git](https://github.com/PannacottaFu90/Property_Price_Predictor_Belgium.git)
    cd Property_Price_Predictor_Belgium

    pip install -r requirements.txt

    uvicorn app:app --host 0.0.0.0 --port 8000

    streamlit run streamlit_app.py

## 🔌 API Integration
You can request a prediction programmatically via terminal or any HTTP client:

    curl -X 'POST' \
    '[https://belgian-real-estate-price-estimator.onrender.com/predict](https://belgian-real-estate-price-estimator.onrender.com/predict)' \
    -H 'Content-Type: application/json' \
    -d '{
    "data": {
        "livable_surface_m2": 150,
        "property_type": "house",
        "zip_code": 1000,
        "building_condition": "Good",
        "has_swimming_pool": false,
        "kitchen_equipped": true
    }
    }'

## 🤝 Contact & Development

Developed by PannacottaFu90.


![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-ebba2d.svg)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED.svg)