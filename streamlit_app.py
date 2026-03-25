import streamlit as st
import requests

st.set_page_config(
    page_title="Immo Eliza - Predictor", layout="centered", page_icon="🏠"
)

st.title("🏠 Valutatore Immobiliare Belgio")
st.write(
    "Ottieni una stima immediata del valore di mercato basata sui dati più recenti."
)

# --- SEZIONE INPUT ---
tab1, tab2, tab3 = st.tabs(["📍 Posizione e Tipo", "🏠 Caratteristiche", "🌳 Extra"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        zip_code = st.number_input(
            "Codice Postale (es. 1000)", value=1000, min_value=1000, max_value=9999
        )
        prop_type = st.selectbox("Tipo Proprietà", ["house", "apartment"])
    with col2:
        surface = st.number_input("Superficie Abitabile (m²)", value=100, min_value=10)
        # Allineato ai Literal di Pydantic e alla logica .upper() del preprocess
        condition = st.selectbox(
            "Condizione dell'edificio",
            ["To Rebuild", "To Renovate", "Good", "New"],
            index=2,
        )

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        garages = st.number_input("Posti Auto / Garage", value=0, min_value=0)
        kitchen = st.checkbox("Cucina Equipaggiata", value=True)
    with col_b:
        furnished = st.checkbox("Arredato")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        has_terrace = st.checkbox("Terrazza")
        terrace_area = (
            st.number_input("Area Terrazza (m²)", value=0) if has_terrace else 0
        )
    with c2:
        has_garden = st.checkbox("Giardino")
        garden_area = (
            st.number_input("Area Giardino (m²)", value=0) if has_garden else 0
        )
        has_swimming_pool = st.checkbox("Piscina")

# --- LOGICA DI INVIO ---
if st.button("🚀 Calcola Valore Mercato", use_container_width=True):

    # Costruiamo il payload ESATTAMENTE come lo vuole PropertyData in app.py
    payload = {
        "data": {
            "livable_surface_m2": int(surface),
            "property_type": prop_type,
            "zip_code": int(zip_code),
            "garages_final": int(garages),
            "terrace_area_m2": int(terrace_area),
            "garden_area_m2": int(garden_area),
            "has_swimming_pool": bool(has_swimming_pool),
            "has_terrace": bool(has_terrace),
            "has_garden": bool(has_garden),
            "furnished": bool(furnished),
            "kitchen_equipped": bool(kitchen),
            "building_condition": condition,
        }
    }

    try:
        # Quando sarai su Render, sostituisci con l'URL dell'istanza web
        # es: https://tua-api.onrender.com/predict
        API_URL = "http://127.0.0.1:8000/predict"

        with st.spinner("Interrogando l'intelligenza artificiale..."):
            res = requests.post(API_URL, json=payload)

        if res.status_code == 200:
            prediction = res.json()["prediction"]
            st.balloons()
            st.markdown(f"### 💎 Valore Stimato: **€ {prediction:,.2f}**")
            st.info(f"Prezzo stimato al m²: € {prediction/surface:,.2f}")
        else:
            st.error(
                f"Errore API ({res.status_code}): {res.json().get('detail', res.text)}"
            )

    except Exception as e:
        st.error(
            f"Impossibile connettersi all'API. Assicurati che il server FastAPI sia attivo. Errore: {e}"
        )

# --- FOOTER ---
st.divider()
st.caption(
    "Modello di predizione basato su XGBoost. I prezzi sono puramente indicativi."
)
