import streamlit as st
import requests

# Configurazione Pagina
st.set_page_config(page_title="ImmoEliza Analytics", layout="wide", page_icon="🏠")

# --- COSTANTI ---
API_URL = "https://belgian-real-estate-price-estimator.onrender.com/predict"

# --- SIDEBAR (Navigazione) ---
with st.sidebar:
    st.title("🏠 ImmoEliza Pro")
    st.subheader("Real Estate Intelligence")
    tool = st.radio(
        "Seleziona Strumento:",
        ["Estimatore Valore", "Analizzatore Deal"],
        help="Scegli se calcolare una stima o valutare un annuncio esistente.",
    )
    st.divider()
    st.caption("Powered by XGBoost & FastAPI")


# --- FUNZIONE INPUT COMUNI ---
def get_property_inputs():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 📍 Localizzazione")
        zip_code = st.number_input(
            "Codice Postale", value=1000, min_value=1000, max_value=9999
        )
        prop_type = st.selectbox("Tipo Proprietà", ["house", "apartment"])
        surface = st.number_input("Superficie Abitabile (m²)", value=100, min_value=10)

    with col2:
        st.markdown("##### 🏗️ Struttura")
        condition = st.selectbox(
            "Condizione", ["To Rebuild", "To Renovate", "Good", "New"], index=2
        )
        garages = st.number_input("Posti Auto / Garage", value=0, min_value=0)
        kitchen = st.checkbox("Cucina Equipaggiata", value=True)
        furnished = st.checkbox("Arredato")

    with col3:
        st.markdown("##### 🌳 Esterni & Plus")
        has_terrace = st.checkbox("Terrazza")
        terrace_area = (
            st.number_input("Area Terrazza (m²)", value=0) if has_terrace else 0
        )
        has_garden = st.checkbox("Giardino")
        garden_area = (
            st.number_input("Area Giardino (m²)", value=0) if has_garden else 0
        )
        has_swimming_pool = st.checkbox("Piscina")

    return {
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


# --- TOOL 1: ESTIMATORE VALORE ---
if tool == "Estimatore Valore":
    st.title("🔍 Estimatore Valore di Mercato")
    st.write("Inserisci i dettagli dell'immobile per ottenere una stima professionale.")

    data_payload = get_property_inputs()

    if st.button("🚀 Calcola Stima", use_container_width=True):
        try:
            with st.spinner("Analisi di mercato in corso..."):
                res = requests.post(API_URL, json={"data": data_payload})

            if res.status_code == 200:
                output = res.json()
                pred = output["prediction"]
                low = output["lower_bound"]
                high = output["upper_bound"]

                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Valore Stimato", f"€ {pred:,.0f}")
                m2.metric(
                    "Prezzo al m²", f"€ {pred/data_payload['livable_surface_m2']:,.2f}"
                )
                m3.metric("Confidenza (MAE)", f"± € {output['mae']:,.0f}")

                st.subheader("Intervallo di Confidenza")
                st.info(
                    f"Sulla base della variabilità di mercato, il valore probabile è compreso tra **€ {low:,.0f}** e **€ {high:,.0f}**."
                )
            else:
                st.error("Errore nella comunicazione con il server.")
        except Exception as e:
            st.error(f"Errore di connessione: {e}")

# --- TOOL 2: ANALIZZATORE DEAL ---
elif tool == "Analizzatore Deal":
    st.title("💰 Analizzatore Opportunità (Deal Checker)")
    st.write("Confronta il prezzo di un annuncio con il valore reale stimato dall'AI.")

    # Prezzo richiesto in evidenza
    asking_price = st.number_input(
        "💵 Prezzo richiesto dall'annuncio (€)", min_value=10000, step=5000
    )
    st.divider()

    data_payload = get_property_inputs()

    if st.button("⚖️ Valuta Opportunità", use_container_width=True):
        try:
            with st.spinner("Valutazione convenienza..."):
                res = requests.post(API_URL, json={"data": data_payload})

            if res.status_code == 200:
                output = res.json()
                pred = output["prediction"]
                low = output["lower_bound"]
                mae = output["mae"]

                st.divider()

                # Logica del Verdetto
                if asking_price < low:
                    st.success(
                        f"🔥 **POSSIBILE DEAL**: Il prezzo richiesto (€ {asking_price:,.0f}) è significativamente inferiore al valore di mercato (€ {pred:,.0f})."
                    )
                elif asking_price < (pred + (mae * 0.5)):
                    st.info(
                        f"⚖️ **PREZZO EQUO**: L'annuncio è in linea con le stime di mercato attuali."
                    )
                else:
                    st.error(
                        f"⚠️ **SOVRAPPREZZO**: L'immobile sembra quotato sopra il valore di mercato stimato."
                    )

                # Visualizzazione grafica sintetica
                st.progress(min(max((asking_price / (pred * 1.5)), 0.0), 1.0))
                st.caption(
                    f"Stima di mercato: € {pred:,.0f} (Range: € {low:,.0f} - € {output['upper_bound']:,.0f})"
                )

            else:
                st.error("Errore API.")
        except Exception as e:
            st.error(f"Errore: {e}")

# --- FOOTER ---
st.divider()
st.caption(
    "Nota: Le stime sono generate da un modello statistico e non sostituiscono una perizia tecnica."
)
