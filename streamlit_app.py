import streamlit as st
import requests

st.set_page_config(page_title="ImmoEliza Analytics", layout="wide", page_icon="🏠")

# --- url ---
API_URL = "https://belgian-real-estate-price-estimator.onrender.com/predict"

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏠 ImmoEliza Pro")
    st.subheader("Real Estate Intelligence")
    tool = st.radio(
        "Select tool:",
        ["Estimator", "Deal"],
        help="Select one of the two",
    )
    st.divider()
    st.caption("Powered by XGBoost & FastAPI")


# --- INPUT ---
def get_property_inputs():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 📍 Localization")
        zip_code = st.number_input("ZIP", value=1000, min_value=1000, max_value=9999)
        prop_type = st.selectbox("Property type", ["house", "apartment"])
        surface = st.number_input("Surface (m²)", value=100, min_value=10)

    with col2:
        st.markdown("##### 🏗️ Structure")
        condition = st.selectbox(
            "Condition", ["To Rebuild", "To Renovate", "Good", "New"], index=2
        )
        garages = st.number_input("Parking", value=0, min_value=0)
        kitchen = st.checkbox("Kitchen", value=True)
        furnished = st.checkbox("Furnished")

    with col3:
        st.markdown("##### 🌳 Plus")
        has_terrace = st.checkbox("Terrace")
        terrace_area = (
            st.number_input("Terrace area (m²)", value=0) if has_terrace else 0
        )
        has_garden = st.checkbox("Garden")
        garden_area = st.number_input("Garden area (m²)", value=0) if has_garden else 0
        has_swimming_pool = st.checkbox("Swimming pool")

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


# --- TOOL 1 ---
if tool == "Estimator":
    st.title("🔍 Market Value Estimator")
    st.write("Insert all the info you want")

    data_payload = get_property_inputs()

    if st.button("🚀 Result: ", use_container_width=True):
        try:
            with st.spinner("Processing..."):
                res = requests.post(API_URL, json={"data": data_payload})

            if res.status_code == 200:
                output = res.json()
                pred = output["prediction"]
                low = output["lower_bound"]
                high = output["upper_bound"]

                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Value: ", f"€ {pred:,.0f}")
                m2.metric(
                    "Price/m²", f"€ {pred/data_payload['livable_surface_m2']:,.2f}"
                )
                m3.metric("Confidence (MAE)", f"± € {output['mae']:,.0f}")

                st.subheader("Confidence")
                st.info(
                    f"Depending on the specific context, the real value could be between **€ {low:,.0f}** and **€ {high:,.0f}**."
                )
            else:
                st.error("Communication error.")
        except Exception as e:
            st.error(f"Connection error: {e}")

# --- TOOL 2: ANALIZZATORE DEAL ---
elif tool == "Deal":
    st.title("💰 Deal Checker")
    st.write("Compare thepriceof an isertion with AI prediction.")

    # Prezzo richiesto in evidenza
    asking_price = st.number_input("💵 Price (€)", min_value=10000, step=5000)
    st.divider()

    data_payload = get_property_inputs()

    if st.button("⚖️ Deal Checker", use_container_width=True):
        try:
            with st.spinner("Processing.."):
                res = requests.post(API_URL, json={"data": data_payload})

            if res.status_code == 200:
                output = res.json()
                pred = output["prediction"]
                low = output["lower_bound"]
                mae = output["mae"]

                st.divider()

                if asking_price < low:
                    st.success(
                        f"🔥 **POSSIBLE DEAL**: The price (€ {asking_price:,.0f}) is lower than market value (€ {pred:,.0f})."
                    )
                elif asking_price < (pred + (mae * 0.5)):
                    st.info(
                        f"⚖️ **CORRECT PRICE**: The insertion seems aligned with the market."
                    )
                else:
                    st.error(f"⚠️ **DANGER!**: The property seems too expensive.")

                st.progress(min(max((asking_price / (pred * 1.5)), 0.0), 1.0))
                st.caption(
                    f" € {pred:,.0f} (Range: € {low:,.0f} - € {output['upper_bound']:,.0f})"
                )

            else:
                st.error("Error API.")
        except Exception as e:
            st.error(f"Error: {e}")

# --- FOOTER ---
st.divider()
st.caption("Note: This is a simulation based on ")
