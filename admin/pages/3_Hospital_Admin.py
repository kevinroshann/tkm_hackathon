import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Hospital Admin - Intelligence Hub", layout="wide")

st.title("Hospital Intelligence & Procurement Hub")

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    # Check current dir
    if os.path.exists("hospital_forecast.pkl") and os.path.exists("patient_forecast.pkl"):
        res_mod = joblib.load("hospital_forecast.pkl")
        pat_mod = joblib.load("patient_forecast.pkl")
        return res_mod, pat_mod
    # Check parent dir
    elif os.path.exists("../hospital_forecast.pkl") and os.path.exists("../patient_forecast.pkl"):
        res_mod = joblib.load("../hospital_forecast.pkl")
        pat_mod = joblib.load("../patient_forecast.pkl")
        return res_mod, pat_mod
    return None, None

resource_model, patient_model = load_models()

if not resource_model or not patient_model:
    st.error("Model files missing! Please run `train_model.py` to generate `hospital_forecast.pkl` and `patient_forecast.pkl`.")
    st.stop()

targets = ["med_A", "med_B", "med_C", "oxygen", "syringes"]

# --- SIDEBAR ---
with st.sidebar:
    st.header("Forecast Settings")
    st.subheader("Weather Input")
    temp = st.slider("Temperature (°C)", 10, 45, 30)
    humidity = st.slider("Humidity (%)", 20, 100, 60)
    rainfall = st.slider("Rainfall (mm)", 0.0, 50.0, 5.0)
    season_idx = st.slider("Seasonal Index", -5.0, 5.0, 0.0)
    
    st.divider()
    st.subheader("Inventory Levels")
    current_stock = {}
    for item in targets:
        current_stock[item] = st.number_input(f"Current {item.title()}", min_value=0, value=100)
    
    st.divider()
    safety_buffer = st.slider("Safety Buffer (%)", 0, 50, 10) / 100

# --- PHASE 1: PATIENT PREDICTION ---
weather_df = pd.DataFrame([[temp, humidity, rainfall, season_idx]], 
                          columns=["temp", "humidity", "rainfall", "season_index"])
predicted_patients = int(patient_model.predict(weather_df)[0])

st.subheader("Phase 1: Patient Load Forecasting")
col1, col2 = st.columns(2)
with col1:
    st.info(f"Based on: Temp {temp}°C, Humidity {humidity}%, Rain {rainfall}mm")
    st.metric("Predicted Daily Patients", f"{predicted_patients}", delta="AI Forecast")
with col2:
    suggested_icu = int(predicted_patients * 0.12)
    icu_cases = st.number_input("Expected ICU Cases (Adjustable)", value=suggested_icu)

st.divider()

# --- PHASE 2: RESOURCE PREDICTION ---
st.subheader("Phase 2: Resource Procurement Planner")

input_df = pd.DataFrame([[predicted_patients, icu_cases, season_idx, temp]], 
                        columns=["patients", "icu_cases", "season_index", "temp"])
daily_pred = resource_model.predict(input_df)[0]
weekly_demand = daily_pred * 7

results = []
for i, item in enumerate(targets):
    demand = weekly_demand[i]
    stock = current_stock[item]
    target_with_buffer = demand * (1 + safety_buffer)
    to_purchase = max(0, target_with_buffer - stock)
    
    results.append({
        "Resource": item.replace("_", " ").title(),
        "Weekly Demand": round(demand, 1),
        "Target (w/ Buffer)": round(target_with_buffer, 1),
        "In Stock": stock,
        "To Order": round(to_purchase, 1),
        "Status": "Sufficient" if to_purchase == 0 else "Order Required"
    })

res_df = pd.DataFrame(results)

# Summary
total_to_order = res_df[res_df["To Order"] > 0].shape[0]
if total_to_order > 0:
    st.warning(f"Recommendation: Restock **{total_to_order}** items to meet demand.")
else:
    st.success("Current inventory is sufficient for predicted load.")

# Visuals
c1, c2 = st.columns([2, 1])

with c1:
    fig = go.Figure()
    fig.add_trace(go.Bar(name='In Stock', x=res_df["Resource"], y=res_df["In Stock"], marker_color='#2ecc71'))
    fig.add_trace(go.Bar(name='Required', x=res_df["Resource"], y=res_df["Target (w/ Buffer)"], marker_color='#e74c3c'))
    fig.update_layout(barmode='group', title="Stock vs Requirement", height=350)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.dataframe(res_df[["Resource", "To Order", "Status"]], hide_index=True, use_container_width=True)

# Export
csv = res_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Purchase Order (CSV)", csv, "procurement_list.csv", "text/csv")
