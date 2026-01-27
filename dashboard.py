import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hospital Intelligence Hub", layout="wide")

@st.cache_resource
def load_models():
    res_mod = joblib.load("hospital_forecast.pkl")
    pat_mod = joblib.load("patient_forecast.pkl")
    return res_mod, pat_mod

try:
    resource_model, patient_model = load_models()
except:
    st.error("Model files missing! Ensure both 'hospital_forecast.pkl' and 'patient_forecast.pkl' exist.")
    st.stop()

targets = ["med_A", "med_B", "med_C", "oxygen", "syringes"]

# --- SIDEBAR: WEATHER INPUTS (MODEL 1) ---
st.sidebar.header("🌡️ Weather Forecast (Model 1)")
temp = st.sidebar.slider("Temperature (°C)", 10, 45, 30)
humidity = st.sidebar.slider("Humidity (%)", 20, 100, 60)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 50.0, 5.0)
season_idx = st.sidebar.slider("Seasonal Index", -5.0, 5.0, 0.0)

# --- SIDEBAR: INVENTORY (MODEL 2) ---
st.sidebar.divider()
st.sidebar.header("📦 Current Inventory")
current_stock = {}
for item in targets:
    current_stock[item] = st.sidebar.number_input(f"Current {item.title()}", min_value=0, value=100)

st.sidebar.divider()
safety_buffer = st.sidebar.slider("Safety Buffer (%)", 0, 50, 10) / 100

# --- STEP 1: PATIENT PREDICTION (MODEL 1) ---
st.title("🏥 Hospital Intelligence & Procurement Hub")

# Create input for Model 1
weather_df = pd.DataFrame([[temp, humidity, rainfall, season_idx]], 
                          columns=["temp", "humidity", "rainfall", "season_index"])
predicted_patients = int(patient_model.predict(weather_df)[0])

st.subheader("📊 Phase 1: Patient Load Prediction")
col_p1, col_p2 = st.columns(2)
with col_p1:
    st.metric("Predicted Daily Patients", f"{predicted_patients} people", 
              help="Calculated based on weather patterns and seasonality.")
with col_p2:
    # Heuristic: ICU cases usually scale with total patients (~12%)
    suggested_icu = int(predicted_patients * 0.12)
    icu_cases = st.number_input("Adjust Expected ICU Cases", value=suggested_icu)

st.divider()

# --- STEP 2: RESOURCE PREDICTION (MODEL 2) ---
st.subheader("📋 Phase 2: Resource & Procurement Planner")

# Use Model 1's output as Model 2's input
input_df = pd.DataFrame([[predicted_patients, icu_cases, season_idx, temp]], 
                        columns=["patients", "icu_cases", "season_index", "temp"])
daily_pred = resource_model.predict(input_df)[0]
weekly_demand = daily_pred * 7

# --- CALCULATIONS ---
results = []
for i, item in enumerate(targets):
    demand = weekly_demand[i]
    stock = current_stock[item]
    target_with_buffer = demand * (1 + safety_buffer)
    to_purchase = max(0, target_with_buffer - stock)
    
    results.append({
        "Resource": item.replace("_", " ").title(),
        "Weekly Demand": round(demand, 1),
        "With Buffer": round(target_with_buffer, 1),
        "In Stock": stock,
        "To Purchase": round(to_purchase, 1),
        "Status": "✅ Sufficient" if to_purchase == 0 else "🚨 Order Required"
    })

res_df = pd.DataFrame(results)

# --- UI DISPLAY ---
total_to_order = res_df[res_df["To Purchase"] > 0].shape[0]
if total_to_order > 0:
    st.warning(f"Action Required: Restock **{total_to_order}** items to meet demand for {predicted_patients} patients.")
else:
    st.success("Current inventory is sufficient for the predicted patient load.")

# Visual Comparison
fig = go.Figure()
fig.add_trace(go.Bar(name='In Stock', x=res_df["Resource"], y=res_df["In Stock"], marker_color='#3498db'))
fig.add_trace(go.Bar(name='Weekly Demand', x=res_df["Resource"], y=res_df["Weekly Demand"], marker_color='#e67e22'))
fig.update_layout(barmode='group', height=400, margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# Table
def highlight_shortfall(s):
    return ['background-color: #ffcccc' if v == "🚨 Order Required" else '' for v in s]

st.table(res_df.style.apply(highlight_shortfall, subset=['Status']))

# Download
csv = res_df.to_csv(index=False).encode('utf-8')
st.download_button("📩 Download Purchase Order (CSV)", csv, "procurement_list.csv", "text/csv")