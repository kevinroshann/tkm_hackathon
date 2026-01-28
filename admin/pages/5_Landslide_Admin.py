import streamlit as st
import pandas as pd
import joblib
import os
import requests
import plotly.graph_objects as go
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Landslide Admin", page_icon="🌋", layout="wide")

# ===============================
# MODEL LOADING
# ===============================
@st.cache_resource
def load_landslide_model():
    paths = ["../../landslide_model.pkl", "../landslide_model.pkl", "landslide_model.pkl"]
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p)
    return None

data_payload = load_landslide_model()

# ===============================
# HEADER
# ===============================
st.markdown("""
<div style="background:linear-gradient(135deg,#7c2d12,#ea580c);
padding:2rem;border-radius:12px;margin-bottom:2rem;">
<h1 style="color:white;text-align:center;">🌋 Landslide Prevention & Admin</h1>
<p style="color:#fed7aa;text-align:center;">
Simulate conditions and generate risk assessments.
</p>
</div>
""", unsafe_allow_html=True)

if not data_payload:
    st.error("⚠️ Model file not found. Please run training script.")
    st.stop()

model = data_payload["model"]
feature_names = data_payload["features"]

# ===============================
# LIVE WEATHER (Open-Meteo)
# ===============================
# Default Placeholders
live_rain = 0.0
live_soil = 0.5
live_wind = 0.0
live_temp = 30.0

st.subheader("🌍 Real-Time Environmental Data")

# 1. Location Settings (Expander to keep UI clean)
with st.expander("📍 Configuration (Location)", expanded=False):
    wc1, wc2, wc3 = st.columns([1, 1, 1])
    lat = wc1.number_input("Latitude", value=10.8505)
    lon = wc2.number_input("Longitude", value=76.2711)
    if wc3.button("Refresh Data"):
        st.cache_data.clear()

# 2. Auto-Fetch Function
@st.cache_data(ttl=300) # Cache for 5 mins
def fetch_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=rain,showers,soil_moisture_0_to_1cm,wind_speed_10m,temperature_2m"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except:
        return None
    return None

# 3. Execute Fetch
weather_data = fetch_weather(lat, lon)

if weather_data:
    current = weather_data.get('current', {})
    live_rain = (current.get('rain', 0.0) + current.get('showers', 0.0)) * 24 # 24h Projection
    live_soil = current.get('soil_moisture_0_to_1cm', 0.0) * 1.5 # Scale correction
    live_wind = current.get('wind_speed_10m', 0.0)
    live_temp = current.get('temperature_2m', 0.0)
    
    # Clamp Soil
    live_soil = max(0.0, min(1.0, live_soil))
else:
    st.warning("⚠️ Offline Mode: Using default weather values.")

# 4. Display Metrics (NO SLIDERS for these)
m1, m2, m3, m4 = st.columns(4)
m1.metric("🌧️ Rainfall (24h Proj)", f"{live_rain:.1f} mm", delta="Auto-Fetched")
m2.metric("💧 Soil Saturation", f"{live_soil*100:.0f}%", delta="Sensor Data")
m3.metric("🌬️ Wind Speed", f"{live_wind} km/h")
m4.metric("🌡️ Temperature", f"{live_temp} °C")

st.divider()

# ===============================
# CONTROLS (MANUAL ONLY)
# ===============================
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("Simulated Geophysics")
    with st.container(border=True):
        st.caption("Adjust geological parameters manually:")
        # Rain and Soil sliders REMOVED - using live_rain and live_soil variables directly
        
        slope = st.slider("⛰️ Slope Angle (°)", 0, 90, 35)
        vegetation = st.slider("🌱 Vegetation Cover", 0.0, 1.0, 0.4)
        earthquake = st.selectbox("🌐 Earthquake Activity", [0, 1])
        proximity = st.slider("🚰 Dist. to Water (km)", 0.0, 10.0, 1.5)
        soil_type = st.selectbox("🪨 Soil Type", ["Gravel", "Sand", "Silt"])

# Prepare Input
soil_gravel = 1 if soil_type == "Gravel" else 0
soil_sand = 1 if soil_type == "Sand" else 0
soil_silt = 1 if soil_type == "Silt" else 0

input_data = pd.DataFrame([[
    live_rain, slope, live_soil, vegetation, earthquake, proximity, soil_gravel, soil_sand, soil_silt
]], columns=feature_names)


# Prediction
probs = model.predict_proba(input_data)[0]
risk_score = probs[1] # Probability of Landslide
risk_percent = round(risk_score * 100, 2)

if risk_score < 0.3:
    level = "LOW"
    color = "green"
elif risk_score < 0.6:
    level = "MEDIUM"
    color = "orange"
else:
    level = "HIGH"
    color = "red"

# ===============================
# RESULTS
# ===============================
with c2:
    st.subheader("Risk Assessment")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Risk", f"{risk_percent}%", delta="AI Confidence")
    m2.metric("Risk Level", level, delta_color="off")
    m3.metric("Simulated Time", datetime.now().strftime("%H:%M"))
    
    st.divider()
    
    if level == "HIGH":
        st.error(f"### 🚨 CRITICAL THREAT DETECTED\n**Recommendation**: Immediate Evacuation of sectors with >{slope}° slope.")
    elif level == "MEDIUM":
        st.warning(f"### ⚠️ ELEVATED RISK\n**Recommendation**: Deploy monitoring teams to high-saturation zones.")
    else:
        st.success(f"### ✅ STABLE CONDITIONS\n**Recommendation**: Routine monitoring.")
        
    # Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_percent,
        title = {'text': "Landslide Probability"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "lightyellow"},
                {'range': [60, 100], 'color': "salmon"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)



st.divider()

# ===============================
# GEMINI REPORT GENERATION
# ===============================
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Landslide Risk Assessment Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(data, prediction, sitrep):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # 1. Header Info
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    pdf.ln(5)
    
    # 2. Environmental Data
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Environmental Conditions", 0, 1)
    pdf.set_font("Arial", size=11)
    
    # Create simple list
    env_data = [
        f"Rainfall (24h Projected): {data['rain']:.1f} mm",
        f"Soil Saturation: {data['soil']*100:.1f}%",
        f"Slope Angle: {data['slope']} degrees",
        f"Seismic Activity: {'Active' if data['quake'] else 'None'}",
        f"Wind Speed: {data['wind']} km/h",
        f"Temperature: {data['temp']} C"
    ]
    
    for item in env_data:
        pdf.cell(10) # Indent
        pdf.cell(0, 7, f"- {item}", 0, 1)
    pdf.ln(5)
    
    # 3. AI Prediction
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. AI Risk Analysis", 0, 1)
    pdf.set_font("Arial", size=11)
    
    pdf.cell(10)
    pdf.cell(0, 7, f"Risk Probability: {prediction['prob']}%", 0, 1)
    pdf.cell(10)
    pdf.set_text_color(220, 50, 50) if prediction['level'] == 'HIGH' else pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f"Risk Level: {prediction['level']}", 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    # 4. Gemini SITREP
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Tactical Situation Report (AI)", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, sitrep)
    
    return pdf.output(dest='S').encode('latin-1', 'replace')

if st.button("📝 Generate Report & Download PDF"):
    conf_key = os.getenv("GEMINI_API_KEY")
    if not conf_key:
        st.error("Gemini API Key not found.")
    else:
        with st.spinner("Analyzing data and generating documents..."):
            try:
                genai.configure(api_key=conf_key)
                try:
                    g_model = genai.GenerativeModel('models/gemini-2.5-flash-lite-preview-09-2025')
                except:
                    g_model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Construct Prompt
                prompt = (
                    f"Act as a Disaster Response Specialist. Generate a concise Situation Report for a Landslide Risk Assessment.\n"
                    f"DATA:\n"
                    f"- Rainfall (24h): {live_rain} mm\n"
                    f"- Slope Angle: {slope} degrees\n"
                    f"- Soil Saturation: {live_soil*100}%\n"
                    f"- Vegetation Cover: {vegetation*100}%\n"
                    f"- Seismic Activity: {'YES' if earthquake else 'NO'}\n"
                    f"- Proximity to Water: {proximity} km\n"
                    f"\n"
                    f"AI PREDICTION:\n"
                    f"- Risk Probability: {risk_percent}%\n"
                    f"- Risk Level: {level}\n"
                    f"\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Summarize the critical threats.\n"
                    f"2. Recommend 3 specific tactical actions for the ground team.\n"
                    f"3. Keep it professional, urgent, and concise (max 150 words)."
                )
                
                response = g_model.generate_content(prompt)
                ai_text = response.text
                
                # Show on Screen
                st.info(f"**AI SITREP**\n\n{ai_text}")
                
                # Generate PDF
                report_data = {
                    "rain": live_rain, "soil": live_soil, "slope": slope, 
                    "quake": earthquake, "wind": live_wind, "temp": live_temp
                }
                pred_data = {"prob": risk_percent, "level": level}
                
                pdf_bytes = generate_pdf_report(report_data, pred_data, ai_text)
                
                st.download_button(
                    label="📄 Download Official PDF Report",
                    data=pdf_bytes,
                    file_name=f"Landslide_SITREP_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"Generation Failed: {str(e)}")
