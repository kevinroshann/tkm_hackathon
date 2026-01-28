import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
import joblib
import os
import plotly.graph_objects as go


st.set_page_config(page_title="Orchestrator - Emergency AI", layout="wide")

API_URL = "http://localhost:5000"

# --- Session State for Auto-Refresh ---
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# --- MODEL LOADING ---
@st.cache_resource
def load_forecast_models():
    # Check current dir
    if os.path.exists("hospital_forecast.pkl") and os.path.exists("patient_forecast.pkl"):
        return joblib.load("hospital_forecast.pkl"), joblib.load("patient_forecast.pkl")
    # Check parent dir (for admin/ subfolder)
    elif os.path.exists("../hospital_forecast.pkl") and os.path.exists("../patient_forecast.pkl"):
        return joblib.load("../hospital_forecast.pkl"), joblib.load("../patient_forecast.pkl")
    return None, None

res_model, pat_model = load_forecast_models()

# --- Title & Controls ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Emergency Orchestrator")
with col2:
    st.toggle("Real-time Updates", key="auto_refresh", value=True)
    if st.button("Manual Refresh"):
        st.rerun()

# --- Data Fetching ---
@st.cache_data(ttl=2)  # Cache for 2 seconds to act as "real-time" buffer
def fetch_logs():
    try:
        response = requests.get(f"{API_URL}/logs", timeout=2)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

logs_data = fetch_logs()
df = pd.DataFrame(logs_data)

# --- Process Data ---
if not df.empty:
    # Ensure columns exist
    cols = ['id', 'priority', 'status', 'severity', 'sos', 'resolved', 'departments', 'timestamp', 'complaint']
    for c in cols:
        if c not in df.columns:
            df[c] = None
    
    # Calculate counts
    total_cases = len(df)
    active_cases = len(df[~df['resolved']])
    critical_active = len(df[(df['priority'] == 'CRITICAL') & (~df['resolved'])])
    urgent_active = len(df[(df['priority'] == 'URGENT') & (~df['resolved'])])
    resolved_count = len(df[df['resolved']])
else:
    total_cases = active_cases = critical_active = urgent_active = resolved_count = 0

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "All Cases", "Visualizations", "Medicine Info", "Strategic Insights"])

# ... (Tabs 1-4 remain same)

# ================= TAB 5: STRATEGIC INSIGHTS =================
with tab5:
    st.subheader("🤖 AI Strategic Intelligence")
    
    if not df.empty:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### 🧩 Department Co-occurrence Matrix")
            st.caption("Which departments represent joint responses?")
            
            # Logic: Create a matrix of Dept vs Dept Co-occurrence
            all_depts_flat = sorted(list(set([d for sublist in df['departments'] if sublist for d in sublist])))
            matrix = pd.DataFrame(0, index=all_depts_flat, columns=all_depts_flat)
            
            for depts in df['departments']:
                if depts and len(depts) > 1:
                    for i in range(len(depts)):
                        for j in range(i+1, len(depts)):
                            d1, d2 = depts[i], depts[j]
                            if d1 in matrix.index and d2 in matrix.columns:
                                matrix.loc[d1, d2] += 1
                                matrix.loc[d2, d1] += 1
            
            fig_matrix = px.imshow(matrix, text_auto=True, color_continuous_scale='RdBu_r', title="Joint Response Frequency")
            st.plotly_chart(fig_matrix, use_container_width=True)
            
        with c2:
            st.markdown("### 📊 Volume Anomaly Detection")
            st.caption("Is current case volume abnormal?")
            
            current_volume = len(df[~df['resolved']])
            # Simplified Z-score simulation (In real app, fetch historical avg)
            historical_avg = 15 
            std_dev = 5
            
            z_score = (current_volume - historical_avg) / std_dev
            
            score_col = "green"
            status = "Normal"
            if z_score > 2:
                score_col = "red"
                status = "Critical Surge"
            elif z_score > 1:
                score_col = "orange"
                status = "Elevated"
            
            st.metric("Current Z-Score", f"{z_score:.2f}", delta=status)
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = z_score,
                title = {'text': "Surge Index"},
                gauge = {
                    'axis': {'range': [-3, 5]},
                    'bar': {'color': score_col},
                    'steps': [
                        {'range': [-3, 1], 'color': "lightgreen"},
                        {'range': [1, 2], 'color': "lightyellow"},
                        {'range': [2, 5], 'color': "Salmon"}],
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
    else:
        st.info("Insufficient data for strategic analysis.")


# ================= TAB 1: DASHBOARD =================
with tab1:
    # Top Level Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Emergencies", active_cases, f"{total_cases} Total")
    m2.metric("Critical (Active)", critical_active, delta="High Risk", delta_color="inverse")
    m3.metric("Urgent (Active)", urgent_active, delta="Medium Risk", delta_color="off")
    m4.metric("Fully Resolved", resolved_count, delta="Completed", delta_color="normal")
    
    st.divider()
    
    # Recent Active Critical Cases
    st.subheader("Critical Action Required")
    if not df.empty:
        critical_df = df[(df['priority'] == 'CRITICAL') & (~df['resolved'])]
        if not critical_df.empty:
            for _, row in critical_df.iterrows():
                with st.expander(f"CRITICAL - #{row['id']} - {row['timestamp']}", expanded=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Description:** {row['complaint']}")
                        st.markdown(f"**Depts:** {', '.join(row['departments']) if row['departments'] else 'None'}")
                        
                        # Resolution Progress
                        resolved_by = row.get('resolved_by')
                        if not isinstance(resolved_by, list):
                            resolved_by = []
                            
                        assigned = row.get('departments')
                        if not isinstance(assigned, list):
                            assigned = []

                        if resolved_by:
                            st.info(f"⏳ Partial Resolution: {len(resolved_by)}/{len(assigned)} ({', '.join(resolved_by)})")
                        else:
                            st.caption("Status: Pending Action")
                    with c2:
                        st.metric("Severity", f"{row['severity']}%")
                        st.caption("Action: Monitor Only. Resolve in Dept Portal.")
                    
                    # AI Co-Pilot for Orchestrator
                    if st.button("🧠 Generate AI Insights", key=f"orch_ai_{row['id']}"):
                        with st.spinner("AI analyzing..."):
                            try:
                                payload = {
                                    "complaint": row['complaint'],
                                    "departments": row['departments'],
                                    "severity": row['severity']
                                }
                                resp = requests.post(f"{API_URL}/generate_advice", json=payload)
                                if resp.status_code == 200:
                                    st.markdown(resp.json().get("advice"))
                                else:
                                    error_msg = resp.json().get("error", "Unknown Error")
                                    st.error(f"AI Service Error: {error_msg}")
                            except:
                                st.error("Connection Error")
        else:
            st.success("No active critical cases at the moment.")
    else:
        st.info("No data available.")

# ================= TAB 2: ALL CASES =================
with tab2:
    c_head, c_btn = st.columns([4, 1])
    with c_head:
        st.subheader("Case Management Registry (Read-Only)")
    with c_btn:
        if not df.empty:
             csv = df.to_csv(index=False).encode('utf-8')
             st.download_button("Export to CSV", csv, "emergency_registry.csv", "text/csv")
    
    if not df.empty:
        # Filters
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            filter_status = st.selectbox("Filter Status", ["All", "Active", "Resolved"])
        with f_col2:
            filter_dept = st.selectbox("Filter Dept", ["All"] + sorted(list(set([d for depts in df['departments'] if depts for d in depts]))))

        # Apply Filters
        filtered_df = df.copy()
        if filter_status == "Active":
            filtered_df = filtered_df[~filtered_df['resolved']]
        elif filter_status == "Resolved":
            filtered_df = filtered_df[filtered_df['resolved']]
            
        if filter_dept != "All":
            filtered_df = filtered_df[filtered_df['departments'].apply(lambda x: filter_dept in x if x else False)]

        # Display Table
        st.dataframe(
            filtered_df[['id', 'timestamp', 'priority', 'severity', 'resolved', 'departments', 'action']],
            column_config={
                "resolved": st.column_config.CheckboxColumn("Resolved?", disabled=True),
                "severity": st.column_config.ProgressColumn("Severity", min_value=0, max_value=100, format="%d%%"),
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No cases found.")

# ================= TAB 3: VISUALIZATIONS =================
with tab3:
    if not df.empty:
        g1, g2 = st.columns(2)
        
        with g1:
            # Severity Distribution
            fig_sev = px.histogram(df, x="severity", nbins=20, title="Severity Distribution", color_discrete_sequence=['#ff4b4b'])
            st.plotly_chart(fig_sev, use_container_width=True)
            
        with g2:
            # Priority Breakdown
            fig_pri = px.pie(df, names='priority', title="Case Priority Breakdown", hole=0.4, color_discrete_sequence=['#ff4b4b', '#ffa421', '#00cc96'])
            st.plotly_chart(fig_pri, use_container_width=True)
            
        # Department Load Chart
        st.divider()
        st.subheader("Department Load Analysis")
        all_depts = []
        for depts in df['departments']:
            if depts: all_depts.extend(depts)
        dept_counts = pd.Series(all_depts).value_counts().reset_index()
        dept_counts.columns = ['Department', 'Count']
        
        fig_dept = px.bar(dept_counts, x='Department', y='Count', title="Total Cases per Department", color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig_dept, use_container_width=True)

        # Time Heatmap
        st.divider()
        if 'timestamp' in df.columns:
            df['dt'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['dt'].dt.hour
            df['day'] = df['dt'].dt.day_name()
            
            heatmap_data = df.groupby(['day', 'hour']).size().reset_index(name='count')
            
            fig_heat = px.density_heatmap(heatmap_data, x='hour', y='day', z='count', 
                                         title="Emergency Intensity Heatmap (Day vs Hour)",
                                         nbinsx=24, color_continuous_scale='Viridis')
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # Line Chart
            cases_over_time = df.set_index('dt').resample('H').size().reset_index(name='count')
            fig_time = px.line(cases_over_time, x='dt', y='count', title="Hourly Emergency Trends", markers=True)
            st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Insufficient data for visualization.")

# ================= TAB 4: MEDICINE INFO =================
with tab4:
    st.subheader("Predictive Medicine Intelligence")
    st.markdown("Forecasting based on **Weather Patterns** and **Historical Data** (Not active cases).")
    
    if res_model and pat_model:
        c1, c2, c3 = st.columns(3)
        with c1:
            temp = st.number_input("Forecast: Temp (°C)", value=30)
        with c2:
            humidity = st.number_input("Forecast: Humidity (%)", value=60)
        with c3:
            season = st.number_input("Season Index", value=0.0)
            
        weather_input = pd.DataFrame([[temp, humidity, 5.0, season]], 
                            columns=["temp", "humidity", "rainfall", "season_index"])
        
        try:
            # Predict Patients
            pred_patients = int(pat_model.predict(weather_input)[0])
            
            # Predict Resources
            res_input = pd.DataFrame([[pred_patients, int(pred_patients*0.1), season, temp]], 
                            columns=["patients", "icu_cases", "season_index", "temp"])
            pred_resources = res_model.predict(res_input)[0]
            
            st.divider()
            
            m1, m2 = st.columns(2)
            m1.metric("Predicted Daily Patient Load", pred_patients, help="Based on weather conditions")
            m2.metric("Expected ICU Admissions", int(pred_patients * 0.12))
            
            st.subheader("Resource Demand Forecast (Next 7 Days)")
            
            targets = ["Med A", "Med B", "Med C", "Oxygen", "Syringes"]
            chart_data = pd.DataFrame({
                "Resource": targets,
                "Predicted Demand": pred_resources * 7  # Weekly demand
            })
            
            st.bar_chart(chart_data, x="Resource", y="Predicted Demand", color="#ff4b4b")
            
            with st.expander("View Detailed Numbers"):
                st.dataframe(chart_data, use_container_width=True)
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            
    else:
        st.warning("Forecasting models not found. Please run Training module.")

# --- Auto Refresh Logic ---
if st.session_state.auto_refresh:
    time.sleep(2)
    st.rerun()
