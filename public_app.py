import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time

st.set_page_config(page_title="Public Safety Portal", layout="centered")

API_URL = "http://localhost:5000"

st.title("🛡️ Community Safety Portal")

# --- TABS ---
tab1, tab2 = st.tabs(["📢 Report Emergency", "📊 Public Overview"])

# ================= TAB 1: REPORTING =================
with tab1:
    st.markdown("### Report an Incident")
    st.info("Your safety is our priority. Please provide details below.")
    
    with st.form("emergency_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_name = st.text_input("Your Name", placeholder="John Doe")
        with col2:
            user_contact = st.text_input("Contact Number", placeholder="+1 234 567 890")
        
        complaint = st.text_area("What is happening?", height=150, 
                                placeholder="Describe the situation clearly... (e.g. 'Fire on 4th floor')")
        
        submitted = st.form_submit_button("Submit Report", type="primary")

    if submitted:
        if not user_name or not user_contact or not complaint:
            st.error("Please fill in all fields.")
        else:
            with st.spinner("Connecting to Emergency Dispatch..."):
                try:
                    payload = {
                        "text": complaint,
                        "user_name": user_name,
                        "user_contact": user_contact,
                        "sos": 0
                    }
                    response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Report Received! Reference ID: #{data['id']}")
                        
                        # --- CIVILIAN AI SAFETY ADVICE ---
                        st.divider()
                        st.subheader("⚠️ Safety Instructions")
                        with st.spinner(" Generating safety advice..."):
                            try:
                                ai_payload = {
                                    "complaint": complaint,
                                    "departments": data['departments'],
                                    "severity": data['severity'],
                                    "audience": "civilian"  # Requesting Civilian Persona
                                }
                                ai_resp = requests.post(f"{API_URL}/generate_advice", json=ai_payload)
                                if ai_resp.status_code == 200:
                                    advice = ai_resp.json().get("advice", "")
                                    st.markdown(advice)
                                else:
                                    st.warning("Stay safe. Help is on the way.")
                            except:
                                st.warning("Stay safe. Help is on the way.")
                        
                        st.divider()
                        st.caption("Emergency Services have been notified.")
                        
                    else:
                        st.error(f"Server Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Connection Failed: {str(e)}")

# ================= TAB 2: PUBLIC OVERVIEW =================
with tab2:
    st.subheader("Community Safety Status")
    
    try:
        response = requests.get(f"{API_URL}/logs", timeout=2)
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            
            if not df.empty:
                # High Level Metrics (Safe to show public)
                total = len(df)
                resolved = len(df[df['resolved']])
                active = total - resolved
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Incidents Handled", total)
                c2.metric("Active Response", active)
                c3.metric("Resolved Cases", resolved)
                
                st.divider()
                
                # Anonymous Heatmap
                if 'timestamp' in df.columns:
                    df['dt'] = pd.to_datetime(df['timestamp'])
                    df['hour'] = df['dt'].dt.hour
                    df['day'] = df['dt'].dt.day_name()
                    heatmap_data = df.groupby(['day', 'hour']).size().reset_index(name='count')
                    
                    # --- NEW CHARTS ---
                    st.markdown("### 📈 Real-time Community Insights")
                    g1, g2 = st.columns(2)
                    
                    with g1:
                        # Incident Type Distribution (Based on assigned departments as proxy for type)
                        all_depts = []
                        for depts in df['departments']:
                            if depts: all_depts.extend(depts)
                        if all_depts:
                            dept_counts = pd.Series(all_depts).value_counts().reset_index()
                            dept_counts.columns = ['Type', 'Count']
                            fig_pie = px.pie(dept_counts, names='Type', values='Count', title="Incidents by Service Type", hole=0.4)
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                    with g2:
                        # Activity Timeline
                        cases_over_time = df.set_index('dt').resample('H').size().reset_index(name='count')
                        fig_line = px.line(cases_over_time, x='dt', y='count', title="Emergency Activity (Last 24h)", markers=True)
                        st.plotly_chart(fig_line, use_container_width=True)

                    st.divider()
                    
                    fig = px.density_heatmap(heatmap_data, x='hour', y='day', z='count', 
                                            title="Incident Density (Past 7 Days)",
                                            nbinsx=24, color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)
                
                st.info("Detailed information is restricted to authorized personnel.")
            else:
                st.write("No data available.")
        else:
            st.error("Could not fetch status.")
    except:
        st.error("System Maintenance Mode")

