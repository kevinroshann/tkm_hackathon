import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime

st.set_page_config(page_title="Department Portal", layout="wide")

API_URL = "http://localhost:5000"
DEPARTMENTS = ["Medical", "Fire", "Police", "Utilities", "Public Works", "Social Services"]

st.title("Department Response Portal")

# --- LOGIN / SELECTOR ---
with st.sidebar:
    st.markdown("### Department Login")
    selected_dept = st.selectbox("Select Department", DEPARTMENTS)
    st.info(f"Viewing as: **{selected_dept} Department**")
    
    st.divider()
    auto_refresh = st.toggle("Auto Refresh", value=True)
    if st.button("Refresh Now"):
        st.rerun()

# --- FETCH DATA ---
@st.cache_data(ttl=2)
def fetch_dept_logs(dept):
    try:
        # Fetching all logs and filtering client-side for flexibility
        response = requests.get(f"{API_URL}/logs/{dept}", timeout=2)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

logs = fetch_dept_logs(selected_dept)
df = pd.DataFrame(logs)

# --- ANALYTICS ---
# Calculate Average Resolution Time
avg_res_time = "N/A"
if not df.empty and 'resolved_at' in df.columns:
    resolved_df = df[df['resolved'] == True].copy()
    if not resolved_df.empty and 'timestamp' in resolved_df.columns:
        resolved_df['start'] = pd.to_datetime(resolved_df['timestamp'])
        resolved_df['end'] = pd.to_datetime(resolved_df['resolved_at'])
        resolved_df['duration_mins'] = (resolved_df['end'] - resolved_df['start']).dt.total_seconds() / 60
        minutes = resolved_df['duration_mins'].mean()
        avg_res_time = f"{minutes:.1f} mins"

# --- METRICS ---
col1, col2, col3, col4 = st.columns(4)
active_count = len([x for x in logs if not x.get('resolved')]) if logs else 0
resolved_count = len([x for x in logs if x.get('resolved')]) if logs else 0

col1.metric("Active Cases", active_count, delta="Assigned to you")
col2.metric("Resolved Cases", resolved_count)
col3.metric("Response Efficiency", avg_res_time, help="Average time from report to resolution")
col4.metric("Total History", len(logs) if logs else 0)

st.divider()

# --- ACTIVE CASES VIEW ---
st.subheader(f"Active Assignments: {selected_dept}")

if not df.empty and active_count > 0:
    active_df = df[~df['resolved']].copy()
    
    for _, row in active_df.iterrows():
        # Card style
        priority_color = "red" if row['priority'] == 'CRITICAL' else "orange" if row['priority'] == 'URGENT' else "green"
        
        with st.container():
            c1, c2, c3 = st.columns([4, 2, 1])
            with c1:
                st.markdown(f"#### #{row['id']} | :{priority_color}[**{row['priority']}**]") 
                st.write(f"**Desc:** {row['complaint']}")
                
                # --- VISIBILITY: WHO IS HERE? ---
                init_depts = row.get('initial_depts', row['departments']) # Fallback
                all_depts = row['departments']
                
                # 1. First Initiated
                if init_depts:
                    init_badges = " ".join([f"`{d}`" for d in init_depts])
                    st.markdown(f"**First Initiated:** {init_badges}")
                
                # 2. Joint / Requested
                # Find depts that are in all_depts but NOT in init_depts
                joint_list = [d for d in all_depts if d not in init_depts]
                if joint_list:
                    joint_badges = " ".join([f"`{d}`" for d in joint_list])
                    st.markdown(f"**Joint Response:** {joint_badges}")
                
                # --- ALERTS: AM I BEING CALLED? ---
                alerts = row.get('backup_requests', [])
                for alert in alerts:
                    # If I am the target, show BIG
                    if alert['to'] == selected_dept:
                        st.error(f"🚨 **INCOMING REQUEST from {alert['from']}**: {alert['message']}")

                st.caption(f"From: {row['user_name']} ({row['user_contact']}) | {row['timestamp']}")
            
            with c2:
                st.markdown(f"**Action:** {row['action']}")
                st.metric("Severity", f"{row['severity']}%")
                
            with c3:
                st.write("") # Spacer
                
                # Check if already resolved by this department
                resolved_by_list = row.get('resolved_by', [])
                is_resolved_by_dept = selected_dept in resolved_by_list
                
                if is_resolved_by_dept:
                    st.success("✔ Dept Resolved")
                else:
                    if st.button("Mark Resolved", key=f"resolve_{row['id']}"):
                        try:
                            # Send department context
                            payload = {"department": selected_dept}
                            response = requests.post(f"{API_URL}/resolve/{row['id']}", json=payload)
                            
                            if response.status_code == 200:
                                data = response.json()
                                if data.get('fully_resolved'):
                                    st.success("Case Fully Resolved! 🎉")
                                else:
                                    st.info("Dept Resolution Recorded. Waiting for others.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Error updating status")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            # --- AI CO-PILOT ---
            with st.expander("🧠 AI Tactical Co-Pilot", expanded=False):
                if st.button("Generate Tactical Plan", key=f"ai_{row['id']}"):
                    with st.spinner("Disconnecting... AI analyzing tactical options..."):
                        try:
                            payload = {
                                "complaint": row['complaint'],
                                "departments": row['departments'],
                                "severity": row['severity']
                            }
                            resp = requests.post(f"{API_URL}/generate_advice", json=payload)
                            if resp.status_code == 200:
                                advice = resp.json().get("advice", "No advice generated.")
                                st.markdown(advice)
                            else:
                                error_msg = resp.json().get("error", "Unknown Error")
                                st.error(f"AI Service Unavailable: {error_msg}")
                        except Exception as e:
                            st.error(f"Error: {e}")

            # --- REQUEST BACKUP UI ---
            with st.expander("📡 Request Inter-Dept Support (Send New)", expanded=False):
                st.caption("Need more help? Alert another department.")
                with st.form(key=f"backup_form_{row['id']}"):
                    target_dept = st.selectbox("Select Target Dept", [d for d in DEPARTMENTS if d != selected_dept])
                    help_msg = st.text_input("Message", value=f"Need backup for #{row['id']}")
                    
                    if st.form_submit_button("📢 Send Alert"):
                        try:
                            payload = {
                                "origin_dept": selected_dept,
                                "target_dept": target_dept,
                                "message": help_msg,
                                "severity": row['severity'], # Auto-copy severity
                                "ref_id": row['id']
                            }
                            resp = requests.post(f"{API_URL}/request_backup", json=payload)
                            if resp.status_code == 200:
                                st.success(f"Alert sent to {target_dept}!")
                            else:
                                st.error("Failed to send alert.")
                        except Exception as e:
                            st.error(f"Connection Error: {e}")
            
            st.divider()
elif not df.empty:
    st.success("No active cases! All assigned tasks completed.")
else:
    st.info("No records found for this department.")

# --- HISTORY ---
with st.expander("Case History (Resolved)"):
    if not df.empty and resolved_count > 0:
        hist_df = df[df['resolved']]
        st.dataframe(hist_df[['id', 'timestamp', 'priority', 'complaint', 'resolved_at']], hide_index=True, use_container_width=True)
    else:
        st.write("No history available.")

# --- AUTO REFRESH ---
if auto_refresh:
    time.sleep(3)
    st.rerun()
