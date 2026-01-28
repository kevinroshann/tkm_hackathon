import streamlit as st
import requests

st.set_page_config(
    page_title="Admin Command Console",
    layout="wide"
)

st.title("👮 Emergency Command & Admin Console")

API_URL = "http://localhost:5000"

# Health Check
try:
    response = requests.get(f"{API_URL}/health", timeout=1)
    if response.status_code == 200:
        st.success("Network Status: SECURE | Neural Core: ONLINE")
    else:
        st.error("Network Status: UNSTABLE")
except:
    st.error("Network Status: OFFLINE (Check API Server)")

st.markdown("""
### Authorized Access Only
This console is restricted to **authorized personnel only**.

**Modules:**
- **Orchestrator**: Command Center for real-time incident management.
- **Hospital Admin**: Medical resource forecasting and procurement.
- **Department Portal**: Field operations and case resolution.

*For public reporting, please use the Public Safety App.*
""")

with st.sidebar:
    st.warning("RESTRICTED AREA")
    st.info("Use the navigation menu above to access specific modules.")
