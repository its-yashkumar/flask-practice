import streamlit as st
import pandas as pd
import numpy as np
import time
# import tft_bg
# from tft_bg import predictions as pred, train_model, df, timestamps
from streamlit_autorefresh import st_autorefresh  # Import auto-refresh utility

# Function to fetch real-time lagging storm topologies
def get_lagging_storms():
    """Simulate fetching real-time lagging storms. Replace with actual logic."""
    return list(np.random.choice(["Topology A", "Topology B", "Topology C"], 
                                 size=np.random.randint(1, 4), 
                                 replace=False))

# Initialize session state for storing lagging storms
if "lagging_storms" not in st.session_state:
    st.session_state["lagging_storms"] = get_lagging_storms()

# Automatically refresh every 5 seconds **without blocking execution**
st_autorefresh(interval=5000, key="lag_update")

# Update the storm topology list dynamically
st.session_state["lagging_storms"] = get_lagging_storms()
storm_list = ", ".join(st.session_state["lagging_storms"])

# Display real-time updated storm list
st.markdown(
    f"""
    <style>
    .storm-container {{
        display: flex;
        align-items: center;
        margin-bottom: 1em;
    }}
    .storm-label {{
        font-weight: bold;
        margin-right: 1em;
        border: 1px solid black;
        border-radius: 0.5em;
        padding: 0.5em;
    }}
    .storm-list {{
        border: 1px solid black;
        border-radius: 0.5em;
        padding: 0.5em;
        width: auto;
    }}
    </style>
    <div class="storm-container">
        <span class="storm-label">Lag alert at Topologies:</span>
        <div class="storm-list">{storm_list}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("Execution done")
