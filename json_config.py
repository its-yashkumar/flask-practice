import streamlit as st
import json

# Initialize session state variables
if 'topology_model' not in st.session_state:
    st.session_state.topology_model = []

if "update_text" not in st.session_state:
    st.session_state.update_text = None

# Function to store configuration
def store_temp_config(config_topology, default_model):
    if not config_topology or not default_model:
        return False  # Ensure valid inputs

    existing_topologies = {topo for topo, _ in st.session_state.topology_model}

    if config_topology in existing_topologies:
        return "overwrite_prompt"

    st.session_state.topology_model.append((config_topology, default_model))
    st.success(f"Added new topology '{config_topology}' with model '{default_model}'")
    return True  # Indicate successful addition

# UI elements
st.title("Storm Topology Model Configuration")

topologies = ["topo1", "topo2", "topo3"]  # Example topologies
models_dict = {
    "topo1": ["modelA", "modelB"],
    "topo2": ["modelX", "modelY"],
    "topo3": ["modelP", "modelQ"],
}

# Select topology & model
config_topology = st.selectbox("Select storm topology", topologies, index=None, key="input_topology")

if config_topology:
    config_default = st.selectbox(
        "Select trained model", models_dict[config_topology], key="input_model"
    )

# Button for saving topology
if st.button("Save Topology"):
    st.session_state.update_text = store_temp_config(st.session_state.input_topology, st.session_state.input_model)

# Overwrite prompt if topology exists
if st.session_state.update_text == "overwrite_prompt":
    st.warning("Topology already exists. Do you want to overwrite it?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Yes, Overwrite"):
            for i, (topo, _) in enumerate(st.session_state.topology_model):
                if topo == st.session_state.input_topology:
                    st.session_state.topology_model[i] = (
                        st.session_state.input_topology, 
                        st.session_state.input_model
                    )
                    st.success(f"Updated '{st.session_state.input_topology}' to '{st.session_state.input_model}'")
                    st.session_state.update_text = True
                    st.rerun()  # Forces UI refresh
    
    with col2:
        if st.button("No, Keep Existing"):
            st.warning("Topology not updated.")
            st.session_state.update_text = False  # Prevent further updates

# Display stored topologies
st.write("### Stored Topologies:")
st.write(st.session_state.topology_model)
