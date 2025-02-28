# import streamlit as st
# import json

# # File path for the JSON configuration
# CONFIG_FILE_PATH = "config.json"

# # Load topologies from JSON
# def load_config(file_path):
#     try:
#         with open(file_path, "r") as file:
#             return json.load(file)
#     except FileNotFoundError:
#         st.error("Configuration file not found!")
#         return {}
#     except json.JSONDecodeError:
#         st.error("Error decoding JSON file!")
#         return {}
# def save_config(file_path, config):
#     with open(file_path, "w") as file:
#         json.dump(config, file, indent=4)

# # Load topologies from the JSON file
# config = load_config(CONFIG_FILE_PATH)

# # Initialize session state variables
# if 'topology_model' not in st.session_state:
#     st.session_state.topology_model = []

# if "update_text" not in st.session_state:
#     st.session_state.update_text = None

# # Function to store configuration
# def store_temp_config(config_topology, default_model):
#     if not config_topology or not default_model:
#         return False  # Ensure valid inputs

#     existing_topologies = {topo for topo, _ in st.session_state.topology_model}

#     if config_topology in existing_topologies:
#         return "overwrite_prompt"

#     st.session_state.topology_model.append((config_topology, default_model))
#     st.success(f"Added new topology '{config_topology}' with model '{default_model}'")
#     return True  # Indicate successful addition

# # UI Layout
# st.title("Storm Topology Model Configuration")

# topologies = list(config.keys())  # Extract topology names from JSON

# # Form for selecting topology & model
# with st.form(key='topology_form'):
#     columns = st.columns(2)

#     with columns[0]:  # Left Column: Topology Selection
#         config_topology = st.selectbox(
#             "Select Storm Topology",
#             topologies,
#             index=None,
#             key="input_topology"
#         )

#     with columns[1]:  # Right Column: Model Selection
#         if config_topology:
#             default_model = config[config_topology].get("default", "")
#             model_list = config[config_topology].get("models", [])

#             config_default = st.selectbox(
#                 "Select Trained Model",
#                 model_list,
#                 index=model_list.index(default_model) if default_model in model_list else 0,
#                 key="input_model"
#             )
    
#     # Submit button inside the form
#     submit = st.form_submit_button("Save Topology")

# # Handling form submission
# if submit:
#     st.session_state.update_text = store_temp_config(
#         st.session_state.input_topology, st.session_state.input_model
#     )

# # Overwrite prompt if topology exists
# if st.session_state.update_text == "overwrite_prompt":
#     st.warning("Topology already exists. Do you want to overwrite it?")

#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Yes, Overwrite"):
#             for i, (topo, _) in enumerate(st.session_state.topology_model):
#                 if topo == st.session_state.input_topology:
#                     st.session_state.topology_model[i] = (
#                         st.session_state.input_topology, 
#                         st.session_state.input_model
#                     )
#                     st.success(f"Updated '{st.session_state.input_topology}' to '{st.session_state.input_model}'")
#                     st.session_state.update_text = True
#                     st.rerun()  # Refresh UI
    
#     with col2:
#         if st.button("No, Keep Existing"):
#             st.warning("Topology not updated.")
#             st.session_state.update_text = False  # Prevent further updates

# # Display stored topologies
# st.write("### Stored Topologies:")
# st.write(st.session_state.topology_model)




import streamlit as st
import json
import os

# File path for the JSON configuration
CONFIG_FILE_PATH = "config.json"

# Function to load the JSON configuration
def load_config(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}  # Return empty config if file doesn't exist or is corrupted

# Function to save updated configurations to the JSON file
def save_config(file_path, config_data):
    with open(file_path, "w") as file:
        json.dump(config_data, file, indent=4)

# Load existing topologies from the JSON file
config = load_config(CONFIG_FILE_PATH)

# Initialize session state variables
if "topology_model" not in st.session_state:
    st.session_state.topology_model = []  # Start with an empty list

if "update_text" not in st.session_state:
    st.session_state.update_text = None

# Function to store and update topology
def store_temp_config(config_topology, default_model):
    if not config_topology or not default_model:
        return False  # Ensure valid inputs

    existing_topologies = {topo for topo, _ in st.session_state.topology_model}

    if config_topology in existing_topologies:
        return "overwrite_prompt"

    # Add new topology in session state
    st.session_state.topology_model.append((config_topology, default_model))

    # Update config dictionary
    if config_topology not in config:
        config[config_topology] = {"default": default_model, "models": [default_model]}
    else:
        if default_model not in config[config_topology]["models"]:
            config[config_topology]["models"].append(default_model)
        config[config_topology]["default"] = default_model

    # Save updated configuration to file
    save_config(CONFIG_FILE_PATH, config)

    st.success(f"Added new topology '{config_topology}' with model '{default_model}'")
    return True  # Indicate successful addition

# UI Layout
st.title("Storm Topology Model Configuration")

topologies = list(config.keys())  # Extract topology names from JSON

txtColumns = st.columns(2)  # Creates two columns for dropdowns

with txtColumns[0]:
    config_topology = st.selectbox(
        "Select Storm Topology",
        topologies,
        placeholder="Choose a Storm Topology",
        index=None,
        key="input_topology"
    )

with txtColumns[1]:
    if config_topology is None:
        config_default = st.selectbox(
            "Select Trained Model",
            options=[],
            placeholder="Choose a storm topology first",
            key="Null Topology"
        )
    else:
        default_model = config[config_topology]['default']
        config_default = st.selectbox(
            "Select Trained Model",
            config[config_topology]['models'],
            placeholder="Choose a trained model",
            index=config[config_topology]['models'].index(default_model),
            key="input_model"
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

                    # Update config dictionary
                    config[st.session_state.input_topology]["default"] = st.session_state.input_model
                    if st.session_state.input_model not in config[st.session_state.input_topology]["models"]:
                        config[st.session_state.input_topology]["models"].append(st.session_state.input_model)

                    # Save updated configuration to file
                    save_config(CONFIG_FILE_PATH, config)

                    st.success(f"Updated '{st.session_state.input_topology}' to '{st.session_state.input_model}'")
                    st.session_state.update_text = True
                    st.rerun()  # Forces UI refresh
    
    with col2:
        if st.button("No, Keep Existing"):
            st.warning("Topology not updated.")
            st.session_state.update_text = False  # Prevent further updates

# Display stored topologies
st.write("### Stored Topologies:")
st.write(st.session_state.topology_model)  # Will start as an empty list


if st.button("Save Configuration"):
   for i, (topo,de_model) in enumerate(st.session_state.topology_model):
       config[topo]['default']=de_model
   save_config(CONFIG_FILE_PATH, config) 
