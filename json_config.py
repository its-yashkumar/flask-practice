import streamlit as st
    import json
    import os
    from pathlib import Path

# Default file path (adjust as needed)
    DEFAULT_FILE_PATH = "json/config.json"

    def load_config(file_path):
        """Loads JSON configuration from a file."""
        try:
          with open(file_path, 'r') as file:
              return json.load(file)
        except FileNotFoundError:
            return {}  # Return an empty dictionary if the file doesn't exist
        except json.JSONDecodeError:
            st.error(f"Error decoding JSON in {file_path}")
            return {}


    def save_config(file_path, config_data):
        """Saves JSON configuration to a file."""
        with open(file_path, 'w') as file:
            json.dump(config_data, file, indent=4)
    def display_widgets(config,input_container,count):
        count+=1
        with input_container:
            t_partition,m_partition= st.columns(2)
            with t_partition:
                    config_topology=st.selectbox("Select storm topology",config.keys() , placeholder="Choose a Storm Topology",index=None,key=f"Select Topology {count}")
            with m_partition:
                    if  config_topology==None:
                        config_default=st.selectbox("Select trained model",None , placeholder="Choose a storm topology first",key=f"Null Topology {count}")
                    else:
                        default_model=config[config_topology]['default']
                        config_default=st.selectbox("Select trained model",config[config_topology]['models'] , placeholder="Choose a trained model",index=config[config_topology]['models'].index(default_model),key=f"Model Topology {count}")
        return config_topology,config_default,count

    def main():

            count=1
            st.header("Storm Topology Model Configuration")
            config = load_config(DEFAULT_FILE_PATH)
            if not config:
                config = {}
            # st.write(config.keys())
            # for topology_name in config.keys():
            #     st.write(config[topology_name]["default"])
            topologies=list(config.keys())
            input_container = st.empty()
            default_model=""
            _,_,count=display_widgets(config,input_container,count)
            count+=1
            display_widgets(config,input_container,count)
            st.button("Add Topology",on_click=display_widgets,args=(config,input_container,count))

            # import streamlit as st
            topology_model=[]
            st.write('# Solution without using a dataframe')

            if 't_partition' not in st.session_state:
                st.session_state.t_partition = ''
            if 'm_partition' not in st.session_state:
                st.session_state.m_partition = ''
            # if 'col3' not in st.session_state:
            #     st.session_state.col3 = ''
            # if 'col4' not in st.session_state:
            #     st.session_state.col4 = ''

            dataColumns = st.columns(2)
            with dataColumns[0]:
                st.write('#### Topology Name')
                st.session_state.t_partition
            with dataColumns[1]:
                st.write('#### Default Model')
                st.session_state.m_partition

            def add_txtForm(config_topology,default_model):
                st.session_state.t_partition += (st.session_state.input_topology + '  \n')
                st.session_state.m_partition += (st.session_state.input_model + '  \n')

            # txtForm = st.form(key='txtForm')
            txtForm = st.form(key='txtForm')
            # with txtForm:
            txtColumns = st.columns(2)
            with txtColumns[0]:
                 config_topology=st.selectbox("Select storm topology",topologies , placeholder="Choose a Storm Topology",index=None,key="input_topology")
            with txtColumns[1]:
                if  config_topology==None:
                    config_default=st.selectbox("Select trained model",None , placeholder="Choose a storm topology first",key=f"Null Topology {count}")
                else:
                    default_model=config[config_topology]['default']
                    config_default=st.selectbox("Select trained model",config[config_topology]['models'] , placeholder="Choose a trained model",index=config[config_topology]['models'].index(default_model),key="input_model")
            topology_model.append([config_topology,default_model])
            st.write(topology_model)
            st.button('Add Topology',on_click=add_txtForm,key="Add topology",args=(config_topology,default_model))

    main()


