import os
import importlib
import sys
import streamlit as st

MODEL_DIR = "src/models"

def list_models():
    """List all Python scripts in the models directory (excluding __init__.py)."""
    return [f[:-3] for f in os.listdir(MODEL_DIR) if f.endswith(".py") and f != "__init__.py"]

def load_and_run_model(model_name):
    """Dynamically import and run a selected model script."""
    if model_name not in list_models():
        st.error(f"Model '{model_name}' not found! Available models: {list_models()}")
        return
    
    sys.path.append(MODEL_DIR)  # Add models folder to import path
    module = importlib.import_module(model_name)
    
    if hasattr(module, "main"):  # Ensure the model script has a main function
        st.success(f"Running model: {model_name}")
        module.main()
    else:
        st.error(f"Error: '{model_name}' does not have a 'main()' function.")

# Streamlit UI
st.title("Model Runner")
st.sidebar.header("Select a Model")

available_models = list_models()

if available_models:
    selected_model = st.sidebar.selectbox("Choose a model to run", available_models)
    if st.sidebar.button("Run Model"):
        load_and_run_model(selected_model)
else:
    st.sidebar.warning("No models found in the directory.")
