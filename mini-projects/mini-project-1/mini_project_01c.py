###############################################################################
# File Name: mini_project_01c.py
#
# Description: This Streamlit script provides a web interface where users can 
# enter symptoms, sends the input to a Flask API hosting a BERT-based medical 
# diagnosis model, and displays the predicted diagnosis. It includes input 
# validation, error handling, and a simple two-column layout for instructions 
# and user input.
#
# Record of Revisions (Date | Author | Change):
# 09/21/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import streamlit as st
import requests

st.set_page_config(page_title="Diagnosis App", layout="wide")
st.title("Diagnosis App")

colLeft, colRight = st.columns([1, 1])

with colLeft:
    st.header('Instructions') # Header for left column
    st.text('Enter your symptoms. Hit run and the model will then return a diagnosis.') # Instructions
with colRight:
    symptoms = st.text_area('Enter Symptoms:')  # Text input for symptoms
    run = st.button('Run') # Run Button

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if run:
        if not symptoms:
            st.error('Symptoms are required!')
        else:
            try: # Send data to the Flask API
                response = requests.post('http://127.0.0.1:5000/predict', json={'symptoms': symptoms})
                response.raise_for_status() # Raise an exception for bad status codes
                prediction = response.json()
                st.header('Model Prediction:')
                st.success(f"Diagnosis: {prediction['prediction']}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the Flask API. Make sure it is running.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error during API call: {e}") 