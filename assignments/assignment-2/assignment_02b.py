###############################################################################
# File Name: assignment_02b.py
#
# Description: This Streamlit app provides a simple UI for entering NFL team 
# stats (PF, PA, PD, SoS) and sends them to a Flask API to predict whether the 
# team had a winning or losing record, then displays the prediction.
#
# Record of Revisions (Date | Author | Change):
# 08/27/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import streamlit as st, joblib
import requests
import json

st.set_page_config(page_title="NFL Model Prediction App", layout="wide")
st.title("NFL Model Prediction App")

PF = st.text_input("PF (number):")
PA = st.text_input("PA (number):")
PD = st.text_input("PD (number):")
SoS = st.text_input("SoS (number):")

# Communicate with API
if st.button("Predict"):
    # Prepare data for sending to API
    input_data = {'features': [float(PF), float(PA), float(PD), float(SoS)]}
        
    # Send data to the Flask API
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
        response.raise_for_status() # Raise an exception for bad status codes
        prediction_result = response.json()
        st.success(f"Prediction: {prediction_result['prediction']} Record")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the Flask API. Make sure it is running.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error during API call: {e}")
