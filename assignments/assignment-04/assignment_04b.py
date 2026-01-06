###############################################################################
# File Name: assignment_04b.py
#
# Description: This program implements a Streamlit web application that allows 
# users to upload a CT scan image, sends it to a Flask API for inference, and 
# displays the model’s COVID vs. normal prediction.
#
# Record of Revisions (Date | Author | Change):
# 09/12/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import streamlit as st, joblib
import requests
import json

st.set_page_config(page_title="Covid CT Prediction App", layout="wide")
st.title("Covid CT Prediction App")

colLeft, colRight = st.columns([1, 1])

with colLeft:
    st.header('Instructions') # Header for left column
    st.text('Upload a CT scan image, in jpg format, to have the model predict if the CT scan is showing covid or normal.') # Instructions
with colRight:
    image = st.file_uploader('Upload Image:', type = ['jpg'], accept_multiple_files = False) # Image input
    run = st.button('Run') # Run Button

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if run:
        if not image:
            st.error('Image is required!')
        else:
            try: # Send data to the Flask API
                response = requests.post('http://127.0.0.1:5000/predict', files={'image': image})
                response.raise_for_status() # Raise an exception for bad status codes
                prediction = response.json()
                st.header('Model Prediction:')
                st.image(image)
                st.success(f"This is a {prediction['prediction']} CT scan.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the Flask API. Make sure it is running.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error during API call: {e}") 