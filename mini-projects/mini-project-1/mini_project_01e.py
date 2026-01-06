###############################################################################
# File Name: mini_project_01e.py
#
# Description: This Streamlit script creates a web-based physics Q&A chatbot 
# interface where students can enter a physics question, send it to a Flask API, 
# and view the model-generated answer in real time. It provides instructions, 
# handles empty inputs, and displays API connection errors gracefully.
#
# Record of Revisions (Date | Author | Change):
# 09/21/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import streamlit as st
import requests

st.set_page_config(page_title="Physics Q&A Chatbot", layout="wide")
st.title("Physics Q&A Chatbot")

colLeft, colRight = st.columns([1, 1])

with colLeft:
    st.header('Instructions') # Header for left column
    st.text('Enter question pertaining to physics. Hit enter and the chatbot will answer your question.') # Instructions
with colRight:
    question = st.text_area('Enter Physics Question:')  # Text input for question
    run = st.button('Run') # Run Button

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if run:
        if not question:
            st.error('A question is required!')
        else:
            try: # Send data to the Flask API
                response = requests.post('http://127.0.0.1:5000/predict', json={'question': question})
                response.raise_for_status() # Raise an exception for bad status codes
                prediction = response.json()
                st.header('Model Prediction:')
                st.success(prediction['prediction'])
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the Flask API. Make sure it is running.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error during API call: {e}") 