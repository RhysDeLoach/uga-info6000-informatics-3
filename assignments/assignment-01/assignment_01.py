###############################################################################
# File Name: assignment_01.py
#
# Description: This Streamlit app creates a two-column UI where the left column 
# collects user inputs (name, number of points, optional image) and the right 
# column displays outputs, including a greeting, a random scatter plot, and 
# the uploaded image if provided. It also includes an instructions expander 
# and tabs to organize output explanations and visuals.
#
# Record of Revisions (Date | Author | Change):
# 08/15/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import streamlit as st
import numpy as np

st.set_page_config(page_title = 'Exercise 1', layout = 'wide') # Set the page configuration with wide layout
st.title('Exercise 1: Building a Streamlit UI'); st.caption('Made by Rhys DeLoach') # Set the title and caption

colLeft, colRight = st.columns([1, 1]) # Create two equal-width columns (See AI Reference 1)

with colLeft:
    st.header('Inputs') # Header for left column
    name = st.text_input('Enter your name (Required):') # Name input
    numPoint = st.number_input('Enter number of points (Required):', min_value=1) # Points input
    image = st.file_uploader('Upload Image (Optional):', type = ['png', 'jpg', 'jpeg']) # Image input
    run = st.button('Run') # Run Button
    with st.expander('Instructions'): # Instructions expander
        st.text('Enter your name and desired number of points to be graphed. Optionally, you can also upload an image (.png, .jpg, or .jpeg).')

with colRight:
    st.header('Outputs') # Header for right column
    results = st.container() # Container for output
    tabSummary, tabDetails, tabPeanut = st.tabs(['Summary', 'Details', 'Peanuts']) # Creates the information tabs
    with tabSummary:
        st.text('The output will contain a greeting to the user, a scatter plot of the input number of points, and if provided, an image.')
    with tabDetails:
        st.text('The scatter plot x-axis is based on point index, and the y-axis is a randomly generated number between 0 and 1')
    with tabPeanut:
        st.header(':peanuts:')

if run:
    if not name.strip():
        st.error('Name is required!') # Name error if name input is left empty
    else:
        with results: # Outputs
            st.text(f'Hello, {name}!')
            st.scatter_chart(np.random.rand(numPoint, 1))
            if image:
                st.image(image)
