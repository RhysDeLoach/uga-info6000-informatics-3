###############################################################################
# File Name: assignment_02c.py
#
# Description: This Flask API receives JSON data containing NFL team stats, 
# uses a pre-trained logistic regression model to predict whether the team had 
# a winning or losing record, appends the prediction and input data to an SQL 
# database, and returns the prediction as JSON.
#
# Record of Revisions (Date | Author | Change):
# 08/27/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
from flask import Flask, request, jsonify
import numpy as np
import joblib
import sqlite3 as sql
import pandas as pd

app = Flask(__name__)

# Load the trained model
def load_model(path="nflModel.joblib"): return joblib.load(path)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Assuming 'data' contains features in a list or dict
        features = np.array(data['features']).reshape(1, -1) # Reshape for a single sample
        prediction = model.predict(features)[0] # Get the prediction (0 or 1)

        sqlData = np.append(features, prediction).reshape(1, -1) # Append prediction to features
        sqlData = pd.DataFrame(sqlData, columns=['PF', 'PA', 'PD', 'SoS', 'winningRecord']) # Create DataFrame
        
        conn = sql.connect('NFL.db') # Connect to database
        sqlData.to_sql('stats', conn, if_exists='append', index = False) # Append test data to SQL table
        conn.close() # Close connection

        prediction = "Winning" if prediction == 1 else "Losing" # Convert to readable format
        return jsonify({'prediction': prediction})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True) # Set debug=False for production