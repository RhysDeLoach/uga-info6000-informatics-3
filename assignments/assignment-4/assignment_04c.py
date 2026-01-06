###############################################################################
# File Name: assignment_04c.py
#
# Description: This program implements a Flask API that loads a trained 
# DenseNet121 model for grayscale CT scan images, preprocesses uploaded images, 
# performs inference to classify them as COVID or normal, and returns the 
# prediction as JSON.
#
# Record of Revisions (Date | Author | Change):
# 09/12/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
from flask import Flask, request, jsonify
import numpy as np
import joblib
import sqlite3 as sql
import pandas as pd
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

app = Flask(__name__)

# Set device
device = torch.device('mps' if torch.mps.is_available() else 'cpu')

# Load the trained model
model = models.densenet121(weights = None) 
model.features.conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False).to(device) # Change Input Layer to Accept Grayscale
stateDict = torch.load("output/covidCTModel.pth", map_location=device)
model.load_state_dict(stateDict)
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1), # Convert to Grayscale
    transforms.ToTensor() # Convert to Tensor
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = Image.open(request.files['image'])

        model.eval() # Set model to evaluation mode

        with torch.inference_mode(): 
            image = transform(img).unsqueeze(0).to(device) # Preprocess image and add batch dimension
            imagePred = model(image.to(device)) # Make a prediction on image with an extra dimension and send it to the target device

        imagePredProb = torch.softmax(imagePred, dim=1) # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)

        prediction = torch.argmax(imagePredProb, dim=1) # Convert prediction probabilities -> prediction labels
        print(prediction)
        prediction = "covid" if prediction == 0 else "normal" # Convert to readable format
        print(prediction)
        return jsonify({'prediction': prediction})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True) # Set debug=False for production
