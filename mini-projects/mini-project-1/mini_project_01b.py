###############################################################################
# File Name: mini_project_01b.py
#
# Description: This script creates a Flask API that exposes the fine-tuned BERT 
# medical text classifier. Users can send POST requests with a JSON containing 
# symptoms, and the API returns the predicted diagnosis using the Hugging Face 
# pipeline.
#
# Note: Model not included due to GitHub size constraints.
#
# Record of Revisions (Date | Author | Change):
# 09/21/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import torch
from torchvision import models
import torch.nn as nn
from transformers import pipeline, AutoTokenizer,AutoModelForSequenceClassification
import pickle


app = Flask(__name__)

# Set device
device = torch.device('mps' if torch.mps.is_available() else 'cpu')

# Initialize Model
model_path = "output/bert/checkpoint-540/" 
task = "text-classification"

# Recreate Encode Labels
with open("output/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

id2label = {i: label for i, label in enumerate(le.classes_)}

# Initialize the pipeline and if the model was trained with a specifc tokenizer use the same
# Load the model with the custom id2label mapping
model = AutoModelForSequenceClassification.from_pretrained(model_path, id2label=id2label, num_labels=24)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Pipeline
classifier = pipeline(task, model=model, tokenizer=tokenizer)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symptoms = request.json['symptoms']
        diagnosis = classifier(symptoms)[0]['label'] # Get the predicted label from the pipeline output
        return jsonify({'prediction': diagnosis})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True) # Set debug=False for production