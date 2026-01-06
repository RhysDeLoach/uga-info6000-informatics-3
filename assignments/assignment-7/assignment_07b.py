###############################################################################
# File Name: assignment_07b.py
#
# Description: This script loads the previously trained Random Forest 
# classifier, scaler, and SVD transformer to make predictions on new audio 
# files. It extracts the same features (MFCCs, spectral centroid, bandwidth, 
# zero-crossing rate) from each audio file, preprocesses them, and prints the 
# predicted class for each file.
#
# Record of Revisions (Date | Author | Change):
# 10/16/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import joblib
import numpy as np
import librosa
import os
import warnings

warnings.filterwarnings( # Ignores feature name warning to declutter prints
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)

# Load models
rf = joblib.load('output/random_forest_model.pkl')
svd = joblib.load('output/svd_transform.pkl')
scaler = joblib.load('output/scaler.pkl')

audioPaths = ['data/valid/power_drill.mp3','data/valid/dog_bark.mp3','data/valid/street_carnival.mp3']

for path in audioPaths:
    y, sr = librosa.load(path)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    feats = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(spec_centroid),
        np.mean(spec_bandwidth),
        np.mean(zcr),
    ]).reshape(1, -1)

    # Preprocessing
    feats_scaled = scaler.transform(feats)
    feats_reduced = svd.transform(feats_scaled)

    # Predict class
    prediction = rf.predict(feats_reduced)
    file = os.path.basename(path)
    print(f'Actual class: {os.path.splitext(file)[0]}')
    print(f"Predicted class: {prediction[0]}\n\n")
