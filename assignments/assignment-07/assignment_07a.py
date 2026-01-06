###############################################################################
# File Name: assignment_07a.py
#
# Description: This script processes the UrbanSound8K audio dataset by 
# extracting features (MFCCs, spectral centroid, bandwidth, and 
# zero-crossing rate), normalizes them, reduces dimensionality using Truncated 
# SVD, and trains a Random Forest classifier to predict audio classes. It 
# evaluates accuracy on a test split and saves the trained model, scaler, 
# and SVD transform for later use.
#
# Note: Dataset not included due to GitHub size constraints.
#
# Record of Revisions (Date | Author | Change):
# 10/16/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libraries
import numpy as np
import pandas as pd
import librosa
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load metadata
metadata = pd.read_csv('UrbanSound_Filtered/filtered_metadata.csv')

# Define feature columns
feature_names = [f'mfcc{i}' for i in range(1, 21)] + ['centroid', 'bandwidth', 'zeroCrossing']
features = pd.DataFrame(columns=feature_names)
labels = []

# Feature extraction loop
for index in range(len(metadata)):
    file = metadata['slice_file_name'][index]
    labels.append(metadata['class'][index]) 
    audioPath = f'UrbanSound_Filtered/audio/{file}'
    y, sr = librosa.load(audioPath)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    
    feats = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(spec_centroid),
        np.mean(spec_bandwidth),
        np.mean(zcr),
    ])
    
    features.loc[len(features)] = feats

# Convert labels to array
labels = np.array(labels)

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Reduce dimensionality with Truncated SVD
svd = TruncatedSVD(n_components=20, random_state=42)
X_reduced = svd.fit_transform(features_scaled)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_reduced, labels, test_size=0.2, random_state=42)

# Train
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save models
joblib.dump(rf, 'output/random_forest_model.pkl')
joblib.dump(svd, 'output/svd_transform.pkl')
joblib.dump(scaler, 'output/scaler.pkl')