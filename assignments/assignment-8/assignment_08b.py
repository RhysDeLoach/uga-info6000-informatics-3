###############################################################################
# File Name: assignment_08b.py
#
# Description: This script performs anomaly detection on equipment data using 
# DBSCAN. It first reduces the features to 2D with PCA, fits a DBSCAN 
# clustering model, and then plots the results, highlighting anomalies in red 
# and normal points in blue.
#
# Record of Revisions (Date | Author | Change):
# 11/02/2025 | Rhys DeLoach | Initial creation
###############################################################################

# Import Libaries
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv('data/equipment_anomaly_data.csv') # Load data to df

X = data.drop(columns=['faulty','equipment','location']) # Drop labels and equipment/location because DBscan is distance based

pca = PCA(n_components=2) # Initialize 2D PCA
XReduced = pca.fit_transform(X) # Reduce features using PCA

dbscan = DBSCAN(eps=2.5, min_samples=9) # Intialize DBScan
labels = dbscan.fit_predict(XReduced) # Fit PCA features to DBScan

# Plot Data
plt.figure(figsize=(8,6))
plt.scatter(XReduced[labels==-1,0], XReduced[labels==-1,1], c='red', label='Anomaly')
plt.scatter(XReduced[labels!=-1,0], XReduced[labels!=-1,1], c='blue', alpha=0.6, label='Normal')
plt.title('DBSCAN Anomaly Detection')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

